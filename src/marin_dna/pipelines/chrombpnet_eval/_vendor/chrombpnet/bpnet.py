# Author: Lei Xiong <jsxlei@gmail.com>

import torch
import torch.nn.functional as F

# adapted from BPNet in bpnet-lite, credit goes to Jacob Schreiber <jmschreiber91@gmail.com>


class BPNet(torch.nn.Module):
    """A basic BPNet model with stranded profile and total count prediction.

    This is a reference implementation for BPNet. The model takes in
    one-hot encoded sequence, runs it through: 

    (1) a single wide convolution operation 

    THEN 

    (2) a user-defined number of dilated residual convolutions

    THEN

    (3a) profile predictions done using a very wide convolution layer 
    that also takes in stranded control tracks 

    AND

    (3b) total count prediction done using an average pooling on the output
    from 2 followed by concatenation with the log1p of the sum of the
    stranded control tracks and then run through a dense layer.

    This implementation differs from the original BPNet implementation in
    two ways:


    (1) The model concatenates stranded control tracks for profile
    prediction as opposed to adding the two strands together and also then
    smoothing that track 

    (2) The control input for the count prediction task is the log1p of
    the strand-wise sum of the control tracks, as opposed to the raw
    counts themselves.

    (3) A single log softmax is applied across both strands such that
    the logsumexp of both strands together is 0. Put another way, the
    two strands are concatenated together, a log softmax is applied,
    and the MNLL loss is calculated on the concatenation. 

    (4) The count prediction task is predicting the total counts across
    both strands. The counts are then distributed across strands according
    to the single log softmax from 3.

    Note that this model is also used as components in the ChromBPNet model,
    as both the bias model and the accessibility model. Both components are
    the same BPNet architecture but trained on different loci.


    Parameters
    ----------
    n_filters: int, optional
        The number of filters to use per convolution. Default is 64.

    n_layers: int, optional
        The number of dilated residual layers to include in the model.
        Default is 8.

    n_outputs: int, optional
        The number of profile outputs from the model. Generally either 1 or 2 
        depending on if the data is unstranded or stranded. Default is 2.

    n_control_tracks: int, optional
        The number of control tracks to feed into the model. When predicting
        TFs, this is usually 2. When predicting accessibility, this is usualy
        0. When 0, this input is removed from the model. Default is 2.

    alpha: float, optional
        The weight to put on the count loss.

    profile_output_bias: bool, optional
        Whether to include a bias term in the final profile convolution.
        Removing this term can help with attribution stability and will usually
        not affect performance. Default is True.

    count_output_bias: bool, optional
        Whether to include a bias term in the linear layer used to predict
        counts. Removing this term can help with attribution stability but
        may affect performance. Default is True.

    name: str or None, optional
        The name to save the model to during training.

    trimming: int or None, optional
        The amount to trim from both sides of the input window to get the
        output window. This value is removed from both sides, so the total
        number of positions removed is 2*trimming.

    verbose: bool, optional
        Whether to display statistics during training. Setting this to False
        will still save the file at the end, but does not print anything to
        screen during training. Default is True.
    """

    def __init__(
        self, 
        out_dim=1000,
        n_filters=64, 
        n_layers=8, 
        rconvs_kernel_size=3,
        conv1_kernel_size=21,
        profile_kernel_size=75,
        n_outputs=1, 
        n_control_tracks=0, 
        profile_output_bias=True, 
        count_output_bias=True, 
        name=None, 
        verbose=False,
        custom_dilations=False
    ):
        super().__init__()

        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.verbose = verbose
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)

        # first convolution without dilation
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=conv1_kernel_size, padding='valid')
        self.irelu = torch.nn.ReLU()

        if custom_dilations:
            rconvs_kernel_size = 5
            dilation_sched = [2,8,32,64,128]
            # dilation_sched = [8,16,32,64,128,256]
            self.rconvs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_filters, n_filters, kernel_size=rconvs_kernel_size, padding='valid', 
                    dilation=dilation_sched[i-1]) for i in range(1, self.n_layers+1)
            ])

        # residual dilated convolutions
        else:
            self.rconvs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_filters, n_filters, kernel_size=rconvs_kernel_size, padding='valid', 
                    dilation=2**i) for i in range(1, self.n_layers+1)
            ])

        self.rrelus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

        # profile prediction
        self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
            kernel_size=profile_kernel_size, padding='valid', bias=profile_output_bias)
        
        # count prediction
        n_count_control = 1 if n_control_tracks > 0 else 0
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
            bias=count_output_bias)


    def forward(self, x, x_ctl=None):
        """A forward pass of the model.

        This method takes in a nucleotide sequence x, a corresponding
        per-position value from a control track, and a per-locus value
        from the control track and makes predictions for the profile 
        and for the counts. This per-locus value is usually the
        log(sum(X_ctl_profile)+1) when the control is an experimental
        read track but can also be the -output from another model.

        Parameters
        ----------
        x: torch.tensor, shape=(batch_size, 4, length)
            The one-hot encoded batch of sequences.

        X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
            A value representing the signal of the control at each position in 
            the sequence. If no controls, pass in None. Default is None.

        Returns
        -------
        pred_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
            The output predictions for each strand trimmed to the output
            length.
        pred_count: torch.tensor, shape=(batch_size, 1)
        """
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1)
        x = self.get_embs_after_crop(x)

        if self.verbose: print(f'trunk shape: {x.shape}')

        if x_ctl is not None:
            crop_size = (x_ctl.shape[2] - x.shape[2]) // 2
            if self.verbose: print(f'crop_size: {crop_size}')
            if crop_size > 0:
                x_ctl = x_ctl[:, :, crop_size:-crop_size]
            else:
                x_ctl = F.pad(x_ctl, (-crop_size, -crop_size))

        pred_profile = self.profile_head(x, x_ctl=x_ctl) # before log_softmax
        pred_count = self.count_head(x, x_ctl=x_ctl) #.squeeze(-1) # (batch_size, 1)

        return pred_profile, pred_count
    

    def get_embs_after_crop(self, x):
        x = self.irelu(self.iconv(x))
        for i in range(self.n_layers):
            conv_x = self.rrelus[i](self.rconvs[i](x))
            crop_len = (x.shape[2] - conv_x.shape[2]) // 2
            if crop_len > 0:
                x = x[:, :, crop_len:-crop_len]
            x = torch.add(x, conv_x)
        
        return x
    
    def profile_head(self, x, x_ctl=None):
        """
        Profile head of the model.
        output: (batch_size, n_outputs, out_window)
        """

        if x_ctl is not None:
            x = torch.cat([x, x_ctl], dim=1)

        pred_profile = self.fconv(x)

        crop_size = (pred_profile.shape[2] - self.out_dim) // 2
        if crop_size > 0:
            pred_profile = pred_profile[:, :, crop_size:-crop_size]
        else:
            pred_profile = F.pad(pred_profile, (-crop_size, -crop_size)) # pad if out_window > in_window
        
        return pred_profile

        
    def count_head(self, x, x_ctl=None):

        # pred_count = torch.mean(x, dim=2)
        pred_count = self.global_avg_pool(x).squeeze(-1)
        if x_ctl is not None:
            x_ctl = torch.sum(x_ctl, dim=(1, 2)).unsqueeze(-1)
            pred_count = torch.cat([pred_count, torch.log1p(x_ctl)], dim=-1)
        pred_count = self.linear(pred_count)
        return pred_count
    
    
    @classmethod
    def from_keras(cls, filename, name='chrombpnet'):
        """Loads a model from ChromBPNet TensorFlow format.
    
        This method will load one of the components of a ChromBPNet model
        from TensorFlow format. Note that a full ChromBPNet model is made up
        of an accessibility model and a bias model and that this will load
        one of the two. Use `ChromBPNet.from_chrombpnet` to end up with the
        entire ChromBPNet model.


        Parameters
        ----------
        filename: str
            The name of the h5 file that stores the trained model parameters.


        Returns
        -------
        model: BPNet
            A BPNet model compatible with this repository in PyTorch.
        """
        if filename.endswith('.h5'):
            import h5py

            h5 = h5py.File(filename, "r")
            w = h5['model_weights']
        else:
            import os
            os.system('conda activate chrombpnet')
            import tensorflow as tf

            model = tf.keras.models.load_model(filename)
            w = model.get_weights()
            os.system('conda deactivate')

        if 'bpnet_1conv' in w.keys():
            prefix = ""
        else:
            prefix = "wo_bias_"
        # print(f"Loading {name} model from {filename}", flush=True)

        namer = lambda prefix, suffix: '{0}{1}/{0}{1}'.format(prefix, suffix)
        k, b = 'kernel:0', 'bias:0'

        n_layers = 0
        for layer_name in w.keys():
            try:
                idx = int(layer_name.split("_")[-1].replace("conv", ""))
                n_layers = max(n_layers, idx)
            except:
                pass

        name = namer(prefix, "bpnet_1conv")
        n_filters = w[name][k].shape[2]

        model = BPNet(n_layers=n_layers, n_filters=n_filters, n_outputs=1,
            n_control_tracks=0)

        convert_w = lambda x: torch.nn.Parameter(torch.tensor(
            x[:]).permute(2, 1, 0))
        convert_b = lambda x: torch.nn.Parameter(torch.tensor(x[:]))

        iname = namer(prefix, 'bpnet_1st_conv')

        model.iconv.weight = convert_w(w[iname][k])
        model.iconv.bias = convert_b(w[iname][b])

        for i in range(1, n_layers+1):
            lname = namer(prefix, 'bpnet_{}conv'.format(i))

            model.rconvs[i-1].weight = convert_w(w[lname][k])
            model.rconvs[i-1].bias = convert_b(w[lname][b])

        prefix = prefix + "bpnet_" if prefix != "" else ""

        fname = namer(prefix, 'prof_out_precrop')
        model.fconv.weight = convert_w(w[fname][k])
        model.fconv.bias = convert_b(w[fname][b])

        name = namer(prefix, "logcount_predictions")
        model.linear.weight = torch.nn.Parameter(torch.tensor(w[name][k][:].T))
        model.linear.bias = convert_b(w[name][b])
        return model

    @classmethod
    def to_keras(cls, filename):
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Conv1D, Cropping1D, Add, GlobalAveragePooling1D, Dense, ReLU
        from tensorflow.keras.models import Model
        import numpy as np

        def build_keras_bpnet(sequence_len, n_dil_layers, filters, conv1_kernel_size, profile_kernel_size, num_tasks, out_pred_len):
            inp = Input(shape=(sequence_len, 4), name='sequence')

            # First convolution without dilation
            x = Conv1D(filters, kernel_size=conv1_kernel_size, padding='valid', activation='relu', name='conv1')(inp)
            
            for i in range(1, n_dil_layers + 1):
                conv_x = Conv1D(filters, kernel_size=3, padding='valid', activation='relu', dilation_rate=2**i, name=f'dilated_conv_{i}')(x)
                x_len = x.shape[1]
                conv_x_len = conv_x.shape[1]
                assert (x_len - conv_x_len) % 2 == 0
                crop_len = (x_len - conv_x_len) // 2
                x = Cropping1D(cropping=(crop_len, crop_len), name=f'crop_{i}')(x)
                x = Add(name=f'add_{i}')([conv_x, x])

            # Profile prediction
            prof_out_precrop = Conv1D(num_tasks, kernel_size=profile_kernel_size, padding='valid', name='profile_conv')(x)
            cropsize = (prof_out_precrop.shape[1] - out_pred_len) // 2
            prof = Cropping1D(cropping=(cropsize, cropsize), name='profile_crop')(prof_out_precrop)
            profile_out = tf.reshape(prof, (tf.shape(prof)[0], -1), name='profile_out')

            # Counts prediction
            gap_combined_conv = GlobalAveragePooling1D(name='gap')(x)
            count_out = Dense(num_tasks, name='count_out')(gap_combined_conv)

            model = Model(inputs=[inp], outputs=[profile_out, count_out])
            return model

        # # Build the Keras model
        keras_model = build_keras_bpnet(sequence_len=2114, n_dil_layers=8, filters=512, conv1_kernel_size=21, profile_kernel_size=75, num_tasks=1, out_pred_len=1000)

        # # Summary of the model
        # keras_model.summary()

        def transfer_weights(pytorch_model, keras_model):
            # Transfer conv1 weights
            keras_model.get_layer('conv1').set_weights([
                pytorch_model.conv1.weight.detach().numpy().transpose(2, 1, 0),
                pytorch_model.conv1.bias.detach().numpy()
            ])

            # Transfer dilated conv layers' weights
            for i, layer in enumerate(pytorch_model.dilated_convs):
                keras_layer = keras_model.get_layer(f'dilated_conv_{i + 1}')
                keras_layer.set_weights([
                    layer.weight.detach().numpy().transpose(2, 1, 0),
                    layer.bias.detach().numpy()
                ])

            # Transfer profile conv weights
            keras_model.get_layer('profile_conv').set_weights([
                pytorch_model.profile_conv.weight.detach().numpy().transpose(2, 1, 0),
                pytorch_model.profile_conv.bias.detach().numpy()
            ])

            # Transfer dense layer weights
            keras_model.get_layer('count_out').set_weights([
                pytorch_model.dense.weight.detach().numpy().transpose(1, 0),
                pytorch_model.dense.bias.detach().numpy()
            ])

        transfer_weights(self, keras_model)
        keras_model.save(filename)





class DoubleBPNet(BPNet):

    def __init__(
        self, 
        out_dim=1000,
        n_filters=64, 
        n_layers=8, 
        rconvs_kernel_size=3,
        conv1_kernel_size=21,
        profile_kernel_size=75,
        n_outputs=1, 
        n_control_tracks=0,
        input_sizes=[4,768], 
        profile_output_bias=True, 
        count_output_bias=True, 
        name=None, 
        verbose=False,
        dropout=False
    ):
        super().__init__()

        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.verbose = verbose
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
        self.input_sizes = input_sizes

        # first convolution without dilation
        self.iconv1 = torch.nn.Conv1d(input_sizes[0], n_filters, kernel_size=conv1_kernel_size, padding='valid')
        self.irelu1 = torch.nn.ReLU()

        self.iconv2 = torch.nn.Conv1d(input_sizes[1], n_filters, kernel_size=conv1_kernel_size, padding='valid')
        self.irelu2 = torch.nn.ReLU()


        # residual dilated convolutions
        self.rconvs1 = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=rconvs_kernel_size, padding='valid', 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])

        self.rrelus1 = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

        self.rconvs2 = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=rconvs_kernel_size, padding='valid', 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])

        self.rrelus2 = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])


        # profile prediction
        self.fconv = torch.nn.Conv1d(n_filters * 2, n_outputs, 
            kernel_size=profile_kernel_size, padding='valid', bias=profile_output_bias)
        
        # count prediction
        n_count_control = 1 if n_control_tracks > 0 else 0
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(n_filters * 2, 1, 
            bias=count_output_bias)

    def get_embs_after_crop(self, x):
        assert x.shape[1] == self.input_sizes[0] + self.input_sizes[1]
        x_onehot = x[:,:self.input_sizes[0],:]
        x_lm = x[:,self.input_sizes[0]:,:]

        x_onehot = self.irelu1(self.iconv1(x_onehot))
        for i in range(self.n_layers):
            conv_x = self.rrelus1[i](self.rconvs1[i](x_onehot))
            crop_len = (x_onehot.shape[2] - conv_x.shape[2]) // 2
            if crop_len > 0:
                x_onehot = x_onehot[:, :, crop_len:-crop_len]
            x_onehot = torch.add(x_onehot, conv_x)
        
        x_lm = self.irelu2(self.iconv2(x_lm))
        for i in range(self.n_layers):
            conv_x = self.rrelus2[i](self.rconvs2[i](x_lm))
            crop_len = (x_lm.shape[2] - conv_x.shape[2]) // 2
            if crop_len > 0:
                x_lm = x_lm[:, :, crop_len:-crop_len]
            x_lm = torch.add(x_lm, conv_x)

        x = torch.cat([x_onehot, x_lm], dim=1)
        return x



class ConvBlock(torch.nn.Module):
    """
    Basic convolutional block.
    Consists of a convolutional layer, a max pooling layer and a dropout layer.
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        pool_size: int, 
        dropout: float
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.mp = torch.nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.do = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv(x))  # (batch_size, out_channels, seq_len)
        x = self.mp(x)  # (batch_size, out_channels, seq_len // pool_size)
        x = self.do(x)  # (batch_size, out_channels, seq_len // pool_size)
        return x
    
class DreamRNN(BPNet):

    def __init__(
        self, 
        out_dim=1000,
        n_filters=64, 
        n_layers=8, 
        rconvs_kernel_size=3,
        conv1_kernel_size=21,
        profile_kernel_size=75,
        n_outputs=1, 
        n_control_tracks=0, 
        profile_output_bias=True, 
        count_output_bias=True, 
        name=None, 
        verbose=False,
        dropout=False
    ):
        super().__init__()
        in_channels = 512
        out_channels = 320
        lstm_hidden_channels = 320
        kernel_sizes = [9, 15]
        pool_size = 1
        dropout1 = 0.2
        dropout2 = 0.2
        self.do = torch.nn.Dropout(dropout2)
        self.lstm = torch.nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_channels, batch_first=True, bidirectional=True)
        self.conv_list1 = torch.nn.ModuleList([
            ConvBlock(8, in_channels // len(kernel_sizes), k, pool_size, dropout1) for k in kernel_sizes
        ])
        self.conv_list2 = torch.nn.ModuleList([
            ConvBlock(2 * lstm_hidden_channels, out_channels // len(kernel_sizes), k, pool_size, dropout1) for k in kernel_sizes
        ])

        self.fconv = torch.nn.Conv1d(out_channels, n_outputs, 
            kernel_size=profile_kernel_size, padding='valid', bias=profile_output_bias)
        
        # count prediction
        self.linear = torch.nn.Linear(out_channels, 1, 
            bias=count_output_bias)


    def get_embs_after_crop(self, x):
        conv_outputs = [conv(x) for conv in self.conv_list1]  # [(batch_size, each_out_channels, seq_len // pool_size), ...]
        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, out_channels, seq_len // pool_size)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, in_channels)
        x, _ = self.lstm(x)  # (batch_size, seq_len, 2 * lstm_hidden_channels)
        x = x.permute(0, 2, 1)  # (batch_size, 2 * lstm_hidden_channels, seq_len)
        conv_outputs = [conv(x) for conv in self.conv_list2]  # [(batch_size, each_conv_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, conv_out_channels, seq_len // pool_size)
        x = self.do(x)

        return x





