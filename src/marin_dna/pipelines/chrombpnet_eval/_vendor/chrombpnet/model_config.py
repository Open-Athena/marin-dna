# Author: Lei Xiong <jsxlei@gmail.com>
from .parse_utils import add_argparse_args, from_argparse_args, parse_argparser, get_init_arguments_and_types
from argparse import ArgumentParser, Namespace
from typing import Any, List, Tuple, Union

class BaseConfig:

    model_type = "base"

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any):
        """Extends existing argparse by default `LightningDataModule` attributes.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningDataModule.add_argparse_args(parser)
        """
        return add_argparse_args(cls, parent_parser, **kwargs)


    @classmethod
    def from_argparse_args(
        cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ) -> Union["pl.LightningDataModule", "pl.Trainer"]:
        """Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid DataModule arguments.

        Example::

            module = LightningDataModule.from_argparse_args(args)
        """
        return from_argparse_args(cls, args, **kwargs)


    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the DataModule signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        return get_init_arguments_and_types(cls)
    


class ChromBPNetConfig(BaseConfig):
    r"""
    
    ```"""

    model_type = "chrombpnet"

    def __init__(
        self,

        out_dim:int=1000,
        n_filters:int=512, 
        n_layers:int=8, 
        conv1_kernel_size:int=21,
        profile_kernel_size:int=75,
        n_outputs:int=1, 
        n_control_tracks:int=0, 
        profile_output_bias:int=True, 
        count_output_bias:int=True, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.profile_output_bias = profile_output_bias
        self.count_output_bias = count_output_bias



class ArsenalChromBPNetConfig(BaseConfig):
    r"""
    
    ```"""

    model_type = "arsenal-chrombpnet"

    def __init__(
        self,
        input_len:int=2114,
        out_dim:int=1000,
        n_filters:int=512, 
        n_layers:int=8, 
        conv1_kernel_size:int=21,
        profile_kernel_size:int=75,
        n_outputs:int=1, 
        n_control_tracks:int=0, 
        profile_output_bias:int=True, 
        count_output_bias:int=True, 
        finetune_arsenal: bool = False,
        arsenal_output_type: str = "embedding",
        input_embedding_dim: int = 512,
        arsenal_input_size: int = 2114,
        num_layers_avg: int = 4,
        category : int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_len = input_len
        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.n_outputs = n_outputs
        self.n_control_tracks = n_control_tracks
        self.profile_output_bias = profile_output_bias
        self.count_output_bias = count_output_bias
        self.finetune_arsenal = finetune_arsenal
        self.arsenal_output_type = arsenal_output_type
        self.input_embedding_dim = input_embedding_dim
        self.arsenal_input_size = arsenal_input_size
        self.num_layers_avg = num_layers_avg
        self.category = category