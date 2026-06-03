# Author: Lei Xiong <jsxlei@gmail.com>

import numpy as np 
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import jensenshannon
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from .metrics_utils import * 
from .data_utils import write_bigwig, read_chrom_sizes, expand_3col_to_10col
from .genome import hg38_datasets   

plt.rcParams["figure.figsize"]=10,5
font = {'weight' : 'bold',
        'size'   : 10}
matplotlib.rc('font', **font)

import os
import h5py
import pandas as pd
import json

def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def get_regions(regions, seqlen, regions_used=None):
    # regions file is assumed to be centered at summit (2nd + 10th column)
    # it is adjusted to be of length seqlen centered at summit

    assert(seqlen%2==0)

    #with open(regions_file) as r:
    #    regions = [x.strip().split('\t') for x in r]

    # regions = pd.read_csv(regions_file,sep='\t',header=None)
    #print(regions)
    if regions_used is None:
        regions = [[x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])] for x in np.array(regions.values)]
    else:
        regions = [[x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])] for x in np.array(regions.values)[regions_used]]

    return regions


def write_predictions_h5py(profile, logcts, coords, out_dir='.'):
    # open h5 file for writing predictions
    os.makedirs(out_dir, exist_ok=True)
    output_h5_fname = os.path.join(out_dir, "predictions.h5")
    h5_file = h5py.File(output_h5_fname, "w")
    # create groups
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")

    num_examples=len(coords)

    coords_chrom_dset =  [str(coords[i][0]) for i in range(num_examples)]
    coords_center_dset =  [int(coords[i][1]) for i in range(num_examples)]
    coords_peak_dset =  [int(coords[i][3]) for i in range(num_examples)]

    dt = h5py.special_dtype(vlen=str)

    # create the "coords" group datasets
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom", data=np.array(coords_chrom_dset, dtype=dt),
        dtype=dt, compression="gzip")
    coords_start_dset = coord_group.create_dataset(
        "coords_center", data=coords_center_dset, dtype=int, compression="gzip")
    coords_end_dset = coord_group.create_dataset(
        "coords_peak", data=coords_peak_dset, dtype=int, compression="gzip")

    # create the "predictions" group datasets
    profs_dset = pred_group.create_dataset(
        "profs",
        data=profile,
        dtype=float, compression="gzip")
    logcounts_dset = pred_group.create_dataset(
        "logcounts", data=logcts,
        dtype=float, compression="gzip")

    # close hdf5 file
    h5_file.close()


def save_predictions(output, regions, chrom_sizes, out_dir='./'):
    """
    Save the predictions to an HDF5 file and write regions to a CSV file.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(chrom_sizes) as f:
        gs = [x.strip().split('\t') for x in f]
    gs = [(x[0], int(x[1])) for x in gs if len(x)==2]
    if regions.shape[1] < 10:
        regions = expand_3col_to_10col(regions)
    # gs = read_chrom_sizes(chrom_sizes); print(gs)

    seqlen = 1000
    regions_array = [[x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])] for x in np.array(regions.values)]

    # parse output
    parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}

    data = softmax(parsed_output['pred_profile']) * (np.expand_dims(np.exp(parsed_output['pred_count']), axis=1))

    write_bigwig(
        data,
        regions_array,
        gs,
        os.path.join(out_dir, "pred.bw"),
        outstats_file=None,
        debug_chr=None,
        use_tqdm=True)

    # save predictions into h5py file
    # write_predictions_h5py(parsed_output['pred_profile'], parsed_output['pred_count'], regions_array, out_dir)

    return

def load_output_to_regions(output, regions, out_dir='./'):
    """
    Load the output to regions
    """
    regions = regions.reset_index(drop=True).copy()
    os.makedirs(out_dir, exist_ok=True)
    parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}
    if 'is_peak' in regions.columns:
        regions['is_peak'] = regions['is_peak'].astype(int)
    regions['pred_count'] = parsed_output['pred_count']
    regions['true_count'] = parsed_output['true_count']
    regions.to_csv(os.path.join(out_dir, 'regions.csv'), sep='\t', index=False)
    return regions, parsed_output

def compare_with_observed(regions, parsed_output, out_dir='./', tag='all_regions'):
    """
    """
    os.makedirs(out_dir, exist_ok=True)
    # chrom_sizes = os.path.expanduser('~/.cache/regnet/hg38.chrom.sizes')
    # gs = bigwig_helper.read_chrom_sizes(chrom_sizes) #list(chrom_sizes.items())

    # #gs = bigwig_helper.read_chrom_sizes(chrom_sizes)
    # seqlen = 1000
    # regions_array = [[x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])] for x in np.array(regions.values)]
    
    # # parse output
    # parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}

    # data = softmax(parsed_output['pred_profile']) * (np.expand_dims(np.exp(parsed_output['pred_count']),axis=1))

    # bigwig_helper.write_bigwig(
    #                     data, 
    #                     regions_array, 
    #                     gs, 
    #                     os.path.join(out_dir, "pred.bw"), 
    #                     outstats_file=None, 
    #                     debug_chr=None, 
    #                     use_tqdm=True)

    # save predictions into h5py file
    # write_predictions_h5py(parsed_output['pred_profile_prob'], parsed_output['true_profile'], coords, out_dir)

    # regions = pd.DataFrame(coords, columns=['chrom', 'summit', 'forward_reverse', 'is_peak'])
    # regions['is_peak'] = regions['is_peak'].astype(int)
    # regions['pred_count'] = parsed_output['pred_count']
    # regions['true_count'] = parsed_output['true_count']
    # regions.to_csv(os.path.join(out_dir, 'regions.csv'), sep='\t', index=False)

    # print(peak_regions.head())

    metrics_dictionary={}
    metrics_dictionary["counts_metrics"] = {}
    # save count metrics
    spearman_cor, pearson_cor, mse = counts_metrics(regions['true_count'], regions['pred_count'], os.path.join(out_dir, 'all_regions'))
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"] = {}
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["spearmanr"] = spearman_cor
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["pearsonr"] = pearson_cor
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["mse"] = mse

    metrics_dictionary["profile_metrics"] = {}
    mnll_pw, mnll_norm, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, mnll_rnd, mnll_rnd_norm = profile_metrics(parsed_output['true_profile'], softmax(parsed_output['pred_profile']))
    plot_histogram(jsd_pw, jsd_rnd, os.path.join(out_dir, 'all_regions_jsd'), '')
    metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"] = {}
    metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"]["median_jsd"] = np.nanmedian(jsd_pw)        
    metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)
    

    if 'is_peak' in regions.columns:
        peak_regions = regions[regions['is_peak']==1].copy()
        peak_index = peak_regions.index
        peak_regions = peak_regions.reset_index(drop=True)
        print('peak_regions', peak_regions.head())

        spearman_cor, pearson_cor, mse = counts_metrics(peak_regions['true_count'], peak_regions['pred_count'], os.path.join(out_dir, 'peaks'))
        metrics_dictionary["counts_metrics"]["peaks"] = {}
        metrics_dictionary["counts_metrics"]["peaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["peaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["peaks"]["mse"] = mse

        mnll_pw, mnll_norm, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, mnll_rnd, mnll_rnd_norm = profile_metrics(parsed_output['true_profile'][peak_index], softmax(parsed_output['pred_profile'])[peak_index])
        plot_histogram(jsd_pw, jsd_rnd, os.path.join(out_dir, 'peaks_jsd'), '')
        metrics_dictionary["profile_metrics"]["peaks"] = {}
        metrics_dictionary["profile_metrics"]["peaks"]["median_jsd"] = np.nanmedian(jsd_pw)        
        metrics_dictionary["profile_metrics"]["peaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)

        nonpeak_regions = regions[regions['is_peak']==0].copy()
        nonpeak_index = nonpeak_regions.index
        nonpeak_regions = nonpeak_regions.reset_index(drop=True)
        print('nonpeak_regions', nonpeak_regions.head())

        spearman_cor, pearson_cor, mse = counts_metrics(nonpeak_regions['true_count'], nonpeak_regions['pred_count'], os.path.join(out_dir, 'nonpeaks'))
        metrics_dictionary["counts_metrics"]["nonpeaks"] = {}
        metrics_dictionary["counts_metrics"]["nonpeaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["nonpeaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["nonpeaks"]["mse"] = mse
        mnll_pw, mnll_norm, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, mnll_rnd, mnll_rnd_norm = profile_metrics(parsed_output['true_profile'][nonpeak_index], softmax(parsed_output['pred_profile'])[nonpeak_index])
        plot_histogram(jsd_pw, jsd_rnd, os.path.join(out_dir, 'nonpeaks_jsd'), '')
        metrics_dictionary["profile_metrics"]["nonpeaks"] = {}
        metrics_dictionary["profile_metrics"]["nonpeaks"]["median_jsd"] = np.nanmedian(jsd_pw)        
        metrics_dictionary["profile_metrics"]["nonpeaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)

    print(json.dumps(metrics_dictionary, indent=4, default=lambda o: float(o)))

    # os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'metrics_{tag}.json'), 'w') as fp:
        json.dump(metrics_dictionary, fp,  indent=4, default=lambda x: float(x))
    return metrics_dictionary

def counts_metrics(labels,preds,outf=None,title='',fontsize=20, xlab='Log Count Labels', ylab='Log Count Predictions'):
    '''
    Get count metrics
    '''
    spearman_cor=spearmanr(labels,preds)[0]
    pearson_cor=pearsonr(labels,preds)[0]  
    mse=((labels - preds)**2).mean(axis=0)

    #print("spearman:"+str(spearman_cor))
    #print("pearson:"+str(pearson_cor))
    #print("mse:"+str(mse))

    plt.rcParams["figure.figsize"]=8,8
    # fig=plt.figure() 
    ax = density_scatter(labels,
                    preds,
                    xlab=xlab,
                    ylab=ylab,
                    fontsize=fontsize
                    )
    plt.suptitle("count: spearman R="+str(round(spearman_cor,3))+"\nPearson R="+str(round(pearson_cor,3))+"\nmse="+str(round(mse,3)), y=0.9, fontsize=20)
    # plt.legend(loc='best')

    if outf is not None:
        plt.savefig(outf+'.counts_pearsonr.png',format='png',dpi=300)
    plt.show()
    plt.close()
    
    return spearman_cor, pearson_cor, mse

def profile_metrics(true_counts,pred_probs,pseudocount=0.001):
    '''
    Get profile metrics
    '''
    mnll_pw = []
    mnll_norm = []

    jsd_pw = []
    jsd_norm = []
    jsd_rnd = []
    jsd_rnd_norm = []
    mnll_rnd = []
    mnll_rnd_norm = []

    num_regions = true_counts.shape[0]
    for idx in range(num_regions):
        # mnll
        #curr_mnll = mnll(true_counts[idx,:],  probs=pred_probs[idx,:])
        #mnll_pw.append(curr_mnll)
        # normalized mnll
        #min_mnll, max_mnll = mnll_min_max_bounds(true_counts[idx,:])
        #curr_mnll_norm = get_min_max_normalized_value(curr_mnll, min_mnll, max_mnll)
        #mnll_norm.append(curr_mnll_norm)

        # jsd
        cur_jsd=jensenshannon(true_counts[idx,:]/(pseudocount+np.nansum(true_counts[idx,:])),pred_probs[idx,:])
        jsd_pw.append(cur_jsd)
        # normalized jsd
        min_jsd, max_jsd = jsd_min_max_bounds(true_counts[idx,:])
        curr_jsd_norm = get_min_max_normalized_value(cur_jsd, min_jsd, max_jsd)
        jsd_norm.append(curr_jsd_norm)

        # get random shuffling on labels for a worst case performance on metrics - labels versus shuffled labels
        shuffled_labels=np.random.permutation(true_counts[idx,:])
        shuffled_labels_prob=shuffled_labels/(pseudocount+np.nansum(shuffled_labels))

        # mnll random
        #curr_rnd_mnll = mnll(true_counts[idx,:],  probs=shuffled_labels_prob)
        #mnll_rnd.append(curr_rnd_mnll)
        # normalized mnll random
        #curr_rnd_mnll_norm = get_min_max_normalized_value(curr_rnd_mnll, min_mnll, max_mnll)
        #mnll_rnd_norm.append(curr_rnd_mnll_norm)   

        # jsd random
        curr_jsd_rnd=jensenshannon(true_counts[idx,:]/(pseudocount+np.nansum(true_counts[idx,:])),shuffled_labels_prob)
        jsd_rnd.append(curr_jsd_rnd)
        # normalized jsd random
        curr_rnd_jsd_norm = get_min_max_normalized_value(curr_jsd_rnd, min_jsd, max_jsd)
        jsd_rnd_norm.append(curr_rnd_jsd_norm)

    return np.array(mnll_pw), np.array(mnll_norm), np.array(jsd_pw), np.array(jsd_norm), np.array(jsd_rnd), np.array(jsd_rnd_norm), np.array(mnll_rnd), np.array(mnll_rnd_norm)

def plot_histogram(region_jsd, shuffled_labels_jsd, output_prefix, title):

    #generate histogram distributions 
    num_bins=100
    plt.rcParams["figure.figsize"]=8,8
    
    #plot mnnll histogram 
    #plt.figure()
    #n,bins,patches=plt.hist(mnnll_vals,num_bins,facecolor='blue',alpha=0.5,label="Predicted vs Labels")
    #n1,bins1,patches1=plt.hist(shuffled_labels_mnll,num_bins,facecolor='black',alpha=0.5,label='Shuffled Labels vs Labels')
    #plt.xlabel('Multinomial Negative LL Profile Labels and Predictions in Probability Space')
    #plt.title("MNNLL: "+ tile)
    #plt.legend(loc='best')
    #plt.savefig(output_prefix+".mnnll.png",format='png',dpi=300)
    
    #plot jsd histogram
    plt.figure()
    n,bins,patches=plt.hist(region_jsd,num_bins,facecolor='blue',alpha=0.5,label="Predicted vs Labels")
    n1,bins1,patches1=plt.hist(shuffled_labels_jsd,num_bins,facecolor='black',alpha=0.5,label='Shuffled Labels vs Labels')
    plt.xlabel('Jensen Shannon Distance Profile Labels and Predictions in Probability Space')
    plt.title("JSD Dist: "+title)
    plt.legend(loc='best')
    plt.savefig(output_prefix+".profile_jsd.png",format='png',dpi=300)
    plt.close()

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)