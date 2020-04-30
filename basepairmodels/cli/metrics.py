import argparsers
import logger
import tensorflow as tf

from batchgenutils import *
from losses import multinomial_nll
from utils import *

import json
import numpy as np
import pandas as pd
import pyBigWig

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, binned_statistic
from scipy.special import logsumexp


def metrics_main():
    # parse the command line arguments
    parser = argparsers.metrics_argsparser()
    args = parser.parse_args()

    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        logging.error("Directory {} does not exist".format(
            args.output_dir))
        
        return

    if args.automate_filenames:
        # create a new directory using current date/time to store the
        # metrics outputs & logs 
        date_time_str = local_datetime_str(args.time_zone)
        metrics_dir = '{}/{}'.format(args.output_dir, date_time_str)
        os.mkdir(metrics_dir)
    elif os.path.isdir(args.output_dir):
        metrics_dir = args.output_dir        
    else:
        logging.error("Directory does not exist {}.".format(args.output_dir))
        return

    # filename to write debug logs
    logfname = "{}/metrics.log".format(metrics_dir)
    
    # set up the loggers
    logger.init_logger(logfname)
    
    
    if args.peaks is not None:
        peaks_df = pd.read_csv(args.peaks, 
                               sep='\t', header=None, 
                               names=['chrom', 'st', 'end', 'name', 'score',
                                      'strand', 'signal', 'p', 'q', 'summit'])

        # keep only those rows corresponding to the required 
        # chromosomes
        peaks_df = peaks_df[peaks_df['chrom'].isin(args.chroms)]
        
        # create new column for peak pos
        peaks_df['summit_pos'] = peaks_df['st'] + peaks_df['summit']

        # create new column for start pos
        peaks_df['start_pos'] = peaks_df['summit_pos'] - \
                                    args.metrics_seq_len // 2

        # create new column for end pos
        peaks_df['end_pos'] = peaks_df['summit_pos'] + \
                                    args.metrics_seq_len // 2

        # sort based on chromosome number and right flank coordinate
        peaks_df = peaks_df.sort_values(['chrom', 'summit_pos']).reset_index(
            drop=True)

        # select only the chrom & summit positon columns
        allPositions = peaks_df[['chrom', 'start_pos', 'end_pos']]

        allPositions = allPositions.reset_index(drop=True)
        
    else:
        
        allPositions = getChromPositions(args.chroms, args.chrom_sizes, 
                                         args.metrics_seq_len // 2, 
                                         args.step_size, mode='sequential',
                                         num_positions=-1)
            
    print(allPositions.shape)
   
   # open the two bigWig files
    try:
        bigWigProfileA = pyBigWig.open(args.profileA)
        bigWigProfileB = pyBigWig.open(args.profileB)
        
        if args.countsA:
            bigWigCountsA = pyBigWig.open(args.countsA)
        if args.countsB:
            bigWigCountsB = pyBigWig.open(args.countsB)
        
    except Exception as e:
        logging.error("Problems occurred when opening one of the input files: "
                      "{}".format(str(e)))

    
    # for pearson on counts
    countsA = []
    countsB = []
    
    # initialize arrays to hold metrics values
    array_len = len(allPositions.index)
    pearson = np.zeros(array_len, dtype=np.float64)
    spearman = np.zeros(array_len, dtype=np.float64)
    jsd = np.zeros(array_len, dtype=np.float64)
    mse = np.zeros(array_len, dtype=np.float64)
    
    idx = 0
    for chrom, start, end in allPositions.itertuples(index=False, name=None):
        profileA = np.nan_to_num(np.array(bigWigProfileA.values(chrom, start, end)))
        profileB = np.nan_to_num(np.array(bigWigProfileB.values(chrom, start, end)))

        if args.countsA:
            # since every base is assigned the total counts in the 
            # region we have to take the mean
            valsCountsA = np.mean(np.nan_to_num(np.array(bigWigCountsA.values(chrom, start, end))))
        else:
            valsCountsA = np.sum(profileA)

        if args.countsB:
            # since every base is assigned the total counts in the 
            # region we have to take the mean
            valsCountsB = np.mean(np.nan_to_num(np.array(bigWigCountsB.values(chrom, start, end))))
        else:
            valsCountsB = np.sum(profileB)
            
        # check to see if we fetched the correct numnber of values
        # if the two array lengths dont match we cant compute the 
        # metrics
        if len(profileA) != (end - start) or \
            len(profileB) != (end - start):
            logging.warning("Unable to fetch {} values on chrom {} from "
                            "{} to {}. Skipping.".format(end-start, chrom, 
                                                         start, end))
            continue
        
        
        if sum(profileA) != 0:
            if args.apply_softmax_to_profileA:
                # we use log softmax to circumvent numerical instability
                # and then exponetiate 
                probProfileA = profileA - logsumexp(profileA)
                probProfileA = np.exp(probProfileA)
                
                # we need actual counts to compute mse
                valsProfileA = np.multiply(valsCountsA, probProfileA)
                
                if len(args.smooth_profileA) > 0:
                    sigma = float(args.smooth_profileA[0])
                    width = float(args.smooth_profileA[1])
                    truncate = (((width - 1)/2)-0.5)/sigma

                    valsProfileA = gaussian_filter1d(
                        valsProfileA, sigma=sigma, truncate=truncate)
                    
                    # recompute probabilities
                    probProfileA = valsProfileA / sum(valsProfileA)
                
            else:
                if args.smooth_profileA:
                    sigma = float(args.smooth_profileA[0])
                    width = float(args.smooth_profileA[1])
                    truncate = (((width - 1)/2)-0.5)/sigma

                    profileA = gaussian_filter1d(
                        profileA, sigma=sigma, truncate=truncate)
                    
                # convert to probabilities by diving by sum
                probProfileA = profileA / sum(profileA)
                
                # if we are in the else block it implies profileA has
                # actual counts
                valsProfileA = profileA
        else:
            # uniform distribution
            probProfileA = 1.0/len(profileA) * np.ones(len(profileA), 
                                                       dtype=np.float32)
            
        if sum(profileB) != 0:
            if args.apply_softmax_to_profileB:
                # we use log softmax to circumvent numerical instability
                # and then exponetiate 
                probProfileB = profileB - logsumexp(profileB)
                probProfileB = np.exp(probProfileB)

                # we need actual counts to compute mse
                valsProfileB = np.multiply(valsCountsB, probProfileB)
                
                if len(args.smooth_profileB) > 0:
                    sigma = float(args.smooth_profileB[0])
                    width = float(args.smooth_profileB[1])
                    truncate = (((width - 1)/2)-0.5)/sigma

                    valsProfileB = gaussian_filter1d(
                        valsProfileB, sigma=sigma, truncate=truncate)
                    
                    # recompute probabilities
                    probProfileB = valsProfileB / sum(valsProfileB)
            else:
                if args.smooth_profileB:
                    sigma = float(args.smooth_profileB[0])
                    width = float(args.smooth_profileB[1])
                    truncate = (((width - 1)/2)-0.5)/sigma
                    
                    profileB = gaussian_filter1d(
                        profileB, sigma=sigma, truncate=truncate)
                    
                # convert to probabilities by diving by sum
                probProfileB = profileB / sum(profileB)
                
                # if we are in the else block it implies profileB has
                # actual counts
                valsProfileB = profileB

        else:
            # uniform distribution
            probProfileB = 1.0/len(profileB) * np.ones(len(profileB), 
                                                       dtype=np.float32)
                        
        # pearson & spearman
        # with pearson we need to check if either of the arrays
        # has zero standard deviation (i.e having all same elements,
        # a zero or any other value). Unfortunately np.std
        # returns a very small non-zero value, so we'll use a 
        # different approach to check if the array has the same value.
        # If true then pearson correlation is undefined 
        if np.unique(probProfileA).size == 1 or \
            np.unique(probProfileB).size == 1:
            pearson[idx] = np.nan
            spearman[idx] = np.nan
        else:
            pearson[idx] = pearsonr(probProfileA, probProfileB)[0]
            spearman[idx] = spearmanr(valsProfileA, valsProfileB)[0]

        # jsd
        jsd[idx] = jensenshannon(probProfileA, probProfileB)

        # mse
        mse[idx] = np.square(np.subtract(valsProfileA, valsProfileB)).mean()

        # add to the counts list
        countsA.append(np.sum(valsProfileA))
        countsB.append(np.sum(valsProfileB))
        idx += 1

    counts_pearson = pearsonr(countsA, countsB)[0]
    
    print("|-|", metrics_dir.split('/')[-2], np.median(pearson), np.median(spearman), np.median(jsd), np.median(mse), counts_pearson)
    print("pearson", np.median(pearson), max(pearson), min(pearson)) 
    print("spearman", np.median(spearman), max(spearman), min(spearman))
    print("jsd", np.median(jsd), max(jsd), min(jsd))
    print("mse", np.median(mse), max(mse), min(mse))
    print("counts pearson", counts_pearson)

    np.savez_compressed('{}/mse'.format(metrics_dir), mse=mse)
    np.savez_compressed('{}/pearson'.format(metrics_dir), pearson=pearson)
    np.savez_compressed('{}/spearman'.format(metrics_dir), spearman=spearman)
    np.savez_compressed('{}/jsd'.format(metrics_dir), jsd=jsd)
    np.savez_compressed('{}/counts_pearson'.format(metrics_dir), pearson=pearson)
    
    
    # write all the command line arguments to a json file
    config_file = '{}/config.json'.format(metrics_dir)
    with open(config_file, 'w') as fp:
        json.dump(vars(args), fp)
    
if __name__ == '__main__':
    tf.enable_eager_execution()
    metrics_main()

