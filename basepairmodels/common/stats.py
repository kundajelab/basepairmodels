"""

    This module contains functions to 


"""

import os 
import pyBigWig
import numpy as np

from mseqgen.quietexception import QuietException


def get_recommended_counts_loss_weight(input_bigwigs, training_intervals, 
                                       alpha=1.0):
    """
        This function computes the hyper parameter lambda (l) as
        suggested in the BPNet paper on pg. 28
        https://www.biorxiv.org/content/10.1101/737981v2.full.pdf
        
        if lambda `l` is set to 1/2 * n_obs, where n_obs is the 
        average number of total counts in the training set, the 
        profile loss and the  total counts loss will be roughly given 
        equal weight. We can use the `alpha` parameter to upweight 
        the profile predictions relative to the total count 
        predictions as shown below
        
        l = (alpha / 2) * n_obs
    
        Args:
            input_bigwigs (list): list of bigwig files with assay
                signal. n_obs will computed as a global average
                across all the input bigwigs
            
            training_intervals (pandas.Dataframe): 3 column dataframe  
                with 'chrom', 'start' and 'end' columns, representing 
                range [start, end) for the spans of interest 

            alpha (float): parameter to scale profile loss relative
                to the counts loss. A value < 1.0 will upweight the
                profile loss
    
        Returns
            float: counts loss weight (lambda)
            
    """

    # check to make sure all bigwigs are valid files
    for bigwig in input_bigwigs:
        if not os.path.exists(bigwig):
            raise QuietException("File {} does not exist".format(bigwig))

    # open each bigwig and add file pointers to a list
    bigwigs = []
    for bigwig in input_bigwigs:
        bigwigs.append(pyBigWig.open(bigwig))
    
    # total counts from all training windows across all bigwigs
    total_counts = 0
    
    # iterate over training windows
    for _idx, row in training_intervals.iterrows():
        
        # chrom window
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        # iterate over bigwigs for each window and accumulate the sum
        for bw in bigwigs:
            total_counts += np.sum(np.nan_to_num(bw.values(chrom, start, end)))
        
    # average of the total counts
    n_obs = total_counts / (len(bigwigs) * len(training_intervals))
    
    return (alpha / 2) * n_obs


            
            
            
        
        
        
    
