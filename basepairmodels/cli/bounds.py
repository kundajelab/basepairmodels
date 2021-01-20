"""
    Python script for compute lower and upper bounds for the 
    mnll and jsd metrics

    License:
    
    MIT License

    Copyright (c) 2020 Kundaje Lab

    Permission is hereby granted, free of charge, to any person 
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be 
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

"""

import logging
import numpy as np
import os
import pandas as pd
import pyBigWig
import sys

from basepairmodels.cli.argparsers import bounds_argsparser

from mseqgen import quietexception

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp

from tqdm import tqdm


def expLogP(true_counts, probs=None, logits=None):
    """
        Compute the expected log probability

        Args:
          true_counts (numpy.array): observed count values
          
          probs (numpy.array): predicted values as probabilities
          
          logits (numpy.array): predicted logits values

        Return
            float: expected log probability
    
    """
    
    if logits is not None:
        # check for length mismatch
        if len(probs) != len(logits):
            raise quietexception.QuietException(
                "Length of logits does not match length of true_counts")
        
        # convert logits to softmax probabilities
        probs = logits - logsumexp(logits)
        probs = np.exp(probs)
        
    if probs is not None:      
        # check for length mistmatch
        if len(probs) != len(true_counts):
            raise quietexception.QuietException(
                "Length of probs does not match length of true_counts")
        
        # check if probs sums to 1
        if abs(np.sum(probs) - 1.0) > 1e-3:
            raise quietexception.QuietException(
                "'probs' array does not sum to 1")

        # compute expected log probabilities
        return np.sum(np.multiply(true_counts, np.log(probs + 1e-6)))
                    
    else:
        # both 'probs' and 'logits' are None
        raise quietexception.QuietException(
            "At least one of probs or logits must be provided. Both are None.")
        
        
def average_profile(input_bigWig, peaks_df, peak_width):
    """
        Function to compute the average profile across all peaks
        
        Args:
            input_bigWig (str): path to bigWig file
            
            peaks_df (str): pandas dataframe containing peaks 
                information. 
                
                The dataframe should have 'chrom', 'start', and 'end'
                as first 3 columns. Each peak should have the same
                width (equal to peak_width) i.e 'end' - 'start' is the
                same for all rows in the dataframe.
                
            peak_width (int): width of each peak.
        
        Returns:
            np.array: numpy array of length peak_width
                    
    """
    
    # open the bigWig file for reading
    bw = pyBigWig.open(input_bigWig)
    
    # initialize numpy array for average profile
    average_profile = np.zeros(peak_width)

    # iterate through all peaks and compute the average
    for idx, row in peaks_df.iterrows():
        
        # raise exception if 'end' - 'start' is not equal to peak_width
        if (row['end'] - row['start']) != peak_width:
            raise quietexception.QuietException(
                "Inconsistent peak width found at: {}:{}-{}".format(
                    row['chrom'], row['start'], row['end']))
        
        # read values from bigWig
        average_profile += np.nan_to_num(
            bw.values(row['chrom'], row['start'], row['end']))

    # average profile
    average_profile /= peaks_df.shape[0]
    
    # close bigWig file
    bw.close()
    
    return average_profile
        
    
def gaussian1D_smoothing(input_array, sigma, window_size):
    """
        Function to smooth input array using 1D gaussian smoothing
        
        Args:
            input_array (numpy.array): input array of values
            
            sigma (float): sigma value for gaussian smoothing
            
            window_size (int): window size for gaussian smoothing
            
        Returns:
            numpy.array: smoothed output array
        
    """

    # compute truncate value (#standard_deviations)
    truncate = (((window_size - 1)/2)-0.5)/sigma
    
    return gaussian_filter1d(input_array, sigma=sigma, truncate=truncate)


def lower_bounds(input_bigWig, peaks_df, peak_width, smoothing_params):
    """
        Function to compute expLogP and jsd lower bounds
        
        Args:
            input_bigWig (str): path to bigWig file
            
            peaks_df (str): pandas dataframe containing peaks 
                information. 
                
                The dataframe should have 'chrom', 'start', and 'end'
                as first 3 columns. Each peak should have the same
                width (equal to peak_width) i.e 'end' - 'start' is the
                same for all rows in the dataframe.
                
            peak_width (int): width of each peak.
            
            smoothing_params (list): list of length 2, containing sigma
                and window_size values for 1D gaussian smoothing

        Returns:
            tuple: (numpy array of expLogP lower bounds, 
                numpy array of jsd lower bounds)
    
    """
    
    # get the average profile
    avg_profile = average_profile(input_bigWig, peaks_df, peak_width)
    
    # open the bigWig file for reading
    bw = pyBigWig.open(input_bigWig)
    
    # convert average profile to probabilities
    avg_profile_prob = avg_profile / np.sum(avg_profile)
    
    # uniform distribution profile 
    uniform_profile = np.ones(peak_width) * (1.0 / peak_width)
    
    # arrays to hold the lower bound values
    expLogP_lower_bounds = np.zeros(peaks_df.shape[0])
    jsd_lower_bounds = np.zeros(peaks_df.shape[0])
    
    # iterate through all peaks
    for idx, row in peaks_df.iterrows():
        
        # raise exception if 'end' - 'start' is not equal to peak_width
        if (row['end'] - row['start']) != peak_width:
            raise quietexception.QuietException(
                "Inconsistent peak width found at: {}:{}-{}".format(
                    row['chrom'], row['start'], row['end']))
            
        # get bigWig profile
        profile = np.nan_to_num(
            bw.values(row['chrom'], row['start'], row['end']))
        
        # if we find that the profile at this peak is all zeors
        if sum(profile) == 0:
            print("Found 'zero' profile at {}: ({}, {})".format(
                row['chrom'], row['start'], row['end']))
            expLogP_lower_bounds[idx] = np.nan
            jsd_lower_bounds[idx] = np.nan
            continue
            
        # expLogP lower bound with average profile 
        b1 = expLogP(profile, probs=avg_profile_prob)
        
        # mnll lower bound with uniform profile
        b2 = expLogP(profile, probs=uniform_profile)
        
        # actual mnll lower bound
        expLogP_lower_bounds[idx] = max(b1, b2)
        
        # smooth profile for jsd 
        smoothed_profile = gaussian1D_smoothing(profile, smoothing_params[0],
                                                smoothing_params[1])
        
        # convert smoothed profile to probabilities
        profile_prob = smoothed_profile / sum(smoothed_profile)
        
        # jsd lower bound with average profile 
        b1 = jensenshannon(profile_prob, avg_profile_prob)
        
        # jsd lower bound with uniform profile
        b2 = jensenshannon(profile_prob, uniform_profile)
        
        # actual jsd lower bound
        jsd_lower_bounds[idx] = min(b1, b2)
    
    return [expLogP_lower_bounds, jsd_lower_bounds]


def get_nonzero_pseudoreplicate_pair(true_counts):
    """
        Function to generate pseudoreplicate pair from true counts
        where each pseudoreplicate has nonzero sum(counts)
        
        Args:
            true_counts (numpy.array): 1D numpy array containing base
                level true counts
                
        Returns:
            tuple: (numpy array of counts for replicate 1, 
                numpy array of counts for replicate 2)
            
    """
    
    # pseudoreplicate that will be treated as the "observed" profile
    obs = np.zeros(len(true_counts))
    
    # pseudoreplicate that will be treated as the "predicted" profile
    pred = np.zeros(len(true_counts))
    
    while True:        
        # generate one coin toss for each true_count
        coin_tosses = np.random.binomial(1, 0.5, sum(true_counts))

        coin_toss_idx = 0

        # assign each count to one of the two pseudoreplicates
        for i in range(len(true_counts)): # along the width of the profile
            for j in range(true_counts[i]): # for each count at that position
                # coin toss value is 0 assign it to 'obs' else assign it to 'pred'
                if coin_tosses[coin_toss_idx] == 0:
                    obs[i] += 1
                else:
                    pred[i] += 1

                coin_toss_idx += 1
        
        if sum(obs) == 0 or sum(pred) == 0:
            continue
        else:
            break
                    
    return (obs, pred)
            

def upper_bounds(input_bigWig, peaks_df, peak_width, 
                 num_pseudoreplicate_pairs=1, aggregation_method='max', 
                 smoothing_params=[7, 81]):
    
    """
        Function to compute expLogP and jsd upper bounds
        
        Args:
            input_bigWig (str): path to bigWig file
            
            peaks_df (str): pandas dataframe containing peaks 
                information. 
                
                The dataframe should have 'chrom', 'start', and 'end'
                as first 3 columns. Each peak should have the same
                width (equal to peak_width) i.e 'end' - 'start' is the
                same for all rows in the dataframe.
                
            peak_width (int): width of each peak.
            
            num_pseudoreplicate_pairs (int): number of pseudoreplicate
                pairs to be generated
                
            aggregation_method (str): either 'max' or 'avg'. If 
                num_pseudoreplicate_pairs > 1, this specifies how the
                upper bound values should be aggregated
                
            smoothing_params (list): list of length 2, containing sigma
                and window_size values for 1D gaussian smoothing

        Returns:
            tuple: (numpy array of expLogP upper bounds, 
                numpy array of jsd upper bounds)
    
    """
        
    # open the bigWig file for reading
    bw = pyBigWig.open(input_bigWig)
    
    # arrays to hold the lower bound values
    expLogP_upper_bounds = np.zeros(peaks_df.shape[0])
    jsd_upper_bounds = np.zeros(peaks_df.shape[0])
    
    # iterate through all peaks
    for idx, row in peaks_df.iterrows():
        
        # raise exception if 'end' - 'start' is not equal to peak_width
        if (row['end'] - row['start']) != peak_width:
            raise quietexception.QuietException(
                "Inconsistent peak width found at: {}:{}-{}".format(
                    row['chrom'], row['start'], row['end']))
            
        # get bigWig profile
        profile = np.nan_to_num(
            bw.values(row['chrom'], row['start'], row['end']))
        
        # if we find that the profile at this peak is all zeros
        if sum(profile) == 0:
            print("Found 'zero' profile at {}: ({}, {})".format(
                row['chrom'], row['start'], row['end']))
            expLogP_upper_bounds[idx] = np.nan
            jsd_upper_bounds[idx] = np.nan
            continue

        # if we find that the sum(profile) at this peak <2 then we
        # cannot generate a non-zero pseudoreplicate
        if sum(profile) < 2:
            print("Found sum(profile) < 2 at {}: ({}, {})".format(
                row['chrom'], row['start'], row['end']))
            expLogP_upper_bounds[idx] = np.nan
            jsd_upper_bounds[idx] = np.nan
            continue

        # list of values for each pseudoreplicate pair
        # These will be aggregrated later using the user specified
        # method
        _expLogP_upper_bounds = []
        _jsd_upper_bounds = []
        
        # generate multiple pairs of pseudoreplicates and compute
        # separate sets of bounds for each pair
        for i in range(num_pseudoreplicate_pairs):
            
            (obs, pred) = get_nonzero_pseudoreplicate_pair(profile.astype(int))
 
            # smooth the "observed" and "predicted" pseudo replicates
            smooth_obs = gaussian1D_smoothing(obs, smoothing_params[0],
                                                smoothing_params[1])            
            smooth_pred = gaussian1D_smoothing(pred, smoothing_params[0],
                                                smoothing_params[1])
            
            # convert to probabilities
            smooth_obs_prob = smooth_obs / np.sum(smooth_obs)
            smooth_pred_prob = smooth_pred / np.sum(smooth_pred)
            
            scale = np.sum(profile) / np.sum(obs)
            _expLogP_upper_bounds.append(expLogP(obs, smooth_pred_prob) * scale)
            
            _jsd_upper_bounds.append(jensenshannon(smooth_obs_prob, 
                                                 smooth_pred_prob))
            
        
        # aggregate values using user specified method
        if aggregation_method == 'max':
            expLogP_upper_bounds[idx] = max(_expLogP_upper_bounds)
            jsd_upper_bounds[idx] = max(_jsd_upper_bounds)
        else:
            # average value from all pseudoreplicate pairs
            expLogP_upper_bounds[idx] = sum(_expLogP_upper_bounds) / \
                num_pseudoreplicate_pairs
            jsd_upper_bounds[idx] = sum(_jsd_upper_bounds) / \
                num_pseudoreplicate_pairs
        
    return [expLogP_upper_bounds, jsd_upper_bounds]


def bounds_main():
    
    # parse the command line arguments
    parser = bounds_argsparser()
    args = parser.parse_args()
    
    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        raise quietexception.QuietException(
            "Directory {} does not exist".format(args.output_dir))

    # check to make sure at least one input profile was provided
    if len(args.input_profiles) == 0:
        raise quietexception.QuietException(
            "At least one input file is required to compute upper and "
            "lower bound")

    # check to see if the number of output names is equal to the number
    # of input profiles that were provided
    if len(args.output_names) != len(args.input_profiles) :
        raise quietexception.QuietException(
            "There should be same number of output names as the number "
            "of input files")

    # check if each input profile bigWig file exists
    for fname in args.input_profiles:
        if not os.path.exists(fname):
            raise quietexception.QuietException(
                "File not found! {}".format(fname))

    # check if the peaks file exists
    if not os.path.exists(args.peaks):
        raise quietexception.QuietException(
            "Peaks file {} does not exist".format(args.peaks))

    # read the peaks bed file into a pandas dataframe
    peaks_df = pd.read_csv(args.peaks, usecols=[0,1,2], 
                           names=['chrom', 'start', 'end'], 
                           header=None, sep='\t')    
    
    # if --chroms paramter is provided filter the dataframe rows
    peaks_df = peaks_df[peaks_df['chrom'].isin(args.chroms)]
    
    # modified start and end based on specified peak_width
    peaks_df['start'] = peaks_df['start'] + \
                         (peaks_df['end'] - peaks_df['start']) // 2 - \
                         args.peak_width // 2 
    peaks_df['end'] = peaks_df['start'] + args.peak_width
    
    # reset index in case rows have been filtered
    peaks_df = peaks_df.reset_index()
    
    # iterate through each input profile
    for i in range(len(args.input_profiles)):
        
        # path to input profile bigWig
        input_profile_bigWig = args.input_profiles[i]
        
        print("Processing ... ", input_profile_bigWig)
        
        # compute lower bounds 
        print("Computing lower bounds ...")
        expLogP_lower_bounds, jsd_lower_bounds = lower_bounds(
            input_profile_bigWig, peaks_df, args.peak_width, 
            args.smoothing_params)

        # compute upper bounds
        print("Computing upper bounds ...")
        expLogP_upper_bounds, jsd_upper_bounds = upper_bounds(
            input_profile_bigWig, peaks_df, args.peak_width,
            args.num_upper_bound_pseudoreplicate_pairs, 
            args.upper_bound_aggregation_method,
            args.smoothing_params)

        # path to the output bounds file
        output_fname = "{}/{}.bds".format(args.output_directory, 
                                          args.output_names[i])
        
        # create a pandas dataframe to hold the upper and lower bound values 
        column_names = ['expLogP_lower_bound', 'expLogP_upper_bound', 
                        'jsd_lower_bound', 'jsd_upper_bound']
        df = pd.DataFrame(columns = column_names)
        
        # assign values to the dataframe columns
        df['expLogP_lower_bound'] = expLogP_lower_bounds
        df['expLogP_upper_bound'] = expLogP_upper_bounds
        df['jsd_lower_bound'] = jsd_lower_bounds
        df['jsd_upper_bound'] = jsd_upper_bounds
        
        # write the dataframe to a csv file
        df.to_csv(output_fname, index=False)
    
        
if __name__ == '__main__':
    bounds_main()

