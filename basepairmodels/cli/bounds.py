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
from basepairmodels.cli.exceptionhandler import NoTracebackException
from basepairmodels.cli.metrics import mnll, profile_cross_entropy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr, multinomial
from tqdm import tqdm
        

def get_average_profile(input_bigWig, peaks_df, peak_width):
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
            raise NoTracebackException(
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
        # along the width of the profile
        for i in range(len(true_counts)): 
            
            # for each count at that position
            for j in range(true_counts[i]): 
                
                # if coin toss value is 0 assign it to 'obs' else
                # assign it to 'pred'
                if coin_tosses[coin_toss_idx] == 0:
                    obs[i] += 1
                else:
                    pred[i] += 1

                coin_toss_idx += 1
        
        # if by chance one of the two pseudoreplicates doesn't
        # get any counts
        if sum(obs) == 0 or sum(pred) == 0:
            
            # reinitialize the arrays
            obs = np.zeros(len(true_counts))
            pred = np.zeros(len(true_counts))
            
            continue
            
        else:
            
            break
                    
    return (obs, pred)


def bounds(input_bigWig, peaks_df, peak_width, smoothing_params=[7, 81]):
    """
        Function to compute lower & upper bounds, and average profile
        performance for cross entropy and jsd metrics
        
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
                and window_size values for 1D gaussian smoothing of 
                profiles
        
        Returns:
            tuple: (numpy array of average profile, pandas dataframe
                with bounds values in columns)
                
    """
    
    # compute the average profile
    print("Computing average profile ...")
    avg_profile = get_average_profile(input_bigWig, peaks_df, peak_width)
    
    # get average profile as probabilities
    avg_profile_prob = avg_profile / np.sum(avg_profile)
    
    # open the bigWig file for reading
    bw = pyBigWig.open(input_bigWig)
        
    # arrays to hold metrics values for mnll, cross entropy, jsd, 
    # pearson and spearman correlation of the peak profile computed 
    # against uniform, average and self(observed peak) profile

    # mnll
    mnll_uniform = np.zeros(peaks_df.shape[0])
    mnll_average = np.zeros(peaks_df.shape[0])
    mnll_self = np.zeros(peaks_df.shape[0])    
    
    # cross entropy
    ce_uniform = np.zeros(peaks_df.shape[0])
    ce_average = np.zeros(peaks_df.shape[0])
    ce_self = np.zeros(peaks_df.shape[0])
    
    # jsd
    jsd_uniform = np.zeros(peaks_df.shape[0])
    jsd_average = np.zeros(peaks_df.shape[0])
    jsd_self = np.zeros(peaks_df.shape[0])
    
    # pearson
    pearson_uniform = np.zeros(peaks_df.shape[0])
    pearson_average = np.zeros(peaks_df.shape[0])
    pearson_self = np.zeros(peaks_df.shape[0])
    
    # spearman
    spearman_uniform = np.zeros(peaks_df.shape[0])
    spearman_average = np.zeros(peaks_df.shape[0])
    spearman_self = np.zeros(peaks_df.shape[0])

    print("Computing bounds ...")

    # iterate through all peaks
    for idx, row in tqdm(peaks_df.iterrows(), desc='peak', 
                         total=peaks_df.shape[0]):

        # raise exception if 'end' - 'start' is not equal to peak_width
        if (row['end'] - row['start']) != peak_width:

            raise NoTracebackException(
                "Inconsistent peak width found at: {}:{}-{}".format(
                    row['chrom'], row['start'], row['end']))

        # get bigWig profile
        profile = np.nan_to_num(
            bw.values(row['chrom'], row['start'], row['end']))

        # if we find that the profile at this peak is all zeros
        if sum(profile) == 0:

            print("Found 'zero' profile at {}: ({}, {})".format(
                row['chrom'], row['start'], row['end']))

            # assign nans to all 
            mnll_uniform[idx] = np.nan
            mnll_average[idx] = np.nan
            mnll_self[idx] = np.nan

            ce_uniform[idx] = np.nan
            ce_average[idx] = np.nan
            ce_self[idx] = np.nan

            jsd_uniform[idx] = np.nan
            jsd_average[idx] = np.nan
            jsd_self[idx] = np.nan

            pearson_uniform[idx] = np.nan
            pearson_average[idx] = np.nan
            pearson_self[idx] = np.nan

            spearman_uniform[idx] = np.nan
            spearman_average[idx] = np.nan
            spearman_self[idx] = np.nan

            continue

        # uniform distribution profile
        uniform_profile = np.ones(peak_width) * (1.0 / peak_width)

        # smoothed profile 
        profile_smooth = gaussian1D_smoothing(profile, smoothing_params[0], 
                                              smoothing_params[1])

        # smoothed profile as probabilities 
        profile_smooth_prob = profile_smooth / np.sum(profile_smooth)

        # profile as probabilities
        profile_prob = profile / np.sum(profile)

        # mnll of profile with uniform profile
        mnll_uniform[idx] = mnll(profile, probs=uniform_profile)

        # mnll of profile with average profile
        mnll_average[idx] = mnll(profile, probs=avg_profile_prob)

        # mnll of profile with itself
        mnll_self[idx] = mnll(profile, probs=profile_prob)

        # cross entropy of profile with uniform profile
        ce_uniform[idx] = profile_cross_entropy(profile, 
                                                probs=uniform_profile)

        # cross entropy of profile with average profile
        ce_average[idx] = profile_cross_entropy(profile, 
                                                probs=avg_profile_prob)

        # cross entropy of profile with itself
        ce_self[idx] = profile_cross_entropy(profile, probs=profile_prob)

        # jsd of profile with uniform profile
        jsd_uniform[idx] = jensenshannon(profile_prob, uniform_profile)

        # jsd of profile with average profile
        jsd_average[idx] = jensenshannon(profile_prob, avg_profile_prob)

        # jsd of profile with itself (upper bound)
        jsd_self[idx] = 0.0

        # pearson of profile with uniform profile
        ### nothing to do ... leave it as zeros

        # pearson of profile with average profile
        pearson_average[idx] = pearsonr(profile, avg_profile_prob)[0]
        
        # pearson of profile with itself
        pearson_self[idx] = pearsonr(profile, profile)[0]
        
        # spearman of profile with uniform profile
        ### nothing to do ... leave it as zeros

        # spearman of profile with average profile
        spearman_average[idx] = spearmanr(profile, avg_profile_prob)[0]

        spearman_self[idx] = spearmanr(profile, profile)[0]

    # create a pandas dataframe to hold the upper & lower bound, 
    # and avg profile performance values 
    column_names = ['mnll_uniform', 'mnll_average', 'mnll_self',
                    'ce_uniform', 'ce_average', 'ce_self',
                    'jsd_uniform', 'jsd_average', 'jsd_self',
                    'pearson_uniform', 'pearson_average', 'pearson_self', 
                    'spearman_uniform', 'spearman_average', 'spearman_self']
    
    # create a pandas dataframe to store all the bounds values
    bounds_df = pd.DataFrame(columns = column_names)
        
    # assign values to the dataframe columns
    bounds_df['mnll_uniform'] = np.nan_to_num(mnll_uniform)
    bounds_df['mnll_average'] = np.nan_to_num(mnll_average)
    bounds_df['mnll_self'] = np.nan_to_num(mnll_self)
    bounds_df['ce_uniform'] = np.nan_to_num(ce_uniform)
    bounds_df['ce_average'] = np.nan_to_num(ce_average)
    bounds_df['ce_self'] = np.nan_to_num(ce_self)
    bounds_df['jsd_uniform'] = np.nan_to_num(jsd_uniform)
    bounds_df['jsd_average'] = np.nan_to_num(jsd_average)
    bounds_df['jsd_self'] = np.nan_to_num(jsd_self)
    bounds_df['pearson_uniform'] = np.nan_to_num(pearson_uniform)
    bounds_df['pearson_average'] = np.nan_to_num(pearson_average)
    bounds_df['pearson_self'] = np.nan_to_num(pearson_self)
    bounds_df['spearman_uniform'] = np.nan_to_num(spearman_uniform)
    bounds_df['spearman_average'] = np.nan_to_num(spearman_average)
    bounds_df['spearman_self'] = np.nan_to_num(spearman_self)

    return avg_profile, bounds_df


def bounds_main():
    """
        The main entry point for the bounds computation script
    """
    
    # parse the command line arguments
    parser = bounds_argsparser()
    args = parser.parse_args()
    
    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        raise NoTracebackException(
            "Directory {} does not exist".format(args.output_directory))

    # check to make sure at least one input profile was provided
    if len(args.input_profiles) == 0:
        raise NoTracebackException(
            "At least one input file is required to compute upper and "
            "lower bound")

    # check to see if the number of output names is equal to the number
    # of input profiles that were provided
    if len(args.output_names) != len(args.input_profiles) :
        raise NoTracebackException(
            "There should be same number of output names as the number "
            "of input files")

    # check if each input profile bigWig file exists
    for fname in args.input_profiles:
        if not os.path.exists(fname):
            raise NoTracebackException(
                "File not found! {}".format(fname))

    # check if the peaks file exists
    if not os.path.exists(args.peaks):
        raise NoTracebackException(
            "Peaks file {} does not exist".format(args.peaks))

    # read the peaks bed file into a pandas dataframe
    peaks_df = pd.read_csv(args.peaks, sep='\t', header=None, 
                           names=['chrom', 'st', 'en', 'name', 'score',
                                  'strand', 'signalValue', 'p', 'q', 'summit'])
    
    # if --chroms paramter is provided filter the dataframe rows
    if args.chroms is not None:
        peaks_df = peaks_df[peaks_df['chrom'].isin(args.chroms)]
    
    # modified start and end based on summit & specified peak_width
    peaks_df['start'] = peaks_df['st'] + peaks_df['summit'] - \
                            (args.peak_width // 2)
    peaks_df['end'] = peaks_df['st'] + peaks_df['summit'] + \
                            (args.peak_width // 2)
    
    print("Peaks shape", peaks_df.shape[0])
    # reset index in case rows have been filtered
    peaks_df = peaks_df.reset_index()
    
    # iterate through each input profile
    for i in range(len(args.input_profiles)):
        
        # path to input profile bigWig
        input_profile_bigWig = args.input_profiles[i]
        
        print("Processing ... ", input_profile_bigWig)
        
        # compute upper & lower bounds, and avg profile performance
        average_profile, bounds_df = bounds(
            input_profile_bigWig, peaks_df, args.peak_width, 
            args.smoothing_params)

        # path to output average profile file
        average_profile_filename = "{}/{}_average_profile.csv".format(
            args.output_directory, args.output_names[i])
        
        # write average profile to csv file
        print("Saving average profile ...")
        np.savetxt(average_profile_filename, average_profile,
                   delimiter=",")
        
        # path to the output bounds file
        output_fname = "{}/{}.bds".format(args.output_directory, 
                                          args.output_names[i])
        
        # write the dataframe to a csv file
        print("Saving bounds file ...")
        bounds_df.to_csv(output_fname, index=False)


if __name__ == '__main__':
    bounds_main()
