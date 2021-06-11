import logging
import numpy as np
import os
import pandas as pd
import pyBigWig
import sys

from basepairmodels.cli.argparsers import logits2profile_argsparser
from basepairmodels.cli.exceptionhandler import NoTracebackException
from scipy.special import logsumexp
from tqdm import tqdm

def logits2profile_main():
    # parse the command line arguments
    parser = logits2profile_argsparser()
    args = parser.parse_args()
    
    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        raise NoTracebackException(
            "Directory {} does not exist".format(args.output_dir))
        return
    
    # check if the logits file exists
    if not os.path.exists(args.logits_file):
        raise NoTracebackException(
            "Logits file {} does not exist".format(args.logits_file))
        return

    # check if the counts file exists
    if not os.path.exists(args.counts_file):
        raise NoTracebackException(
            "Counts file {} does not exist".format(args.counts_file))
        return

    # check if the peaks file exists
    if not os.path.exists(args.peaks):
        raise NoTracebackException(
            "Peaks file {} does not exist".format(args.peaks))
        return
    
    # check if the chrom sizes file exists
    if not os.path.exists(args.chrom_sizes):
        raise NoTracebackException(
            "Peaks file {} does not exist".format(args.chrom_sizes))
        return

    # construct header for the output bigWig file
    header = []
    # dataframe with chromosome sizes
    chrom_sizes_df = pd.read_csv(args.chrom_sizes, sep = '\t', header=None, 
                                 names = ['chrom', 'size'])
    chrom_sizes_df = chrom_sizes_df.set_index('chrom')
    # sort chromosomes, to be consistent with how pandas sorts
    # chromosomes ... for e.g. chrom21 is < chrom8
    chroms = args.chroms[:]
    chroms.sort()
    for chrom in chroms:
        size = chrom_sizes_df.at[chrom, 'size']
        header.append((chrom, int(size)))

    logging.debug("bigWig HEADER - {}".format(header))
    
    # open logits bigWig for reading
    logits_bigWig = pyBigWig.open(args.logits_file)

    # open counts bigWig for reading
    counts_bigWig = pyBigWig.open(args.counts_file)

    # open output bigWig for writing 
    output_bigWig_fname = '{}/{}.bw'.format(args.output_directory, 
                                            args.output_filename)
    output_bigWig = pyBigWig.open(output_bigWig_fname, 'w')
    
    # add the header to the bigWig files
    output_bigWig.addHeader(header, maxZooms=0)    
    
    # read the peaks file into a dataframe 
    peaks_df = pd.read_csv(args.peaks, usecols=[0, 1 ,2], 
                           names=['chrom', 'start', 'end'], header=None,
                           sep='\t')
    peaks_df = peaks_df[peaks_df['chrom'].isin(args.chroms)]
    peaks_df['_start'] = peaks_df['start'] + \
                         (peaks_df['end'] - peaks_df['start']) // 2 - \
                         args.window_size // 2 
    peaks_df['_end'] = peaks_df['_start'] + args.window_size
    peaks_df = peaks_df.sort_values(by=['chrom', '_start'])    
    print(peaks_df)
    
    # maintain a dictionary to record chrom coordinates that are
    # written to the output bigWig, this will make inserting 
    # overlapping coordinates easy to handle. pyBigWig's addEntries
    # function will scream if you write to a position to which
    # an entry was already added previously 
    # Note: since chromosome's are sorted we can delete the 
    # previous chromosome entries to save memory
    write_log = {}
    
    prev_chrom = ''
    for index, row in tqdm(peaks_df.iterrows(), total=peaks_df.shape[0]):
        chrom = row['chrom']
        start = row['_start']
        end = row['_end']

        # delete write log entries of the previous chromosome
        if chrom != prev_chrom:
            write_log.pop(prev_chrom, None)
            # new dict for new chrom
            write_log[chrom] = {}
        prev_chrom = chrom
            
        try:
            logits_vals = np.nan_to_num(logits_bigWig.values(chrom, start, end))
        except RuntimeError as e:
            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()
            print("Skipping peak ({}, {}, {}) in logits bigWig. No data "
                  "found. Make sure to use the same peaks and "
                  "output-window-size that were used in the predict "
                  "step".format(chrom, start, end))
            continue

        try:
            counts_vals = np.nan_to_num(counts_bigWig.values(chrom, start, end))
        except RuntimeError as e:
            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()
            print("Skipping peak ({}, {}, {}) in counts bigWig. No data "
                  "found. Make sure to use the same peaks and "
                  "output-window-size that were used in the predict "
                  "step".format(chrom, start, end))
            continue

        chroms = [chrom] * args.window_size
        starts = list(range(start, end, 1))
        ends = list(range(start + 1, end + 1, 1))

        # scale logits: first softmax, then multiply by counts
        probVals = logits_vals - logsumexp(logits_vals)
        probVals = np.exp(probVals)
        profile = np.multiply(counts_vals, probVals)
        
        for i in range(len(chroms)):
            try:
                _ = write_log[chroms[i]][starts[i]]
            except KeyError as e:
                # write to bigWig only if the location was not written to
                # before
                output_bigWig.addEntries(
                    [chroms[i]], [starts[i]], ends=[ends[i]], 
                    values=[profile[i]])

                # add entry into write log
                write_log[chrom][start] = 0
        
if __name__ == '__main__':
    logits2profile_main()

