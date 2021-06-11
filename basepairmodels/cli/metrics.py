import json
import numpy as np
import pandas as pd
import pyBigWig

from basepairmodels.cli.argparsers import metrics_argsparser
from basepairmodels.cli.bpnetutils import *
from basepairmodels.cli.exceptionhandler import NoTracebackException
from basepairmodels.cli.logger import *
from mseqgen.sequtils import getChromPositions
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, multinomial
from scipy.special import logsumexp
from tqdm import tqdm


def mnll(true_counts, logits=None, probs=None):
    """
        Compute the multinomial negative log-likelihood between true
        counts and predicted values of a BPNet-like profile model
        
        One of `logits` or `probs` must be given. If both are
        given `logits` takes preference.

        Args:
            true_counts (numpy.array): observed counts values
            
            logits (numpy.array): predicted logits values
            
            probs (numpy.array): predicted values as probabilities
          
        Returns:
            float: cross entropy
    
    """

    dist = None 
    
    if logits is not None:
        
        # check for length mismatch
        if len(logits) != len(true_counts):
            raise NoTracebackException(
                "Length of logits does not match length of true_counts")
        
        # convert logits to softmax probabilities
        probs = logits - logsumexp(logits)
        probs = np.exp(probs)
        
    elif probs is not None:      
        
        # check for length mistmatch
        if len(probs) != len(true_counts):
            raise NoTracebackException(
                "Length of probs does not match length of true_counts")
        
        # check if probs sums to 1
        if abs(1.0 - np.sum(probs)) > 1e-3:
            raise NoTracebackException(
                "'probs' array does not sum to 1")   
           
    else:
        
        # both 'probs' and 'logits' are None
        raise NoTracebackException(
            "At least one of probs or logits must be provided. "
            "Both are None.")
  
    # compute the nmultinomial distribution
    mnom = multinomial(np.sum(true_counts), probs)
    return -(mnom.logpmf(true_counts) / len(true_counts))
    
def profile_cross_entropy(true_counts, logits=None, probs=None):
    """
        Compute the cross entropy between true counts and predicted 
        values of a BPNet-like profile model
        
        One of `logits` or `probs` must be given. If both are
        given `logits` takes preference.

        Args:
            true_counts (numpy.array): observed counts values
            
            logits (numpy.array): predicted logits values
            
            probs (numpy.array): predicted values as probabilities
          
        Returns:
            float: cross entropy
    
    """

    if logits is not None:
        
        # check for length mismatch
        if len(logits) != len(true_counts):
            raise NoTracebackException(
                "Length of logits does not match length of true_counts")
        
        # convert logits to softmax probabilities
        probs = logits - logsumexp(logits)
        probs = np.exp(probs)
        
    elif probs is not None:      
        
        # check for length mistmatch
        if len(probs) != len(true_counts):
            raise NoTracebackException(
                "Length of probs does not match length of true_counts")
        
        # check if probs sums to 1
        if abs(1.0 - np.sum(probs)) > 1e-3:
            raise NoTracebackException(
                "'probs' array does not sum to 1")        
    else:
        
        # both 'probs' and 'logits' are None
        raise NoTracebackException(
            "At least one of probs or logits must be provided. "
            "Both are None.")
        
    # convert true_counts to probabilities
    true_counts_prob = true_counts / np.sum(true_counts)
    
    return -np.sum(np.multiply(true_counts_prob, np.log(probs + 1e-7)))


def get_min_max_normalized_value(val, minimum, maximum):
    
    ret_val = (val - minimum) / (maximum - minimum)
    
    if ret_val < 0:
        return 0
    
    if ret_val > 1:
        return 1
    
    return ret_val


def metrics_main():
    # parse the command line arguments
    parser = metrics_argsparser()
    args = parser.parse_args()

    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        raise NoTracebackException(
            "Directory {} does not exist".format(args.output_dir))
    
    # check if the peaks file exists
    if args.peaks is not None and not os.path.exists(args.peaks):
        raise NoTracebackException(
            "File {} does not exist".format(args.peaks))
            
    # check if the bounds file exists
    if args.bounds_csv is not None and not os.path.exists(args.bounds_csv):
        raise NoTracebackException(
            "File {} does not exist".format(args.bounds_csv))
        
    # check if profile A exists
    if not os.path.exists(args.profileA):
        raise NoTracebackException(
            "File {} does not exist".format(args.profileA))
    
    # check if profile B exists
    if not os.path.exists(args.profileB):
        raise NoTracebackException(
            "File {} does not exist".format(args.profileB))

    # check if counts A exists
    if args.countsA is not None and not os.path.exists(args.countsA):
        raise NoTracebackException(
            "File {} does not exist".format(args.countsA))
    
    # check if counts B exists
    if args.countsB is not None and not os.path.exists(args.countsB):
        raise NoTracebackException(
            "File {} does not exist".format(args.countsB))

    # check if we need to auto generate the output directory
    if args.automate_filenames:
        # create a new directory using current date/time to store the
        # metrics outputs & logs 
        date_time_str = local_datetime_str(args.time_zone)
        metrics_dir = '{}/{}'.format(args.output_dir, date_time_str)
        os.mkdir(metrics_dir)
    elif os.path.isdir(args.output_dir):
        metrics_dir = args.output_dir        
    else:
        raise NoTracebackException(
            "{} is not a directory".format(args.output_dir))

    # filename to write debug logs
    logfname = "{}/metrics.log".format(metrics_dir)
    
    # set up the loggers
    init_logger(logfname)

    # read the bounds csv into a pandas DataFrame
    if args.bounds_csv is not None:
        logging.info("Loading lower and upper bounds ...")
        bounds_df = pd.read_csv(args.bounds_csv, header=0)
    else:
        bounds_df = None
    
    # check if peaks file has been supplied
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

        # select only the chrom & summit positon columns
        allPositions = peaks_df[['chrom', 'start_pos', 'end_pos']]

        allPositions = allPositions.reset_index(drop=True)
     
    # else generate geome wide positions
    else:
        
        allPositions = getChromPositions(args.chroms, args.chrom_sizes, 
                                         args.metrics_seq_len // 2, 
                                         mode='sequential',
                                         num_positions=-1, step=args.step_size)

    # check that there are exactly the same number of rows in the 
    # bounds dataframe as compared to allPositions
    if bounds_df is not None and (bounds_df.shape[0] != allPositions.shape[0]):
        raise NoTracebackException(
            "Bounds row count does not match chrom positions row "
            "count".format(args.peaks)) 
     
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
    multinomial_nll = np.zeros(array_len, dtype=np.float64)
    ce = np.zeros(array_len, dtype=np.float64)
    jsd = np.zeros(array_len, dtype=np.float64)
    pearson = np.zeros(array_len, dtype=np.float64)
    spearman = np.zeros(array_len, dtype=np.float64)
    mse = np.zeros(array_len, dtype=np.float64)
    
    for idx, row in tqdm(allPositions.iterrows(), total=allPositions.shape[0]):
        
        chrom = row['chrom']
        start = row['start_pos']
        end = row['end_pos']
        
        # get all the bounds values
        if bounds_df is not None:
            mnll_min = bounds_df.loc[idx, 'mnll_self']
            mnll_max = bounds_df.loc[idx, 'mnll_uniform']
            ce_min = bounds_df.loc[idx, 'ce_self']
            ce_max = bounds_df.loc[idx, 'ce_uniform']
            jsd_min = bounds_df.loc[idx, 'jsd_self']
            jsd_max = bounds_df.loc[idx, 'jsd_uniform']
            pearson_min = bounds_df.loc[idx, 'pearson_uniform']
            pearson_max = bounds_df.loc[idx, 'pearson_self']
            spearman_min = bounds_df.loc[idx, 'spearman_uniform']
            spearman_max = bounds_df.loc[idx, 'spearman_self']
        try:
            profileA = np.nan_to_num(np.array(
                bigWigProfileA.values(chrom, start, end)))
            profileB = np.nan_to_num(np.array(
                bigWigProfileB.values(chrom, start, end)))
        except Exception as e:
            raise NoTracebackException(
                "Error retrieving values {}, {}, {}".format(chrom, start, end))
            
        if args.countsA:
            # since every base is assigned the total counts in the 
            # region we have to take the mean
            valsCountsA = np.mean(np.nan_to_num(
                np.array(bigWigCountsA.values(chrom, start, end))))
        else:
            valsCountsA = np.sum(profileA)

        if args.countsB:
            # since every base is assigned the total counts in the 
            # region we have to take the mean
            valsCountsB = np.mean(np.nan_to_num(
                np.array(bigWigCountsB.values(chrom, start, end))))
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
        
        elif args.exclude_zero_profiles:
                continue
        
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
        
        elif args.exclude_zero_profiles:
                continue
        
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
            pearson[idx] = 0
            spearman[idx] = 0
        else:
            pearson[idx] = pearsonr(valsProfileA, valsProfileB)[0]
            spearman[idx] = spearmanr(valsProfileA, valsProfileB)[0]

        # mnll
        multinomial_nll[idx] = mnll(valsProfileA, probs=probProfileB)
        # cross entropy
        ce[idx] = profile_cross_entropy(valsProfileA, probs=probProfileB)
        # jsd
        jsd[idx] = jensenshannon(probProfileA, probProfileB)
 
        # apply min max normlization
        if bounds_df is not None:
            multinomial_nll[idx] = get_min_max_normalized_value(
                multinomial_nll[idx], mnll_min, mnll_max)
            ce[idx] = get_min_max_normalized_value(ce[idx], ce_min, ce_max)
            jsd[idx] = get_min_max_normalized_value(jsd[idx], jsd_min, jsd_max)
            pearson[idx] = get_min_max_normalized_value(
                pearson[idx], pearson_min, pearson_max)
            spearman[idx] = get_min_max_normalized_value(
                spearman[idx], spearman_min, spearman_max)

        # mse
        mse[idx] = np.square(np.subtract(valsProfileA, valsProfileB)).mean()

        # add to the counts list
        countsA.append(np.sum(valsProfileA))
        countsB.append(np.sum(valsProfileB))

            
    counts_pearson = pearsonr(countsA, countsB)[0]
    counts_spearman = spearmanr(countsA, countsB)[0]
    
    logging.info("\t\tmin\t\tmax\t\tmedian")
    logging.info("mnll\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(multinomial_nll), np.max(multinomial_nll), 
        np.median(multinomial_nll)))
    logging.info("cross_entropy\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(ce), np.max(ce), np.median(ce)))
    logging.info("jsd\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(jsd), np.max(jsd), np.median(jsd)))
    logging.info("pearson\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
       np. min(pearson), np.max(pearson), np.median(pearson)))
    logging.info("spearman\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(spearman), np.max(spearman), np.median(spearman)))
    logging.info("mse\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(mse), np.max(mse), np.median(mse)))
    logging.info("==============================================")
    logging.info("counts pearson: {}".format(counts_pearson))
    logging.info("counts spearman: {}".format(counts_spearman))

    np.savez_compressed('{}/mnll'.format(metrics_dir), mnll=multinomial_nll)
    np.savez_compressed('{}/cross_entropy'.format(metrics_dir), 
                        cross_entropy=ce)
    np.savez_compressed('{}/mse'.format(metrics_dir), mse=mse)
    np.savez_compressed('{}/pearson'.format(metrics_dir), pearson=pearson)
    np.savez_compressed('{}/spearman'.format(metrics_dir), spearman=spearman)
    np.savez_compressed('{}/jsd'.format(metrics_dir), jsd=jsd)
    np.savez_compressed('{}/counts_pearson'.format(metrics_dir), 
                        counts_pearson=counts_pearson)
    np.savez_compressed('{}/counts_spearman'.format(metrics_dir), 
                        counts_spearman=counts_spearman)
    
    
    # write all the command line arguments to a json file
    config_file = '{}/config.json'.format(metrics_dir)
    with open(config_file, 'w') as fp:
        json.dump(vars(args), fp)
    
if __name__ == '__main__':
    metrics_main()

