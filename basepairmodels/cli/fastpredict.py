import datetime
import h5py
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import sys
import time

from basepairmodels.cli import argparsers
from basepairmodels.cli import bigwigutils
from basepairmodels.cli import logger

from basepairmodels.cli.bpnetutils import *
from basepairmodels.cli.exceptionhandler import NoTracebackException
from basepairmodels.cli.losses import MultichannelMultinomialNLL, multinomial_nll
from basepairmodels.cli.metrics import mnll, profile_cross_entropy
from mseqgen import generators
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp, softmax
from scipy.stats import pearsonr, spearmanr, multinomial
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope


def _fix_sum_to_one(probs):
    """
      Fix probability arrays whose sum is fractinally above or 
      below 1.0
      
      Args:
          probs (numpy.ndarray): An array whose sum is almost equal
              to 1.0
              
      Returns:
          np.ndarray: array that sums to 1
    """
    
    _probs = np.copy(probs)
    
    if np.sum(_probs) > 1.0:        
        _probs[np.argmax(_probs)] -= np.sum(_probs) - 1.0    
    
    if np.sum(_probs) < 1.0:
        _probs[np.argmin(_probs)] += 1.0 - np.sum(_probs)
    
    return _probs
    
def mnll_min_max_bounds(profile):
    """
        Min Max bounds for the mnll metric
        
        Args:
            profile (numpy.ndarray): true profile 
        Returns:
            tuple: (min, max) bounds values
    """
    
    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile = profile.astype(np.float64)
    
    # profile as probabilities
    profile_prob = profile / np.sum(profile)
    
    # the scipy.stats.multinomial function is very sensitive to 
    # profile_prob summing to exactly 1.0, if not you get NaN as the
    # resuls. In majority of the cases we can fix that problem by
    # adding or substracting the difference (but unfortunately it
    # doesnt always and there are cases where we still see NaNs, and
    # those we'll set to 0)
    profile_prob = _fix_sum_to_one(profile_prob)

    # mnll of profile with itself
    min_mnll = mnll(profile, probs=profile_prob)
    
    # if we still find a NaN, even after the above fix, set it to zero
    if math.isnan(min_mnll):
        min_mnll = 0.0

    if math.isinf(min_mnll):
        min_mnll = 0.0

    # mnll of profile with uniform profile
    max_mnll = mnll(profile, probs=uniform_profile)
    
    if math.isnan(max_mnll):
        print("max_mnll is nan")

    return (min_mnll, max_mnll)


def cross_entropy_min_max_bounds(profile):
    """
        Min Max bounds for the cross entropy metric
        
        Args:
            profile (numpy.ndarray): true profile 
            
        Returns:
            tuple: (min, max) bounds values
    """

    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # mnll of profile with itself
    min_cross_entropy = profile_cross_entropy(profile, probs=profile_prob)
    
    # mnll of profile with uniform profile
    max_cross_entropy = profile_cross_entropy(profile, probs=uniform_profile)

    return (min_cross_entropy, max_cross_entropy)


def jsd_min_max_bounds(profile):
    """
        Min Max bounds for the jsd metric
        
        Args:
            profile (numpy.ndarray): true profile 
            
        Returns:
            tuple: (min, max) bounds values
    """
    
    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # jsd of profile with uniform profile
    max_jsd = jensenshannon(profile_prob, uniform_profile)

    # jsd of profile with itself (upper bound)
    min_jsd = 0.0

    return (min_jsd, max_jsd)

def min_max_normalize(metric_value, min_max_bounds, default=1.0):
    """
        Min Max normalize a metric value
        
        Args:
            metric_value (float): unnormalized metric value
            min_max_bounds (tuple): (min, max) bounds
            default (float): the default value to return in case
                of runtime exceptions
    """
    
    min_bound = min_max_bounds[0]
    max_bound = min_max_bounds[1]
    
    if (max_bound - min_bound) == 0:
        return default
    
    norm_value = (metric_value - min_bound) / (max_bound - min_bound) 
    
    if norm_value < 0:
        return 0
    
    if norm_value > 1:
        return 1
    
    return norm_value


def metrics_update(
    metrics_tracker, true_profile, true_logcounts, pred_profile,
    pred_logcounts, true_profile_smoothing=[7.0, 81]):
    """
        Update metrics with new true/predicted profile & counts
        
        Args:
            metrics_tracker (dict): dictionary to track & update 
                metrics info 
            true_profile (numpy.ndarray): ground truth profile
            true_counts (float): sum of true profile
            pred_profile (numpy.ndarray): predicted profile output
            pred_counts (float): predicted 
            true_profile_smoothing (list): list of 2 values, sigma &
                window size for gaussuan 1d smoothing
            
    
    """
    
    # Step 1 - smooth true profile
    sigma = true_profile_smoothing[0]
    width = float(true_profile_smoothing[1])
    truncate = (((width - 1)/2)-0.5)/sigma

    # Step 2 - smooth true profile and convert profiles to 
    # probabilities
    if np.sum(true_profile) != 0 and np.sum(pred_profile) != 0:
        # smoothing
        true_profile_smooth = gaussian_filter1d(
            true_profile, sigma=sigma, truncate=truncate)

        # convert to probabilities
        true_profile_smooth_prob = true_profile_smooth / np.sum(
            true_profile_smooth)
        pred_profile_prob = pred_profile / np.sum(pred_profile)
        pred_profile_prob = _fix_sum_to_one(pred_profile_prob)

        # metrics 
        # profile pearson & spearman
        # with pearson we need to check if either of the arrays
        # has zero standard deviation (i.e having all same elements,
        # a zero or any other value). Unfortunately np.std
        # returns a very small non-zero value, so we'll use a 
        # different approach to check if the array has the same value.
        # If true then pearson correlation is undefined 
        if np.unique(true_profile_smooth_prob).size == 1 or \
            np.unique(pred_profile_prob).size == 1:
            metrics_tracker['profile_pearsonrs'].append(0)
            metrics_tracker['profile_spearmanrs'].append(0)
        else:
            metrics_tracker['profile_pearsonrs'].append(
                pearsonr(true_profile_smooth_prob, pred_profile_prob)[0])

            metrics_tracker['profile_spearmanrs'].append(
                spearmanr(true_profile_smooth_prob, pred_profile_prob)[0])
            
        # mnll
        _mnll = np.nan_to_num(mnll(true_profile, probs=pred_profile_prob))
        _mnll = min_max_normalize(_mnll, mnll_min_max_bounds(true_profile))
        if math.isnan(_mnll):
            print("found nan")
        metrics_tracker['profile_mnlls'].append(_mnll)

        # cross entropy
        metrics_tracker['profile_cross_entropys'].append(
            min_max_normalize(
                profile_cross_entropy(true_profile, probs=pred_profile_prob), 
                cross_entropy_min_max_bounds(true_profile)))

        # jsd
        metrics_tracker['profile_jsds'].append(
            min_max_normalize(
                jensenshannon(true_profile_smooth_prob, pred_profile_prob), 
                jsd_min_max_bounds(true_profile)))

        # mse
        metrics_tracker['profile_mses'].append(
            np.square(np.subtract(true_profile, pred_profile)).mean())
    else:
        metrics_tracker['profile_pearsonrs'].append(0)
        metrics_tracker['profile_spearmanrs'].append(0)
        metrics_tracker['profile_mnlls'].append(0)
        metrics_tracker['profile_cross_entropys'].append(0)
        metrics_tracker['profile_jsds'].append(0)
        metrics_tracker['profile_mses'].append(0)
        
            

        
    metrics_tracker['all_true_logcounts'].append(true_logcounts)
    metrics_tracker['all_pred_logcounts'].append(pred_logcounts)


def predict(args, pred_dir):    
    # load the model
    model = load_model(args.model)

    # parameters that are specific to the batch generation process.
    # for prediction we dont use jitter, reverse complement 
    # augmentation and negative sampling
    batch_gen_params = {}    
    batch_gen_params['sequence_generator_name'] = args.sequence_generator_name
    batch_gen_params['input_seq_len'] = args.input_seq_len
    batch_gen_params['output_len'] = args.output_len
    batch_gen_params['rev_comp_aug'] = False
    batch_gen_params['negative_sampling_rate'] = 0.0
    batch_gen_params['max_jitter'] = 0
    batch_gen_params['shuffle'] = False
    batch_gen_params['mode'] = 'test'

    # instantiate the batch generator class for testing
        # get the corresponding batch generator class for this model
    sequence_generator_class_name = generators.find_generator_by_name(
        batch_gen_params['sequence_generator_name'])
    logging.info("SEQGEN Class Name: {}".format(sequence_generator_class_name))
    BatchGenerator = getattr(generators, sequence_generator_class_name)

    test_gen = BatchGenerator(
        args.input_data, batch_gen_params, args.reference_genome, 
        args.chrom_sizes, args.chroms, num_threads=args.threads, 
        batch_size=args.batch_size)

    # testing generator function
    test_generator = test_gen.gen()

    # extract the basename from the model filename and 
    # remove the extension
    model_tag = os.path.basename(args.model).split('.')[0]
    
    # total number of batches that will be generated
    num_batches = test_gen.len()
    
    # number of tracks in the prediction output 
    num_output_tracks = test_gen._total_signal_tracks
    
    # hash table keep track of predicted coordinates, the key is
    # "chrom" + '_' + start and is mapped to a running index of
    # the returned predictions
    # Since in 'test' mode the batch generator pads with extra rows
    # this will ensure that wedont duplicate the predictions
    coordinate_hash = {}
    
    # open h5 file for writing predictions    
    output_h5_fname = "{}/{}_predictions.h5".format(args.output_dir, model_tag)    
    h5_file = h5py.File(output_h5_fname, "w")
    logging.info("Writing predictions to {}".format(output_h5_fname))

    # create groups 
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")
    
    # the batch generator pads data with extra rows for efficient
    # batch generation with uniform load across multiple threads,
    # but the hdf5 will/should only have the predictions from the 
    # original peaks samples (unpadded data)
    num_examples = test_gen.get_samples_len()

    # create the "coords" group datasets
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom", (num_examples,),
        dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip")
    coords_start_dset = coord_group.create_dataset(
        "coords_start", (num_examples,), dtype=int, compression="gzip")
    coords_end_dset = coord_group.create_dataset(
        "coords_end", (num_examples,), dtype=int, compression="gzip")
    
    # create the "predictions" group datasets
    pred_profs_dset = pred_group.create_dataset(
        "pred_profs", 
        (num_examples, args.output_window_size, num_output_tracks),
        dtype=float, compression="gzip")
    pred_logcounts_dset = pred_group.create_dataset(
        "pred_logcounts", (num_examples, num_output_tracks),
        dtype=float, compression="gzip")
    true_profs_dset = pred_group.create_dataset(
        "true_profs", 
        (num_examples, args.output_window_size, num_output_tracks),
        dtype=float, compression="gzip")
    true_logcounts_dset = pred_group.create_dataset(
        "true_logcounts", (num_examples, num_output_tracks), 
        dtype=float, compression="gzip")
    
    # begin time for predictions
    t1 = time.time()
    
    metrics_tracker = {
        # profile metrics 
        'profile_mnlls': [],
        'profile_cross_entropys': [],
        'profile_jsds': [],
        'profile_mses': [],
        'profile_pearsonrs': [],
        'profile_spearmanrs': [],

        # for counts correlation
        'all_true_logcounts': [],
        'all_pred_logcounts': []
    }
    
    # running counter to count all non repeating examples across
    # all batches
    cnt_examples = 0

    # run predict on each batch separately
    for batch in tqdm(test_generator, desc='batch', total=num_batches):
        
        coordinates = batch['coordinates']
        true_profiles = batch['true_profiles']
        true_logcounts = batch['true_logcounts']

        # predict on the batch
        predictions = model.predict(batch)
        
        # arrays to hold required values for each batch before we 
        # write to HDF5 file
        pred_profiles = np.zeros(
            (args.batch_size, args.output_window_size, num_output_tracks))
        pred_logcounts = np.zeros((args.batch_size, num_output_tracks))
        
        # lists for chrom positions for each batch
        chroms = []
        starts = []
        ends = []
        
        # count the number of valid non repeating (padded examples)
        # in this batch
        cnt_batch_examples = 0
        
        # for each coordinate in the batch
        for idx in range(len(coordinates)):
            (chrom, start, end) = coordinates[idx]
            start = int(start)
            end = int(end)
                
            hash_key = chrom + '_' + str(start)
                
            # skip this prediction if we have already seen it
            if hash_key in coordinate_hash:
                continue
                
            # set hash value to current example index
            coordinate_hash[hash_key] = cnt_examples + cnt_batch_examples
            
            # append the chrom positions
            chroms.append(chrom)
            starts.append(start)
            ends.append(end)

            # process predictions for each track
            for j in range(num_output_tracks):

                # mid section of profiles based on 
                # args.output_window_size, if user decides they dont
                # want the full output to written to HDF5/bigWig
                _start = args.output_len // 2 - args.output_window_size // 2  
                _end = _start + args.output_window_size
                
                # counts prediction
                logcounts_prediction = predictions[1][idx, j]
                pred_logcounts[cnt_batch_examples, j] = logcounts_prediction

                # predicted profile
                pred_profile_logits = predictions[0][idx, _start:_end, j]
                pred_profiles[cnt_batch_examples, :, j] = np.exp(
                    pred_profile_logits - logsumexp(pred_profile_logits)) * \
                    (np.exp(logcounts_prediction) - 1)
            
                # true profile
                true_profiles[cnt_batch_examples, :, j] = \
                    true_profiles[idx, _start:_end, j]

                metrics_update(
                    metrics_tracker,
                    true_profiles[idx, :, j],
                    true_logcounts[idx, j],
                    pred_profiles[cnt_batch_examples, :, j],
                    pred_logcounts[cnt_batch_examples, j])
                
                ### 
                
            # increment the batch examples counter
            cnt_batch_examples += 1

        # assign values at correct index locations 
        # in the hdf5 datsets
        start_idx = cnt_examples
        end_idx = cnt_examples + cnt_batch_examples
        pred_profs_dset[start_idx:end_idx, :, :] = \
            pred_profiles[:cnt_batch_examples]
        pred_logcounts_dset[start_idx:end_idx, :] = \
            pred_logcounts[:cnt_batch_examples]
        true_profs_dset[start_idx:end_idx, :, :] = \
            true_profiles[:cnt_batch_examples]
        true_logcounts_dset[start_idx:end_idx, :] = \
            true_logcounts[:cnt_batch_examples]

        coords_chrom_dset[start_idx:end_idx] = chroms
        coords_start_dset[start_idx:end_idx] = starts
        coords_end_dset[start_idx:end_idx] = ends

        # increment the total examples counter
        cnt_examples += cnt_batch_examples

    # close hdf5 file
    h5_file.close()

    # end time for writing predictions to hdf5 file
    t2 = time.time() 
    logging.info('Elapsed Time: {} secs'.format(t2-t1))
    print('cnt_examples', cnt_examples)

#     if args.generate_predicted_profile_bigWigs:
        
#         chrom_sizes_df = pd.read_csv(
#             args.chrom_sizes, sep='\t', header=None, names=['chrom', 'size']) 
#         chrom_sizes_df = chrom_sizes_df.set_index('chrom')
        
#         h5_file = h5py.File(output_h5_fname, 'r')

#         # construct header for the bigWig file
#         header = []
#         # sort chromosomes, to be consistent with how pandas sorts
#         # chromosomes ... for e.g. chrom21 is < chrom8
#         chroms = args.chroms[:]
#         chroms.sort()
#         for chrom in chroms:
#             size = chrom_sizes_df.at[chrom, 'size']
#             header.append((chrom, int(size)))

# #         chrom_sizes_df = 
#         model_tasks = test_gen.get_input_tasks()
#         for model_task_name in model_tasks:
#             task_id = model_tasks[model_task_name]['task_id']
#             strand_id = model_tasks[model_task_name]['strand']
            
# #         # create new bigWig
        
# #         # write header
        
#             for chrom in chroms:
#                 size = chrom_sizes_df.at[chrom, 'size']
#                 h5_file['coords_chrom'] 
                
#                 indices = chrom_sizes_df[
#                     chrom_sizes_df['chrom'] == chrom].index.values
#                 log_pred_profs = h5_file['log_pred_profs'][
#                     indices, task_id,:,strand_id]
#                 print(chrom, log_pred_profs.shape)
                
    # write all the command line arguments to a json file
    config_file = '{}/config.json'.format(pred_dir)
    with open(config_file, 'w') as fp:
        json.dump(vars(args), fp)
                    
            
    logging.info("\t\tmin\t\tmax\t\tmedian")
    
    logging.info("mnll\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format( 
        np.min(metrics_tracker['profile_mnlls']), 
        np.max(metrics_tracker['profile_mnlls']), 
        np.median(metrics_tracker['profile_mnlls'])))
                 
    logging.info("cross_entropy\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(metrics_tracker['profile_cross_entropys']), 
        np.max(metrics_tracker['profile_cross_entropys']), 
        np.median(metrics_tracker['profile_cross_entropys'])))
    
    logging.info("jsd\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(metrics_tracker['profile_jsds']), 
        np.max(metrics_tracker['profile_jsds']), 
        np.median(metrics_tracker['profile_jsds'])))
    
    logging.info("pearson\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(metrics_tracker['profile_mses']), 
        np.max(metrics_tracker['profile_mses']), 
        np.median(metrics_tracker['profile_mses'])))
    
    logging.info("spearman\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(metrics_tracker['profile_pearsonrs']), 
        np.max(metrics_tracker['profile_pearsonrs']), 
        np.median(metrics_tracker['profile_pearsonrs'])))
    
    logging.info("mse\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(
        np.min(metrics_tracker['profile_spearmanrs']),
        np.max(metrics_tracker['profile_spearmanrs']), 
        np.median(metrics_tracker['profile_spearmanrs'])))
    
    logging.info("==============================================")
    counts_pearson = pearsonr(metrics_tracker['all_true_logcounts'], 
                              metrics_tracker['all_pred_logcounts'])[0]
    logging.info("counts pearson: {}".format(counts_pearson))
    counts_spearman = spearmanr(metrics_tracker['all_true_logcounts'], 
                                metrics_tracker['all_pred_logcounts'])[0]
    logging.info("counts spearman: {}".format(counts_spearman))

    np.savez_compressed(
        '{}/mnll'.format(args.output_dir), 
        mnll=metrics_tracker['profile_mnlls'])
    np.savez_compressed(
        '{}/cross_entropy'.format(args.output_dir), 
        cross_entropy=metrics_tracker['profile_cross_entropys'])
    np.savez_compressed(
        '{}/jsd'.format(args.output_dir), jsd=metrics_tracker['profile_jsds'])
    np.savez_compressed(
        '{}/mse'.format(args.output_dir), mse=metrics_tracker['profile_mses'])
    np.savez_compressed(
        '{}/pearson'.format(args.output_dir), 
        pearson=metrics_tracker['profile_pearsonrs'])
    np.savez_compressed(
        '{}/spearman'.format(args.output_dir), 
        spearman=metrics_tracker['profile_spearmanrs'])
    np.savez_compressed('{}/counts_pearson'.format(args.output_dir), 
                        counts_pearson=counts_pearson)
    np.savez_compressed('{}/counts_spearman'.format(args.output_dir), 
                        counts_spearman=counts_spearman)        
         

def predict_main():
    # parse the command line arguments
    parser = argparsers.fastpredict_argsparser()
    args = parser.parse_args()

    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        logging.error("Directory {} does not exist".format(
            args.output_dir))
        
        return

    if args.automate_filenames:
        # create a new directory using current date/time to store the
        # predictions and logs
        date_time_str = local_datetime_str(args.time_zone)
        pred_dir = '{}/{}'.format(args.output_dir, date_time_str)
        os.mkdir(pred_dir)
    elif os.path.isdir(args.output_dir):
        pred_dir = args.output_dir        
    else:
        logging.error("Directory does not exist {}.".format(args.output_dir))
        return

    # filename to write debug logs
    logfname = "{}/predict.log".format(pred_dir)
    
    # set up the loggers
    logger.init_logger(logfname)

    # predict
    logging.info("Loading {}".format(args.model))
    with CustomObjectScope({'MultichannelMultinomialNLL': 
                            MultichannelMultinomialNLL}):
            
        predict(args, pred_dir)
    
if __name__ == '__main__':
    predict_main()
