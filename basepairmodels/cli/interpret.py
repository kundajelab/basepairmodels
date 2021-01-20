import deepdish as dd
import json
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import shap

import tensorflow as tf
import tensorflow_probability as tfp

from keras.models import load_model
from keras.utils import CustomObjectScope

from basepairmodels.cli.argparsers import interpret_argsparser
from basepairmodels.cli.batchgenutils import *
from basepairmodels.cli.bpnetutils import *
from basepairmodels.cli.shaputils import *
from basepairmodels.cli.logger import *
from basepairmodels.cli.losses import MultichannelMultinomialNLL

from scipy.ndimage import gaussian_filter1d

def interpret(args, interpret_dir):
    # load the model
    model = load_model(args.model)
    
    # read all the peaks into a pandas dataframe
    peaks_df = pd.read_csv(args.bed_file, sep='\t', header=None, 
                           names=['chrom', 'st', 'end', 'name', 'score',
                                  'strand', 'signal', 'p', 'q', 'summit'])

    if len(args.chroms) > 0:
        # keep only those rows corresponding to the required 
        # chromosomes
        peaks_df = peaks_df[peaks_df['chrom'].isin(args.chroms)]
           
    if args.sample is not None:
        # randomly sample rows
        print("sampling {} rows".format(args.sample))
        peaks_df = peaks_df.sample(args.sample)
    
    if args.presort_bed_file:
        # sort the bed file in descending order of peak strength
        peaks_df = peaks_df.sort_values(['signal'], 
                                        ascending=False).reset_index(drop=True)
        
    # reference file to fetch sequences
    fasta_ref = pysam.FastaFile(args.reference_genome)

    # if controls have been specified
    if args.controls is not None:
        # log of sum of counts of the control track, (sum of control
        # tracks in case of stranded)
        bias_counts_input = np.zeros((peaks_df.shape[0], 1))

        # the control track (sum of control tracks in case of stranded)
        # and the smooted versions of the control track
        bias_profile_input = np.zeros((peaks_df.shape[0], args.control_len, 
                                      len(args.control_smoothing)))
    

        control_bigwigs = []
        # open the control bigwig files for reading
        print("opening control bigwigs")
        for control_file in args.controls:
            control_bigwigs.append(pyBigWig.open(control_file))

    # get all the sequences for the peaks
    sequences = []
    
    # get a list of valid rows to store only the peaks on which 
    # the contribution scores are computed, excluding those that
    # have may some exceptions, later we'll convert these rows
    # to a dataframe and write to a new file
    rows = []
    
    rowidx = 0
    for idx, row in peaks_df.iterrows():
        
        # fetch sequence
        start = row['st'] + row['summit'] - (args.input_seq_len // 2)
        end =  row['st'] + row['summit'] + (args.input_seq_len // 2)
        
        try:
            seq = fasta_ref.fetch(row['chrom'], start, end).upper()        
        except ValueError: # start/end out of range
            continue
        
        # row passes exception handling
        rows.append(dict(row))
            
        if len(seq) != args.input_seq_len:
            logging.warn("Reference genome doesn't have required sequence " 
                         "length ({}) at peak: {} {}-{}. Returned length {}. "
                         "Skipped.".format(args.input_seq_len, row['chrom'], 
                                           start, end, len(seq)))
            continue        
        sequences.append(seq)
        
        # fetch control values
        if args.controls is not None:
            start = row['st'] + row['summit'] - (args.control_len // 2)
            end =  row['st'] + row['summit'] + (args.control_len // 2)
            for i in range(len(control_bigwigs)):
                vals = control_bigwigs[i].values(row['chrom'], start, end)
                bias_counts_input[rowidx, 0] += np.sum(vals)
                bias_profile_input[rowidx, :, 0] += vals

            # compute the smoothed control values
            for i in range(len(args.control_smoothing)):
                unsmoothed_controls = np.copy(bias_profile_input[rowidx, :, 0])
                if  args.control_smoothing[i] > 1:
                    bias_profile_input[rowidx, :, i] = gaussian_filter1d(
                        unsmoothed_controls, args.control_smoothing[i])
            
        rowidx += 1

    # one hot encode all the sequences
    X = one_hot_encode(sequences)
    print(X.shape)
        
    weightedsum_meannormed_logits = get_weightedsum_meannormed_logits(
        model, task_id=args.task_id, stranded=True)

    def data_func(model_inputs):
        rng = np.random.RandomState(args.seed)
        return [dinuc_shuffle(model_inputs[0], args.num_shuffles, rng)] + \
        [
            np.tile(
                np.zeros_like(model_inputs[i]),
                (args.num_shuffles,) + (len(model_inputs[i].shape) * (1,))
            ) for i in range(1, len(model_inputs))
        ]
    
    profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
        ([model.input[0], model.input[1]], 
         tf.reduce_sum(model.outputs[1], axis=-1)),
        data_func, 
        combine_mult_and_diffref=combine_mult_and_diffref)

    profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
        ([model.input[0], model.input[2]], weightedsum_meannormed_logits),
        data_func, 
        combine_mult_and_diffref=combine_mult_and_diffref)

    logging.info("Generating 'counts' shap scores")
    counts_shap_scores = profile_model_counts_explainer.shap_values(
        [X, np.zeros((X.shape[0], 1))], progress_message=100)
    
    # construct a dictionary for the 'counts' shap scores & the
    # the projected 'counts' shap scores
    # MODISCO workflow expects one hot sequences with shape (?,4,1000)
    projected_shap_scores = np.multiply(X, counts_shap_scores[0])
    counts_scores = {'raw': {'seq': np.transpose(X, (0, 2, 1))},
                     'shap':
                     {'seq': np.transpose(counts_shap_scores[0], (0, 2, 1))},
                     'projected_shap':
                     {'seq': np.transpose(projected_shap_scores, (0, 2, 1))}}
    
    # save the dictionary in HDF5 formnat
    logging.info("Saving 'counts' scores")
    dd.io.save('{}/counts_scores.h5'.format(interpret_dir), counts_scores)
    
    logging.info("Generating 'profile' shap scores")
    profile_shap_scores = profile_model_profile_explainer.shap_values(
        [X, np.zeros((X.shape[0], args.control_len, 2))], progress_message=100)
          
    # construct a dictionary for the 'profile' shap scores & the
    # the projected 'profile' shap scores
    projected_shap_scores = np.multiply(X, profile_shap_scores[0])
    profile_scores = {'raw': {'seq': np.transpose(X, (0, 2, 1))},
                     'shap':
                      {'seq': np.transpose(profile_shap_scores[0], (0, 2, 1))},
                     'projected_shap':
                      {'seq': np.transpose(projected_shap_scores, (0, 2, 1))}}
    
    # save the dictionary in HDF5 formnat
    logging.info("Saving 'profile' scores")
    dd.io.save('{}/profile_scores.h5'.format(interpret_dir), profile_scores)

    # create dataframe from all rows that were sucessfully processed
    df_valid_scores = pd.DataFrame(rows) 
    
    # save the dataframe as a new .bed file 
    df_valid_scores.to_csv('{}/peaks_valid_scores.bed'.format(interpret_dir), 
                           sep='\t', header=False, index=False)
    
    # write all the command line arguments to a json file
    config_file = '{}/config.json'.format(interpret_dir)
    with open(config_file, 'w') as fp:
        config = vars(args)
        json.dump(config, fp)
        
        
def interpret_main():
    # parse the command line arguments
    parser = interpret_argsparser()
    args = parser.parse_args()

    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        logging.error("Directory {} does not exist".format(
            args.output_dir))
        return
    
    # create a new directory using current date/time to store the
    # interpretation scores
    date_time_str = local_datetime_str(args.time_zone)
    interpret_dir = '{}/{}'.format(args.output_dir, date_time_str)
    os.mkdir(interpret_dir)
    
    # filename to write debug logs
    logfname = "{}/interpret.log".format(interpret_dir)
    
    # set up the loggers
    init_logger(logfname)
    
    # interpret
    logging.info("Loading {}".format(args.model))
    with CustomObjectScope({'MultichannelMultinomialNLL': 
                            MultichannelMultinomialNLL}):
            
        interpret(args, interpret_dir)
    
if __name__ == '__main__':
    interpret_main()
