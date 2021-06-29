"""
    Python script for network training via the CLI

    License:
    
    MIT License

    Copyright (c) 2021 Kundaje Lab

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


import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import time

from basepairmodels.cli import argparsers
from basepairmodels.cli import bigwigutils
from basepairmodels.cli import logger
from basepairmodels.cli.exceptionhandler import NoTracebackException
from basepairmodels.cli.losses import MultichannelMultinomialNLL
from basepairmodels.cli.losses import multinomial_nll
from basepairmodels.common.attribution_prior import AttributionPriorModel
from mseqgen import generators
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm

def predict(args, input_data, pred_dir):    
    # load the model
    model = load_model(args.model)

    # input params
    input_params = {}
    input_params['data'] = args.input_data
    input_params['stranded'] = args.stranded
    input_params['has_control'] = args.has_control

    # network params
    network_params = {}
    network_params['control_smoothing'] = args.control_smoothing
    
    # parameters that are specific to the batch generation process.
    # for prediction we dont use jitter, reverse complement 
    # augmentation and negative sampling
    batch_gen_params = {
        'sequence_generator_name': args.sequence_generator_name,
        'input_seq_len': args.input_seq_len,
        'output_len': args.output_len,
        'sampling_mode': 'peaks',
        'shuffle': False,
        'max_jitter': 0, 
        'rev_comp_aug': False,
        'negative_sampling_rate': 0.0, 
        'mode': 'test'}
    
    # get the corresponding batch generator class
    sequence_generator_class_name = generators.find_generator_by_name(
        batch_gen_params['sequence_generator_name'])
    logging.info("SEQGEN Class Name: {}".format(sequence_generator_class_name))
    BatchGenerator = getattr(generators, sequence_generator_class_name)

    # instantiate the batch generator class for training
    test_gen = BatchGenerator(input_params, batch_gen_params,
                               args.reference_genome, 
                               args.chrom_sizes,
                               args.chroms, 
                               num_threads=1,
                               epochs=1, 
                               batch_size=args.batch_size,
                               **network_params)
    
    # testing generator function
    test_generator = test_gen.gen(1)

    # begin time for training
    t1 = time.time()
              
    # extract the basename from the model filename and 
    # remove the extension
    model_tag = os.path.basename(args.model).split('.')[0]
    
    # prepare bigWig files for writing
    prepare_output_files = getattr(
        bigwigutils, 'prepare_{}_output_files'.format(args.model_name))
    (profile_fileobjs, counts_fileobjs) = prepare_output_files(
        input_data, pred_dir, args.chroms, args.chrom_sizes, model_tag, 
        args.exponentiate_counts, args.other_tags) 
    
    # get the correspoding write function for this model
    write_predictions =  getattr(
        bigwigutils, 'write_{}_predictions'.format(args.model_name))

    # write buffers - we'll use a list of pandas dataframes 
    # each dataframe is a write buffer for one task
    profile_write_buffers = []
    counts_write_buffers = []
    
    # initialize the write buffers
    for i in range(len(input_data)):
        profile_write_buffers.append(pd.DataFrame(
            columns=['chrom', 'starts', 'ends', 'vals_sum', 'counts']))
        counts_write_buffers.append(pd.DataFrame(
            columns=['chrom', 'starts', 'ends', 'vals_sum', 'counts']))

        # set the index of the write buffer
        profile_write_buffers[i] = profile_write_buffers[i].set_index(
            ['chrom', 'starts', 'ends'])
        counts_write_buffers[i] = counts_write_buffers[i].set_index(
            ['chrom', 'starts', 'ends'])
    
    # total number of batches that will be generated
    num_batches = test_gen.len()

    # run predict on each batch separately
    cnt_batches = 0
    for coordinates, _, batch in tqdm(
        test_generator, desc='batch', total=num_batches):

        # predict on the batch
        predictions = model.predict(batch)
        
        # for each task
        for j in range(len(input_data)):

            # lists of entries to be added to the bigwigs
            # we'll merge these values into the write buffers later
            chroms = []
            starts = []
            ends = []
            profile_vals = []
            counts_vals = []
                    
            # for each coordinate in the batch
            for idx in range(len(coordinates)):
                (chrom, start, end) = coordinates[idx]

                # mid section in chrom coordinates based on 
                # output_window_size
                _start = start + (end - start) // 2 - \
                            args.output_window_size // 2  
                _end = start + args.output_window_size

                # for each base
                for i in range(_start, _end):
                    
                    # get the profile value for this task
                    # i - start gives an index between 0 and output_len
                    # and takes into account the output_window_size
                    profile_val =  predictions[0][idx][i-start][j]
                    
                    # get the counts value for this task
                    counts_val = predictions[1][idx][j]
                    
                    if args.exponentiate_counts:
                        counts_val = np.exp(counts_val)
                    
                    chroms.append(chrom)
                    starts.append(i)
                    ends.append(i+1)
                    profile_vals.append(profile_val)
                    counts_vals.append(counts_val)

            # create dataframe from the lists of values created
            # earlier
            tmp_profile_vals_df = pd.DataFrame(
                {'chrom': chroms, 'starts':starts, 'ends': ends, 
                 'vals_sum':profile_vals, 'counts': np.ones(len(chroms))})
            tmp_counts_vals_df = pd.DataFrame(
                {'chrom': chroms, 'starts':starts,'ends': ends, 
                 'vals_sum':counts_vals, 'counts': np.ones(len(chroms))})

            # create indices for the above dataframes
            tmp_profile_vals_df = tmp_profile_vals_df.set_index(
                ['chrom', 'starts', 'ends'])
            tmp_counts_vals_df = tmp_counts_vals_df.set_index(
                ['chrom', 'starts', 'ends'])

            # merge values into the write buffer, entries with same
            # index will get added
            profile_write_buffers[j] = profile_write_buffers[j].add(
                tmp_profile_vals_df, fill_value=0)
            counts_write_buffers[j] = counts_write_buffers[j].add(
                tmp_counts_vals_df, fill_value=0)

        cnt_batches += 1

        # if the write buffers have reached capacity or we have
        # reached the end, its time to flush the write buffer to
        # the bigWigs
        if (len(profile_write_buffers[0].index) >= args.write_buffer_size) or \
            (cnt_batches == num_batches):
            
            for j in range(len(input_data)):
                # the final value is an average in cases where
                # prediction windows overlap
                profile_write_buffers[j]['val'] = \
                    profile_write_buffers[j]['vals_sum'] / \
                        profile_write_buffers[j]['counts']
                counts_write_buffers[j]['val'] = \
                    counts_write_buffers[j]['vals_sum'] / \
                        counts_write_buffers[j]['counts']

                # if we reached the end, then the all the index 
                # positions in the write buffer will written to the 
                # bigWig
                if (cnt_batches == num_batches):
                    profile_write_index = profile_write_buffers[j].index
                    counts_write_index = counts_write_buffers[j].index
                # if not, then we only take those index postions that 
                # will not get updated in future iterations (the 
                # last output_window_size chunk could get updated)
                else:
                    profile_write_index = \
                      profile_write_buffers[j][:-args.output_window_size].index
                    counts_write_index = \
                      counts_write_buffers[j][:-args.output_window_size].index

                # convert nans to 0's
                profile_write_buffers[j]['val'] = \
                    profile_write_buffers[j]['val'].fillna(0)
                counts_write_buffers[j]['val'] = \
                    counts_write_buffers[j]['val'].fillna(0)
                
                # get all the chroms, starts, ends & values that 
                # will be written to the profile & counts bigWigs
                profile_chroms = profile_write_index.get_level_values(0)
                profile_starts = profile_write_index.get_level_values(1)
                profile_ends = profile_write_index.get_level_values(2)
                profile_vals = \
                    profile_write_buffers[j].loc[profile_write_index]['val']
                counts_chroms = counts_write_index.get_level_values(0)
                counts_starts = counts_write_index.get_level_values(1)
                counts_ends = counts_write_index.get_level_values(2)
                counts_vals = \
                    counts_write_buffers[j].loc[counts_write_index]['val']
                
                # convert to lists and add the entries
                try:
                    profile_fileobjs[j].addEntries(profile_chroms.tolist(), 
                                                   profile_starts.tolist(),
                                                   ends=profile_ends.tolist(),
                                                   values=profile_vals.tolist())  
                except RuntimeError:
                    print("j", j,
                          "profile chroms", profile_chroms.tolist(), "\n",
                          "profile starts", profile_starts.tolist(), "\n",
                          "profile ends", profile_ends.tolist(), "\n", 
                          "profile vals", profile_vals.tolist())
                    sys.exit(0)
                    
                try:
                    counts_fileobjs[j].addEntries(counts_chroms.tolist(), 
                                                  counts_starts.tolist(),
                                                  ends=counts_ends.tolist(),
                                                  values=counts_vals.tolist())
                except RuntimeError:
                    print("j", j,
                          "counts chroms", counts_chroms.tolist(), "\n",
                          "counts starts", counts_starts.tolist(), "\n",
                          "counts ends", counts_ends.tolist(), "\n", 
                          "counts vals", counts_vals.tolist())
                    sys.exit(0)
                    
                
                # delete the rows that have now been written to the 
                # bigWig files
                profile_write_buffers[j] = profile_write_buffers[j].drop(
                    index=profile_write_index)
                counts_write_buffers[j] = counts_write_buffers[j].drop(
                    index=counts_write_index)

    # close bigWig files
    for file_obj in profile_fileobjs:
        file_obj.close()
    for file_obj in counts_fileobjs:
        file_obj.close()

    # end time for training
    t2 = time.time() 
    logging.info('Total Elapsed Time: {} secs'.format(t2-t1))

    # write all the command line arguments to a json file
    config_file = '{}/config.json'.format(pred_dir)
    with open(config_file, 'w') as fp:
        json.dump(vars(args), fp)
        
def predict_main():
    # parse the command line arguments
    parser = argparsers.predict_argsparser()
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
    
    # make sure the input_data json file exists
    if not os.path.isfile(args.input_data):
        raise NoTracebackException(
            "File not found: {} OR you may have accidentally "
            "specified a directory path.".format(args.input_data))
        
    # load the json file
    with open(args.input_data, 'r') as inp_json:
        try:
            #: dictionary of tasks for training
            input_data = json.loads(inp_json.read())
        except json.decoder.JSONDecodeError:
            raise NoTracebackException(
                "Unable to load json file {}. Valid json expected. "
                "Check the file for syntax errors.".format(
                    args.input_data))

    logging.info("INPUT DATA -\n{}".format(input_data))

    # predict
    logging.info("Loading {}".format(args.model))
    with CustomObjectScope({'MultichannelMultinomialNLL': 
                            MultichannelMultinomialNLL, 
                            'AttributionPriorModel': AttributionPriorModel}):
            
        predict(args, input_data, pred_dir)
    
if __name__ == '__main__':
    predict_main()
