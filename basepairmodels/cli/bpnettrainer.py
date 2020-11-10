"""
    Python script for network training via the CLI

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


from basepairmodels.cli import argparsers
import json
import logging
import multiprocessing as mp
import os
import sys

stderr = sys.stderr
sys.stderr = open('keras.stderr', 'w')
from basepairmodels.common import model_archs, training
sys.stderr = stderr
from mseqgen import quietexception


def main():
    # change the way processes are started, default = 'fork'
    # had to do this to prevent Keras multi gpu from deadlocking
    mp.set_start_method('forkserver')
    
    # inform user of the keras stderr log file
    logging.warning("For all keras related error logs refer to "
                    "keras.stderr in your local directory")

    # parse the command line arguments
    parser = argparsers.training_argsparser()
    args = parser.parse_args()

    # input params
    input_params = {}
    input_params['data'] = args.input_data
    input_params['stranded'] = args.stranded
    input_params['has_control'] = args.has_control

    # output params 
    output_params = {}
    output_params['automate_filenames'] = args.automate_filenames
    output_params['time_zone'] = args.time_zone
    output_params['tag_length'] = args.tag_length
    output_params['output_dir'] = args.output_dir
    output_params['model_output_filename']= args.model_output_filename
    
    # genome params
    genome_params = {}
    genome_params['reference_genome'] = args.reference_genome
    genome_params['chrom_sizes'] = args.chrom_sizes
    genome_params['chroms'] = args.chroms
    genome_params['exclude_chroms'] = args.exclude_chroms

    # batch generation parameters
    batch_gen_params = {}    
    batch_gen_params['sequence_generator_name'] = args.sequence_generator_name
    batch_gen_params['input_seq_len'] = args.input_seq_len
    batch_gen_params['output_len'] = args.output_len
    batch_gen_params['sampling_mode'] = args.sampling_mode
    batch_gen_params['rev_comp_aug'] = args.reverse_complement_augmentation
    batch_gen_params['negative_sampling_rate'] = args.negative_sampling_rate
    batch_gen_params['max_jitter'] = args.max_jitter
    batch_gen_params['shuffle'] = args.shuffle
    
    # hyper parameters
    hyper_params = {}
    hyper_params['epochs'] = args.epochs
    hyper_params['batch_size'] = args.batch_size
    hyper_params['learning_rate'] = args.learning_rate
    hyper_params['min_learning_rate'] = args.min_learning_rate
    hyper_params['early_stopping_patience'] = args.early_stopping_patience
    hyper_params['early_stopping_min_delta'] = args.early_stopping_min_delta
    hyper_params['reduce_lr_on_plateau_patience'] = \
        args.reduce_lr_on_plateau_patience
    
    # parallelization parms
    parallelization_params = {}
    parallelization_params['threads'] = args.threads
    parallelization_params['gpus'] = args.gpus
    
    # network params
    network_params = {}
    network_params['name'] = args.model_arch_name
    network_params['filters'] = args.filters
    network_params['counts_loss_weight'] = args.counts_loss_weight
    network_params['control_smoothing'] = args.control_smoothing
    
    if not os.path.exists(output_params['output_dir']):
        raise quietexception.QuietException(
            "Directory {} does not exist".format(output_params['output_dir']))

    if not output_params['automate_filenames'] and \
        output_params['automate_filenames'] is None:
        raise quietexception.QuietException(
            "Model output filename not specified")

    if not os.path.exists(genome_params['reference_genome']):
        raise quietexception.QuietException(
            "Reference genome file {} does not exist".format(
                genome_params['reference_genome'] ))
    
    if not os.path.exists(genome_params['chrom_sizes']):
        raise quietexception.QuietException(
            "Chromosome sizes file {} does not exist".format(
            genome_params['chrom_sizes']))
        
    try:
        get_model = getattr(model_archs, network_params['name'])
    except AttributeError:
        raise quietexception.QuietException(
            "Network {} not found in model definitions".format(
                network_params['name']))
    
    if not os.path.isfile(args.splits):
        raise quietexception.QuietException("File not found: {}", args.splits)
                
    # load splits from json file
    with open(args.splits, "r") as splits_json:
        splits = json.loads(splits_json.read())
    
    # training and validation
    training.train_and_validate_ksplits(
        input_params, output_params, genome_params, batch_gen_params, 
        hyper_params, parallelization_params, network_params, splits)

if __name__ == '__main__':
    main()


