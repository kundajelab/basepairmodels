"""
    This module containins training functions that are common to
    the CLI & the API
    
    Functions:
    
        train_and_validate: Train and validate on a single train and
            validation set
        
        train_and_validate_ksplits: Train and validate on one or 
            more train/val splits specified via a json file
        
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


import copy
import datetime
import json
import logging
import multiprocessing as mp
import os
import pandas as pd
import sys
import tensorflow.keras.backend as kb
import time
import warnings

from basepairmodels.common import model_archs
from basepairmodels.cli.bpnetutils import *
from basepairmodels.cli.losses import MultichannelMultinomialNLL
from basepairmodels.cli import experiments
from basepairmodels.cli import logger
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from mseqgen import generators 


def early_stopping_check(losses, patience=5, min_delta=1e-3):
    """
        Function to check if early stopping criteria are met
        
        Args:
            losses (list): list of all losses in order of the epochs,
                these could be training or validation losses
            patience (int): the number of epochs with no improvement
                greater than `min_delta`
            min_delta (float): The smallest amount that signals  
                sufficienct decrease in validation loss to justify
                continuation of training for a further #patience
                epochs
                
        Returns:
            bool: True, if early stopping criteria are satisfied, 
                False otherwise
    """
    
    # if sufficient number of epochs have not elapsed yet
    if len(losses) <= patience:
        return False
    
    # the loss value upon which the patience check will be performed
    anchor_loss = losses[-(patience+1)]
    
    for i in range(patience):
        if (anchor_loss - losses[i-patience]) > min_delta:
            return False
    
    return True

    
def reduce_lr_on_plateau(losses, current_lr, factor=0.5, patience=2, 
                         min_lr=1e-4):
    """
        Function to compute the new learning rate if loss is
        plateauing 
        
        Args:
            losses (list): list of all losses in order of the epochs,
                these could be training or validation losses
            current_lr (float): current learning rate
            factor (float): the factor by which the learning rate is
                to be reduced in case the plateau criteria is met
            patience (int): number of epochs with no improvement after 
                which learning rate will be reduced.
            min_lr (float): lower bound on the learning rate
                
        Returns:
            float: new learning rate
                
    """
    
    # if sufficient number of epochs have not elapsed yet 
    if len(losses) <= patience:
        return current_lr
    
    # the loss value upon which the patience check will be performed
    anchor_loss = losses[-(patience+1)]
    
    for i in range(patience):
        # improvement found
        if losses[i-patience] < anchor_loss:
            # no change in learning rate
            return current_lr
    
    # new learning rate
    new_lr = current_lr * factor
    
    # check if it's below lower bound
    if new_lr < min_lr:
        return current_lr
    
    return new_lr


def train_and_validate(input_params, output_params, genome_params, 
                       batch_gen_params, hyper_params, parallelization_params, 
                       network_params, train_chroms, val_chroms, 
                       model_dir, suffix_tag=None):

    """
        Train and validate on a single train and validation set
        
        Note: the list & description for each of the required keys
            in all of the json parameter files passed to this 
            fucntion can be found here:
            http://
        
        Args:
            input_params (dict): dictionary containing input parameters
            
            output_params (dict): dictionary containing output 
                parameters
            
            genome_params (dict): dictionary containing genome
                parameters
            
            batch_gen_params (dict): dictionary containing batch
                generation parameters
            
            hyper_params (dict): dictionary containing containing 
                training & validation hyper parameters
            
            parallelization_params (dict): dictionary containing
                parameters for parallelization options
            
            network_params (dict): dictionary containing parameters
                specific to the deep learning architecture
                
            use_attribution_prior (bool): indicate whether attribution
                prior loss model should be used

            attribution_prior_params (dict): dictionary containing
                attribution prior parameters
            
            train_chroms (list): list of training chromosomes
            
            val_chroms (list): list of validation chromosomes
            
            model_dir (str): the path to the output directory
            
            suffix_tag (str): optional tag to add as a suffix to files
                (model, log, history & config params files) created in
                the model directory
         
         Returns:
             keras.models.Model
             
    """
    
    # filename to write debug logs
    if suffix_tag is not None:
        logfname = '{}/trainer_{}.log'.format(model_dir, suffix_tag)
    else:
        logfname = '{}/trainer.log'.format(model_dir)
        
    # we need to initialize the logger for each process
    logger.init_logger(logfname)
    
    # parameters that are specific to the training batch generation
    # process
    train_batch_gen_params = batch_gen_params
    train_batch_gen_params['mode'] = 'train'
    
    # parameters that are specific to the validation batch generation
    # process. For validation we dont use jitter, reverse complement 
    # augmentation and negative sampling
    val_batch_gen_params = copy.deepcopy(batch_gen_params)
    val_batch_gen_params['max_jitter'] = 0
    val_batch_gen_params['rev_comp_aug'] = False
    val_batch_gen_params['negative_sampling_rate'] = 0.0
    val_batch_gen_params['mode'] = 'val'

    # get the corresponding batch generator class for this model
    sequence_generator_class_name = generators.find_generator_by_name(
        batch_gen_params['sequence_generator_name'])
    logging.info("SEQGEN Class Name: {}".format(sequence_generator_class_name))
    BatchGenerator = getattr(generators, sequence_generator_class_name)

    # instantiate the batch generator class for training
    train_gen = BatchGenerator(input_params, train_batch_gen_params, 
                               network_params, 
                               genome_params['reference_genome'], 
                               genome_params['chrom_sizes'],
                               train_chroms, 
                               num_threads=parallelization_params['threads'],
                               epochs=hyper_params['epochs'], 
                               batch_size=hyper_params['batch_size'])


    # instantiate the batch generator class for validation
    val_gen = BatchGenerator(input_params, val_batch_gen_params, 
                             network_params,
                             genome_params['reference_genome'], 
                             genome_params['chrom_sizes'],
                             val_chroms, 
                             num_threads=parallelization_params['threads'],
                             epochs=hyper_params['epochs'], 
                             batch_size=hyper_params['batch_size'])

    # lets make sure the sizes look reasonable
    logging.info("TRAINING SIZE - {}".format(train_gen._samples.shape))
    logging.info("VALIDATION SIZE - {}".format(val_gen._samples.shape))

    # we need to calculate the number of training steps and 
    # validation steps in each epoch, fit/evaluate requires this
    # to determine the end of an epoch
    train_steps = train_gen.len()
    val_steps = val_gen.len()

    # we may have to reduce the --threads sometimes
    # if the peak file has very few peaks, so we need to
    # check if these numbers will be 0
    logging.info("TRAINING STEPS - {}".format(train_steps))
    logging.info("VALIDATION STEPS - {}".format(val_steps))

    # get an instance of the model
    logging.debug("New {} model".format(network_params['name']))
    get_model = getattr(model_archs, network_params['name'])
    model = get_model(train_batch_gen_params['input_seq_len'], 
                      train_batch_gen_params['output_len'],
                      len(network_params['control_smoothing']) + 1,
                      filters=network_params['filters'], 
                      num_tasks=train_gen._num_tasks,
                      use_attribution_prior=use_attribution_prior,
                      attribution_prior_params=attribution_prior_params)
    
    # print out the model summary
    model.summary()

#     # if running in multi gpu mode
#     if parallelization_params['gpus'] > 1:
#         logging.debug("Multi GPU model")
#         model = multi_gpu_model(model, gpus=parallelization_params['gpus'])

    # compile the model
    logging.debug("Compiling model")
    logging.info("counts_loss_weight - {}".format(
        network_params['counts_loss_weight']))
    model.compile(Adam(learning_rate=hyper_params['learning_rate']),
                    loss=[MultichannelMultinomialNLL(
                        train_gen._num_tasks), 'mse'], 
                    loss_weights=[1, network_params['counts_loss_weight']])
    
    # begin time for training
    t1 = time.time()

    # track validation losses for early stopping and learning rate
    # updates
    val_losses = []
    
    # track best loss so we can restore weights 
    best_loss = 1e6
    
    # keep a copy of the best weights
    best_weights = None
    
    # the epoch with the best validation loss
    best_epoch = 1
    
    # start training
    logging.debug("Training started ...")
    for epoch in range(hyper_params['epochs']):
        # First, let's train for one epoch
        logging.info("Training Epoch {}".format(epoch + 1))
        train_start_time = time.time()
        # training generator function that will be passed to fit
        train_generator = train_gen.gen(epoch)
        model.fit(train_generator, epochs=1, steps_per_epoch=train_steps)
        train_end_time = time.time()
        
        # Then, we evaluate on the validation set
        logging.info("Validation Epoch {}".format(epoch + 1))
        val_start_time = time.time()
        # validation generator function that will be passed to evaluate 
        val_generator = val_gen.gen(epoch)
        val_loss = model.evaluate(
            val_generator, steps=val_steps, return_dict=True)
        val_losses.append(val_loss['loss'])
        val_end_time = time.time()
        
        # update best weights and loss 
        if val_loss['loss'] < best_loss:
            best_weights = model.get_weights()
            best_loss = val_loss['loss']
            best_epoch = epoch + 1
        
        # check if early stopping criteria are satisfied
        if early_stopping_check(
            val_losses,
            patience=hyper_params['early_stopping_patience'],
            min_delta=hyper_params['early_stopping_min_delta']):
            
            # restore best weights
            logging.info("Restoring best weights from epoch {}".format(
                best_epoch))
            model.set_weights(best_weights)
            break

        # lower learning rate if criteria are satisfied
        new_lr = reduce_lr_on_plateau(
            val_losses,
            model.optimizer.learning_rate,
            factor=hyper_params['lr_reduction_factor'], 
            patience=hyper_params['reduce_lr_on_plateau_patience'],
            min_lr=hyper_params['min_learning_rate'])
        
        # set the new learning rate
        model.optimizer.lr.assign(new_lr)

        # display current learning rate and training status
        logging.info("Current learning rate - {}, Stop Training - {}".format(
            model.optimizer.learning_rate, model.stop_training))

    # end time for training
    t2 = time.time() 
    logging.info("Total Elapsed Time: {}".format(t2-t1))

    # base model filename
    if output_params['automate_filenames']:
        # get random alphanumeric tag for model
        model_tag = getAlphaNumericTag(output_params['tag_length'])
        model_fname = "{}/{}".format(model_dir, model_tag)
    elif output_params['model_output_filename'] is not None:
        model_fname = "{}/{}".format(model_dir, 
                                     output_params['model_output_filename'])
    else:
        model_fname = "{}/model".format(model_dir)
    
    # add suffix tag to model name
    if suffix_tag is not None:
        model_fname += "_{}".format(suffix_tag)
    
    # extension
    model_fname += ".h5"

    # save HDF5 model file
    model.save(model_fname)
    logging.info("Finished saving model: {}".format(model_fname))

#     # save history to json:  
#     # Step 1. create a custom history object with a new key for 
#     # epoch times
#     custom_history = copy.deepcopy(history.history)
#     custom_history['times'] = time_tracker.times

#     # Step 2. convert the custom history dict to a pandas DataFrame:  
#     hist_df = pd.DataFrame(custom_history) 

#     # file name for json file
#     hist_json = model_fname.replace('.h5', '.history.json')

#     # Step 3. write the dataframe to json
#     with open(hist_json, mode='w') as f:
#         hist_df.to_json(f)
    
#     logging.info("Finished saving training and validation history: {}".format(
#         hist_json))

    # write all the command line arguments to a json file
    # & include the number of epochs the training lasted for, and the
    # validation and testchroms
    config_file = '{}/config'.format(model_dir)
    # add suffix tag to model name
    if suffix_tag is not None:
        config_file += "_{}".format(suffix_tag)
    # extension
    config_file += ".json"
    
    with open(config_file, 'w') as fp:
        config = {}        
        config['input_params'] = input_params
        config['output_params'] = output_params
        config['genome_params'] = genome_params
        config['batch_gen_params'] = batch_gen_params
        config['hyper_params'] = hyper_params
        config['parallelization_params'] = parallelization_params
        config['network_params'] = network_params
        
        # the number of epochs the training lasted
        config['training_epochs'] = epoch + 1

        # the epoch with best validation loss
        config['best_epoch'] = best_epoch 
        
        config['train_chroms'] = train_chroms
        config['val_chroms'] = val_chroms
        config['model_filename'] = model_fname

        json.dump(config, fp)

    return model
        
def train_and_validate_ksplits(
    input_params, output_params, genome_params, batch_gen_params, hyper_params, 
    parallelization_params, network_params, use_attribution_prior, 
    attribution_prior_params, splits):

    """
        Train and validate on one or more train/val splits
        
        Args:
            input_params (dict): dictionary containing input parameters
            
            output_params (dict): dictionary containing output 
                parameters
            
            genome_params (dict): dictionary containing genome
                parameters
            
            batch_gen_params (dict): dictionary containing batch
                generation parameters
            
            hyper_params (dict): dictionary containing containing 
                training & validation hyper parameters
            
            parallelization_params (dict): dictionary containing
                parameters for parallelization options
            
            network_params (dict): dictionary containing parameters
                specific to the deep learning architecture
                
            use_attribution_prior (bool): indicate whether attribution
                prior loss model should be used

            attribution_prior_params (dict): dictionary containing
                attribution prior parameters
            
            splits (str): path to the json file containing train & 
                validation splits
    """
    
    
    # list of chromosomes after removing the excluded chromosomes
    chroms = set(genome_params['chroms']).difference(
        set(genome_params['exclude_chroms']))
        
    # list of models from all of the splits
    models = []
    
    # run training for each validation/test split
    num_splits = len(list(splits.keys()))
    for i in range(num_splits):
        
        if output_params['automate_filenames']:
            # create a new directory using current date/time to store the
            # model, the loss history and logs 
            date_time_str = local_datetime_str(output_params['time_zone'])
            model_dir = '{}/{}_split{:03d}'.format(
                output_params['output_dir'], date_time_str, i)
            os.mkdir(model_dir)
            split_tag = None
        elif os.path.isdir(output_params['output_dir']):
            model_dir = output_params['output_dir']     
            split_tag = "split{:03d}".format(i)
        else:
            logging.error("Directory does not exist {}.".format(
                output_params['output_dir']))
            return
            
        # filename to write debug logs
        logfname = '{}/trainer.log'.format(model_dir)
        # set up logger for main procecss
        logger.init_logger(logfname)
    
        # train & validation chromosome split
        if 'val' not in splits[str(i)]:
            logging.error("KeyError: 'val' required for split {}".format(i))
            return
        val_chroms = splits[str(i)]['val']
        # if 'train' key is present
        if 'train' in splits[str(i)]:
            train_chroms = splits[str(i)]['train']
        # if 'test' key is present but train is not
        elif 'test' in splits[str(i)]:
            test_chroms = splits[str(i)]['test']
            # take the set difference of the whole list of
            # chroms with the union of val and test
            train_chroms = list(chroms.difference(
                set(val_chroms + test_chroms)))
        else:
            # take the set difference of the whole list of
            # chroms with val
            train_chroms = list(chroms.difference(val_chroms))
        
        logging.info("Split #{}".format(i))
        logging.info("Train: {}".format(train_chroms))
        logging.info("Val: {}".format(val_chroms))
            
        # Start training for the split in a separate process
        # This ensures that all resources are freed, when the 
        # process terminates, & available for training the next split
        # Mitigates the problem where training subsequent splits
        # is considerably slow
        logging.debug("Split {}: Creating training process".format(i))
        p = mp.Process(
            target=train_and_validate, 
            args=[input_params, output_params, genome_params, 
                  batch_gen_params, hyper_params, parallelization_params, 
                  network_params, use_attribution_prior, 
                  attribution_prior_params, train_chroms, val_chroms, 
                  model_dir, split_tag])
        p.start()
        
        # wait for the process to finish
        p.join()
