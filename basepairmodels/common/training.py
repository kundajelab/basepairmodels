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


import sys

stderr = sys.stderr
sys.stderr = open('keras.stderr', 'w')
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
sys.stderr = stderr

from basepairmodels.cli.batchgenutils import *
from basepairmodels.cli.bpnetutils import *
from basepairmodels.cli.callbacks import BatchController, TimeHistory
from basepairmodels.cli.losses import MultichannelMultinomialNLL
from basepairmodels.cli import experiments
from basepairmodels.cli import logger
from basepairmodels.common import model_archs
from mseqgen import generators 

import copy
import datetime
import json
import multiprocessing as mp
import pandas as pd
import time
import warnings

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
            input_params (str): path to json file containing input
                parameters
            
            output_params (str): path to json file containing output
                parameters
            
            genome_params (str): path to json file containing genome
                parameters
            
            batch_gen_params (str): path to json file containing batch
                generation parameters
            
            hyper_params (str): path to json file containing training &
                validation hyper parameters
            
            parallelization_params (str): path to json file containing
                parameters for parallelization options
            
            network_params (str): path to json file containing
                parameters specific to the deep learning architecture
            
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
    BatchGenerator = getattr(generators, 'M{}SequenceGenerator'.format(
        network_params['name']))

    # instantiate the batch generator class for training
    train_gen = BatchGenerator(input_params, train_batch_gen_params, 
                               network_params, 
                               genome_params['reference_genome'], 
                               genome_params['chrom_sizes'],
                               train_chroms, 
                               num_threads=parallelization_params['threads'],
                               epochs=hyper_params['epochs'], 
                               batch_size=hyper_params['batch_size'])

    # training generator function that will be passed to 
    # fit_generator
    train_generator = train_gen.gen()

    # instantiate the batch generator class for validation
    val_gen = BatchGenerator(input_params, val_batch_gen_params, 
                                 network_params,
                             genome_params['reference_genome'], 
                             genome_params['chrom_sizes'],
                             val_chroms, 
                             num_threads=parallelization_params['threads'],
                             epochs=hyper_params['epochs'], 
                             batch_size=hyper_params['batch_size'])
    
    # validation generator function that will be passed to 
    # fit_generator
    val_generator = val_gen.gen()

    # lets make sure the sizes look reasonable
    logging.info("TRAINING SIZE - {}".format(train_gen.data.shape))
    logging.info("VALIDATION SIZE - {}".format(val_gen.data.shape))

    # we need to calculate the number of training steps and 
    # validation steps in each epoch, fit_generator requires this
    # to determine the end of an epoch
    train_steps = train_gen.get_num_batches_per_epoch()
    val_steps = val_gen.get_num_batches_per_epoch()

    # we may have to reduce the --threads sometimes
    # if the peak file has very few peaks, so we need to
    # check if these numbers will be 0
    logging.info("TRAINING STEPS - {}".format(train_steps))
    logging.info("VALIDATION STEPS - {}".format(val_steps))
  
    # Here we specify all our callbacks
    # 1. Early stopping if validation loss doesn't decrease
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                       patience=hyper_params['early_stopping_patience'],
                       min_delta=hyper_params['early_stopping_min_delta'], 
                       restore_best_weights=True)

    # 2. Reduce learning rate if validation loss is plateuing
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, 
        patience=hyper_params['reduce_lr_on_plateau_patience'], 
        min_lr=hyper_params['min_learning_rate'])

    # 3. Timing hook to record start, end & elapsed time for each 
    # epoch
    time_tracker = TimeHistory()

    # 4. Batch controller callbacks to ensure that batches are 
    # generated only on a per epoch basis, also ensures graceful
    # termination of the batch generation 
    train_batch_controller = BatchController(train_gen)
    val_batch_controller = BatchController(val_gen)

    # get an instance of the model
    logging.debug("New {} model".format(network_params['name']))
    get_model = getattr(model_archs, network_params['name'])
    model = get_model(train_batch_gen_params['input_seq_len'], 
                      train_batch_gen_params['output_len'],
                      len(input_params['control_smoothing']) + 1,
                      filters=network_params['filters'], 
                      num_tasks=train_gen.num_tasks)

    # if running in multi gpu mode
    if parallelization_params['gpus'] > 1:
        logging.debug("Multi GPU model")
        model = multi_gpu_model(model, gpus=parallelization_params['gpus'])

    # compile the model
    logging.debug("Compiling model")
    model.compile(Adam(lr=hyper_params['learning_rate']),
                    loss=[MultichannelMultinomialNLL(
                        train_gen.num_tasks), 'mse'], 
                    loss_weights=[1, network_params['counts_loss_weight']])

    model.summary()
    
    # begin time for training
    t1 = time.time()

    # start training
    logging.debug("Training started ...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        history = model.fit_generator(train_generator, 
                                      validation_data=val_generator,
                                      epochs=hyper_params['epochs'], 
                                      steps_per_epoch=train_steps, 
                                      validation_steps=val_steps, 
                                      callbacks=[es, reduce_lr, time_tracker,
                                                 train_batch_controller,
                                                 val_batch_controller])

    # end time for training
    t2 = time.time() 
    logging.info("Total Elapsed Time: {}".format(t2-t1))

    # send the stop signal to the generators
    train_gen.set_stop()
    val_gen.set_stop()
      
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

    # save history to json:  
    # Step 1. create a custom history object with a new key for 
    # epoch times
    custom_history = copy.deepcopy(history.history)
    custom_history['times'] = time_tracker.times

    # Step 2. convert the custom history dict to a pandas DataFrame:  
    hist_df = pd.DataFrame(custom_history) 

    # file name for json file
    hist_json = model_fname.replace('.h5', '.history.json')

    # Step 3. write the dataframe to json
    with open(hist_json, mode='w') as f:
        hist_df.to_json(f)
    
    logging.info("Finished saving training and validation history: {}".format(
        hist_json))

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
        epochs = len(history.history['val_loss'])
        config['training_epochs'] = epochs
        
        config['train_chroms'] = train_chroms
        config['val_chroms'] = val_chroms
        config['model_filename'] = model_fname

        json.dump(config, fp)

    return model
        
def train_and_validate_ksplits(input_params, output_params, genome_params, 
                               batch_gen_params, hyper_params, 
                               parallelization_params, network_params, 
                               splits):

    """
        Train and validate on one or more train/val splits

        Note: the list & description for each of the required keys
            in all of the json parameter files passed to this 
            function can be found here:
            http://
        
        Args:
            input_params (str): path to json file containing input
                parameters
            
            output_params (str): path to json file containing output
                parameters
            
            genome_params (str): path to json file containing genome
                parameters
            
            batch_gen_params (str): path to json file containing batch
                generation parameters
            
            hyper_params (str): path to json file containing training &
                validation hyper parameters
            
            parallelization_params (str): path to json file containing
                parameters for parallelization options
            
            network_params (str): path to json file containing
                parameters specific to the deep learning architecture
            
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
        p = mp.Process(target=train_and_validate, args=
                       [input_params, output_params, genome_params, 
                        batch_gen_params, hyper_params, parallelization_params, 
                        network_params, train_chroms, val_chroms, model_dir,
                        split_tag])
        p.start()
        
        # wait for the process to finish
        p.join()
