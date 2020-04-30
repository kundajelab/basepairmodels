import os
import sys

stderr = sys.stderr
sys.stderr = open('keras.stderr', 'w')
from keras import layers, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
sys.stderr = stderr

from batchgenutils import *
from utils import *
from callbacks import BatchController, TimeHistory
from losses import MultichannelMultinomialNLL

import MTBatchGenerator
import argparsers
import logger
import model_archs
import datetime
import experiments
import time
import warnings


import multiprocessing as mp
import pandas as pd
import copy
import json


def training_process(args, input_data, train_chroms, val_chroms, test_chroms, 
                     model_dir, logfname):
    # we need to initialize the logger for each process
    logger.init_logger(logfname)

    logging.info("INPUT DATA -\n{}".format(input_data))
    
    logging.info("args.stranded {}".format(args.stranded))
    
    # parameters that are specific to the training batch generation
    # process
    train_batchgen_params = {'seq_len': args.input_seq_len,
                             'output_len': args.output_len,
                             'max_jitter': args.max_jitter, 
                             'rev_comp_aug': 
                                 args.reverse_complement_augmentation,
                             'negative_sampling_rate': 
                                 args.negative_sampling_rate}
    
    # parameters that are specific to the validation batch generation
    # process. For validation we dont use jitter, reverse complement 
    # augmentation and negative sampling
    val_batchgen_params = {'seq_len': args.input_seq_len,
                           'output_len': args.output_len,
                           'max_jitter': 0, 
                           'rev_comp_aug': False,
                           'negative_sampling_rate': False}
    
    if args.train_peaks:
        # get a pandas dataframe for the training data with peak 
        # positions
        train_data = getPeakPositions(input_data, train_chroms, 
                                      args.chrom_sizes, args.input_seq_len // 2)

        # get a pandas dataframe for the validation data with peak
        # positions
        val_data = getPeakPositions(input_data, val_chroms, 
                                    args.chrom_sizes, args.input_seq_len // 2)
        
    elif args.train_sequential is not None:
        num_positions, step = args.train_sequential

        # get a pandas dataframe for the training data with
        # sequential positions at equal intervals
        train_data = getChromPositions(train_chroms, args.chrom_sizes, 
                                       args.input_seq_len // 2, 
                                       mode='sequential', 
                                       num_positions=num_positions, step=step)

        # get a pandas dataframe for the validation data with
        # sequential positions at equal intervals
        val_data = getChromPositions(val_chroms, args.chrom_sizes, 
                                    args.input_seq_len // 2, 
                                    mode='sequential', 
                                    num_positions=num_positions, step=step)
        
        batchen_params['max_jitter'] = 0
    
    elif args.train_random is not None:
        # get a pandas dataframe for the training data with random
        # positions
        train_data = getChromPositions(train_chroms, args.chrom_sizes, 
                                       args.input_seq_len // 2, 
                                       mode='random', 
                                       num_positions=args.train_random)

        # get a pandas dataframe for the validation data with random
        # positions
        val_data = getChromPositions(val_chroms, args.chrom_sizes, 
                                    args.input_seq_len // 2, 
                                    mode='random', 
                                    num_positions=args.train_random)

        batchen_params['max_jitter'] = 0

    # get the corresponding batch generator class for this model
    BatchGenerator = getattr(MTBatchGenerator, 
                             'MT{}BatchGenerator'.format(args.model_arch_name))
            
    # instantiate the batch generator class for training
    train_gen = BatchGenerator(input_data, args.reference_genome, 
                               args.chrom_sizes, train_chroms, 
                               train_batchgen_params, 'train_gen', 
                               args.stranded, num_threads=args.threads,
                               epochs=args.epochs, 
                               batch_size=args.batch_size)

    # training generator function that will be passed to 
    # fit_generator
    train_generator = train_gen.gen(train_data)

    # instantiate the batch generator class for validation
    val_gen = BatchGenerator(input_data, args.reference_genome, 
                             args.chrom_sizes, val_chroms, 
                             val_batchgen_params, 'val_gen', 
                             args.stranded, num_threads=args.threads,
                             epochs=args.epochs, 
                             batch_size=args.batch_size)

    # validation generator function that will be passed to 
    # fit_generator
    val_generator = val_gen.gen(val_data)

    # lets make sure the sizes look reasonable
    logging.info("TRAINING SIZE - {}".format(train_data.shape))
    logging.info("VALIDATION SIZE - {}".format(val_data.shape))

    # we need to calculate the number of training steps and 
    # validation steps in each epoch, fit_generator requires this
    # to determine the end of an epoch
    train_steps = train_data.shape[0] // args.batch_size
    val_steps = val_data.shape[0] // args.batch_size

    # we may have to reduce the --threads sometimes
    # if the peak file has very few peaks, so we need to
    # check if these numbers will be 0
    logging.info("TRAINING STEPS - {}".format(train_steps))
    logging.info("VALIDATION STEPS - {}".format(val_steps))
    
    # Here we specify all our callbacks
    # 1. Early stopping if validation loss doesn't decrease
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5,
                       min_delta=1e-3, restore_best_weights=True)

    # 2. Reduce learning rate if validation loss is plateuing
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                                  min_lr=args.min_learning_rate)

    # 3. Timing hook to record start, end & elapsed time for each 
    # epoch
    time_tracker = TimeHistory()

    # 4. Batch controller callbacks to ensure that batches are 
    # generated only on a per epoch basis, also ensures graceful
    # termination of the batch generation 
    train_batch_controller = BatchController(train_gen)
    val_batch_controller = BatchController(val_gen)

    # get an instance of the model
    logging.debug("New {} model".format(args.model_arch_name))
    get_model = getattr(model_archs, args.model_arch_name)
    model = get_model(train_batchgen_params['seq_len'], 
                      train_batchgen_params['output_len'], 
                      filters=args.filters, 
                      num_tasks=len(list(input_data.keys())))

    # if running in multi gpu mode
    if args.gpus > 1:
        logging.debug("Multi GPU model")
        model = multi_gpu_model(model, gpus=args.gpus)

    # compile the model
    logging.debug("Compiling model")
    model.compile(Adam(lr=args.learning_rate),
                    loss=[MultichannelMultinomialNLL(len(
                        list(input_data.keys()))), 'mse'], 
                    loss_weights=[1, args.counts_loss_weight])

    model.summary()
    
    # begin time for training
    t1 = time.time()

    # start training
    logging.debug("Training started ...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        history = model.fit_generator(train_generator, 
                                      validation_data=val_generator,
                                      epochs=args.epochs, 
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
      
    if args.automate_filenames:
        # get random alphanumeric tag for model
        model_tag = getAlphaNumericTag(args.tag_length)

        other_tags = '_'.join(args.other_tags)
        if len(args.other_tags) > 0:
            model_fname = "{}/{}_{}.h5".format(
                model_dir, model_tag, '_'.join(args.other_tags))
        else:
            model_fname = "{}/{}.h5".format(
                model_dir, model_tag)
    elif args.model_output_filename is not None:
        model_fname = "{}/{}.h5".format(model_dir, args.model_output_filename)

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
    config_file = '{}/config.json'.format(model_dir)
    with open(config_file, 'w') as fp:
        config = vars(args)
        
        # the number of epochs the training lasted
        epochs = len(history.history['val_loss'])
        config['training_epochs'] = epochs
        
        # val chroms
        config['val_chroms'] = val_chroms
        
        # test chroms
        config['test_chroms'] = test_chroms
        
        # model file name
        config['model_filename'] = model_fname

        json.dump(config, fp)

def trainer_main():
    # inform user of the keras stderr log file
    logging.info("For all keras related error logs refer to keras.stderr in "
                 "your local directory")
    
    # parse the command line arguments
    parser = argparsers.training_argsparser()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        logging.error("Directory {} does not exist".format(
            args.output_dir))
        
        return
    
    if not args.automate_filenames and args.model_output_filename is None:
        logging.error("Model output filename not specified")
        return
    
    # get the dictionary of the input tasks
    input_data = getInputTasks(args.input_dir, stranded=args.stranded, 
                               has_control=args.has_control, mode='training',
                               require_peaks=args.train_peaks)
            
    # if there was a problem constructing the input tasks dictionary
    if input_data is None:
        return
    
    # function string to retrieve splits
    splits = args.splits
    
    # list of chromosomes after removing the excluded chromosomes
    chroms = set(args.chroms).difference(set(args.exclude_chroms))
    
    # get the validation and test splits
    get_splits = getattr(experiments, 'get_' + splits)
    splits = get_splits()
    
    num_splits = len(list(splits.keys()))
        
    # run training for each validation/test split
    for i in range(num_splits):
        
        if args.automate_filenames:
            # create a new directory using current date/time to store the
            # model, the loss history and logs 
            date_time_str = local_datetime_str(args.time_zone)
            model_dir = '{}/{}'.format(args.output_dir, date_time_str)
            os.mkdir(model_dir)
        elif os.path.isdir(args.output_dir):
            model_dir = args.output_dir        
        else:
            logging.error("Directory does not exist {}.".format(args.output_dir))
            return
            
        # filename to write debug logs
        logfname = '{}/trainer.log'.format(model_dir)

        # set up logger for main procecss
        logger.init_logger(logfname)
    
        # train, val, test split (train derived from val & test)
        val_chroms = splits[i]['val']
        test_chroms = splits[i]['test']
        train_chroms = list(chroms.difference(set(val_chroms + test_chroms)))
        
        logging.info("Split #{}".format(i))
        logging.info("Train: {}".format(train_chroms))
        logging.info("Val: {}".format(val_chroms))
        logging.info("Test: {}".format(test_chroms))
            
        # Start training for the split in a separate process
        # This ensures that all resources are freed, when the 
        # process terminates, & available for training the next split
        # Mitigates the problem where training subsequent splits
        # is considerably slow
        logging.debug("Split {}: Creating training process".format(i))
        p = mp.Process(target=training_process, args=
                       [args, input_data, train_chroms, val_chroms, 
                        test_chroms, model_dir, logfname])
        p.start()
        
        # wait for the process to finish
        p.join()


if __name__ == '__main__':
    # change the way processes are started, default = 'fork'
    # had to do this to prevent Keras multi gpu from deadlocking
    mp.set_start_method('forkserver')
    
    # call the real main function
    trainer_main()
