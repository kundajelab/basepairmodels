import multiprocessing as mp
import logging
import pandas as pd 
import numpy as np
import pysam
import pyBigWig
import random
import batchgenutils
from threading import Thread
from queue import Queue
from scipy.ndimage import gaussian_filter1d
from batchgenutils import *


class MTBatchGenerator:
    """ MultiTask batch data generation 
    
    """
    
    
    def __init__(self, tasks, reference, chrom_sizes, chroms, batchgen_params, 
                 name_str, stranded=True, num_threads=28, epochs=100,
                 batch_size=64, shuffle=True, mode='train'):
        
        self.name_str = name_str

        self.tasks = tasks
        self.num_tasks = len(list(self.tasks.keys()))
        self.reference = reference

        # read the chrom sizes into a dataframe and filter rows
        # from unwanted chromosomes
        self.chrom_sizes_df = pd.read_csv(chrom_sizes, sep = '\t', 
                              header=None, names = ['chrom', 'size']) 

        # keep only those chrom_sizes rows corresponding to the 
        # required chromosomes
        self.chrom_sizes_df = self.chrom_sizes_df[
            self.chrom_sizes_df['chrom'].isin(chroms)]

        # generate a new column for sampling weights of the chromosomes
        self.chrom_sizes_df['weights'] = (self.chrom_sizes_df['size'] / 
                                          self.chrom_sizes_df['size'].sum())

        self.stranded = stranded
        self.num_threads = num_threads
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        
        self.flank =  int(batchgen_params['seq_len'] / 2)
        self.output_flank = int(batchgen_params['output_len'] / 2)
        if self.mode =='train':
            self.max_jitter = batchgen_params['max_jitter']
            self.rev_comp_aug = batchgen_params['rev_comp_aug'] 
            self.negative_sampling_rate = \
                batchgen_params['negative_sampling_rate']
        
        # control batch generation for next epoch
        # if the value is not set to True, batches are not generated
        # Use an external controller to set value to True/False
        self.ready_for_next_epoch = False
        
        # early stopping flag
        self.stop = False
    
    
    def set_ready_for_next_epoch(self, val):
        """Set the variable that controls batch generation 
           for the next epoch
        
            Args: None
                
            Returns: None
                
        """
        self.ready_for_next_epoch = val


    def set_stop(self):
        """ Set stop Flag to True
        
            Args: None
                
            Returns: None
                
        """
        self.stop = True


    def set_early_stopping(self):
        """ Set early stopping flag to True
        
            Args: None
                
            Returns: None
                
        """
        self.set_stop()

        
    def generate_batch(self, coords):
        """ Generate one batch of inputs and outputs
            
        """
        
        raise NotImplementedError("Implement this method in a derived class")

    
    def get_negative_batch(self):
        """
            get chrom positions for the negative samples using
            uniform random sampling from across the all chromosomes
            in self.chroms
            
            Returns:
                pandas.DataFrame: dataframe of coordinates 
                    ('chrom' & 'pos')
        """

        # Step 1: select chromosomes, using sampling weights 
        # according to sizes
        chrom_df = self.chrom_sizes_df.sample(
            n=int(self.batch_size*self.negative_sampling_rate),
            weights=self.chrom_sizes_df.weights, replace=True)

        # Step 2: generate 'n' random numbers where 'n' is the length
        # of chrom_df 
        r = [random.random() for _ in range(chrom_df.shape[0])]

        # Step 3. multiply the random numbers with the size column.
        # Additionally, factor in the flank size and jitter while 
        # computing the position
        chrom_df['pos'] = ((chrom_df['size'] - 
                            ((self.flank + self.max_jitter)*2)) * r
                           + self.flank + self.max_jitter).astype(int)

        return chrom_df[['chrom', 'pos']]

    
    def proc_target(self, coords_df, mpq, proc_idx):
        """
            Function that will be executed in a separate process.
            Takes a dataframe of peak coordinates and parses them in 
            batches, to get one hot encoded sequences and corresponding
            outputs, and adds the batches to the multiprocessing queue.
            Optionally, samples negative locations and adds them to 
            each batch
            
            Args:
                coords_df (pandas.DataFrame): dataframe containing
                    the chrom & peak pos
                mpq (multiprocessing.Queue): The multiprocessing queue
                    to hold the batches
                
            Return
                ---
        """
        
        # divide the coordinates dataframe into batches
        cnt = 0
        for i in range(0, coords_df.shape[0], self.batch_size):   
            # we need to make sure we dont try to fetch 
            # data beyond the length of the dataframe
            if (i + self.batch_size) > coords_df.shape[0]:
                break
                
            batch_df = coords_df.iloc[i:i + self.batch_size]
            
            # add equal number of negative samples
            if self.mode == 'train' and self.negative_sampling_rate > 0.0:
                neg_batch = self.get_negative_batch()
                batch_df = pd.concat([batch_df, neg_batch])
            
            # generate a batch of one hot encoded sequences and 
            # corresponding outputs
            batch = self.generate_batch(batch_df)
            
            # add batch to the multiprocessing queue
            mpq.put(batch)
    
            cnt += 1
        
        logging.debug("{} process {} put {} batches into mpq".format(
            self.name_str, proc_idx, cnt))
            
    def stealer(self, mpq, q, num_batches, thread_id):
        """
            Thread target function to "get" (steal) from the
            multiprocessing queue and "put" in the regular queue

            Args:
                mpq (multiprocessing.Queue): The multiprocessing queue
                    to steal from
                q (Queue): The regular queue to put the batch into
                num_batches (int): the number of batches to "steal"
                    from the mp queue
                thread_id (int): thread id for debugging purposes

            Returns:
                ---
        """
        for i in range(num_batches):            
            q.put(mpq.get())

        logging.debug("{} stealer thread {} got {} batches from mpq".format(
            self.name_str, thread_id, num_batches))

            
    def epoch_run(self, data):
        """
            Manage batch generation processes & threads
            for one epoch

            Args:
                data (pandas.DataFrame): dataframe with 'chrom' &
                    'pos' columns
                            
            Returns:
                ---
        """
        
        # list of processes that are spawned
        procs = []     
        
        # list of multiprocessing queues corresponding to each 
        # process
        mp_queues = [] 

        # list of stealer threads (that steal the items out of 
        # the mp queues)
        threads = []   
                       
        # the regular queue
        q = Queue()    

        # to make sure we dont flood the user with warning messages
        warning_dispatched = False
        
        # number of data samples to assign to each processor
        samples_per_processor = batchgenutils.roundToMultiple(
            int(data.shape[0] / self.num_threads), 
            self.batch_size)

        # batches that will be generated by each process thread
        num_batches = []
        
        # spawn processes that will generate batches of data and "put"
        # into the multiprocessing queues
        for i in range(self.num_threads):
            mpq = mp.Queue()

            # give each process a slice of the dataframe of positives
            df = data[i*samples_per_processor : 
                      (i+1)*samples_per_processor][['chrom', 'pos']]

            # the last process gets the leftover data points
            if i == (self.num_threads-1):
                df = pd.concat([df, data[(i+1)*samples_per_processor:]])
                
            num_batches.append(len(df) // self.batch_size)
            
            if df.shape[0] != 0:
                logging.debug("{} spawning process {}, df size {}, "
                              "sum(num_batches) {}".format(
                              self.name_str, i, df.shape, sum(num_batches)))

                # spawn and start the batch generation process 
                p = mp.Process(target = self.proc_target, args = [df, mpq, i])
                p.start()
                procs.append(p)
                mp_queues.append(mpq)
                
            else:
                if not warning_dispatched:
                    logging.warn("One or more process threads are not being "
                                 "assigned data for parallel batch "
                                 "generation. You should reduce the number "
                                 "of threads using the --threads option "
                                 "for better performance. Inspect logs for "
                                 "batch assignments.")
                    warning_dispatched = True
                
                logging.debug("{} skipping process {}, df size {}, "
                              "num_batches {}".format(
                              self.name_str, i, df.shape, sum(num_batches)))
                
                procs.append(None)
                mp_queues.append(None)

        logging.debug("{} num_batches list {}".format(self.name_str, 
                                                      num_batches))
                
        # the threads that will "get" from mp queues 
        # and put into the regular queue
        # this speeds up yielding of batches, because "get"
        # from mp queue is very slow
        for i in range(self.num_threads):
            # start a stealer thread only if data was assigned to
            # the i-th  process
            if num_batches[i] > 0:
                
                logging.debug("{} starting stealer thread {} [{}] ".format(
                    self.name_str, i, num_batches[i]))
                
                mp_q = mp_queues[i]
                stealerThread = Thread(target=self.stealer, 
                                       args=[mp_q, q, num_batches[i], i])
                stealerThread.start()
                threads.append(stealerThread)
            else:
                threads.append(None)
                
                logging.debug("{} skipping stealer thread {} ".format(
                    self.name_str, i, num_batches))

        return procs, threads, q, sum(num_batches)

    def gen(self, data):
        """
            generator function to yield batches of data

            Args:
                data (pandas.DataFrame): dataframe with 'chrom' &
                    'pos' columns
                            
            Returns:
                ---
        """
        for i in range(self.epochs):
            # set this flag to False and wait for the
            self.ready_for_next_epoch = False
            
            logging.debug("{} ready set to FALSE".format(self.name_str))
            
            if self.shuffle: # shuffle at the beginning of each epoch
                data = data.sample(frac = 1.0)
                logging.debug("{} Shuffling complete".format(self.name_str))

            # spawn multiple processes to generate batches of data in
            # parallel for each epoch
            procs, threads, q, total_batches = self.epoch_run(data)

            logging.debug("{} Batch generation for epoch {} started".format(
                self.name_str, i+1))
            
            # yield the correct number of batches for each epoch
            for j in range(total_batches):                
                yield q.get()

            # wait for batch generation processes to finish once the
            # required number of batches have been yielded
            for j in range(self.num_threads):
                if procs[j] is not None:
                    procs[j].join()
                    
                if threads[j] is not None:
                    threads[j].join()
                
                logging.debug("{} join complete for process {}".format(
                    self.name_str, j))
            
            logging.debug("{} Finished join for epoch {}".format(
                self.name_str, i+1))
            
            # wait here for the signal 
            while (not self.ready_for_next_epoch) and (not self.stop):
                continue

            logging.debug("{} Ready for next epoch".format(self.name_str))
            
            if self.stop:
                logging.debug("{} Terminating batch generation".format(
                    self.name_str))
                break
                

class MTBPNetBatchGenerator(MTBatchGenerator):
    """MultiTask batch data generation for BPNet
    
    """

    def __init__(self, tasks, reference, chrom_sizes, chroms, batchgen_params,
                 name_str, stranded=True, num_threads=28, epochs=100, 
                 batch_size=64, shuffle=True, mode='train', 
                 control_smoothing=[1, 50]):
        
        # call base class constructor
        super().__init__(tasks, reference, chrom_sizes, chroms, 
                         batchgen_params, name_str, stranded, 
                         num_threads, epochs, batch_size, shuffle, mode)
        
        
        self.control_smoothing = control_smoothing

        
    def generate_batch(self, coords):
        """Generate one batch of inputs and outputs
            
            For all coordinates in "coords" fetch sequences &
            one hot encode the sequences. Fetch corresponding
            signal values (for e.g. from a bigwig file), 
            and aggregate if specified. Package the one hot
            encoded sequences and the ouput values as a tuple.
            
            Args:
                coords (pandas.DataFrame): dataframe with 'chrom' and
                    'pos' columns specifying the chromosome and the 
                    coordinate
                
            Returns:
                tuple: A batch tuple with one hot encoded sequences 
                and corresponding outputs 
        """
        
        # reference file to fetch sequences
        fasta_ref = pysam.FastaFile(self.reference)

        # TODO: currently in the multitask case the control is
        #       summed across all tasks and strands. Not sure 
        #       if this the correct way to do it, but user should
        #       have the option to map the corresponding task to
        #       their corresponding control tracks
        
        # Initialization
        # (batch_size, output_len, #smoothing_window_sizes)
        control_profile = np.zeros((coords.shape[0], self.output_flank*2, 
                                    len(self.control_smoothing)), 
                                   dtype=np.float32)
        
        # (batch_size)
        control_profile_counts = np.zeros((coords.shape[0]), 
                                          dtype=np.float32)

        # in 'test' mode we only need the sequence & the control
        if self.mode == 'train':
            # (batch_size, output_len, #tasks)
            profile = np.zeros((coords.shape[0], self.output_flank*2, 
                                self.num_tasks), dtype=np.float32)
        
            # (batch_size, #tasks)
            profile_counts = np.zeros((coords.shape[0], self.num_tasks), 
                                      dtype=np.float32)
        
        # if reverse complement augmentation is enabled then double the sizes
        if self.mode == 'train' and self.rev_comp_aug:
            control_profile = control_profile.repeat(2, axis=0)
            control_profile_counts = control_profile_counts.repeat(2, axis=0)
            profile = profile.repeat(2, axis=0)
            profile_counts = profile_counts.repeat(2, axis=0)
 
        # list of sequences in the batch, these will be one hot
        # encoded together as a single sequence after iterating
        # over the batch
        sequences = []  
        
        # list of chromosome start/end coordinates 
        # useful for tracking test batches
        coordinates = []
        
        # open all the control bigwig files and store the file 
        # objects in a dictionary
        control_files = {}
        for task in self.tasks:
            # the control is not necessary 
            if 'control' in self.tasks[task]:
                control_files[task] = pyBigWig.open(
                    self.tasks[task]['control'])

        # in 'test' mode we only need the sequence & the control
        if self.mode == 'train':
            # open all the required bigwig files and store the file 
            # objects in a dictionary
            signal_files = {}
            for task in self.tasks:
                signal_files[task] = pyBigWig.open(self.tasks[task]['signal'])
            
        # iterate over the batch
        rowCnt = 0
        for _, row in coords.iterrows():
            # randomly set a jitter value to move the peak summit 
            # slightly away from the exact center
            jitter = 0
            if self.mode == 'train' and self.max_jitter:
                jitter = random.randint(-self.max_jitter, self.max_jitter)
            
            # Step 1 get the sequence 
            chrom = row['chrom']
            # we use self.flank here and not self.output_flank because
            # input seq_len is different from output_len
            start = row['pos'] - self.flank + jitter
            end = row['pos'] + self.flank + jitter
            seq = fasta_ref.fetch(chrom, start, end).upper()
            
            # collect all the sequences into a list
            sequences.append(seq)
            
            start = row['pos'] - self.output_flank  + jitter
            end = row['pos'] + self.output_flank + jitter
            
            # collect all the start/end coordinates into a list
            # we'll send this off along with 'test' batches
            coordinates.append((chrom, start, end))

            # iterate over each task
            for task in self.tasks:
                # identifies the +/- strand pair
                task_id = self.tasks[task]['task_id']
                
                # the strand id: 0-positive, 1-negative
                # easy to index with those values
                strand = self.tasks[task]['strand']
                
                # Step 2. get the control values
                if task in control_files:
                    control_values = control_files[task].values(
                        chrom, start, end)

                    # replace nans with zeros
                    if np.any(np.isnan(control_values)): 
                        control_values = np.nan_to_num(control_values)

                    # update row in batch with the control values
                    # the values are summed across all tasks
                    # the axis = 1 dimension accumulates the sum
                    # there are 'n' copies of the sum along axis = 2, 
                    # n = #smoothing_windows
                    control_profile[rowCnt, :, :] += np.expand_dims(
                        control_values, axis=1)
                
                # in 'test' mode we only need the sequence & the control
                if self.mode == 'train':
                    # Step 3. get the signal values
                    # fetch values using the pyBigWig file objects
                    values = signal_files[task].values(chrom, start, end)
                
                    # replace nans with zeros
                    if np.any(np.isnan(values)): 
                        values = np.nan_to_num(values)

                    # update row in batch with the signal values
                    if self.stranded:
                        profile[rowCnt, :, task_id*2 + strand] = values
                    else:
                        profile[rowCnt, :, task_id] = values

            rowCnt += 1
        
        # Step 4. reverse complement augmentation
        if self.mode == 'train' and self.rev_comp_aug:
            # Step 4.1 get list of reverse complement sequences
            rev_comp_sequences = reverse_complement_of_sequences(sequences)
            
            # append the rev comp sequences to the original list
            sequences.extend(rev_comp_sequences)
            
            # Step 4.2 reverse complement of the control profile
            control_profile[rowCnt:, :, :] = reverse_complement_of_profiles(
                control_profile[:rowCnt, :, :], stranded=self.stranded)
            
            # Step 4.3 reverse complement of the signal profile
            profile[rowCnt:, :, :]  = reverse_complement_of_profiles(
                profile[:rowCnt, :, :], stranded=self.stranded)

        # Step 5. one hot encode all the sequences in the batch 
        X = one_hot_encode(sequences)
 
        # we can perform smoothing on the entire batch of control values
        for i in range(len(self.control_smoothing)):
            if self.control_smoothing[i] > 1:
                control_profile[:, :, i] = gaussian_filter1d(
                    control_profile[:, :, i], self.control_smoothing[i])

        # log of sum of control profile without smoothing (idx = 0)
        control_profile_counts = np.log(
            np.sum(control_profile[:, :, 0], axis=-1) + 1)
        
        # in 'test' mode we only need the sequence & the control
        if self.mode == 'train':
            # we can now sum the profiles for the entire batch
            profile_counts = np.log(np.sum(profile, axis=1) + 1)
    
            # return a tuple of input and output dictionaries
            return ({'sequence': X, 
                     'control_profile': control_profile, 
                     'control_logcount': control_profile_counts},
                    {'profile_predictions': profile, 
                     'logcount_predictions': profile_counts})

        # in 'test' mode return a tuple of cordinates & the
        # input dictionary
        return (coordinates, {'sequence': X, 
                            'control_profile': control_profile,
                            'control_logcount': control_profile_counts})

# create alias for the batch generator
MTBPNetSumAllBatchGenerator = MTBPNetBatchGenerator
