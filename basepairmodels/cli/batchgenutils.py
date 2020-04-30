import pandas as pd
import numpy as np
import glob
import logging
import os

from collections import OrderedDict

def getChromPositions(chroms, chrom_sizes, flank, step, mode='sequential', 
                     num_positions=-1):
    """Chromosome positions spanning the entire chromosome at 
       a) regular intervals or b) random locations
        
        Args:
            chroms (space separated list): The list of required 
                chromosomes 
            chrom_sizes (str): path to the chromosome sizes file
            flank (int): Buffer size before & after the position to  
                ensure we dont fetch values at index < 0 & > chrom size
            step (int): the interval between consecutive chromosome
                positions
            mode (str): mode of returned position 'sequential' (from
                the beginning) or 'random
            num_positions (int): number of chromosome positions
                to return on each chromosome, use -1 to return 
                positions across the entrire chromosome for all given
                chromosomes in `chroms`. mode='random' cannot be used
                with num_positions=-1
            
        Returns:
            pandas.DataFrame: two column dataframe of chromosome 
                positions (chrom, pos)
            
    """
    
    if mode == 'random' and num_positions == -1:
        logging.error("Incompatible parameter passed to getTestDataFrame: " + 
                      "'mode' = random, `num_positions`=-1")
        return None

    # read the chrom sizes into a dataframe 
    chrom_sizes = pd.read_csv(chrom_sizes, sep = '\t', 
                          header=None, names = ['chrom', 'size']) 
    
    # keep only those chrom_sizes rows corresponding to the required
    # chromosomes
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]
    chrom_sizes = chrom_sizes.set_index('chrom')
    
    # initialize an empty dataframe with 'chrom' and 'pos' columns
    positions = pd.DataFrame(columns=['chrom', 'pos'])

    # for each chromosome in the list
    for i in range(len(chroms)):
        chrom_size = chrom_sizes.at[chroms[i], 'size']
        
        # keep start & end within bounds
        start = flank
        end = chrom_size - flank + 1
                
        if mode == 'random':
            # randomly sample positions
            pos_array = np.random.randint(start, end, num_positions)

        if mode == 'sequential':
            _end = end
            if num_positions != -1:
                # change the last positon based on the number of 
                # required positions
                _end = start + step*num_positions
                
                # if the newly computed 'end' goes beyond the 
                # chromosome end (we could throw an error here)
                if _end > end:
                    _end = end
        
            # positions at regular intervals
            pos_array = list(range(start, _end, step))    

        # construct a dataframe for this chromosome
        chrom_df = pd.DataFrame({'chrom': [chroms[i]]*len(pos_array), 
                                 'pos': pos_array})
        
        # concatenate to existing df
        positions = pd.concat([positions, chrom_df])
        
    return positions    
    

def getPeakPositions(tasks, chroms, chrom_sizes, flank, drop_duplicates=False, 
                     sort_across_tasks=False):
    """ Peak positions for all the tasks filtered based on required
        chromosomes and other qc filters. Since 'task' here refers 
        one strand of input/output, if the data is stranded the peaks
        will be duplicated for the plus and minus strand.
        
        
        Args:
            tasks (dict): A python dictionary containing the task
                information. Each task in tasks should have the
                key 'peaks' that has the path to he peaks file
            chroms (list): The list of required test chromosomes
            chrom_sizes (str): path to the chromosome sizes file
            flank (int): Buffer size before & after the position to  
                ensure we dont fetch values at index < 0 & > chrom size
            drop_duplicates (boolean): if True duplicates will be
                dropped from returned dataframe. 
            sort_across_tasks (boolean): if True a final sort is done
                such that chrom positions are sorted across all tasks
            
        Returns:
            pandas.DataFrame: two column dataframe of peak 
                positions (chrom, pos)
            
    """

    # read the chrom sizes into a dataframe 
    chrom_sizes = pd.read_csv(chrom_sizes, sep = '\t', 
                          header=None, names = ['chrom', 'size']) 
    
    # keep only those chrom_sizes rows corresponding to the required
    # chromosomes
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]

    # necessary for dataframe apply operation below --->>>
    chrom_size_dict = dict(chrom_sizes.to_records(index=False))

    # initialize an empty dataframe
    allPeaks = pd.DataFrame()

    for task in tasks:   
        peaks_df = pd.read_csv(tasks[task]['peaks'], 
                               sep='\t', header=None, 
                               names=['chrom', 'st', 'end', 'name', 'score',
                                      'strand', 'signal', 'p', 'q', 'summit'])

        print(peaks_df.shape)
        # keep only those rows corresponding to the required 
        # chromosomes
        peaks_df = peaks_df[peaks_df['chrom'].isin(chroms)]
        print(peaks_df.shape)

        # create new column for peak pos
        peaks_df['pos'] = peaks_df['st'] + peaks_df['summit']

        # compute left flank coordinates of the input sequences 
        # (including the allowed jitter)
        peaks_df['flank_left'] = (peaks_df['pos'] - flank).astype(int)

        # compute right flank coordinates of the input sequences 
        # (including the allowed jitter)
        peaks_df['flank_right'] = (peaks_df['pos'] + flank).astype(int)

        # filter out rows where the left flank coordinate is < 0
        peaks_df = peaks_df[peaks_df['flank_left'] >= 0]

        # --->>> create a new column for chrom size
        peaks_df["chrom_size"] = peaks_df['chrom'].apply(
            lambda chrom: chrom_size_dict[chrom])

        # filter out rows where the right flank coordinate goes beyond
        # chromosome size
        peaks_df = peaks_df[peaks_df['flank_right'] <= peaks_df['chrom_size']]

        # sort based on chromosome number and right flank coordinate
        peaks_df = peaks_df.sort_values(['chrom', 'flank_right']).reset_index(
            drop=True)

        # append to all peaks data frame
        allPeaks = allPeaks.append(peaks_df[['chrom', 'pos']])

        allPeaks = allPeaks.reset_index(drop=True)
    
    # drop the duplicate rows, i.e. the peaks that get duplicated
    # for the plus and minus strand tasks
    if drop_duplicates:
        allPeaks = allPeaks.drop_duplicates()
    
    # sort the chrom positions once again, now across all tasks
    if sort_across_tasks:
        allPeaks = allPeaks.sort_values(['chrom', 'pos']).reset_index(
            drop=True)

    return allPeaks

def getInputTasks(input_dir, stranded=False, has_control=False, 
                  mode='training', require_peaks=True):
    """Dictionary specifying the various tasks for the profile model
    
        Args:
            input_dir (str): the path to directory containing the 
                signal & peaks files for each task. For single task
                this points to directory containing the bigWig and 
                peaks.bed files. For multi task this points to the
                directory containing the sub directories for each task
            stranded (boolean): indicates whether the bigwig files 
                in the input_dir are stranded or unstranded
            has_control (boolean): indicates whether control is
                included
            mode (str): either 'training' or 'test'. In 'test' mode
                'signal' bigWigs are not required, only 'control'
                bigWigs maybe required
            require_peaks (boolean): specify whether the 'peaks' file
                is required for each task. If training or testing is
                performed on the whole genome or at randomly sampled
                points, then set to False
    
        Returns:
            collections.OrderedDict: nested python dictionary of tasks,
                specifying the 'signal' and/or 'control' bigWigs,
                'peaks' file, 'task_id' & 'strand'. 'strand' is 0 for
                plus strand and 1 for minus strand
                
    """
    
    # initialize
    tasks = OrderedDict()
    task_id = 0
    
    for root, dirs, filenames in os.walk(input_dir, topdown=True, 
                                         followlinks=True):
        
        # ignore hidden files & directories
        files = [f for f in filenames if not f[0] == '.']
        # dirs should be modified in place
        dirs[:] = [d for d in dirs if not d[0] == '.']
        
        
        # if the dir is one of the sub directories, and we find more
        # sub directories within
        if (root != input_dir) and (len(dirs) > 0):
            logging.error("Incompatible directory structure, " + 
                          "max allowed depth is 2.\n{}".format(root))
            return None

        # if at the top level we find both sub directories & files
        if (root == input_dir) and (len(dirs) > 0) and (len(filenames) > 0):
            logging.error("Incompatible directory structure. " +
                          "Directories can only exclusively have" + 
                          "sub directories or files, not both." + 
                          "\n{}\n{}".format(dirs, filenames))
            return None
        
        # inside a directory with only files & no sub directories
        if len(dirs) == 0:
            # get all the bigWigs that dont begin with 
            # 'control' prefix
            signal_bigwigs = glob.glob(root + '/[!control]*.bw')
            # the control bigWigs
            control_bigwigs = glob.glob(root + '/control*.bw')
            # the peaks file
            peaks = glob.glob(root + '/peaks.bed')
            # maybe the user didnt unzip the bed file
            if len(peaks) == 0:
                peaks = glob.glob(root + '/peaks.bed.gz')

            # if no peaks file found
            if require_peaks and len(peaks) == 0:
                logging.error("Peaks file not found! Peaks file " + 
                              "should be in bed format and named " + 
                              "peaks.bed.\n{}".format(root))
                return None
            else:
                peaks_fname = peaks[0]
    
            # if not in training mode signal bigWigs can be excluded
            # because we are making predictions
            if mode == 'training':
                # if no bigWig file is found
                if len(signal_bigwigs) == 0:
                    logging.error("At least one signal bigWig file " +
                                  "expected. None found.\n{}".format(root))
                    return None

                # if more than two signal bigWig files are found 
                # in stranded mode
                if len(signal_bigwigs) > 2 and stranded:
                    logging.error("At most two signal bigWig files allowed. " + 
                                  "One for plus strand and one for the " + 
                                  "minus strand.\n{}".format(signal_bigwigs))
                    return None
                
                # if two or more signal bigWig files found in 
                # 'unstranded' mode
                if len(signal_bigwigs) >= 2 and not stranded:
                    logging.error("Only one signal bigWig file " +
                                  "(unstranded.bw) must be given for " + 
                                  "unstranded tasks. Use --stranded to " +
                                  "indicate input data is stranded" +
                                  "\n{}".format(signal_bigwigs))
                    return None

                # if only one signal bigWig file is found in 'stranded' 
                # mode
                if len(signal_bigwigs) == 1 and stranded:
                    logging.error("Two signal bigWigs files (plus.bw and " + 
                                  "minus.bw) expected for stranded tasks." +
                                  "\n{}".format(signal_bigwigs))
                    return None


                # if the one signal bigWig file in 'unstranded' mode is 
                # not called 'unstranded.bw'
                if (len(signal_bigwigs) == 1) and \
                    root+'/unstranded.bw' not in signal_bigwigs:

                    logging.error("When using a single bigWig file it" + 
                                  "should be named unstranded.bw" + 
                                  "\n{}".format(signal_bigwigs))
                    return None

                # if the two bigWig files in 'stranded' mode are not 
                # called 'plus.bw' & 'minus.bw'
                if len(signal_bigwigs) == 2 and \
                    (root+'/plus.bw' not in signal_bigwigs or 
                     root+'/minus.bw' not in signal_bigwigs):

                    logging.error("When using two bigWig files the " +
                                  "assumption is that the files are " + 
                                  "for the two strands, so they should " + 
                                  "be named plus.bw and minus.bw." + 
                                  "\n{}".format(signal_bigwigs))
                    return None

            # if control is expected but contol bigWigs are NOT found
            if has_control and len(control_bigwigs) == 0:
                logging.error("Missing control bigWig file(s).  "
                              "--has_control flag indicates that control"
                              "should be present.\n{}".format(control_bigwigs))                
                
            # if control is NOT expected but contol bigWigs are found
            if not has_control and len(control_bigwigs) > 0:
                logging.error("Found control bigWig file(s) when"
                              "--has_control flag indicates that control"
                              "should NOT be present.\n{}".format(
                                  control_bigwigs))                

            # if more than two control bigWig files are found in 
            # 'stranded' mode
            if len(control_bigwigs) > 2 and stranded:
                logging.error("At most two control bigWig files allowed. " + 
                              "One for plus strand and one for the " + 
                              "minus strand.\n{}".format(control_bigwigs))
                return None

            # if two or more control bigWig files found in 
            # 'unstranded' mode
            if len(control_bigwigs) >= 2 and not stranded:
                logging.error("Only one control bigWig file " +
                              "(control_unstranded.bw)  must be given for " + 
                              "unstranded tasks. Use --stranded to " +
                              "indicate input data is stranded" + 
                              "\n{}".format(control_bigwigs))
                return None

            # if only one control bigWig file is found in 'stranded' 
            # mode
            if len(control_bigwigs) == 1 and stranded:
                logging.error("Two control bigWigs files (control_plus.bw  " + 
                              "and control_minus.bw) expected for stranded ." +
                              "tasks\n{}".format(control_bigwigs))

            # if the one control bigWig file in 'unstranded' mode is
            # not called 'control_unstranded.bw'
            if (len(control_bigwigs) == 1) and \
                root+'/control_unstranded.bw' not in control_bigwigs:
                
                logging.error("When using a single control bigWig file it" + 
                              "should be named control_unstranded.bw" + 
                              "\n{}".format(control_bigwigs))
                return None
            
            # if the two bigWig files in 'stranded' mode are not 
            # called 'control_plus.bw' & 'control_minus.bw'
            if len(control_bigwigs) == 2 and \
                (root+'/control_plus.bw' not in control_bigwigs or 
                 root+'/control_minus.bw' not in control_bigwigs):

                logging.error("When using two bigWig files the " +
                              "assumption is that the files are " + 
                              "for the two strands, so they should " + 
                              "be named control_plus.bw and " + 
                              "control_minus.bw.\n{}".format(signal_bigwigs))
                return None
            
            # add new entries to the dictionary
            if stranded:
                # in stranded mode we'll add separate entries for the
                # plus and minus strand
                # a. the `task_id` is the same for the plus-minus pair
                # b. 'strand' is 0 for plus and 1 for minus
                # c. `peaks` is the same for the plus-minus pair
                
                task_name_plus = '{}_plus'.format(os.path.basename(root))
                task_name_minus = '{}_minus'.format(os.path.basename(root))

                tasks[task_name_plus] = {'strand': 0, 
                                         'task_id': task_id}
                
                tasks[task_name_minus] = {'strand': 1, 
                                          'task_id': task_id}
                
                # add the signal track in training mode
                if mode == 'training':
                    tasks[task_name_plus]['signal'] = \
                        '{}/plus.bw'.format(root)
                    tasks[task_name_minus]['signal'] = \
                        '{}/minus.bw'.format(root)
                
                # add peaks file if required
                if require_peaks:
                    tasks[task_name_plus]['peaks'] = '{}'.format(peaks_fname)
                    tasks[task_name_minus]['peaks'] = '{}'.format(peaks_fname)

                # add entries for control if it exists
                if has_control:
                    tasks[task_name_plus]['control'] = \
                        '{}/control_plus.bw'.format(root)
                
                    tasks[task_name_minus]['control'] = \
                        '{}/control_minus.bw'.format(root)
            
            # unstranded
            else:
                task_name = '{}_unstranded'.format(os.path.basename(root))

                tasks[task_name] = {'strand': 0, 
                                    'task_id': task_id}

                # add the signal track in training mode
                if mode == 'training':
                    tasks[task_name]['signal'] = \
                        '{}/unstranded.bw'.format(root)
                    
                # add peaks file if required
                if require_peaks:
                    tasks[task_name]['peaks'] = '{}'.format(peaks_fname)
                                       
                # add entries for control if it exists
                if has_control:
                    tasks[task_name]['control'] = \
                        '{}/control_unstranded.bw'.format(root)
            
            # increment task id for next task
            task_id += 1
            
        else:
            # in place sort of `dirs`, so tasks are always created
            # in sorted order (os.walk will walk through in sorted
            # order)
            dirs.sort()
    
    return tasks

    
def roundToMultiple(x, y): 
    """Return the largest multiple of y < x
        
        Args:
            x (int): the number to round
            y (int): the multiplier
        
        Returns:
            int: largest multiple of y < x
            
    """

    r = (x+int(y/2)) & ~(y-1)
    if r > x:
        r = r - y
    return r


def one_hot_encode(sequences):
    """One hot encoding of a list of DNA sequences
       
       Args:
           sequences (list):: python list of strings of equal length
           
       Returns:
           numpy.ndarray: 3-dimension numpy array with shape
               (len(sequences), len(list_item), 4)

    """
    
    if len(sequences) == 0:
        logging.error("'sequences' is empty")
    
    # make sure all sequences are of equal length
    seq_len = len(sequences[0])
    for sequence in sequences:
        if len(sequence) != seq_len:
            logging.error("Incompatible sequence lengths. All sequences " +
                          "should have the same length.")
            
    # Step 1. convert sequence list into a single string
    _sequences = ''.join(sequences)
    
    # Step 2. translate the alphabet to a string of digits
    transtab = str.maketrans('ACGTN', '01234')    
    sequences_trans = _sequences.translate(transtab)
    
    # Step 3. convert to list of ints
    int_sequences = list(map(int, sequences_trans))
    
    # Step 4. one hot encode using int_sequences to index 
    # into an 'encoder' array
    encoder = np.vstack([np.eye(4), np.zeros(4)])
    X = encoder[int_sequences]

    # Step 5. reshape 
    return X.reshape(len(sequences), len(sequences[0]), 4)

            
def reverse_complement_of_sequences(sequences):
    """Reverse complement of DNA sequences
       
       Args:
           sequences (list): python list of strings of DNA sequence of 
               arbitraty length
    
        Returns:
            list: python list of strings
    
    """

    if len(sequences) == 0:
        logging.error("'sequences' is empty")
    
    # reverse complement translation table
    rev_comp_tab = str.maketrans("ACTG", "TGAC")

    # translate and reverse ([::-1] <= [start:end:step])
    return [seq.translate(rev_comp_tab)[::-1] for seq in sequences]


def reverse_complement_of_profiles(profiles, stranded=True):
    """Reverse complement of an genomics assay signal profile 

       CONVERT (Stranded profile)
                ______
               |      |
               |      |       
        _______|      |___________________________________________
        acgggttttccaaagggtttttaaaacccggttgtgtgtccacacacagtgtgtcaca
        ----------------------------------------------------------
        ----------------------------------------------------------
        ʇƃɔɔɔɐɐɐɐƃƃʇʇʇɔɔɔɐɐɐɐɐʇʇʇʇƃƃƃɔɔɐɐɔɐɔɐɔɐƃƃʇƃʇƃʇƃʇɔɐɔɐɔɐƃʇƃʇ
        ____________________________________________      ________
                                                    \    /
                                                     \  /
                                                      \/

        TO                                                                      

                  /\
                 /  \      
        ________/    \____________________________________________
        tgtgacacactgtgtgtggacacacaaccgggttttaaaaaccctttggaaaacccgt
        ----------------------------------------------------------
        ----------------------------------------------------------
        ɐɔɐɔʇƃʇƃʇƃɐɔɐɔɐɔɐɔɔʇƃʇƃʇƃʇʇƃƃɔɔɔɐɐɐɐʇʇʇʇʇƃƃƃɐɐɐɔɔʇʇʇʇƃƃƃɔɐ
        ___________________________________________        _______
                                                   |      |
                                                   |      | 
                                                   |______|                                                          


        OR 
        
        CONVERT (unstranded profile)
        
                ______
               |      |
               |      |       
        _______|      |___________________________________________
        acgggttttccaaagggtttttaaaacccggttgtgtgtccacacacagtgtgtcaca

        TO                                          
                                                    ______
                                                   |      |
                                                   |      |
        ___________________________________________|      |_______
        tgtgacacactgtgtgtggacacacaaccgggttttaaaaaccctttggaaaacccgt
                                                                  
        

        Args:
            profiles (numpy.ndarray): 3-dimensional numpy array, a 
                batch of multitask profiles of shape 
                (#examples, seq_len, #assays) if unstranded and 
                (#examples, seq_len, #assays*2) if stranded. In the
                stranded case the ssumption is: the postive & negative
                strands occur in pairs on axis=2(i.e. 3rd dimension) 
                e.g. 0th & 1st index, 2nd & 3rd...

        Returns:
            numpy.ndarray: 3-dimensional numpy array 

    """
    
    # check if profiles is 3-dimensional
    if profiles.ndim != 3:
        logging.error("'profiles' should be a 3-dimensional array. " + 
                      "Found {}".format(profiles.ndim))
        
    # check if the 3rd dimension is an even number if profiles are stranded
    if stranded and (profiles.shape[2] % 2) != 0:
        logging.error("3rd dimension of stranded 'profiles' should be even. "
                      "Found {}".format(profiles.shape))

    if stranded:
    
        # get reshaped version of profiles array
        # axis = 2 becomes #assays
        tmp_prof = profiles.reshape(
            (profiles.shape[0], profiles.shape[1], -1, 2))

        # get reverse complement by flipping along axis 1 & 3
        # axis 1 is the sequence length axis & axis 3 is the 
        # +/- strand axis after reshaping
        rev_comp_profile = np.flip(tmp_prof, axis=(1, 3))
        
        # reshape back to match shape of the input
        return rev_comp_profile.reshape(profiles.shape)

    else:
        
        return np.flip(profiles, axis=1)
