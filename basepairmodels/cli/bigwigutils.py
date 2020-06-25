import pyBigWig
import logging
import pandas as pd
import numpy as np

def prepare_BPNet_output_files(tasks, output_dir, chroms, chrom_sizes, 
                               model_tag, exponentiate_counts, other_tags=[]):
    """ prepare output bigWig files for writing bpnet predictions
        a. Construct aprropriate filenames
        b. Add headers to each bigWig file
    
        Args:
            tasks (collections.OrderedDict): nested python dictionary
                of tasks. The predictions of each task will be
                written to a separate bigWig
            output_dir (str): destination directory where the output
                files will be created
            chroms (list): list of chromosomes for which the bigWigs
                will contain predictions
            chrom_sizes (str): the path to the chromosome sizes file.
                The chrom size is used in constructing the header of
                the bigWig file
            model_tag (str): the unique tag of the model that is 
                generating the predictions
            exponentiate_counts (boolean): True if counts predictions
                are to be exponentiated before writing to the bigWigs.
                This will determine if the counts bigWigs have the 
                'exponentiated' tag in the filename
            other_tags (list): list of additional tags to be added as
                suffix to the filenames

        Returns:
            tuple: (list of profile bigWig file objects, 
                    list of counts bigWig file objects)
            
    """
    
    # the lists of file objects 
    profile_fileobjs = []
    counts_fileobjs = []

    # one profile and one counts bigWig for each task
    for task in tasks:
        
        other_tags = '_'.join(other_tags)
        if len(other_tags) > 0:
            profile_fname = "{}/{}_{}_{}.bw".format(
                output_dir, model_tag, '_'.join(other_tags),
                task)

            counts_fname = "{}/{}_{}_{}_counts.bw".format(
                output_dir, model_tag, '_'.join(other_tags),
                task)
        else:
            profile_fname = "{}/{}_{}.bw".format(
                output_dir, model_tag, task)

            counts_fname = "{}/{}_{}_counts.bw".format(
                output_dir, model_tag, task)
                         
        # add 'exponentiated' tag in the counts filename
        if exponentiate_counts:
            counts_fname = counts_fname.replace(
                '_counts.bw', '_exponentiated_counts.bw')

        logging.info("Profile bigWig - {}".format(profile_fname))
        logging.info("Counts bigWig - {}".format(counts_fname))
        # open the bigWig files and add to the list of file objects
        profile_fileobjs.append(pyBigWig.open(profile_fname, 'w'))                     
        counts_fileobjs.append(pyBigWig.open(counts_fname, 'w'))
                 
    # read the chrom sizes into a dataframe 
    # (for constructing the bigWig header)
    chrom_sizes_df = pd.read_csv(chrom_sizes, sep = '\t', header=None, 
                                 names = ['chrom', 'size'])
    chrom_sizes_df = chrom_sizes_df.set_index('chrom')
    
    
    # construct header for the bigWig file
    header = []
    # sort chromosomes, to be consistent with how pandas sorts
    # chromosomes ... for e.g. chrom21 is < chrom8
    chroms.sort()
    for chrom in chroms:
        size = chrom_sizes_df.at[chrom, 'size']
        header.append((chrom, int(size)))

    logging.debug("bigWig HEADER - {}".format(header))
    
    # add the header to all the bigWig files
    for file_obj in profile_fileobjs:
        file_obj.addHeader(header, maxZooms=0)
 
    for file_obj in counts_fileobjs:
        file_obj.addHeader(header, maxZooms=0)

    # return tuple of lists of profile and counts file objects
    return (profile_fileobjs, counts_fileobjs)


def write_BPNet_predictions(profile_predictions, counts_predictions,
                            profile_fileobjs, counts_fileobjs, 
                            coordinates, tasks, exponentiate_counts,
                            output_window_size):
    
    """ write one batch of BPNet predictions to bigWig files
    
        Args:
            profile_predictions (np.ndarray): 3 dimensional numpy 
                array of size (batch_size, output_len, 
                num_tasks*num_strands)
            counts_predictions (np.ndarray): 2 dimensional numpy 
                array of size (batch_size, num_tasks*num_strands)
            profile_fileobjs (list): list of file objects that have
                been opened to write profile predicitions
            counts_fileobjs (list): list of file objects that have
                been opened to write counts predicitions
            coordinates (list): list of (chrom, start, end) for each
                prediction
            tasks (collections.OrderedDict): nested python dictionary
                of tasks
            exponentiate_counts (boolean): True if counts predictions
                are to be exponentiated before writing to the bigWigs
            output_window_size (int): size of the central window of 
                the output 

    """
        
    # see the 'Adding entries to a bigWig file' section here
    # https://github.com/deeptools/pyBigWig, then scroll down to the
    # section on Numpy
    
    # to optimize the write time and avoid multiple writes to the 
    # bigWig files, for each output track we will write the entire 
    # batch of values in one go. To do that we have to pre allocate
    # arrays to hold the chrom strings, the start & end coordinates
    # and the values
    
    # the combined length of the profile outputs for the whole batch
    profile_array_len =  profile_predictions.shape[0] * output_window_size
    
    # pre allocate unicode numpy array with a dummy string of length 5
    # 'chr1' is length 4, 'chr20' is length 5
    profile_chroms =  np.repeat(
        np.array(["CCCCC"] * profile_array_len)[..., np.newaxis], 
        len(profile_fileobjs), axis=1)
    
    # array to hold the start coordinates
    profile_starts = np.zeros((profile_array_len, len(profile_fileobjs)), 
                              dtype=np.int64)
    
    # array to hold the end coordinates
    profile_ends = np.zeros((profile_array_len, len(profile_fileobjs)), 
                            dtype=np.int64)
    
    # array to hold the values
    profile_vals = np.zeros((profile_array_len, len(profile_fileobjs)), 
                            dtype=np.float64)

    # the combined length of the counts outputs for the whole batch
    counts_array_len =  counts_predictions.shape[0]

    # pre allocate unicode numpy array with a dummy string of length 5
    # 'chr1' is length 4, 'chr20' is length 5
    counts_chroms =  np.repeat(
        np.array(["CCCCC"] * counts_array_len)[..., np.newaxis], 
        len(counts_fileobjs), axis=1)
    
    # array to hold the start coordinates
    counts_starts = np.zeros((counts_array_len, len(counts_fileobjs)), 
                              dtype=np.int64)
    
    # array to hold the end coordinates
    counts_ends = np.zeros((counts_array_len, len(counts_fileobjs)), 
                            dtype=np.int64)
    
    # array to hold the values
    counts_vals = np.zeros((counts_array_len, len(counts_fileobjs)), 
                            dtype=np.float64)
    
    # populate the preallocated array
    for i in range(len(coordinates)):
        (chrom, start, end) = coordinates[i]            

        # profile_predictions has the predicted profiles
        # profile_predictions.shape = 
        #    (batchsize, output_len, num_tasks*num_strands)
        # counts_predictions has the predicted log(sum(counts))
        # counts_predictions.shape = (batchsize, num_tasks*num_strands)

        # mid section in chrom coordinates based on output_window_size
        start = start + (end - start) // 2 - output_window_size // 2  
        end = start + output_window_size

        # specify the coordinates for the profiles track,
        # the arrays are of length output_window_size
        profile_starts[i*output_window_size:(i+1)*output_window_size, :] = \
            np.expand_dims(np.arange(start, end, dtype=np.int64), axis=1)
        profile_ends[i*output_window_size:(i+1)*output_window_size, :] = \
            np.expand_dims(np.arange(start+1, end+1, dtype=np.int64), axis=1)
        profile_chroms[i*output_window_size:(i+1)*output_window_size, :] = \
            np.expand_dims(np.array([chrom] * output_window_size), axis=1)

        # now the values
        
        # length of the output 
        output_len = profile_predictions[i].shape[0]

        # start and end indices of the mid section
        # of the predictions corresponding to output_window_size
        s_idx = output_len // 2 - output_window_size // 2
        e_idx = s_idx + output_window_size

        # get the values and populate profile_vals
        for j in range(len(profile_fileobjs)):
            profile_vals[i*output_window_size:(i+1)*output_window_size, j] = \
                profile_predictions[i, s_idx:e_idx, j]  
            
        # specify the coordinates for the counts track,
        # the arrays are of length 1
        counts_starts[i, :] = start
        counts_ends[i, :] = end
        counts_chroms[i, :] = chrom
        
        # now the values        
        # get the values and populate counts_vals
        for j in range(len(counts_fileobjs)):
            val =  counts_predictions[i, j]
            
            if exponentiate_counts:
                val = np.exp(val)
            
            counts_vals[i, j] = val

    # now write the values to the bigWig files
    
    try:
        # add entries to profile bigWigs
        for j in range(len(profile_fileobjs)):
            profile_fileobjs[j].addEntries(profile_chroms[:, j].tolist(), 
                                           profile_starts[:, j].tolist(), 
                                           ends=profile_ends[:, j].tolist(), 
                                           values=profile_vals[:, j].tolist())
    
        # add entries to counts bigWigs
        for j in range(len(counts_fileobjs)):
            counts_fileobjs[j].addEntries(counts_chroms[:, j].tolist(),
                                          counts_starts[:, j].tolist(),
                                          ends=counts_ends[:, j].tolist(), 
                                          values=counts_vals[:, j].tolist())
    except Exception as e:
        logging.error("Skipping the following coordinates due to an error "
                      "{}".format(coordinates))

