def get_10_human_val_test_splits():
    """ Validation and test chromosome splits for the human genome
       
        Returns:
            dict: nested python dictionary of 10 different validation
                and test splits     
    """

    # borrowed from kerasAC/kerasAC/splits.py
    splits = {}
    splits[0] = {'val': ['chr10', 'chr8'], 
                 'test': ['chr1']}
    
    splits[1] = {'val': ['chr1'],
                 'test': ['chr19', 'chr2']}
    
    splits[2] = {'val':['chr19', 'chr2'],
                 'test':['chr3', 'chr20']}
    
    splits[3] = {'val': ['chr3', 'chr20'],
                 'test': ['chr13', 'chr6', 'chr22']}
    
    splits[4] = {'val': ['chr13', 'chr6', 'chr22'],
                 'test': ['chr5', 'chr16', 'chrY']}
    
    splits[5] = {'val': ['chr5', 'chr16', 'chrY'],
                 'test': ['chr4', 'chr15', 'chr21']}
    
    splits[6] = {'val': ['chr4', 'chr15', 'chr21'],
                 'test': ['chr7', 'chr18', 'chr14']}
    
    splits[7] = {'val': ['chr7', 'chr18', 'chr14'],
                 'test': ['chr11', 'chr17', 'chrX']}
    
    splits[8] = {'val': ['chr11', 'chr17', 'chrX'],
                 'test': ['chr12', 'chr9']}
    
    splits[9] = {'val':['chr12','chr9'],
                 'test':['chr10','chr8']}
    
    return splits

def get_1_human_val_test_split():
    """ Validation and test chromosome split for the human genome. 
        Returns the first split from get_10_human_val_test_splits
       
        Returns:
            dict: nested python dictionary with one validation & test
                split
    """
    
    return {0: get_10_human_val_test_splits()[0]}

def get_1_yeast_val_test_split():
    """ Validation and test chromosome split for the yeast genome. 
       
        Returns:
            dict: nested python dictionary with one validation & test
                split
    """
    
    splits = {}
    splits[0] = {'val': ['chrVII', 'chrX'],
                 'test': ['chrII', 'chrV']}
    
    return splits


def get_C2H2_ZNF_split():
    """ Validation and test chromosome split for the human genome
        specifically for the C2H2_ZNF project
        
        Returns:
            dict: nested python dictionary with one validation & test
                split
    """    
    
    splits = {}
    splits[0] = {'val': ['chr1', 'chr9'], 
                 'test': ['chr8', 'chr21']}
    
    return splits

def get_hg38_chroms():
    """Chromosomes in the human genome
    
        Returns:
            list: list of chromosomes
    
    """
    
    chroms = ['chr1','chr2','chr3','chr4','chr5','chr6',
              'chr7','chr8','chr9','chr10','chr11','chr12',
              'chr13','chr14','chr15','chr16','chr17','chr18',
              'chr19','chr20','chr21','chr22','chrX','chrY']
    
    return chroms

def get_hg19_chroms():
    """Chromosomes in the human genome
    
        Returns:
            list: list of chromosomes
    
    """
    
    return get_hg38_chroms()

def get_mm10_chroms():
    """Chromosomes in the mouse genome
    
        Returns:
            list: list of chromosomes
    
    """
    
    chroms = ['chr1','chr2','chr3','chr4','chr5','chr6',
              'chr7','chr8','chr9','chr10','chr11','chr12',
              'chr13','chr14','chr15','chr16','chr17','chr18',
              'chr19','chrX','chrY']
    
    return chroms

def get_10_mouse_val_test_splits():
    """ Validation and test chromosome splits for the mouse genome
    
        Returns:
            dict: nested python dictionary of 10 different validation
                and test splits     
    """

    splits = {}
    splits[0] = {'val': ['chr10', 'chr8'], 
                 'test': ['chr1']}
    
    splits[1] = {'val': ['chr1'],
                 'test': ['chr19', 'chr2']}
    
    splits[2] = {'val':['chr19', 'chr2'],
                 'test':['chr3']}
    
    splits[3] = {'val': ['chr3'],
                 'test': ['chr13', 'chr6']}
    
    splits[4] = {'val': ['chr13', 'chr6'],
                 'test': ['chr5', 'chr16', 'chrY']}
    
    splits[5] = {'val': ['chr5', 'chr16', 'chrY'],
                 'test': ['chr4', 'chr15']}
    
    splits[6] = {'val': ['chr4', 'chr15'],
                 'test': ['chr7', 'chr18', 'chr14']}
    
    splits[7] = {'val': ['chr7', 'chr18', 'chr14'],
                 'test': ['chr11', 'chr17', 'chrX']}
    
    splits[8] = {'val': ['chr11', 'chr17', 'chrX'],
                 'test': ['chr12', 'chr9']}
    
    splits[9] = {'val':['chr12','chr9'],
                 'test':['chr10','chr8']}
    
    return splits

def get_10_mouse_val_test_splits():
    """ Validation and test chromosome split for the mouse genome. 
        Returns the first split from get_10_mouse_val_test_splits
       
        Returns:
            dict: nested python dictionary with one validation & test
                split
    """
    
    return {0: get_10_mouse_val_test_splits()[0]}