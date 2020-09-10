import pandas as pd 
from basepairmodels.common.stats import get_recommended_counts_loss_weight


def test_get_recommended_counts_loss_weight():
    
    # list of bigwigs - source of the training data
    # bw1 has value 1.0 from pos 0 to 9999 on each chromosome
    # bw2 has value 2.0 from pos 10000 to 49999 on each chromosome
    # bw3 has value 3.0 from pos 50000 to 139999 on each chromosome
    bigwigs = ['tests/common/test_data/b1.bw', 
               'tests/common/test_data/b2.bw', 
               'tests/common/test_data/b3.bw']
    
    # list of training data points (e.g. identified peaks)
    intervals = [['chr1', 0, 1000], 
                 ['chr2', 1000, 2000], 
                 ['chr3', 2000, 3000],
                 ['chr4', 3000, 4000],
                 ['chr5', 4000, 5000],
                 ['chr6', 5000, 6000],
                 ['chr7', 6000, 7000],
                 ['chr8', 7000, 8000],
                 ['chr9', 8000, 9000],
                 ['chr10', 9000, 10000],
                 ['chr11', 12000, 13000],
                 ['chr12', 18000, 19000],
                 ['chr13', 24000, 25000],
                 ['chr14', 28000, 29000],
                 ['chr15', 32000, 33000],
                 ['chr16', 39000, 40000],
                 ['chr17', 52000, 53000],
                 ['chr18', 62000, 63000],
                 ['chr19', 72000, 73000],
                 ['chr20', 82000, 83000],
                 ['chr21', 92000, 93000],
                 ['chr22', 102000, 103000],
                 ['chrX', 35000, 36000],
                 ['chrY', 112000, 113000]]
    
    # convert the list of intervals to a pandas dataframe
    intervals_df = pd.DataFrame(intervals, columns=['chrom', 'start', 'end'])  
    
    # alpha value to scale profile loss relative to the counts loss
    alpha = 1.0
    
    # compute the counts loss weight
    counts_loss_weight = get_recommended_counts_loss_weight(
        bigwigs, intervals_df, alpha)
    
    assert counts_loss_weight == ((1.0 / 2.0) * 625.0)
