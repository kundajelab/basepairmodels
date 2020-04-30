import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import deepdish
import modisco.visualization
from modisco.visualization import viz_sequence
import h5py
import numpy as np
import modisco
import modisco.backend
import modisco.nearest_neighbors
import modisco.affinitymat
import modisco.tfmodisco_workflow.seqlets_to_patterns
import modisco.tfmodisco_workflow.workflow
import modisco.aggregator
import modisco.cluster
import modisco.core
import modisco.coordproducers
import modisco.metaclusterers
import modisco.util
import argparse

from modisco.tfmodisco_workflow.seqlets_to_patterns import TfModiscoSeqletsToPatternsFactory
from modisco.tfmodisco_workflow.workflow import TfModiscoWorkflow
from modisco.visualization import viz_sequence

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for running tf-modisco')
    parser.add_argument("-d","--path_to_imp_scores_dir", type=str, help="Path to the importance scores")
    parser.add_argument("-p","--profile_or_counts",type=str,help="scoring method to use, profile or counts scores")
    parser.add_argument("-save","--save_path",type=str,default=None,help="Enter a save directory. If not provided, it is saved in the logdir")
    args = parser.parse_args()
    return args


args = parse_args()
logdir_path = args.path_to_imp_scores_dir


if args.save_path:
    base_path = args.save_path
    print("A path to save has been provided. It is {}".format(base_path))
else:
    base_path  = logdir_path    
    print("No save path provided. Results will be saved at {}".format(base_path))


assert(os.path.exists(logdir_path))
print("The path to the importance scores is {}".format(logdir_path))
scoring_type = args.profile_or_counts
if scoring_type=='profile':
    scores_path = os.path.join(logdir_path,'profile_scores.h5')
    print(" Scores path is {}".format(scores_path))
elif scoring_type=='counts':
    scores_path  = os.path.join(logdir_path,'counts_scores.h5')
    print(" Scores path is {}".format(scores_path))

else:
    print("Enter a valid scoring type: counts or profile")
    
assert(os.path.exists(scores_path))

if scoring_type=='profile':
    save_path = os.path.join(base_path,'modisco_results_allChroms_profile.hdf5')
    seqlet_path = os.path.join(base_path,'seqlets_profile.txt')
elif scoring_type=='counts':
    save_path  = os.path.join(base_path,'modisco_results_allChroms_counts.hdf5')
    seqlet_path = os.path.join(base_path,'seqlets_counts.txt')

##Load the scores
scores = deepdish.io.load(scores_path)
shap_scores_seq = []
proj_shap_scores_seq = []
one_hot_seqs = [] 

center = int(scores['shap']['seq'].shape[-1]/2)
start = center - 200
end = center + 200
for i in scores['shap']['seq']:
    shap_scores_seq.append(i[:,start:end].transpose())


for i in scores['projected_shap']['seq']:
    proj_shap_scores_seq.append(i[:,start:end].transpose())

for i in scores['raw']['seq']:
    one_hot_seqs.append(i[:,start:end].transpose())

tasks = ['task0']
task_to_scores = OrderedDict()
task_to_hyp_scores = OrderedDict()

onehot_data = one_hot_seqs
task_to_scores['task0']  = proj_shap_scores_seq
task_to_hyp_scores['task0']  = shap_scores_seq

# track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
#     task_names=["task0"], 
#     contrib_scores=task_to_scores, 
#     hypothetical_contribs=task_to_hyp_scores, 
#     one_hot=onehot_data)


tfmodisco_patterns_factory = TfModiscoSeqletsToPatternsFactory(
    trim_to_window_size=20, initial_flank_to_add=5, kmer_len=8, num_gaps=1, 
    num_mismatches=0, final_min_cluster_size=20)

tfmodisco_workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
    #Slight modifications from the default settings
    sliding_window_size=20, flank_size=5, target_seqlet_fdr=0.05, 
    max_seqlets_per_metacluster=20000, 
    seqlets_to_patterns_factory=tfmodisco_patterns_factory)

tfmodisco_results = tfmodisco_workflow(task_names=["task0"], 
                                       contrib_scores=task_to_scores, 
                                       hypothetical_contribs=task_to_hyp_scores, 
                                       one_hot=onehot_data)

if os.path.exists(save_path):
    print("Saved file already exists. Removing it")
    os.remove(save_path)

grp = h5py.File(save_path)
tfmodisco_results.save_hdf5(grp)
print("Saved modisco results to file {}".format(str(save_path)))

print("Saving seqlets to %s" % seqlet_path)
seqlets = \
    tfmodisco_results.metacluster_idx_to_submetacluster_results[0].seqlets
bases = np.array(["A", "C", "G", "T"])
with open(seqlet_path, "w") as f:
    for seqlet in seqlets:
        sequence = "".join(
            bases[np.argmax(seqlet["sequence"].fwd, axis=-1)]
        )
        example_index = seqlet.coor.example_idx
        start, end = seqlet.coor.start, seqlet.coor.end
        f.write(">example%d:%d-%d\n" % (example_index, start, end))
        f.write(sequence + "\n")


print("Saving pattern visualizations")

def save_plot(weights, dst_fname):
    """
    
    """
    print(dst_fname)
    colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
    plot_funcs = {0: viz_sequence.plot_a, 1: viz_sequence.plot_c, 
                  2: viz_sequence.plot_g, 3: viz_sequence.plot_t}

    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111) 
    viz_sequence.plot_weights_given_ax(ax=ax, array=weights, 
                                       height_padding_factor=0.2,
                                       length_padding=1.0, 
                                       subticks_frequency=1.0, 
                                       colors=colors, plot_funcs=plot_funcs, 
                                       highlight={}, ylabel="")

    plt.savefig(dst_fname)
    
patterns = (tfmodisco_results
            .metacluster_idx_to_submetacluster_results[0]
            .seqlets_to_patterns_result.patterns)

for idx,pattern in enumerate(patterns):
    print(pattern)
    print("pattern idx",idx)
    print(len(pattern.seqlets))
    save_plot(pattern["task0_contrib_scores"].fwd, 
              '{}/contrib_{}.png'.format(base_path, idx))
    save_plot(pattern["sequence"].fwd,
              '{}/sequence_{}.png'.format(base_path, idx))
