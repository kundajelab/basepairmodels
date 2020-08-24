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

from modisco.tfmodisco_workflow.seqlets_to_patterns import TfModiscoSeqletsToPatternsFactory
from modisco.tfmodisco_workflow.workflow import TfModiscoWorkflow
from modisco.visualization import viz_sequence
from basepairmodels.cli.argparser import modisco_argsparser


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


def modisco_main():
    parser = modisco_argsparser()
    args = parser.parse_args()

    if not os.path.exists(args.scores_path):
        raise quietexception.QuietException(
            "Score file {} does not exist".format(args.scores_path))
        
    if not os.path.exists(args.scores_locations):
        raise quietexception.QuietException(
            "Scores locations file {} does not exist".format(
                args.scores_locations))
        
    if not os.path.exists(args.output_directory):
        raise quietexception.QuietException(
            "Output directiry {} does not exist".format(args.output_directory))

    # Load the scores
    scores = deepdish.io.load(args.scores_path)
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

    if args.suffix_tag is not None:
        modisco_results_path = '{}/modisco_results_{}.hd5'.format(
            args.output_dir, args.suffix_tag)
    else:
        modisco_results_path = '{}/modisco_results.hd5'.format(args.output_dir)
        
    tfmodisco_results.save_hdf5(h5py.File(modisco_results_path))
    print("Saved modisco results to file {}".format(str(modisco_results_path)))

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



    patterns = (tfmodisco_results
                .metacluster_idx_to_submetacluster_results[0]
                .seqlets_to_patterns_result.patterns)

    # generate .pngs of each motif and write motif seqlet to
    # individual files
    for idx,pattern in enumerate(patterns):
        print(pattern)
        print("pattern idx",idx)
        print(len(pattern.seqlets))
        
        pattern_seqlet_path = os.path.join(base_path,
                                           'pattern{}_seqlets.txt'.format(idx))
        with open(pattern_seqlet_path, "w") as f: 
            for seqlet in pattern.seqlets:
                sequence = "".join(
                    bases[np.argmax(seqlet["sequence"].fwd, axis=-1)]
                )
                example_index = seqlet.coor.example_idx
                start, end = seqlet.coor.start, seqlet.coor.end
                f.write(">example%d:%d-%d\n" % (example_index, start, end))
                f.write(sequence + "\n")

        save_plot(pattern["task0_contrib_scores"].fwd, 
                  '{}/contrib_{}.png'.format(base_path, idx))
        save_plot(pattern["sequence"].fwd,
                  '{}/sequence_{}.png'.format(base_path, idx))


if __name__ == '__main__':
    modisco_main()
