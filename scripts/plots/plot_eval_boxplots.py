import json

import sys
sys.path.append('/home/nhilgers/setups/DocGroundEval')

import matplotlib.pyplot as plt
import numpy as np
from src.data_collector import DataCollector, BEGINDataCollector, DSTCDataCollector
from src.eval_collector import BEGINHumanEvalCollector, DSTCHumanEvalCollector

framework_evals_path = "dstc9/test/baseline"
human_dim = "accuracy"
framework_dim = "accurate"
framework_name = "LLEval"
begin_set = False

# Replace this with the path to your evaluation file
evaluation_file = f"outputs/{framework_evals_path}/{framework_name}.json"

if begin_set:
    begin = BEGINDataCollector(dataset_path="../BEGIN-dataset/topicalchat", dataset_split="dev", dataset_name="tc")
    sample_indices = begin.get_samples_with_target()
    begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/topicalchat/begin_dev_tc.tsv")
    sample_indices, _ = begin_eval_collector.get_subset_with_human_eval(sample_indices, None)
    human_ratings = begin_eval_collector.extract_ratings_for_sample_indices(sample_indices)
    index_sets = begin_eval_collector.get_index_sets_disjunctive(sample_indices, human_ratings, human_dim)

else:
    dstc = DSTCDataCollector(dataset_path="../dstc11-track5/data/dstc9", dataset_split="test", dataset_name="dstc9")
    dstc_eval_coll = DSTCHumanEvalCollector(human_eval_path="../dstc11-track5/results/results_dstc9/baseline/entry0.human_eval.json")
    sample_indices = dstc.get_samples_with_target()
    sample_indices, _ = dstc_eval_coll.get_subset_with_human_eval(sample_indices, None)
    human_ratings = dstc_eval_coll.extract_ratings_for_sample_indices(sample_indices)
    index_sets = dstc_eval_coll.get_index_sets_disjunctive(sample_indices, human_ratings, human_dim)

# Load the evaluation data
with open(evaluation_file, "r") as f:
    eval_data = json.load(f)

# Extract the "accur" scores for each index set. For this we need to find the entry in eval_data that has response_index value equal to the index in the index set
accur_scores = []
for index_set in index_sets:
    accur_set = []
    for index in index_set:
        for entry in eval_data:
            if entry["response_index"] == index:
                accur_set.append(entry[framework_dim])
    accur_scores.append(accur_set)

# Create the boxplots
fig1, axes1 = plt.subplots(1, len(index_sets), sharey=True)
colors = ['blue', 'green', 'red']

for i, (accur_set, ax, color) in enumerate(zip(accur_scores, axes1, colors)):
    ax.boxplot(accur_set, patch_artist=True, boxprops=dict(facecolor=color))
    ax.set_title(f"Index Set {i + 1}")
    if i == 0:
        ax.set_ylabel("Assigned Score")
    ax.set_xticks([])

# save the boxplot
plt.savefig(f"outputs/{framework_evals_path}/{framework_name}_boxplot.png", dpi=300)
plt.clf()

# Create the histograms

for scores, name in (([score for accur_set in accur_scores for score in accur_set], framework_name), ([score[human_dim] for score in human_ratings], "Human")):

    fig2, ax = plt.subplots(1, 1)
    bins = np.linspace(0, max(scores) if len(scores) > 0 else 1, 6)
    ax.hist(scores, bins=bins, edgecolor='black', color=color, alpha=0.7)
    ax.set_title(f"Full Set")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")

    # save the histogram
    plt.savefig(f"outputs/{framework_evals_path}/{name}_histogram.png", dpi=300)