import json
import matplotlib.pyplot as plt
from src.data_collector import DataCollector, BEGINDataCollector, DSTCDataCollector
from src.eval_collector import BEGINHumanEvalCollector, DSTCHumanEvalCollector

human_dim = "attributability"
framework_dim = "accurate"

# Replace this with the path to your evaluation file
evaluation_file = "outputs/tc/dev/baseline/LLEval.json"

# Replace these with your disjunctive sets of indices
begin = BEGINDataCollector(dataset_path="../BEGIN-dataset/topicalchat", dataset_split="dev", dataset_name="tc")
sample_indices = begin.get_samples_with_target()
begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/topicalchat/begin_dev_tc.tsv")
sample_indices, _ = begin_eval_collector.get_subset_with_human_eval(sample_indices, None)
human_ratings = begin_eval_collector.extract_ratings(sample_indices)

# Using human_ratings we can create 3 disjunctive sets of indices
index_sets = begin_eval_collector.get_index_sets_disjunctive(sample_indices, human_ratings, human_dim)

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
fig, axes = plt.subplots(1, len(index_sets), sharey=True)
colors = ['blue', 'green', 'red']

for i, (accur_set, ax, color) in enumerate(zip(accur_scores, axes, colors)):
    ax.boxplot(accur_set, patch_artist=True, boxprops=dict(facecolor=color))
    ax.set_title(f"Index Set {i + 1}")
    ax.set_ylabel("Assigned Score")
    ax.set_xticks([])

# save the plot
# high dpi
plt.savefig("outputs/tc/dev/baseline/lleval_boxplot.png", dpi=300)