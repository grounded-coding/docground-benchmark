import json
import matplotlib.pyplot as plt
from src.data_collector import DataCollector, BEGINDataCollector, DSTCDataCollector
from src.eval_collector import BEGINHumanEvalCollector, DSTCHumanEvalCollector

# Replace this with the path to your evaluation file
evaluation_file = "outputs/dstc9/test/baseline/UniEval.json"

# Replace these with your disjunctive sets of indices
begin = DSTCDataCollector(dataset="../dstc11-track5/data/dstc9", dataset_split="test", dataset_name="dstc9")
sample_indices = begin.get_samples_with_target()
begin_eval_collector = DSTCHumanEvalCollector(human_eval_path="../dstc11-track5/results/results_dstc9/baseline/entry0.human_eval.json")
sample_indices, _ = begin_eval_collector.get_subset_with_human_eval(sample_indices)
human_ratings = begin_eval_collector.extract_ratings(sample_indices)

# Using human_ratings (accuracy) which ranges from 1 to 5, we can create 3 disjunctive sets of indices
index_sets = [[i for i, rating in enumerate(human_ratings) if rating["accuracy"] <= 2],
                [i for i, rating in enumerate(human_ratings) if rating["accuracy"] == 3],
                [i for i, rating in enumerate(human_ratings) if rating["accuracy"] >= 4]]

# Load the evaluation data
with open(evaluation_file, "r") as f:
    data = json.load(f)

# Extract the "accur" scores for each index set
accur_scores = [[data[i]["groundedness"] for i in index_set] for index_set in index_sets]

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
plt.savefig("outputs/dstc9/test/baseline/boxplot.png", dpi=300)