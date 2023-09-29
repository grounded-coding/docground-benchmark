import json
import matplotlib.pyplot as plt
from src.data_collector import DataCollector, BEGINDataCollector
from src.eval_collector import BEGINHumanEvalCollector

# Replace this with the path to your evaluation file
evaluation_file = "outputs/tc/dev/baseline/UniEval.json"

# Replace these with your disjunctive sets of indices
begin = BEGINDataCollector(dataset_path="../BEGIN-dataset/topicalchat", dataset_split="dev", dataset_name="tc")
sample_indices = begin.get_samples_with_target()
begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/topicalchat/begin_dev_tc.tsv")
human_ratings = begin_eval_collector.extract_ratings(sample_indices)

# Using human_ratings (which range from 0 to 2), divide sample_indices into three sets
index_sets = [[], [], []]
for i, rating in enumerate(human_ratings):
    index_sets[rating["attributability"]].append(i)

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
plt.savefig("outputs/tc/dev/baseline/boxplot.png", dpi=300)