import glob
import numpy as np
from src.eval_collector import DSTCHumanEvalCollector


# Search for files ending with "human_eval.json" in all subfolders
human_eval_paths = glob.glob("../dstc11-track5/results/results_dstc9/*/*human_eval.json", recursive=True)
print(human_eval_paths)

system_ratings = []
sample_indices = [7]
for (sys_id, human_path) in enumerate(human_eval_paths):
    dstc_collector = DSTCHumanEvalCollector(human_path)
    human_rating = dstc_collector.extract_ratings(sample_indices, human_dims=["accuracy", "appropriateness"])
    # system_ratings should look like this
    # [ [0.7, 0.6], [0.8, 0.9], ... ]
    # where each entry is the average of the human ratings for each dimension
    
    # First, filter out all None entries
    human_ratings = [rating for rating in human_rating if rating is not None]
    # Second, average the ratings for each dimension

    numeric_ratings = []
    for rating in human_ratings:
        numeric_entry = []
        for dim in rating:
            numeric_entry.append(np.mean(rating[dim]))
        numeric_ratings.append(numeric_entry)
    
    system_ratings.append(numeric_ratings)

system_ratings = np.stack(system_ratings, axis=0)

print(system_ratings)