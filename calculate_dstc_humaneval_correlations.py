import json
import numpy as np
import pandas as pd
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bleu_metric import BleuMetric
from scipy.stats import spearmanr, kendalltau

def cleansed_bleu(predictions, labels):
    # Calculate BLEU-4 on the predictions using sacrebleu
    bleu_metric = BleuMetric()
    bleu_score = bleu_metric.evaluate_batch(predictions, labels)['bleu']
    return bleu_score

def cleansed_meteor(predictions, labels):
    # Calculate METEOR on the prediction
    met_metric = MeteorMetric()
    met_score = met_metric.evaluate_batch(predictions, labels)['meteor']
    return met_score

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_ratings(data, metric):
    ratings = [entry[metric] for entry in data if entry is not None]
    return np.array(ratings)

def extract_system_ratings(human_eval_paths, metric, llama_judge=False, labels_file=None):
    # Load the labels
    if labels_file is not None:
        labels = load_data(labels_file)
        # Filter the labels
    if len(human_eval_paths) == 1:
        # Sample-level
        human_evals = load_data(human_eval_paths[0])
    else:
        # System-level
        human_evals = [load_data(json_file) for json_file in human_eval_paths]

    system_ratings = []

    for (sys_id, prediction_scores) in enumerate(human_evals):
        pred_path = human_eval_paths[sys_id].replace(".human_eval.json", ".json")
        predictions = []
        for i, x in enumerate(load_data(pred_path)):
            if x["target"] and labels[i]["target"]:
                predictions.append(x["response"])

        labels_text_only = [x["response"] for x in labels if x["target"]]

        llama_path = human_eval_paths[sys_id].replace(".human_eval.json", "_likert_acc_app_metrics.json")
        llama_file = load_data(llama_path)
        score_path = human_eval_paths[sys_id].replace(".human_eval.json", ".score.json")
        score_file = load_data(score_path)
        ratings = []
        cl_bleu = cleansed_bleu(predictions, labels_text_only)
        cl_meteor = cleansed_meteor(predictions, labels_text_only)

        for i, entry in enumerate(prediction_scores):
            if entry is not None:
                final_entry = entry[metric]
                if llama_judge and len(human_evals) > 1:
                    final_entry.append(float(cl_meteor))
                ratings.append(final_entry)

        # There are ... ratings
        print("System {} has {} ratings and {} predictions".format(sys_id, len(ratings), len(predictions)))
        system_ratings.append(np.mean(np.array(ratings),axis=0))

    system_ratings = np.stack(system_ratings, axis=0)

    # This is the average of the 3 columns
    system_ratings = np.concatenate((np.mean(system_ratings[:, :3], axis=1, keepdims=True), system_ratings[:, 3:]), axis=1)
    
    return system_ratings

def calculate_correlations(ratings, n):
    spearman_corr = spearmanr(ratings)
    kendall_corr = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = kendalltau(ratings[:, i], ratings[:, j])
            kendall_corr.append(corr)
    return spearman_corr, kendall_corr

def print_table(header, data):
    print(header)
    print("-" * len(header))
    for row in data:
        print("{:<20} {:<20} {:<20}".format(*row))

def main(human_evals, labels_file=None):
    n = 2
    llama_judge = True

    accuracy_ratings = extract_system_ratings(human_evals, 'accuracy', llama_judge=llama_judge, labels_file=labels_file)
    appropriateness_ratings = extract_system_ratings(human_evals, 'appropriateness', llama_judge=llama_judge, labels_file=labels_file)

    accuracy_spearman, accuracy_kendall = calculate_correlations(accuracy_ratings, n)
    appropriateness_spearman, appropriateness_kendall = calculate_correlations(appropriateness_ratings, n)


    #print(accuracy_kendall)
    #print(accuracy_spearman)
    #print(appropriateness_kendall)
    #print(appropriateness_spearman)

    if llama_judge:
        cols = ["Human Judges", "METEOR"]

    print("Accuracy Correlations")
    # Create a DataFrame from the correlation matrix
    df = pd.DataFrame(accuracy_spearman[0], columns=cols, index=cols)

    print(df)

    print("\nAppropriateness Correlations")
    df = pd.DataFrame(appropriateness_spearman[0], columns=cols, index=cols)

    print(df)

if __name__ == "__main__":
    import glob

    # Search for files ending with "human_eval.json" in all subfolders
    file_list = glob.glob("results/results_dstc9/*/*human_eval.json", recursive=True)
    # Add the baseline
    # file_list.append("results/results_dstc9/baseline/human_eval.json")
    print(file_list)
    main(file_list, labels_file="data/dstc9/test/labels.json")
