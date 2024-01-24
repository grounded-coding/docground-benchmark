import json
import numpy as np
from utils.file_processing import load_data
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import krippendorff
import matplotlib.pyplot as plt


class HumanEvalCollector(ABC):

    def __init__(self):
        pass

    def extract_ratings_for_sample_indices(self, model=""):
        pass

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=None, model=""):
        return sample_indices, model_responses

    def get_index_sets_disjunctive(self, sample_indices, human_ratings, human_dim):
        sets = [[index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim] <= 0.3],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim] <= 0.7 and human_ratings[i][human_dim] > 0.3],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim] > 0.7]]
        return sets


class DummyEvalCollector(HumanEvalCollector):    

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=None, model=""):
        human_scores = []
        for i in range(len(sample_indices)):
            human_scores.append({})
            for dim in human_dims:
                human_scores[i][dim] = np.random.rand()
        return human_scores


class DialDocEvalCollector(HumanEvalCollector):

    def __init__(self, human_eval_path):
        super().__init__()
        self.human_eval = pd.read_csv(human_eval_path, sep=',')

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=-1, model=""):
        human_evals = self.human_eval
        sample_indices_f = []
        model_responses_f = []

        # model can be ignored here because evaluations are available for all models
        seen = []
        for i, sample_index in enumerate(sample_indices):
            if human_evals.iloc[sample_index] is not None:
                try:
                    sc = int(human_evals.iloc[sample_index]["Answer.match_ref"])
                    if (human_evals.iloc[sample_index]["Input.cond_sys"], human_evals.iloc[sample_index]["Input.ex_id"]) not in seen and sc != exclude_rating:
                        seen.append((human_evals.iloc[sample_index]["Input.cond_sys"], human_evals.iloc[sample_index]["Input.ex_id"]))
                        sample_indices_f.append(sample_index)
                        if model_responses is not None:
                            model_responses_f.append(model_responses[i])
                except:
                    pass
            else:
                pass
        return sample_indices_f, model_responses_f

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["appropriateness", "groundedness"], model=""):
        """
        Important note: If only_model is set, the human ratings are only extracted for the specified model.
        This means that INDICES DO NOT CORRESPOND TO THE SAMPLE INDICES ANYMORE!
        Should ONLY be used for aggregations, averages etc. where the indices are not important.
        """
        dim_map = {"groundedness": "Answer.match_ref",
                   "appropriateness": "Answer.score_appropriateness"}
        # This numeric map simply leaves "contradict": 0 at 0 and everything else at 1
        contra_map = lambda x: 1 if x != 0 else 0
        # This leaves everything unchanged
        identity_map = lambda x: x
        numeric_map = identity_map

        ratings = []
        human_evals = self.human_eval
        
        for sample_index in sample_indices:
            if human_evals.iloc[sample_index] is not None:
                rating = {}
                for dim in human_dims:
                    mapped_dim = dim_map[dim]
                    try:
                        selected_rows = human_evals.loc[(human_evals['Input.ex_id'] == human_evals.iloc[sample_index]['Input.ex_id']) &
                            (human_evals['Input.cond_sys'] == model)]
                        # Aggregate the ratings
                        if dim == "groundedness":
                            selected_rows.loc[:, mapped_dim] = selected_rows.loc[:, mapped_dim].apply(numeric_map)
                        avg_rating = selected_rows[mapped_dim].mean()
                        rating[dim] = avg_rating
                    except:
                        rating[dim] = None
                ratings.append(rating)
        return ratings
    

class BEGINHumanEvalCollector(HumanEvalCollector):
    def __init__(self, human_eval_path):
        super().__init__()
        self.human_eval = pd.read_csv(human_eval_path, sep='\t')

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["attributability"], numeric=True, model="all"):
        ratings = []
        human_evals = self.human_eval

        if model != "all":
            raise NotImplementedError("Model specific human evaluation not implemented for BEGIN yet")

        for sample_index in sample_indices:
            if human_evals.iloc[sample_index] is not None:
                rating = {}
                rating["attributability"] = human_evals.iloc[sample_index]["begin_label"]
                ratings.append(rating)
            else:
                raise ValueError("No human ratings for sample {}".format(sample_index))

        if numeric:
            for rating in ratings:
                if rating["attributability"] == "Not fully attributable":
                    rating["attributability"] = 0
                elif rating["attributability"] == "Generic":
                    rating["attributability"] = 1
                elif rating["attributability"] == "Fully attributable":
                    rating["attributability"] = 2
                else:
                    raise ValueError("Attributability value not recognized")
        return ratings

    def get_index_sets_disjunctive(self, sample_indices, human_ratings, human_dim):
        sets = [[index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim]  == 0],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim]  == 1],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim]  == 2]]

        return sets

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=None, model="all"):
        if model != "all":
            raise NotImplementedError("Model specific human evaluation not implemented for BEGIN yet")

        if exclude_rating is None:
            return sample_indices, model_responses
        else:
            sample_indices_f = []
            model_responses_f = []
            human_evals = self.human_eval
            for i, sample_index in enumerate(sample_indices):
                if human_evals.iloc[sample_index] is not None:
                    rating = human_evals.iloc[sample_index]["begin_label"]
                    if rating != exclude_rating:
                        sample_indices_f.append(sample_index)
                        if model_responses is not None:
                            model_responses_f.append(model_responses[i])
                else:
                    raise ValueError("No human ratings for sample {}".format(sample_index))
            return sample_indices_f, model_responses_f


class DSTCHumanEvalCollector(HumanEvalCollector):
    def __init__(self, human_eval_paths):
        super().__init__()
        data_human_map = {}
        for model in human_eval_paths:
            human_eval_path = human_eval_paths[model]
            data_human_map[model] = load_data(human_eval_path)
        self.data_human_map = data_human_map

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["accuracy", "appropriateness"], model="", print_alpha=True):
        """
        Extracts human ratings from the DSTC human evaluation file
        :param sample_indices: The indices of the samples for which the ratings should be extracted, should never be NONE
        :param human_eval_path: Path to the human evaluation file, a list of dicts, can be NONE
        :param dims: The dimensions for which the ratings should be extracted
        :return: A list of dicts containing the ratings for each sample
        """
        ratings = []
        dimension_ratings = {dim: [] for dim in human_dims}
        human_evals = self.data_human_map[model]
        dimension_sample_ratings = {dim: [] for dim in human_dims}  # To store average ratings for each sample and dimension

        for sample_index in sample_indices:
            rating = {}
            if human_evals[sample_index] is not None:
                for dim in human_dims:
                    # the human file contains a list of 3 ratings, we take the average
                    rating[dim] = np.mean(human_evals[sample_index][dim])
                    dimension_ratings[dim].append(human_evals[sample_index][dim])
                    avg_rating = np.mean(human_evals[sample_index][dim])
                    dimension_sample_ratings[dim].append(avg_rating)
                ratings.append(rating)
            else:
                raise ValueError("No human ratings for sample {}".format(sample_index))
            
        if print_alpha:
            for dim in human_dims:
                alpha = krippendorff.alpha(value_counts=np.array(dimension_ratings[dim]), level_of_measurement="ordinal")
                print(f"Krippendorff's Alpha for '{dim}': {alpha}")
        
        # Plotting
        # fig, axes = plt.subplots(1, len(human_dims), figsize=(len(human_dims)*5, 5))
        # if len(human_dims) == 1:  # If there's only one dimension, axes won't be an array
        #     axes = [axes]
        # for ax, dim in zip(axes, human_dims):
        #     ax.scatter(sample_indices, dimension_sample_ratings[dim])
        #     ax.set_title(f'Average Ratings for {dim}')
        #     ax.set_xlabel('Sample Index')
        #     ax.set_ylabel('Average Rating')
        #     ax.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'{model}_ratings.png')

        return ratings

    def get_subset_with_human_eval(self, sample_indices, candidate_responses=None, exclude_rating=None, model="baseline"):
        """
        Provided with a list of indices that contain responses for the chosen dataset, this function returns a subset of indices that have human ratings"""
        human_eval_indices = []
        model_responses = []
        human_evals = self.data_human_map[model]
        if exclude_rating is not None:
            raise NotImplementedError("Excluding ratings is not implemented for DSTC yet")
        for j, index in enumerate(sample_indices):
            if human_evals[index] is not None:
                human_eval_indices.append(index)
                if candidate_responses is not None:
                    model_responses.append(candidate_responses[j])
        return human_eval_indices, model_responses

    def get_index_sets_disjunctive(self, sample_indices, human_ratings, human_dim):
        sets = [[index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim]  <= 2],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim]  == 3],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim]  >= 4]]
        return sets


class TopicalChatEvalCollector(HumanEvalCollector):
    def __init__(self, human_eval_path):
        super().__init__()
        self.human_eval = load_data(human_eval_path)
        
    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["understandability", "naturalness", "coherence", "engagingness", "groundedness", "overall"], model=""):
        ratings = []
        human_evals = self.human_eval
        
        for sample_index in sample_indices:
            rating = {}
            for dim in human_dims:
                rating[dim] = float(human_evals[str(sample_index)][model]["scores"][dim])
            ratings.append(rating)

        return ratings
    