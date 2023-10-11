import json
import numpy as np
from utils.file_processing import load_data
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path


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

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=None, only_model=""):
        # create a list with dictionaries of length sample_indices where each dictionary contains random ratings for each dimension
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

    # TODO Align this with the method below regarding model extraction
    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=-1, model=""):
        human_evals = self.human_eval
        sample_indices_f = []
        model_responses_f = []
        seen = []
        for i, sample_index in enumerate(sample_indices):
            # isinstance(human_evals.iloc[sample_index]["Answer.match_ref"] must be castable to int without error
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

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["appropriateness", "groundedness"], only_model=""):
        """
        Important note: If only_model is set, the human ratings are only extracted for the specified model.
        This means that INDICES DO NOT CORRESPOND TO THE SAMPLE INDICES ANYMORE!
        Should ONLY be used for aggregations, averages etc. where the indices are not important.
        """
        ratings = []
        human_evals = self.human_eval
        
        for sample_index in sample_indices:
            if human_evals.iloc[sample_index] is not None:
                rating = {}
                for dim in human_dims:
                    mapped_dim = "Answer.match_ref" if dim == "groundedness" else "Answer.score_appropriateness"
                    try:
                        selected_rows = human_evals.loc[(human_evals['Input.ex_id'] == human_evals.iloc[sample_index]['Input.ex_id']) &
                            (human_evals['Input.cond_sys'] == only_model)]
                        avg_rating = selected_rows[mapped_dim].mean()
                        rating[dim] = avg_rating
                    except:
                        rating[dim] = None
                ratings.append(rating)
        return ratings
    

class BEGINHumanEvalCollector(HumanEvalCollector):
    # TODO Fix only_model sample index subset like for DialDoc
    def __init__(self, human_eval_path):
        super().__init__()
        # read from human eval path the file wh ich is a tsv file with columns_ id, model_name, data_source, knowledge_message, response, begin_label
        self.human_eval = pd.read_csv(human_eval_path, sep='\t')

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["attributability"], numeric=True, only_model=""):
        ratings = []
        human_evals = self.human_eval

        for sample_index in sample_indices:
            if human_evals.iloc[sample_index] is not None:
                rating = {}
                # get the entry in column begin_label
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

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=None, model=""):
        if exclude_rating is None:
            return sample_indices, model_responses

        else:
            # Only retrieve samples that do not have rating["attributability"] set to the value specified in exclude_rating
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

    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["accuracy", "appropriateness"], only_model=""):
        """
        Extracts human ratings from the DSTC human evaluation file
        :param sample_indices: The indices of the samples for which the ratings should be extracted, should never be NONE
        :param human_eval_path: Path to the human evaluation file, a list of dicts, can be NONE
        :param dims: The dimensions for which the ratings should be extracted
        :return: A list of dicts containing the ratings for each sample
        """
        ratings = []
        valid_rating = 0

        human_evals = self.data_human_map[only_model]

        for sample_index in sample_indices:
            if human_evals[sample_index] is not None:
                rating = {}
                for dim in human_dims:
                    # the human file contains a list of 3 ratings, we take the average
                    rating[dim] = human_evals[sample_index][dim]
                    rating[dim] = np.mean(rating[dim])
                ratings.append(rating)
                valid_rating += 1
            else:
                # Warn the user that there is no human rating for this sample and skip it
                pass
                # print("No human rating for sample {} - ONLY PROCEED FOR SYSTEM CORRELATIONS".format(sample_index))

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
        
    def extract_ratings_for_sample_indices(self, sample_indices, human_dims=["understandability", "naturalness", "coherence", "engagingness", "groundedness", "overall"], only_model=""):
        ratings = []
        human_evals = self.human_eval
        
        for sample_index in sample_indices:
            rating = {}
            for dim in human_dims:
                rating[dim] = float(human_evals[str(sample_index)][only_model]["scores"][dim])
            ratings.append(rating)

        return ratings
    