import json
import numpy as np
from utils.file_processing import load_data
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path


class HumanEvalCollector(ABC):

    def __init__(self):
        pass

    def extract_ratings(self):
        pass

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=None):
        return sample_indices, model_responses

    def get_index_sets_disjunctive(self, sample_indices, human_ratings, human_dim):
        sets = [[index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim] <= 0.3],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim] <= 0.7 and human_ratings[i][human_dim] > 0.3],
                [index for i, index in enumerate(sample_indices) if human_ratings[i][human_dim] > 0.7]]
        return sets


class DummyEvalCollector(HumanEvalCollector):    

    def extract_ratings(self, sample_indices, human_dims=None):
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

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=-1):
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
                    print("No human ratings for sample {}".format(sample_index))
            else:
                print("No human ratings for sample {}".format(sample_index))
        return sample_indices_f, model_responses_f

    def extract_ratings(self, sample_indices, human_dims=["appropriateness", "groundedness"]):
        ratings = []
        human_evals = self.human_eval
        
        for sample_index in sample_indices:
            if human_evals.iloc[sample_index] is not None:
                rating = {}
                for dim in human_dims:
                    if dim == "appropriateness":
                        rating[dim] = human_evals.loc[(human_evals["Input.cond_sys"] == human_evals.iloc[sample_index]["Input.cond_sys"]) &\
                                                       (human_evals["Input.ex_id"] == human_evals.iloc[sample_index]["Input.ex_id"])]["Answer.score_appropriateness"].mean()
                    else:
                        # TODO Implement a voting strategy for groundedness
                        # For the moment, just take the first one
                        rating[dim] = human_evals.loc[(human_evals["Input.cond_sys"] == human_evals.iloc[sample_index]["Input.cond_sys"]) &\
                                                       (human_evals["Input.ex_id"] == human_evals.iloc[sample_index]["Input.ex_id"])]["Answer.match_ref"].iloc[0]
                ratings.append(rating)
            else:
                raise ValueError("No human ratings for sample {}".format(sample_index))
        return ratings
    

class BEGINHumanEvalCollector(HumanEvalCollector):
    def __init__(self, human_eval_path):
        super().__init__()
        # read from human eval path the file wh ich is a tsv file with columns_ id, model_name, data_source, knowledge_message, response, begin_label
        self.human_eval = pd.read_csv(human_eval_path, sep='\t')

    def extract_ratings(self, sample_indices, human_dims=["attributability"], numeric=True):
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

    def get_subset_with_human_eval(self, sample_indices, model_responses, exclude_rating=None):
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
    def __init__(self, human_eval_path):
        super().__init__()
        self.human_evals = load_data(human_eval_path)

    def extract_ratings(self, sample_indices, human_dims=["accuracy", "appropriateness"]):
        """
        Extracts human ratings from the DSTC human evaluation file
        :param sample_indices: The indices of the samples for which the ratings should be extracted, should never be NONE
        :param human_eval_path: Path to the human evaluation file, a list of dicts, can be NONE
        :param dims: The dimensions for which the ratings should be extracted
        :return: A list of dicts containing the ratings for each sample
        """
        ratings = []
        valid_rating = 0
        human_evals = self.human_evals

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
                raise ValueError("No human ratings for sample {}".format(sample_index))

        # print("Candidate has received {} valid ratings".format(valid_rating))            
        return ratings

    def get_subset_with_human_eval(self, sample_indices, candidate_responses=None, exclude_rating=None):
        """
        Provided with a list of indices that contain responses for the chosen dataset, this function returns a subset of indices that have human ratings"""
        human_eval_indices = []
        model_responses = []
        human_evals = self.human_evals
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
