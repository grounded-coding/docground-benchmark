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

    def get_subset_with_human_eval(self, sample_indices, model_responses):
        return sample_indices, model_responses

class DummyEvalCollector(HumanEvalCollector):    

    def extract_ratings(self, sample_indices, human_dims=None):
        # create a list with dictionaries of length sample_indices where each dictionary contains random ratings for each dimension
        human_scores = []
        for i in range(len(sample_indices)):
            human_scores.append({})
            for dim in human_dims:
                human_scores[i][dim] = np.random.rand()
        return human_scores
    

class BEGINHumanEvalCollector(HumanEvalCollector):
    def __init__(self, human_eval_path):
        super().__init__()
        # read from human eval path the file wh ich is a tsv file with columns_ id, model_name, data_source, knowledge_message, response, begin_label
        self.human_eval = pd.read_csv(human_eval_path, sep='\t')

    def extract_ratings(self, sample_indices, human_dims=["attributability"]):
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

        # TODO Convert to numeric ratings
        return ratings


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

        print("Candidate has received {} valid ratings".format(valid_rating))            
        return ratings

    def get_subset_with_human_eval(self, sample_indices, candidate_responses):
        """
        Provided with a list of indices that contain responses for the chosen dataset, this function returns a subset of indices that have human ratings"""
        human_eval_indices = []
        model_responses = []
        human_evals = self.human_evals
        for j, index in enumerate(sample_indices):
            if human_evals[index] is not None:
                human_eval_indices.append(index)
                model_responses.append(candidate_responses[j])
        return human_eval_indices, model_responses