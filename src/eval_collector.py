import json
import numpy as np
from utils.file_processing import load_data
import pandas as pd
from abc import ABC, abstractmethod


class HumanEvalCollector(ABC):

    def __init__(self):
        pass

    def extract_ratings(self, human_eval_path, dims):
        pass

class DummyEvalCollector(HumanEvalCollector):    

    def extract_ratings(self, human_eval_path, dims=None):
        human_scores = [
            {'dimension1': 0.8, 'dimension2': 0.9},
            {'dimension1': 0.7, 'dimension2': 0.6}
        ]
        return human_scores

class DSTCHumanEvalCollector(HumanEvalCollector):

    def extract_ratings(self, sample_indices, human_eval_path, dims=["accuracy", "appropriateness"]):
        ratings = []
        valid_rating = 0
        human_file = load_data(human_eval_path)

        # Human file is a list of dicts, each dict is a human rating, but can be None
        # Iterate over all sample indices, and if the entry is not None, extract the ratings for the desired dimensions
        for sample_index in sample_indices:
            if human_file[sample_index] is not None:
                rating = {}
                for dim in dims:
                    rating[dim] = human_file[sample_index][dim]
                ratings.append(rating)
                valid_rating += 1
            else:
                # print("No human ratings for sample {}".format(sample_index))
                ratings.append(None)

        print("Candidate has received {} valid ratings".format(valid_rating))            
        return ratings