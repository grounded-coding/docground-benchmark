import numpy as np
import pandas as pd
import unittest
from typing import List, Dict 
from src.eval_collector import DSTCHumanEvalCollector, DummyEvalCollector
from src.data_collector import DataCollector, DummyDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval

class TestPipelineEvaluator(unittest.TestCase):
    def setUp(self):
        self.desired_framework = DummyEval()
        self.data_collector = DummyDataCollector()
        self.type = 'spearman'
        self.framework_dims = ["accur", "app"]
        self.dimension_map = {"accur": "dimension1", "app": "dimension2"}
        self.human_dims = ["dimension1", "dimension2"]
        self.dummy_collector = DummyEvalCollector()

    def test_reference_required(self):
        self.assertFalse(self.desired_framework.reference_required)

    def test_sample_level_correlation(self):
        model_candidates = ['model1']
        pipeline_evaluator = PipelineEvaluator(self.desired_framework, self.dummy_collector, self.data_collector,
                                                    self.framework_dims, self.dimension_map,
                                                     self.type, correlation_level="sample", model_candidates=model_candidates)
        sample_indices = [4, 5]
        framework_scores = [
            {'accur': 0.8, 'app': 0.7},
            {'accur': 0.6, 'app': 0.8}
        ]
        human_scores = self.dummy_collector.extract_ratings(sample_indices, human_dims=self.human_dims)
        correlation = pipeline_evaluator._compute_correlations_for_all_dims(framework_scores, human_scores, self.dimension_map)
    


    def test_run_pipeline(self):
        model_candidates = ['model1']
        pipeline_evaluator = PipelineEvaluator(self.desired_framework, self.dummy_collector, self.data_collector,
                                                    self.framework_dims, self.dimension_map,
                                                     self.type, correlation_level="sample", model_candidates=model_candidates)

        # Load using data_collector class
        sample_indices = [4, 3]

        # Load from some prediction
        model_responses = [
            {'model1': 'response1', 'model2': 'response2'},
            {'model1': 'response3', 'model2': 'response4'}
        ]

        # Load using HumanEvalCollector
        human_framework_correlations = pipeline_evaluator.run_pipeline(
            model_responses, sample_indices
        )

        self.assertIsInstance(human_framework_correlations, List)
        self.assertEqual(len(human_framework_correlations), len(model_candidates))
        print(human_framework_correlations)

if __name__ == '__main__':
    unittest.main()