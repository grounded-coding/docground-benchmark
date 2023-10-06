import numpy as np
import pandas as pd
import unittest
from src.eval_collector import DSTCHumanEvalCollector, DummyEvalCollector
from src.data_collector import DataCollector, DummyDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval

class TestPipelineEvaluator(unittest.TestCase):
    def setUp(self):
        self.desired_framework = DummyEval()
        self.data_collector = DummyDataCollector()
        self.type = 'spearman'
        self.correlation_level = 'sample'
        self.model_candidates = ['model1']
        self.desired_dimensions = ["accur", "app"]
        self.dimension_map = {"accur": "dimension1", "app": "dimension2"}

        self.dummy_collector = DummyEvalCollector()
        self.pipeline_evaluator = PipelineEvaluator(self.desired_framework, self.dummy_collector, self.data_collector,
                                                    self.desired_dimensions, self.dimension_map,
                                                     self.type, self.correlation_level, self.model_candidates)

    def test_reference_required(self):
        self.assertFalse(self.desired_framework.reference_required)

    def test_compute_correlation(self):
        # Test for sample level correlation
        sample_indices = [4, 5]
        framework_scores = [
            {'accur': 0.8, 'app': 0.7},
            {'accur': 0.6, 'app': 0.8}
        ]
        human_scores = self.dummy_collector.extract_ratings(sample_indices, human_dims=["dimension1", "dimension2"])
        correlation = self.pipeline_evaluator._compute_correlations_for_all_dims(framework_scores, human_scores, self.dimension_map)

    def test_run_pipeline(self):
        
        # Load using data_collector class
        sample_indices = [4, 3]

        # Load from some prediction
        model_responses = [
            {'model1': 'response1', 'model2': 'response2'},
            {'model1': 'response3', 'model2': 'response4'}
        ]

        # Load using HumanEvalCollector
        human_framework_correlations = self.pipeline_evaluator.run_pipeline(
            model_responses, sample_indices
        )

        self.assertIsInstance(human_framework_correlations, dict)
        print(human_framework_correlations)

if __name__ == '__main__':
    unittest.main()