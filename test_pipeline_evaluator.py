import numpy as np
import pandas as pd
import unittest
from src.eval_collector import DSTCHumanEvalCollector
from src.pipeline_evaluator import PipelineEvaluator
from scipy.stats import pearsonr, spearmanr
from utils import convert_to_json


class TestPipelineEvaluator(unittest.TestCase):
    def setUp(self):
        self.desired_framework = 'UniEval'
        self.score = 'spearman'
        self.correlation_level = 'sample'
        self.model_candidates = ['model1', 'model2']

        self.pipeline_evaluator = PipelineEvaluator(self.desired_framework, self.score, self.correlation_level, self.model_candidates)

    def test_reference_required(self):
        self.assertFalse(self.pipeline_evaluator.reference_required)

    def test_run_pipeline(self):
        
        # Load using data_collector class
        reference_responses = None
        turn_historys = ['history1', 'history2']
        knowledge_contexts = ['context1', 'context2']

        # Load from some prediction
        model_responses = [
            {'model1': 'response1', 'model2': 'response2'},
            {'model1': 'response3', 'model2': 'response4'}
        ]

        # Load using HumanEvalCollector
        human_scores = [
            {'dimension1': 0.8, 'dimension2': 0.9},
            {'dimension1': 0.7, 'dimension2': 0.6}
        ]

        human_framework_correlations = self.pipeline_evaluator.run_pipeline(
            model_responses, reference_responses, turn_historys, knowledge_contexts, human_scores
        )

        self.assertEqual(len(self.pipeline_evaluator.framework_scores), len(self.model_candidates))
        self.assertIsInstance(human_framework_correlations, dict)

if __name__ == '__main__':
    unittest.main()