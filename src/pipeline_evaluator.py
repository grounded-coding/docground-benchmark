
from abc import ABC, ABCMeta, abstractmethod
import json
from src.eval_collector import HumanEvalCollector
import numpy as np
import pandas as pd
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bleu_metric import BleuMetric
from scipy.stats import spearmanr, kendalltau, pearsonr

class EvaluationFramework(ABC):
    def __init__(self, available_dimensions, reference_required=False, sample_level_support=False):
        self.available_dimensions = available_dimensions
        self.sample_level_support = sample_level_support
        self.reference_required = reference_required
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass

class UniEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['groundedness', 'informativeness', 'fluency', 'engagingness', 'overall'])
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass

class DummyEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['accur', 'app'])
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        scores = []
        for response in model_responses:
            score = {}
            for dim in dims:
                score[dim] = np.random.rand()
            scores.append(score)
        return scores

class LLEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['appropriate', 'accurate'])
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass

class BLEU(EvaluationFramework):
    def __init__(self):
        super().__init__(['bleu-4'], reference_required=True)
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        bleu_metric = BleuMetric()
        bleu_score = bleu_metric.evaluate_batch(model_responses, reference_responses)['bleu']
        return [{"bleu-4": bleu_score}]

class METEOR(EvaluationFramework):
    def __init__(self, reference_required=True):
        super().__init__(['meteor'], reference_required=reference_required)
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        met_metric = MeteorMetric()
        met_score = met_metric.evaluate_batch(model_responses, reference_responses)['meteor'] * 100
        return met_score

class ROUGE(EvaluationFramework):
    def __init__(self, reference_required=True):
        super().__init__(['rouge-l','rouge-1','rouge-2'], reference_required=reference_required)
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass

class PipelineEvaluator:
    def __init__(self, desired_framework: EvaluationFramework, eval_collector: HumanEvalCollector, desired_dimensions: list, 
                 dimension_map, correlation_type, correlation_level, model_candidates: list):
        self.correlation_score = correlation_type
        self.correlation_level = correlation_level
        self.model_candidates = model_candidates
        self.desired_framework = desired_framework
        self.desired_dimensions = desired_dimensions
        self.dimension_map = dimension_map
        self.eval_collector = eval_collector
        self.framework_scores = None

        # Check if desired dimensions are available for the desired framework
        if not set(self.desired_dimensions).issubset(self.desired_framework.available_dimensions):
            raise ValueError("The desired dimensions are not available for the desired framework.")

        if correlation_level == 'system' and len(model_candidates) < 2:
            raise ValueError("For system-level correlation, at least 2 model candidates are required.")

    def run_pipeline(self, model_responses, turn_historys, knowledge_contexts, reference_responses=None):
        if self.desired_framework.reference_required and reference_responses is None:
            raise ValueError("Reference responses are required for the selected evaluation framework.")

        self.framework_scores = self.desired_framework.evaluate(model_responses, reference_responses, turn_historys, knowledge_contexts, self.desired_dimensions)
        human_scores = self.eval_collector.extract_ratings(self.framework_scores, self.desired_dimensions)
        human_framework_correlations = self._compute_correlations(self.framework_scores, self.dimension_map)

        return human_framework_correlations

    def _compute_correlations(self, framework_scores, dimension_map):
        """
        Compute the correlation between the framework scores and the human scores.
        :param framework_scores: A list of framework scores for each data sample
        :param dimension_map: A dictionary mapping the framework scores to the desired human evaluation dimensions
        """
        human_scores = self.eval_collector.extract_ratings(self.framework_scores, self.desired_dimensions)
        print(human_scores)
        correlations = {}
        for framework_dim in self.desired_dimensions:
            human_dim = dimension_map[framework_dim]
            human_scores_dim = np.array([score_dict[human_dim] for score_dict in human_scores])
            framework_scores_dim = np.array([score[framework_dim] for score in framework_scores])

            if self.correlation_score == 'spearman':
                spearman_corr, _ = spearmanr(human_scores_dim, framework_scores_dim)
                correlations[framework_dim] = spearman_corr
            elif self.correlation_score == 'kendall':
                kendall_corr, _ = kendalltau(human_scores_dim, framework_scores_dim)
                correlations[framework_dim] = kendall_corr
            elif self.correlation_score == 'pearson':
                pearson_corr, _ = pearsonr(human_scores_dim, framework_scores_dim)
                correlations[framework_dim] = pearson_corr
        return correlations