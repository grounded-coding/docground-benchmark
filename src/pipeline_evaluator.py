
from abc import ABC, abstractmethod
import json
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
        super().__init__(['meteor'])
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        met_metric = MeteorMetric()
        met_score = met_metric.evaluate_batch(model_responses, reference_responses)['meteor'] * 100
        return met_score

class ROUGE(EvaluationFramework):
    def __init__(self, reference_required=True):
        super().__init__(['rouge-l','rouge-1','rouge-2'])
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass

class PipelineEvaluator:
    def __init__(self, desired_framework, desired_dimensions, correlation_score, correlation_level, model_candidates):
        self.correlation_score = correlation_score
        self.correlation_level = correlation_level
        self.model_candidates = model_candidates
        self.desired_framework = desired_framework
        self.desired_dimensions = desired_dimensions
        self.framework_scores = None

        # Check if desired dimensions are available for the desired framework
        if not set(self.desired_dimensions).issubset(self.available_frameworks[self.desired_framework]['av_dimensions']):
            raise ValueError("The desired dimensions are not available for the desired framework.")

        if correlation_level == 'system' and len(model_candidates) < 2:
            raise ValueError("For system-level correlation, at least 2 model candidates are required.")

    def run_pipeline(self, model_responses, turn_historys, knowledge_contexts, human_scores, reference_responses=None):
        if self.desired_framework.reference_required and reference_responses is None:
            raise ValueError("Reference responses are required for the selected evaluation framework.")

        self.framework_scores = self.desired_framework.evaluate(model_responses, reference_responses, turn_historys, knowledge_contexts, desired_dimensions)
        human_framework_correlations = self._compute_correlations(self.framework_scores, human_scores)

        return human_framework_correlations

    def _compute_correlations(self, framework_scores, human_scores):
        correlations = {}
        if self.correlation_score == 'spearman':
            for dim in self.desired_dimensions:
                human_scores_dim = np.array([data['human_scores'][dim] for data in human_scores])
                framework_scores_dim = np.array([score[dim] for score in framework_scores])

                spearman_corr, _ = spearmanr(human_scores_dim, framework_scores_dim)
                correlations[dim] = spearman_corr
        elif self.correlation_score == 'kendall':
            for dim in self.desired_dimensions:
                human_scores_dim = np.array([data['human_scores'][dim] for data in human_scores])
                framework_scores_dim = np.array([score[dim] for score in framework_scores])

                kendall_corr, _ = kendalltau(human_scores_dim, framework_scores_dim)
                correlations[dim] = kendall_corr
        elif self.correlation_score == 'pearson':
            for dim in self.desired_dimensions:
                human_scores_dim = np.array([data['human_scores'][dim] for data in human_scores])
                framework_scores_dim = np.array([score[dim] for score in framework_scores])

                pearson_corr, _ = pearsonr(human_scores_dim, framework_scores_dim)
                correlations[dim] = pearson_corr
        return correlations