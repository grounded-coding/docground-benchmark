
from src.eval_collector import HumanEvalCollector
from src.data_collector import DataCollector
from src.eval_framework import EvaluationFramework
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from pathlib import Path
from utils.file_processing import load_data
import json


class PipelineEvaluator:
    def __init__(self, desired_framework: EvaluationFramework, eval_collector: HumanEvalCollector,
                 data_collector: DataCollector, desired_dimensions: list,
                 dimension_map, correlation_type, correlation_level, model_candidates: list):
        self.correlation_score = correlation_type
        self.data_collector = data_collector
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

        if len(model_candidates) > 1:
            self.correlation_level = 'system'
            raise NotImplementedError("System-level correlation is not implemented yet.")

    def run_pipeline(self, model_responses, response_indices):
        reference_responses = None

        # Here we filter model responses to only include response for which we have human evaluations
        eval_response_indices, cleaned_model_responses = self.eval_collector.get_subset_with_human_eval(response_indices, model_responses)
        reference_responses, turn_historys, knowledge_contexts = self.data_collector.collect_sample_contexts(eval_response_indices)

        if self.desired_framework.reference_required and reference_responses is None:
            raise ValueError("Reference responses are required for the selected evaluation framework.")

        self.framework_scores = self._evaluate_framework(cleaned_model_responses, reference_responses, turn_historys, knowledge_contexts)
        human_scores = self.eval_collector.extract_ratings(eval_response_indices, self.dimension_map.values())
        human_framework_correlations = self._compute_correlations(self.framework_scores, human_scores, self.dimension_map)

        return human_framework_correlations

    def _evaluate_framework(self, model_responses, reference_responses, turn_historys, knowledge_contexts):
        """
        Evaluate the model responses using the desired evaluation framework.
        This function should use persistent storage to save the evaluation results.
        If the evaluation results are already available, they should be loaded from storage.
        :param model_responses: A list of model responses for each data sample looking like
            [{"model1": "response1", "model2": "response2"}, {"model1": "response1", "model2": "response2"}, ...]
        :param reference_responses: A list of reference responses for each data sample
        :param turn_historys: A list of turn histories for each data sample
        :param knowledge_contexts: A list of knowledge contexts for each data sample
        :return: A list of evaluation scores for each data sample
        """

        score_path = Path(self.data_collector.get_name()) / self.data_collector.dataset_split / self.model_candidates[0] \
                     / (self.desired_framework.get_name() + ".json")
        if score_path.is_file():
            with open(score_path, "r") as read_file:
                framework_scores = json.load(read_file)
        else:
            score_path.parent.mkdir(parents=True, exist_ok=True)
            # Prepare model responses
            specific_responses = [resp[self.model_candidates[0]] for resp in model_responses]
            framework_scores = self.desired_framework.evaluate(specific_responses, reference_responses,
                                                           turn_historys, knowledge_contexts, self.desired_dimensions)
            with open(score_path, "w") as write_file:
                json.dump(framework_scores, write_file)
        return framework_scores

    def _compute_correlations(self, framework_scores, human_scores, dimension_map):
        """
        Compute the correlation between the framework scores and the human scores.
        :param framework_scores: A list of framework scores for each data sample
        :param dimension_map: A dictionary mapping the framework scores to the desired human evaluation dimensions
        """
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