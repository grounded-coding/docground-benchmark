
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

    def get_new_contexts_only(self, existing_framework_scores, response_indices, reference_responses, turn_historys, knowledge_contexts, model_responses):
        """
        Filters response indices, model responses, reference responses, turn historys and knowledge contexts for some model to only those that havent been evaluated yet.
        """
        new_response_indices = []
        for i, response_index in enumerate(response_indices):
            # if for any dimension that we are evaluating the score is not available yet for this sample, we also need to evaluate again
            for dim in self.desired_dimensions:
                if not any(score["response_index"] == response_index and dim in score for score in existing_framework_scores):
                    new_response_indices.append(response_index)
                    break
        
        model_responses = [model_responses[i] for i in range(len(response_indices)) if response_indices[i] in new_response_indices]
        if len(reference_responses) > 0:
            reference_responses = [reference_responses[i] for i in range(len(response_indices)) if response_indices[i] in new_response_indices]
        turn_historys = [turn_historys[i] for i in range(len(response_indices)) if response_indices[i] in new_response_indices]
        knowledge_contexts = [knowledge_contexts[i] for i in range(len(response_indices)) if response_indices[i] in new_response_indices]

        return new_response_indices, model_responses, reference_responses, turn_historys, knowledge_contexts

    def run_pipeline(self, model_responses, response_indices, print_statements=True, exclude_rating=None, verbose=False):
        reference_responses = None
        all_human_framework_correlations = []

        print("--- {} correlation -- {} level ---".format(self.correlation_score, self.correlation_level))
        print("Dataset: {}".format(self.data_collector.get_name()))
        print("Split: {}".format(self.data_collector.dataset_split))
        print("-----------------------------\n")

        if self.correlation_level == 'sample':
            for model in self.model_candidates:
                response_indices, model_responses = self.eval_collector.get_subset_with_human_eval(response_indices, model_responses, exclude_rating=exclude_rating, model=model)
                reference_responses, turn_historys, knowledge_contexts = self.data_collector.collect_sample_contexts(response_indices)
                if verbose:
                    print("--- Sample contexts ---")
                    for i in range(3):
                        print("Sample index: {}".format(response_indices[i]))
                        if reference_responses is not None:
                            print("Reference response: {}".format(reference_responses[i]))
                        print("Model response: {}".format(model_responses[i][model]))
                        print("Turn history: {}".format(turn_historys[i]))
                        print("Knowledge context: {}".format(knowledge_contexts[i]))
                        print("Model: {}".format(model))
                model_sample_correlations = self.compute_sample_correlation_for_model(model_responses, response_indices, reference_responses, turn_historys, knowledge_contexts, model, print_statements)
                all_human_framework_correlations.append(model_sample_correlations)
            # Print the averaged correlations over all models
            avg_correlations = {}
            for dim in self.desired_dimensions:
                avg_correlations[dim] = np.mean([correlations[dim + "-" + self.dimension_map[dim]] for correlations in all_human_framework_correlations])
                print("Average correlation {}: {}".format(dim, avg_correlations[dim]))

        elif self.correlation_level == 'system':
            avg_scores = {}
            for framework_dim in self.desired_dimensions:
                human_dim = self.dimension_map[framework_dim]
                avg_scores[framework_dim] = ([], [])
                for model in self.model_candidates:
                    response_indices, model_responses = self.eval_collector.get_subset_with_human_eval(response_indices, model_responses, exclude_rating=exclude_rating, model=model)
                    reference_responses, turn_historys, knowledge_contexts = self.data_collector.collect_sample_contexts(response_indices)
                    framework_scores = self._get_framework_scores(response_indices, model, reference_responses, turn_historys, knowledge_contexts, model_responses)
                    human_scores = self.eval_collector.extract_ratings_for_sample_indices(response_indices, self.dimension_map.values(), model)

                    framework_scores_dim = np.array([score[framework_dim] for score in framework_scores])
                    human_scores_dim = np.array([score[human_dim] for score in human_scores])
                    avg_scores[framework_dim][0].append(np.mean(human_scores_dim))
                    avg_scores[framework_dim][1].append(np.mean(framework_scores_dim))

            # Compute the actual system level correlations
            system_correlations = self.compute_system_correlation(avg_scores)
            all_human_framework_correlations.append(system_correlations)

            if print_statements:
                print("Models: {}".format(self.model_candidates))
                print("Framework: {}".format(self.desired_framework.get_name()))
                for entry in system_correlations:
                    print("Correlation {}: {}".format(entry, system_correlations[entry]))

        else:
            raise ValueError("Unknown correlation level.")
        print("-----------------------------\n")
        return all_human_framework_correlations

    def compute_system_correlation(self, avg_scores):
        system_correlations = {}
        for framework_dim in self.desired_dimensions:
            human_dim = self.dimension_map[framework_dim]
            key = framework_dim + "-" + human_dim
            human_scores_dim, framework_scdim = avg_scores[framework_dim]

            if self.correlation_score == 'spearman':
                spearman_corr, _ = spearmanr(human_scores_dim, framework_scdim)
                system_correlations[key] = spearman_corr
            elif self.correlation_score == 'kendall':
                kendall_corr, _ = kendalltau(human_scores_dim, framework_scdim)
                system_correlations[key] = kendall_corr
            elif self.correlation_score == 'pearson':
                pearson_corr, _ = pearsonr(human_scores_dim, framework_scdim)
                system_correlations[key] = pearson_corr
        return system_correlations

    def compute_sample_correlation_for_model(self, model_responses, response_indices, reference_responses, turn_historys, knowledge_contexts, model, print_statements):

        if self.desired_framework.reference_required and reference_responses is None:
            raise ValueError("Reference responses are required for the selected evaluation framework.")

        framework_scores = self._get_framework_scores(response_indices, model, reference_responses, turn_historys, knowledge_contexts, model_responses)
        human_scores = self.eval_collector.extract_ratings_for_sample_indices(response_indices, self.dimension_map.values(), model)

        if print_statements and False:
            print("--- Sample responses ---")
            print("Response indices: {}".format(response_indices[:10]))
            print("Human scores: {}".format(human_scores[:10]))
            print("Framework scores: {}".format(framework_scores[:10]))
            print("-------------------------\n")

        human_framework_correlations = self._compute_correlations_for_all_dims(framework_scores, human_scores, self.dimension_map)

        if print_statements:
            print("Framework: {}".format(self.desired_framework.get_name()))
            print("# Samples: {}".format(len(framework_scores)))
            for dim in self.desired_dimensions:
                print("Correlation {}-{}: {}".format(dim, self.dimension_map[dim], human_framework_correlations[dim + "-" + self.dimension_map[dim]]))
                print("Average {}: {}".format(dim, np.mean([score[dim] for score in framework_scores])))
                print("Average human {}: {}".format(self.dimension_map[dim], np.mean([score[self.dimension_map[dim]] for score in human_scores])))

        return human_framework_correlations

    def _get_framework_scores(self, response_indices, model, reference_responses, turn_historys, knowledge_contexts, model_responses): 
        existing_framework_scores = self._load_scores_from_storage(model)
        existing_framework_scores = self._get_subset_of_scores_with_all_dims(existing_framework_scores, self.desired_dimensions)

        new_response_indices, new_model_responses, new_reference_responses, new_turn_historys, new_knowledge_contexts = self.get_new_contexts_only(existing_framework_scores, response_indices, reference_responses, turn_historys, knowledge_contexts, model_responses)
        new_framework_scores = self._evaluate_framework(new_model_responses, new_response_indices, new_reference_responses, new_turn_historys, new_knowledge_contexts, model)
        
        framework_scores = self._merge_existing_and_new_scores(existing_framework_scores, new_framework_scores)
        self._write_scores_to_storage(framework_scores, model)
        # keep only the scores for the desired response indices
        framework_scores = [score for score in framework_scores if score["response_index"] in response_indices]
        return framework_scores

    def _load_scores_from_storage(self, model=""):
        score_path = Path("outputs") / self.data_collector.get_name() / self.data_collector.dataset_split / model \
                     / (self.desired_framework.get_name() + ".json")
        if score_path.is_file():
            with open(score_path, "r") as read_file:
                framework_scores = json.load(read_file)
        else:
            framework_scores = []

        score_path.parent.mkdir(parents=True, exist_ok=True)
        return framework_scores

    def _write_scores_to_storage(self, framework_scores, model=""):
        score_path = Path("outputs") / self.data_collector.get_name() / self.data_collector.dataset_split / model \
                / (self.desired_framework.get_name() + ".json")
        with open(score_path, "w") as write_file:
            json.dump(framework_scores, write_file)
    
    def _get_subset_of_scores_with_all_dims(self, existing_framework_scores, desired_dimensions):
        # only if all dimensions from desired_dimensions are available for a sample, we can use the existing score
        existing_framework_scores = [score for score in existing_framework_scores if all(dim in score for dim in desired_dimensions)]
        return existing_framework_scores

    def _merge_existing_and_new_scores(self, existing_framework_scores, new_framework_scores):
        merged_scores = []
        all_response_indices = [score["response_index"] for score in existing_framework_scores] + [score["response_index"] for score in new_framework_scores]
        for response_index in all_response_indices:
            existing_score_dict = next((score for score in existing_framework_scores if score["response_index"] == response_index), None)
            new_score_dict = next((score for score in new_framework_scores if score["response_index"] == response_index), None)
            if existing_score_dict is None:
                merged_scores.append(new_score_dict)
            elif new_score_dict is None:
                merged_scores.append(existing_score_dict)
            else:
                merged_scores.append({**existing_score_dict, **new_score_dict})
        merged_scores = sorted(merged_scores, key=lambda k: k['response_index'])
        return merged_scores


    def _evaluate_framework(self, model_responses, response_indices, reference_responses, turn_historys, knowledge_contexts, model=""):
        """
        Evaluate the model responses using the desired evaluation framework.
        :param model_responses: A list of model responses for each data sample looking like
            [{"model1": "response1", "model2": "response2"}, {"model1": "response1", "model2": "response2"}, ...]
        :param reference_responses: A list of reference responses for each data sample
        :param response_indices: A list of indices of the data samples for which the model responses should be evaluated‚
        :param turn_historys: A list of turn histories for each data sample
        :param knowledge_contexts: A list of knowledge contexts for each data sample
        :return: A list of evaluation scores for each data sample
        """
        if len(model_responses) > 0:
            model_responses = [resp[model] for resp in model_responses]
            new_framework_scores = self.desired_framework.evaluate(model_responses, reference_responses,
                                                            turn_historys, knowledge_contexts, self.desired_dimensions)
            # Add the response indices to the scores so that we can identify the samples later
            for i, score in enumerate(new_framework_scores):
                score["response_index"] = response_indices[i]

            return new_framework_scores
        else:
            return []

    def _compute_correlations_for_all_dims(self, framework_scores, human_scores, dimension_map):
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

            key = framework_dim + "-" + human_dim

            if self.correlation_score == 'spearman':
                spearman_corr, _ = spearmanr(human_scores_dim, framework_scores_dim)
                correlations[key] = spearman_corr
            elif self.correlation_score == 'kendall':
                kendall_corr, _ = kendalltau(human_scores_dim, framework_scores_dim)
                correlations[key] = kendall_corr
            elif self.correlation_score == 'pearson':
                pearson_corr, _ = pearsonr(human_scores_dim, framework_scores_dim)
                correlations[key] = pearson_corr
        return correlations