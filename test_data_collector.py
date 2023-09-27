from src.data_collector import DSTCDataCollector, DummyDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval
from src.eval_collector import DummyEvalCollector, DSTCHumanEvalCollector
import json
from utils.file_processing import load_data
from typing import List, Dict


if __name__ == "__main__":
    desired_framework = LLEval()
    framework_dimensions = ["accurate"]

    # Load from some prediction
    baseline_pred_path = "../dstc11-track5/results/results_dstc9/baseline/entry0.json"
    model_candidates = ['baseline']
    baseline_human_eval = "../dstc11-track5/results/results_dstc9/baseline/entry0.human_eval.json"

    data_collector = DSTCDataCollector(dataset_path="../dstc11-track5/data/dstc9", dataset_split="test", dataset_name="dstc9")
    response_indices = data_collector.get_samples_with_target(n=200)
    model_responses = data_collector.get_pred_responses(response_indices, model_candidates, baseline_pred_path)

    dimension_map = {"accurate": "accuracy"}
    eval_collector = DSTCHumanEvalCollector(human_eval_path=baseline_human_eval)

    type = 'spearman'
    correlation_level = 'sample'
    pipeline_evaluator = PipelineEvaluator(desired_framework, eval_collector, data_collector, framework_dimensions, dimension_map, type, correlation_level, model_candidates)

    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)

    print(human_framework_correlations)