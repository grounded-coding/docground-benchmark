from src.data_collector import DSTCDataCollector, DummyDataCollector, BEGINDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval
from src.eval_collector import DummyEvalCollector, DSTCHumanEvalCollector, BEGINHumanEvalCollector
import json
from utils.file_processing import load_data
from typing import List, Dict


type = 'spearman'
correlation_level = 'sample'
desired_framework = LLEval()
framework_dimensions = ["accurate"]

## PART 1

begin_cmu_collector = BEGINDataCollector(dataset_path="../BEGIN-dataset/cmu-dog", dataset_split="dev", dataset_name="cmu")
model_candidates = ['baseline']
response_indices = begin_cmu_collector.get_samples_with_target(n=50)
model_responses = begin_cmu_collector.get_pred_responses(response_indices, ["baseline"])
# Implement different model responses for BEGIN
dimension_map = {"accurate": "attributability"}
begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/cmu-dog/begin_dev_cmu.tsv")

pipeline_evaluator = PipelineEvaluator(desired_framework, begin_eval_collector, begin_cmu_collector, framework_dimensions, dimension_map, 
                                       type, correlation_level, model_candidates)
human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices,
                                                               dataset_task_description="")
print(human_framework_correlations)

## PART 1.2

begin_cmu_collector = BEGINDataCollector(dataset_path="../BEGIN-dataset/topicalchat", dataset_split="dev", dataset_name="tc")
model_candidates = ['baseline']
response_indices = begin_cmu_collector.get_samples_with_target(n=50)
model_responses = begin_cmu_collector.get_pred_responses(response_indices, ["baseline"])
# Implement different model responses for BEGIN
dimension_map = {"accurate": "attributability"}
begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/topicalchat/begin_dev_tc.tsv")

pipeline_evaluator = PipelineEvaluator(desired_framework, begin_eval_collector, begin_cmu_collector, framework_dimensions, dimension_map, 
                                       type, correlation_level, model_candidates)
human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices,
                                                               dataset_task_description="")
print(human_framework_correlations)


## PART 2

# Load from some prediction
baseline_pred_path = "../dstc11-track5/results/results_dstc9/baseline/entry0.json"
baseline_human_eval = "../dstc11-track5/results/results_dstc9/baseline/entry0.human_eval.json"

data_collector = DSTCDataCollector(dataset="../dstc11-track5/data/dstc9", dataset_split="test", dataset_name="dstc9")
response_indices = data_collector.get_samples_with_target(n=200)
model_responses = data_collector.get_pred_responses(response_indices, model_candidates, baseline_pred_path)

dimension_map = {"accurate": "accuracy"}
eval_collector = DSTCHumanEvalCollector(human_eval_path=baseline_human_eval)

pipeline_evaluator = PipelineEvaluator(desired_framework, eval_collector, data_collector, framework_dimensions, dimension_map, type, correlation_level, model_candidates)

human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices, dataset_task_description='Context can be reviews from customers or FAQs. FAQs start after token :F: and each new review starts after token :R:.')

print(human_framework_correlations)