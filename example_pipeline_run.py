from src.data_collector import DSTCDataCollector, DummyDataCollector, BEGINDataCollector, DialDocDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval, KnowledgeBLEU
from src.eval_collector import DummyEvalCollector, DSTCHumanEvalCollector, BEGINHumanEvalCollector, DialDocEvalCollector
import json
from utils.file_processing import load_data
from typing import List, Dict


type = 'spearman'
correlation_level = 'sample'
unieval = UniEval()
unieval_dimensions = ["groundedness"]
lleval = LLEval()
lleval_dimensions = ["accurate"]
bleu = KnowledgeBLEU()
bleu_dimensions = ["knowledge-bleu-4"]
kf1 = KnowledgeF1()
kf1_dimensions = ["knowledge-f1"]
model_candidates = ['baseline']

for framework, framework_dimensions in [(lleval, lleval_dimensions), (unieval, unieval_dimensions), (kf1, kf1_dimensions), (bleu, bleu_dimensions)]:
    ## BEGIN CMU DoG
    framework_to_human_dimension_map = {framework_dimensions[0]: "attributability"}
    begin_cmu_collector = BEGINDataCollector(dataset_path="../BEGIN-dataset/cmu-dog", dataset_split="dev", dataset_name="cmu")
    response_indices = begin_cmu_collector.get_samples_with_target(n=-1)
    model_responses = begin_cmu_collector.get_pred_responses(response_indices, ["baseline"])

    begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/cmu-dog/begin_dev_cmu.tsv")
    pipeline_evaluator = PipelineEvaluator(framework, begin_eval_collector, begin_cmu_collector, framework_dimensions, framework_to_human_dimension_map, 
                                        type, correlation_level, model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices,
                                                                dataset_task_description="")

    ## BEGIN TopicalChat
    begin_tc_collector = BEGINDataCollector(dataset_path="../BEGIN-dataset/topicalchat", dataset_split="dev", dataset_name="tc")
    response_indices = begin_tc_collector.get_samples_with_target(n=-1)
    model_responses = begin_tc_collector.get_pred_responses(response_indices, ["baseline"])

    begin_eval_collector = BEGINHumanEvalCollector(human_eval_path="../BEGIN-dataset/topicalchat/begin_dev_tc.tsv")
    pipeline_evaluator = PipelineEvaluator(framework, begin_eval_collector, begin_tc_collector, framework_dimensions, framework_to_human_dimension_map, 
                                        type, correlation_level, model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices, exclude_rating=None,
                                                                dataset_task_description="")

    ## DSTC9
    framework_to_human_dimension_map = {framework_dimensions[0]: "accuracy"}
    baseline_pred_path = "../dstc11-track5/results/results_dstc9/baseline/entry0.json"
    baseline_human_eval = "../dstc11-track5/results/results_dstc9/baseline/entry0.human_eval.json"

    dstc_collector = DSTCDataCollector(dataset_path="../dstc11-track5/data/dstc9", dataset_split="test", dataset_name="dstc9")
    response_indices = dstc_collector.get_samples_with_target(n=-1)
    model_responses = dstc_collector.get_pred_responses(response_indices, model_candidates, baseline_pred_path)

    eval_collector = DSTCHumanEvalCollector(human_eval_path=baseline_human_eval)
    pipeline_evaluator = PipelineEvaluator(framework, eval_collector, dstc_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices, dataset_task_description='Context can be reviews from customers or FAQs. FAQs start after token :F: and each new review starts after token :R:.')

    ## DialDoc

    dialdoc_collector = DialDocDataCollector("../DialDoc-TU2023")
    response_indices = dialdoc_collector.get_samples_with_target(n=-1)
    model_responses = dialdoc_collector.get_pred_responses(response_indices, model_candidates)

    framework_to_human_dimension_map = {framework_dimensions[0]: "groundedness"}
    dialdoc_eval_collector = DialDocEvalCollector("../DialDoc-TU2023/Batch_383409_batch_results_final.csv")
    pipeline_evaluator = PipelineEvaluator(framework, dialdoc_eval_collector, dialdoc_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)