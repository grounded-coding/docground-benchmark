from src.data_collector import DSTCDataCollector, DummyDataCollector, BEGINDataCollector, DialDocDataCollector, TopicalChatDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval, KnowledgeBLEU
from src.eval_collector import DummyEvalCollector, DSTCHumanEvalCollector, BEGINHumanEvalCollector, DialDocEvalCollector, TopicalChatEvalCollector
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

for framework, framework_dimensions in [(lleval, lleval_dimensions), (unieval, unieval_dimensions), (kf1, kf1_dimensions), (bleu, bleu_dimensions)]:
    ## DSTC11
    framework_to_human_dimension_map = {framework_dimensions[0]: "accuracy"}
    baseline_pred_path = "../dstc11-track5/results/team00/entry0.json"
    baseline_human_eval = "../dstc11-track5/results/team00/entry0.human_eval.json"

    dstc_collector = DSTCDataCollector(dataset_path="../dstc11-track5/data", dataset_split="test", dataset_name="dstc11")
    response_indices = dstc_collector.get_samples_with_target(n=200)
    model_responses = dstc_collector.get_pred_responses(response_indices, ["baseline"], baseline_pred_path)

    eval_collector = DSTCHumanEvalCollector(human_eval_path=baseline_human_eval)
    pipeline_evaluator = PipelineEvaluator(framework, eval_collector, dstc_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, ["baseline"])
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)

    gpt_pred_path = "../dstc11-track5/pred/val/rg.chatgpt.json"
    dstc_collector = DSTCDataCollector(dataset_path="../dstc11-track5/data", dataset_split="val", dataset_name="dstc11")
    response_indices = dstc_collector.get_samples_with_target(n=200)
    model_responses = dstc_collector.get_pred_responses(response_indices, ["chatgpt"], baseline_pred_path)

    eval_collector = DSTCHumanEvalCollector(human_eval_path=baseline_human_eval)
    pipeline_evaluator = PipelineEvaluator(framework, eval_collector, dstc_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, ["chatgpt"])
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)
