import sys
sys.path.append('/home/nhilgers/setups/DocGroundEval')
from src.data_collector import DSTCDataCollector, DummyDataCollector, BEGINDataCollector, DialDocDataCollector, TopicalChatDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval, KnowledgeBLEU, GEval
from src.eval_collector import DummyEvalCollector, DSTCHumanEvalCollector, BEGINHumanEvalCollector, DialDocEvalCollector, TopicalChatEvalCollector
import json
from utils.file_processing import load_data
from typing import List, Dict


type = 'spearman'
correlation_level = 'sample'
unieval = UniEval()
unieval_dimensions = ["groundedness", "coherence"]
geval = GEval()
lleval = LLEval()
lleval_dimensions = ["accurate", "appropriate"]
bleu = KnowledgeBLEU()
bleu_dimensions = ["knowledge-bleu-4"]
kf1 = KnowledgeF1()
kf1_dimensions = ["knowledge-f1"]
model_candidates = ["gptllama"]

for framework, framework_dimensions in [(lleval, lleval_dimensions), (unieval, unieval_dimensions), (kf1, kf1_dimensions), (bleu, bleu_dimensions)]:
    ## DialDoc
    dialdoc_collector = DialDocDataCollector("../DialDoc-TU2023")
    response_indices = dialdoc_collector.get_samples_with_target(n=-1)
    model_responses = dialdoc_collector.get_pred_responses(response_indices, model_candidates)

    framework_to_human_dimension_map = {framework_dimensions[0]: "groundedness"}
    if len(framework_dimensions) > 1:
        framework_to_human_dimension_map[framework_dimensions[1]] = "appropriateness"
    dialdoc_eval_collector = DialDocEvalCollector("../DialDoc-TU2023/Batch_383409_batch_results_final.csv")
    pipeline_evaluator = PipelineEvaluator(framework, dialdoc_eval_collector, dialdoc_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)