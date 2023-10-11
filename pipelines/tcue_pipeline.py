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
model_candidates = ['baseline']
    
## TopicalChat UE
for framework, framework_dimensions in [(lleval, lleval_dimensions), (unieval, unieval_dimensions), (kf1, kf1_dimensions), (bleu, bleu_dimensions)]:
    framework_to_human_dimension_map = {framework_dimensions[0]: "groundedness"}
    if len(framework_dimensions) > 1:
        framework_to_human_dimension_map[framework_dimensions[1]] = "coherence"
    topicalchat_ue_collector = TopicalChatDataCollector("../topicalchat")
    response_indices = topicalchat_ue_collector.get_samples_with_target(n=200)
    model_responses = topicalchat_ue_collector.get_pred_responses(response_indices, model_candidates)

    tc_ue_eval_collector = TopicalChatEvalCollector("../topicalchat/topical_chat.json")
    pipeline_evaluator = PipelineEvaluator(framework, tc_ue_eval_collector, topicalchat_ue_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)