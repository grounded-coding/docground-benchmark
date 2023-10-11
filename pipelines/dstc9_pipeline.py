import sys
sys.path.append('/home/nhilgers/setups/DocGroundEval')

from src.data_collector import DSTCDataCollector, DummyDataCollector, BEGINDataCollector, DialDocDataCollector, TopicalChatDataCollector
from src.pipeline_evaluator import PipelineEvaluator
from src.eval_framework import DummyEval, LLEval, KnowledgeF1, BLEU, UniEval, KnowledgeBLEU, GEval
from src.eval_collector import DummyEvalCollector, DSTCHumanEvalCollector, BEGINHumanEvalCollector, DialDocEvalCollector, TopicalChatEvalCollector
import json
from utils.file_processing import load_data
from typing import List, Dict


pred_dict = {"baseline": "../dstc11-track5/results/results_dstc9/baseline/entry0.json",
"entry031": "../dstc11-track5/results/results_dstc9/team03/entry1.json",
"entry100": "../dstc11-track5/results/results_dstc9/team10/entry0.json",
"entry103": "../dstc11-track5/results/results_dstc9/team13/entry3.json",
"entry107": "../dstc11-track5/results/results_dstc9/team17/entry0.json",
"entry109": "../dstc11-track5/results/results_dstc9/team19/entry2.json",
"entry110": "../dstc11-track5/results/results_dstc9/team20/entry4.json",
"entry111": "../dstc11-track5/results/results_dstc9/team21/entry3.json",
"entry113": "../dstc11-track5/results/results_dstc9/team11/entry3.json",
"entry115": "../dstc11-track5/results/results_dstc9/team15/entry3.json",
"entry118": "../dstc11-track5/results/results_dstc9/team18/entry3.json",
"entry123": "../dstc11-track5/results/results_dstc9/team23/entry0.json"}
eval_paths = {"baseline": "../dstc11-track5/results/results_dstc9/baseline/entry0.human_eval.json",
            "entry031": "../dstc11-track5/results/results_dstc9/team03/entry1.human_eval.json",
                "entry100": "../dstc11-track5/results/results_dstc9/team10/entry0.human_eval.json",
"entry103": "../dstc11-track5/results/results_dstc9/team13/entry3.human_eval.json",
"entry107": "../dstc11-track5/results/results_dstc9/team17/entry0.human_eval.json",
"entry109": "../dstc11-track5/results/results_dstc9/team19/entry2.human_eval.json",
"entry110": "../dstc11-track5/results/results_dstc9/team20/entry4.human_eval.json",
"entry111": "../dstc11-track5/results/results_dstc9/team21/entry3.human_eval.json",
"entry113": "../dstc11-track5/results/results_dstc9/team11/entry3.human_eval.json",
"entry115": "../dstc11-track5/results/results_dstc9/team15/entry3.human_eval.json",
"entry118": "../dstc11-track5/results/results_dstc9/team18/entry3.human_eval.json",
"entry123": "../dstc11-track5/results/results_dstc9/team23/entry0.human_eval.json"}

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

model_candidates = ["baseline", "entry031", "entry100", "entry103", "entry107", "entry109", "entry110", "entry111", "entry113", "entry115", "entry118", "entry123"]

for framework, framework_dimensions in [(lleval, lleval_dimensions), (unieval, unieval_dimensions), (kf1, kf1_dimensions), (bleu, bleu_dimensions)]:
    ## DSTC9
    framework_to_human_dimension_map = {framework_dimensions[0]: "accuracy"}
    if len(framework_dimensions) > 1:
        framework_to_human_dimension_map[framework_dimensions[1]] = "appropriateness"

    dstc_collector = DSTCDataCollector(dataset_path="../dstc11-track5/data/dstc9", dataset_split="test", dataset_name="dstc9", pred_dict=pred_dict)
    response_indices = dstc_collector.get_samples_with_target(n=200)
    model_responses = dstc_collector.get_pred_responses(response_indices, model_candidates)

    eval_collector = DSTCHumanEvalCollector(human_eval_paths=eval_paths)

    # Sample level
    pipeline_evaluator = PipelineEvaluator(framework, eval_collector, dstc_collector, framework_dimensions, framework_to_human_dimension_map, type, correlation_level, ["baseline"])
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)

    # Now for system level
    pipeline_evaluator = PipelineEvaluator(framework, eval_collector, dstc_collector, framework_dimensions, framework_to_human_dimension_map, type, "system", model_candidates)
    human_framework_correlations = pipeline_evaluator.run_pipeline(model_responses, response_indices)