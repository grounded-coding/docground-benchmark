
from abc import ABC, ABCMeta, abstractmethod
from utils.file_processing import convert_to_json
import pandas as pd
import numpy as np
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bleu_metric import BleuMetric
from lleval.scorer import PromptScorer
from collections import Counter
from lleval.evaluator import PromptTemplate, DialogEvaluator


class EvaluationFramework(ABC):
    def __init__(self, available_dimensions, reference_required=False, sample_level_support=False):
        self.available_dimensions = available_dimensions
        self.sample_level_support = sample_level_support
        self.reference_required = reference_required

    @abstractmethod
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass

    def get_name(self):
        return self.__class__.__name__


class UniEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['groundedness', 'informativeness', 'fluency', 'engagingness', 'overall'])

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass


class DummyEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['accur', 'app'])

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        scores = []
        for response in model_responses:
            assert isinstance(response, str)
            score = {}
            for dim in dims:
                score[dim] = np.random.rand()
            scores.append(score)
        return scores


class LLEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['appropriate', 'accurate'])

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        data = convert_to_json(output_list=model_responses, src_list=turn_historys, context_list=knowledge_contexts)
        prompt_template = PromptTemplate()
        llama2local = PromptScorer(api_url="http://gpu-19.apptek.local:8080/generate", metric_config_file="metric_likert_config.json", prompt_template=prompt_template, num_retries=3)
        evaluator = DialogEvaluator(llama2local)
        eval_scores, eval_expls = evaluator.evaluate(data, print_result=True)
        return eval_scores


class KnowledgeF1(EvaluationFramework):
    def __init__(self):
        super().__init__(['knowledge-f1'], reference_required=True)

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        # Knowledge F1 is the average tokenlevel F1 overlap of the generated response and each document in the knowledge context
        scores = []
        for response, context in zip(model_responses, knowledge_contexts):
            assert isinstance(response, str)
            assert isinstance(context, list)
            score = {}
            for dim in dims:
                score[dim] = np.mean([self._f1(response, doc) for doc in context])
            scores.append(score)
        return scores
        
    def _f1(self, response, doc):
        # Compute token-level F1 overlap between response and document

        response_tokens = Counter(response.lower().split())
        doc_tokens = Counter(doc.lower().split())

        common_tokens = response_tokens & doc_tokens
        num_common = sum(common_tokens.values())

        if num_common == 0:
            return 0

        precision = num_common / float(sum(response_tokens.values()))
        recall = num_common / float(sum(doc_tokens.values()))

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

class BLEU(EvaluationFramework):
    def __init__(self):
        super().__init__(['bleu-4'], reference_required=True)

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        bleu_metric = BleuMetric()
        scores = []
        for response, reference in zip(model_responses, reference_responses):
            assert isinstance(response, str)
            score = {}
            score["bleu-4"] = bleu_metric.evaluate_example(response, reference)['bleu']
            scores.append(score)
        return scores


class METEOR(EvaluationFramework):
    def __init__(self, reference_required=True):
        super().__init__(['meteor'], reference_required=reference_required)

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        met_metric = MeteorMetric()
        met_score = met_metric.evaluate_batch(model_responses, reference_responses)['meteor'] * 100
        return met_score


class ROUGE(EvaluationFramework):
    def __init__(self, reference_required=True):
        super().__init__(['rouge-l','rouge-1','rouge-2'], reference_required=reference_required)

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        pass