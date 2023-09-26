
from abc import ABC, ABCMeta, abstractmethod
import json
import pandas as pd
import numpy as np
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bleu_metric import BleuMetric


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
        pass


class BLEU(EvaluationFramework):
    def __init__(self):
        super().__init__(['bleu-4'], reference_required=True)

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        bleu_metric = BleuMetric()
        bleu_score = bleu_metric.evaluate_batch(model_responses, reference_responses)['bleu']
        return [{"bleu-4": bleu_score}]


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