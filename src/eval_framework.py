
from abc import ABC, ABCMeta, abstractmethod
from utils.file_processing import convert_to_json
import pandas as pd
import numpy as np
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bleu_metric import BleuMetric
from nltk.translate.bleu_score import sentence_bleu
from lleval.scorer import PromptScorer
from lleval.evaluator import PromptTemplate, DialogEvaluator
from collections import Counter
from uni_eval.evaluator import get_evaluator
from rouge_score import rouge_scorer
from src.openai_scorer import OpenAIScorer
from dotenv import load_dotenv

load_dotenv()

class EvaluationFramework(ABC):
    def __init__(self, available_dimensions, reference_required=False, name=None):
        """
        :param available_dimensions: list of dimensions that this framework can evaluate
        :param reference_required: whether this framework requires a reference response to evaluate
        :param name: Name of the framework, will be used for storing results
        """
        self.available_dimensions = available_dimensions
        self.reference_required = reference_required
        self.name = name

    @abstractmethod
    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims) -> list[dict]:
        """
        :param model_responses: list of model responses which are strings
        :param reference_responses: list of reference responses which are strings
        :param turn_historys: list of turn historys which are lists of strings
        :param knowledge_contexts: list of knowledge contexts which are lists of strings
        :param dims: list of dimensions to evaluate, must be a subset of self.available_dimensions
        :return: list of dictionaries containing the scores where the keys are the dimensions

        This method should call the evaluation framework and return the scores for the specified dimensions on each model response.
        """
        pass

    def get_name(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__


class UniEval(EvaluationFramework):
    def __init__(self):
        super().__init__(['naturalness', 'coherence', 'groundedness', 'understandability', 'overall'])

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        # knowledge_contexts is a list of lists of strings. for unieval these documents are concatenated and joined by a newline separator
        knowledge_contexts = ['\n'.join(context) for context in knowledge_contexts]
        # turn_historys is a list of lists of strings. for unieval these turns are concatenated and joined by a newline separator. at the very end we attach 2 newline separators
        turn_historys = ['\n'.join(turns) + '\n\n' for turns in turn_historys]
        data = convert_to_json(output_list=model_responses, src_list=turn_historys, context_list=knowledge_contexts)
        evaluator = get_evaluator("dialogue")
        eval_scores = evaluator.evaluate(data, dims=dims, print_result=False)
        return eval_scores


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
    def __init__(self, gen_config="configs/llama2/gen_config.json", dim_definitions="configs/dimension_definitions.json",
                 api_url="http://gpu-21.apptek.local:8080/generate", likert_config="configs/llama2/prompt_likert_config.json", name=None):
        super().__init__(['appropriate', 'accurate', "grounded", "coherent"], name=name)
        self.dim_definitions = dim_definitions
        self.gen_config = gen_config
        self.api_url = api_url
        self.likert_config = likert_config

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        data = convert_to_json(output_list=model_responses, src_list=turn_historys, context_list=knowledge_contexts)
        prompt_template = PromptTemplate(self.likert_config)
        llama2local = PromptScorer(api_url=self.api_url, metric_config_file=self.gen_config, prompt_template=prompt_template, num_retries=3)
        evaluator = DialogEvaluator(llama2local, dimension_definitions_file=self.dim_definitions)
        eval_scores, eval_expls = evaluator.evaluate(data, print_result=True, dims=dims)
        merged_scores = []
        for i, score in enumerate(eval_scores):
            for key in eval_expls[i].keys():
                score[key + "_expl"] = eval_expls[i][key]
            merged_scores.append(score)
        return merged_scores


class GPTEval(EvaluationFramework):
    def __init__(self, dim_definitions="configs/dimension_definitions.json", name="GPT4Eval", gpt_model="gpt-4-1106-preview"):
        super().__init__(['appropriate', 'accurate', 'grounded', 'coherent'], name=name)
        self.gpt_model = gpt_model
        self.dim_definitions = dim_definitions

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        data = convert_to_json(output_list=model_responses, src_list=turn_historys, context_list=knowledge_contexts)
        prompt_template = PromptTemplate("configs/gpt3/prompt_likert_config.json")
        gptmodel = OpenAIScorer(metric_config_file="configs/gpt3/gen_config.json", prompt_template=prompt_template, gpt_model=self.gpt_model, num_retries=3)
        evaluator = DialogEvaluator(gptmodel, dimension_definitions_file=self.dim_definitions)
        eval_scores, eval_expls = evaluator.evaluate(data, print_result=True, dims=dims)
        merged_scores = []
        for i, score in enumerate(eval_scores):
            for key in eval_expls[i].keys():
                score[key + "_expl"] = eval_expls[i][key]
            merged_scores.append(score)
        return merged_scores


class SimpleEvaluationFramework(EvaluationFramework):
    """
    A simple evaluation framework does not need a GPU or parallelization to run in real-time.
    Thus we can simply loop for calculating batch scores.
    """
    @abstractmethod
    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        """
        :param model_response: string
        :param reference_response: string
        :param turn_history: list of strings
        :param knowledge_context: list of strings
        :return: dictionary containing the scores where the keys are the dimensions
        """
        pass

    def evaluate(self, model_responses, reference_responses, turn_historys, knowledge_contexts, dims):
        scores = []
        for response, reference, turn_history, knowledge_context in zip(model_responses, reference_responses, turn_historys, knowledge_contexts):
            assert isinstance(response, str)
            assert isinstance(reference, str)
            assert isinstance(turn_history, list)
            assert isinstance(knowledge_context, list)
            score = self.evaluate_example(response, reference, turn_history, knowledge_context, dims)
            scores.append(score)
        return scores
    

class KnowledgeF1(SimpleEvaluationFramework):
    def __init__(self):
        super().__init__(['knowledge-f1'], reference_required=False)

    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        score = {}
        for dim in dims:
            if dim == "knowledge-f1":
                score["knowledge-f1"] = np.mean([self._f1(model_response, doc) for doc in knowledge_context])
            else:
                raise NotImplementedError("Unknown dimension")
        return score
        
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


class BLEU(SimpleEvaluationFramework):
    def __init__(self):
        super().__init__(['bleu-4', 'bleu-1'], reference_required=True)

    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        bleu_metric = BleuMetric()
        score = {}
        for dim in dims:
            if dim == "bleu-4":
                score[dim] = bleu_metric.evaluate_example(model_response, reference_response)['bleu']
            else:
                raise NotImplementedError("BLEU-n needs a different tokenization strategy")
        return score


class SeqLen(SimpleEvaluationFramework):
    def __init__(self):
        super().__init__(['seq-len'], reference_required=False)

    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        score = {}
        for dim in dims:
            if dim == "seq-len":
                score[dim] = len(model_response)
            else:
                raise NotImplementedError("Unknown dimension")
        return score
    

class KnowledgeBLEU(SimpleEvaluationFramework):
    def __init__(self):
        super().__init__(['knowledge-bleu-4', 'knowledge-bleu-1'], reference_required=False)

    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        bleu_metric = BleuMetric()
        score = {}
        for dim in dims:
            if dim == "knowledge-bleu-4":
                score[dim] = np.mean([bleu_metric.evaluate_example(model_response, doc)['bleu'] for doc in knowledge_context])
            else:
                raise NotImplementedError("BLEU-n needs a different tokenization strategy")
        return score


class METEOR(SimpleEvaluationFramework):
    def __init__(self):
        super().__init__(['meteor'], reference_required=True)

    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        met_metric = MeteorMetric()
        score = {}
        for dim in dims:
            if dim == "meteor":
                score["meteor"] = met_metric.evaluate_example(model_response, reference_response)['meteor']
            else:
                raise NotImplementedError("Unknown dimension")
        return score


class ROUGE(SimpleEvaluationFramework):
    def __init__(self):
        super().__init__(['rougeL','rouge-1','rouge-2'], reference_required=True)

    def evaluate_example(self, model_response, reference_response, turn_history, knowledge_context, dims):
        scorer = rouge_scorer.RougeScorer(dims, use_stemmer=True)
        scores = scorer.score(reference_response, model_response)
        rouge_scores = {}
        for dim in dims:
            rouge_scores[dim] = scores[dim].fmeasure * 100
        return rouge_scores