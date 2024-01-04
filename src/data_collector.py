from abc import ABC, abstractmethod
import random
from typing import List, Tuple, Dict
import json
from itertools import groupby
from operator import itemgetter
from utils.file_processing import load_data
import pandas as pd
import re


class DataCollector(ABC):
    def __init__(self, dataset: str, dataset_split: str, dataset_name: str):
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.dataset_name = dataset_name

    def get_name(self):
        if self.dataset_name is None:
            return self.__class__.__name__
        return self.dataset_name

    @abstractmethod
    def get_samples_with_target(self, n=-1) -> Tuple[
        List[int], List[str]]:
        """
        Get all samples with target set to True.

        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: A tuple of sample indices and candidate responses.
        """
        pass

    @abstractmethod
    def collect_sample_contexts(self, sample_indices: List[int]) -> Tuple[
        List[int], List[List[str]], List[List[str]]]:
        """
        Collect sample contexts for the given sample indices.

        :param sample_indices: A list of response ids.
        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: Three lists - reference_responses, turn_historys, and knowledge_contexts.
        """
        pass

    @abstractmethod
    def get_pred_responses(self, sample_indices, model_candidates) -> List[Dict[str, str]]:
        pass


class DummyDataCollector(DataCollector):
    def __init__(self) -> None:
        super().__init__(dataset="dummy_data", dataset_split="dummy_split", dataset_name="dummy")

    def get_pred_responses(self, sample_indices, model_candidates):
        # For each entry in model_candidates, take this name as key and return random strings as values for all sample_indices
        model_responses = []
        for candidate in model_candidates:
            for sample_index in sample_indices:
                model_responses.append({candidate: "Dummy response"})
        return model_responses

    def get_samples_with_target(self, n=-1):
        sample_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if n > 0:
            return sample_indices[:n]
        return sample_indices

    def collect_sample_contexts(self, sample_indices):
        reference_responses = ["Dummy response label"] * len(sample_indices)
        turn_historys = [["Speaker A dummy question", "Speaker B dummy answer"]] * len(sample_indices)
        knowledge_contexts = [["One dummy document", "Another dummy document"]] * len(sample_indices)
        return reference_responses, turn_historys, knowledge_contexts


class BEGINDataCollector(DataCollector):
    """
    Collect sample contexts for the BEGIN benchmark datasets.
    """

    def __init__(self, dataset_path, dataset_split, dataset_name=None) -> None:
        super().__init__(dataset_path, dataset_split, dataset_name)

    def get_samples_with_target(self, n=-1, get_generic=True) -> Tuple[
        List[int], List[str]]:
        """
        Get all samples with target set to True.

        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: A tuple of sample indices and candidate responses.
        """
        sample_indices = []
        j = 0
        data = pd.read_csv(f'{self.dataset}/begin_{self.dataset_split}_{self.dataset_name}.tsv', sep='\t')
        for i in range(len(data)):
            if data.iloc[i]["response"] is not None and not (not get_generic and data.iloc[i]["begin_label"] == "Generic"):
                sample_indices.append(i)
                j += 1
            if n > 0 and j >= n:
                break
        return sample_indices

    def get_pred_responses(self, sample_indices, model_candidates=["baseline"]):
        candidate = model_candidates[0]
        pred_data = pd.read_csv(f'{self.dataset}/begin_{self.dataset_split}_{self.dataset_name}.tsv', sep='\t')
        model_responses = []
        for index in sample_indices:
            x = pred_data.iloc[index]
            model_responses.append({candidate: x["response"]})
        return model_responses

    def collect_sample_contexts(self, sample_indices: List[int]) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        reference_responses = []
        turn_historys = []
        knowledge_contexts = []
        data = pd.read_csv(f'{self.dataset}/begin_{self.dataset_split}_{self.dataset_name}.tsv', sep='\t')

        for index in sample_indices:
            cur_knowledge = data.iloc[index]["knowledge"]
            cur_message = data.iloc[index]["message"]
            if pd.isna(cur_message):
                cur_message = ""
            turn_historys.append([cur_message])
            knowledge_contexts.append([cur_knowledge])

        return reference_responses, turn_historys, knowledge_contexts


class DialDocDataCollector(DataCollector):
    """
    Collect sample contexts for the DialDoc dataset in the format of TU Braunschweig 2023."""

    def __init__(self, dataset_path) -> None:
        super().__init__(dataset=dataset_path, dataset_split="", dataset_name="dialdoc_tu_2023")

    def get_samples_with_target(self, n=-1):
        """
        Get all samples with target set to True.

        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: A tuple of sample indices and candidate responses.
        """
        sample_indices = []
        j = 0
        data = pd.read_csv(f'{self.dataset}/Batch_383409_batch_results_final.csv', sep=',')
        # Track all seen Input.Ans_id values to avoid duplicates
        seen_ans_ids = []

        for i in range(len(data)):
            if data.iloc[i]["Input.human_ref"] is not None and data.iloc[i]["Input.ex_id"] not in seen_ans_ids:
                sample_indices.append(i)
                seen_ans_ids.append(data.iloc[i]["Input.ex_id"])
                j += 1
            if n > 0 and j >= n:
                break
        return sample_indices

    def _extract_turns(self, text):
        pattern = r'<p>(USER|AGENT):\s+(.*?)<\/p>'
        turns_raw = re.findall(pattern, text)
        turns_merged = []
        current_speaker = None
        current_turn = ""

        for speaker, turn in turns_raw:
            if speaker == current_speaker:
                current_turn += " " + turn
            else:
                if current_turn:
                    turns_merged.append(current_turn)
                current_speaker = speaker
                current_turn = turn

        if current_turn:
            turns_merged.append(current_turn)

        return turns_merged

    def collect_sample_contexts(self, sample_indices: List[int]) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        reference_responses = []
        turn_historys = []
        knowledge_contexts = []
        data = pd.read_csv(f'{self.dataset}/Batch_383409_batch_results_final.csv', sep=',')

        for index in sample_indices:
            cur_knowledge = data.iloc[index]["Input.grounding_sec"]
            turn_history = self._extract_turns(data.iloc[index]["Input.dialogue_history"])
            turn_historys.append(turn_history)
            knowledge_contexts.append([cur_knowledge])
            # Remove "AGENT:" string from the start of the reference
            reference_responses.append(data.iloc[index]["Input.human_ref"].replace("AGENT: ", ""))

        return reference_responses, turn_historys, knowledge_contexts

    def get_pred_responses(self, sample_indices, model_candidates):
        # Get response entries as a list for all provided sample indices from pred_data. pred_data is a pandas dataframe with column name response
        model_responses = []
        data = pd.read_csv(f'{self.dataset}/Batch_383409_batch_results_final.csv', sep=',')#
        for index in sample_indices:
            entrys = {}
            for candidate in model_candidates:
                try:
                    entry = data.loc[(data['Input.cond_sys'] == candidate) & (data['Input.ex_id'] == data.iloc[index]['Input.ex_id'])]
                    # Take Input.response_sys from the first entry
                    entrys[candidate] = entry.iloc[0]['Input.response_sys']
                except IndexError:
                    # If there is no entry for this candidate, set it to None
                    entrys[candidate] = None
            model_responses.append(entrys)
        return model_responses
        

class DSTCDataCollector(DataCollector):
    """
    Collect sample contexts for the DSTC11 Track 5 dataset. Also compatible with DSTC9 Track 1 dataset.
    """

    def __init__(self, dataset_path, dataset_split, dataset_name=None, pred_dict=None) -> None:
        super().__init__(dataset_path, dataset_split, dataset_name)
        pred_map = {}
        for model in pred_dict:
            human_eval_path = pred_dict[model]
            pred_map[model] = load_data(human_eval_path)
        self.pred_map = pred_map

    def get_samples_with_target(self, n=-1, models=[]) -> Tuple[
        List[int], List[str]]:
        """
        Get all samples with target set to True.

        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: A tuple of sample indices and candidate responses.
        """
        candidate_responses = []
        sample_indices = []
        j = 0
        with open(f'{self.dataset}/{self.dataset_split}/labels.json') as f:
            labels = json.load(f)
            if n > 0:
                # we sample a random set of indices that have a target but with reproducible seed
                random.seed(42)
                sample_indices = sorted(random.sample([i for i in range(len(labels)) if labels[i]["target"]], n))
                for i in sample_indices:
                    candidate_responses.append(labels[i]["response"])
            else:
                for i in range(len(labels)):
                    if labels[i]["target"]:
                        candidate_responses.append(labels[i]["response"])
                        sample_indices.append(i)
                        j += 1
        return sample_indices

    def get_pred_responses(self, sample_indices, model_candidates):
        model_responses = []
        for index in sample_indices:
            entrys = {}
            for candidate in model_candidates:
                res = self.pred_map[candidate][index]["response"] if self.pred_map[candidate][index]["target"] else ""
                if isinstance(res, list):
                    res = res[0]
                entrys[candidate] = res
            model_responses.append(entrys)
        return model_responses

    def collect_sample_contexts(self, sample_indices: List[int],
                                max_n_sent=15, max_turns=15) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        reference_responses = []
        turn_historys = []
        knowledge_contexts = []

        with open(f'{self.dataset}/{self.dataset_split}/knowledge.json', encoding="utf-8") as f:
            knowledge = json.load(f)

        with open(f'{self.dataset}/{self.dataset_split}/labels.json', encoding="utf-8") as f:
            labels = json.load(f)

        with open(f'{self.dataset}/{self.dataset_split}/logs.json', encoding="utf-8") as f:
            logs = json.load(f)

        for id in sample_indices:
            # Generate sentences from knowledge.json
            sentences = []
            current_doc_id = None
            dstc9_map = False
            n_sent = 0

            q_key = "question"
            a_key = "answer"

            cur_knowledge_set = labels[id]['knowledge']
            if "doc_type" not in cur_knowledge_set[0]:
                dstc9_map = True
                for i, snippet in enumerate(cur_knowledge_set):
                    snippet["sent_id"] = 0
                    snippet["doc_type"] = "faq"
                    cur_knowledge_set[i] = snippet

            # Group the items from labels[id]['knowledge'] by the same entity_id, doc_type and doc_id
            cur_knowledge_set.sort(
                key=lambda x: (x['entity_id'], x['doc_type'], x['doc_id'], x.get('sent_id', float('-inf'))))

            # group them by entity_id, doc_type, and doc_id
            grouped_data = []
            for key, group in groupby(cur_knowledge_set, key=itemgetter('entity_id', 'doc_type', 'doc_id')):
                grouped_data.append(list(group))

            for info_sentence_set in grouped_data:
                for sentence in info_sentence_set:
                    domain = sentence['domain']
                    entity_id = str(sentence['entity_id'])
                    doc_id = str(sentence['doc_id'])
                    doc_type = str(sentence["doc_type"]) + "s"
                    if doc_type != "faqs":
                        sent_id = str(sentence['sent_id'])

                    if doc_type != "faqs":
                        text = knowledge[domain][entity_id][doc_type][doc_id]['sentences'][sent_id]
                    else:
                        if dstc9_map:
                            doc_type = "docs"
                            q_key = "title"
                            a_key = "body"
                        text = knowledge[domain][entity_id][doc_type][doc_id][q_key] + " " + \
                               knowledge[domain][entity_id][doc_type][doc_id][a_key]
                        doc_type = "faqs"
                    entity_name = str(knowledge[domain][entity_id]['name'])

                    if max_n_sent is not None and n_sent + 1 > max_n_sent:
                        break
                    n_sent += 1

                    if doc_id != current_doc_id:
                        if doc_type != "faqs":
                            sentences.append(f":R: ({entity_name}) {text}")
                        else:
                            sentences.append(f":F: ({entity_name}) {text}")
                    current_doc_id = doc_id
                    current_entity_id = entity_id

            # Get all responses of all speakers separated by the speaker's name in the logs.json
            selected_turns = []
            turns = logs[id]
            n_turns = 0
            # If max_turns is set, only use the last max_turns turns
            for log in turns[-max_turns:]:
                if log['speaker'] == 'U':
                    selected_turns.append(log['text'])
                elif log['speaker'] == 'S':
                    selected_turns.append(log['text'])
                n_turns += 1
                if max_turns is not None and n_turns + 1 > max_turns:
                    break

            label = labels[id]['response']
            reference_responses.append(label)
            turn_historys.append(selected_turns)
            knowledge_contexts.append(sentences)

        return reference_responses, turn_historys, knowledge_contexts


class TopicalChatDataCollector(DataCollector):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, dataset_split="", dataset_name="TopicalChat_UE")

    def get_samples_with_target(self, n=-1) -> Tuple[List[int], List[str]]:
        """
        Get all samples which are intended to be used for response generation.

        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: A tuple of sample indices and candidate responses.
        """
        sample_indices = []
        j = 0
        with open(f'{self.dataset}/restructured.json') as f:
            labels = json.load(f)
            for i in range(len(labels)):
                if labels[str(i)]["Original Ground Truth"]["context"] != "_nofact\n":
                    sample_indices.append(i)
                    j += 1
                if n > 0 and j >= n:
                    break
        return sample_indices

    def get_pred_responses(self, sample_indices, model_candidates):
        pred_data = load_data(f'{self.dataset}/restructured.json')
        model_responses = []
        for index in sample_indices:
            entrys = {}
            for model in model_candidates:
                x = pred_data[f"{index}"]
                entrys[model] = x[model]["system_output"]
            model_responses.append(entrys)
        return model_responses

    def collect_sample_contexts(self, sample_indices: List[int]) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        reference_responses = []
        turn_historys = []
        knowledge_contexts = []
        with open(f'{self.dataset}/restructured.json') as f:
            data = json.load(f)
            for index in sample_indices:
                cur_knowledge = data[str(index)]["Original Ground Truth"]['context']
                turn_history = data[str(index)]["Original Ground Truth"]['source']
                if cur_knowledge == "_nofact\n":
                    cur_knowledge = ""
                # turn historys are separated by \n. the ending is marked by \n\n so it will look like
                # hello \n who are you? \n\n
                turn_history = turn_history[:-2].split("\n")
                
                # for knowledge we need to remove the ending marked by \n
                knowledge_contexts.append([cur_knowledge[:-1]])
                reference_responses.append(data[str(index)]["Original Ground Truth"]["system_output"])
                turn_historys.append(turn_history)
        return reference_responses, turn_historys, knowledge_contexts