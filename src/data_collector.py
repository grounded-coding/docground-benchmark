from abc import ABC, abstractmethod
from typing import List, Tuple
import json
from itertools import groupby
from operator import itemgetter
from utils.file_processing import load_data
import pandas as pd


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


class DummyDataCollector(DataCollector):
    def __init__(self) -> None:
        super().__init__(dataset="dummy_data", dataset_split="dummy_split", dataset_name="dummy")

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
        candidate_responses = []
        sample_indices = []
        j = 0
        data = pd.read_csv(f'{self.dataset}/begin_{self.dataset_split}_{self.dataset_name}.tsv', sep='\t')
        for i in range(len(data)):
            if data.iloc[i]["response"] is not None and not (not get_generic and data.iloc[i]["begin_label"] == "Generic"):
                candidate_responses.append(data.iloc[i]["response"])
                sample_indices.append(i)
                j += 1
            if n > 0 and j >= n:
                break
        return sample_indices

    def get_pred_responses(self, sample_indices, model_candidates=["baseline"]):
        candidate = model_candidates[0]
        pred_data = pd.read_csv(f'{self.dataset}/begin_{self.dataset_split}_{self.dataset_name}.tsv', sep='\t')
        # Get response entries as a list for all provided sample indices from pred_data. pred_data is a pandas dataframe with column name response
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
            # the pandas dataframe contains a knowledge column with one knowledge document
            # it also contains  a message column with a single-turn turn history
            # there are no reference responses
            cur_knowledge = data.iloc[index]["knowledge"]
            cur_message = data.iloc[index]["message"]
            # if cur_message is NaN, replace it with an empty string
            if pd.isna(cur_message):
                cur_message = ""
            turn_historys.append([cur_message])
            knowledge_contexts.append([cur_knowledge])

        return reference_responses, turn_historys, knowledge_contexts


class DSTCDataCollector(DataCollector):
    """
    Collect sample contexts for the DSTC11 Track 5 dataset. Also compatible with DSTC9 Track 1 dataset.
    """

    def __init__(self, dataset, dataset_split, dataset_name=None) -> None:
        super().__init__(dataset, dataset_split, dataset_name)

    def get_samples_with_target(self, n=-1) -> Tuple[
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
            for i in range(len(labels)):
                if labels[i]["target"]:
                    candidate_responses.append(labels[i]["response"])
                    sample_indices.append(i)
                    j += 1
                if n > 0 and j >= n:
                    break
        return sample_indices

    def get_pred_responses(self, sample_indices, model_candidates, pred_path):
        candidate = model_candidates[0]
        pred_data = load_data(pred_path)
        model_responses = []
        for index in sample_indices:
            x = pred_data[index]
            model_responses.append({candidate: x["response"]} if x["target"] else None)
        return model_responses

    def collect_sample_contexts(self, sample_indices: List[int],
                                max_n_sent=10, max_turns=10) -> Tuple[List[int], List[List[str]], List[List[str]]]:
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
            current_entity_id = None
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