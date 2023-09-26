from abc import ABC, abstractmethod
from typing import List, Tuple
import json
from itertools import groupby
from operator import itemgetter


class DataCollector(ABC):
    def __init__(self, dataset: str, dataset_split: str):
        self.dataset = dataset
        self.dataset_split = dataset_split

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
        super().__init__(dataset="dummy_data", dataset_split="dummy_split")

    def collect_sample_contexts(self, sample_indices):
        reference_responses = ["Dummy response label"] * len(sample_indices)
        turn_historys = [["Speaker A dummy question", "Speaker B dummy answer"]] * len(sample_indices)
        knowledge_contexts = [["One dummy document", "Another dummy document"]] * len(sample_indices)
        return reference_responses, turn_historys, knowledge_contexts


class DSTCDataCollector(DataCollector):
    """
    Collect sample contexts for the DSTC11 Track 5 dataset. Also compatible with DSTC9 Track 1 dataset.
    """

    def __init__(self, dataset, dataset_split) -> None:
        super().__init__(dataset, dataset_split)

    def get_samples_with_target(self,) -> Tuple[
        List[int], List[str]]:
        """
        Get all samples with target set to True.

        :param dataset_split: The dataset split to use.
        :param dataset: The dataset to use.
        :return: A tuple of sample indices and candidate responses.
        """
        candidate_responses = []
        sample_indices = []
        with open(f'{self.dataset}/{dataset_split}/labels.json') as f:
            data = json.load(f)
            for i in range(len(data)):
                if data[i]["target"]:
                    candidate_responses.append(data[i]["response"])
                    sample_indices.append(i)
        return sample_indices

    def collect_sample_contexts(self, sample_indices: List[int],
                                max_n_sent=10, max_turns=10) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        reference_responses = []
        turn_historys = []
        knowledge_contexts = []

        with open(f'{self.base_path}/{dataset_split}/knowledge.json', encoding="utf-8") as f:
            knowledge = json.load(f)

        with open(f'{self.base_path}/{dataset_split}/labels.json', encoding="utf-8") as f:
            labels = json.load(f)

        with open(f'{self.base_path}/{dataset_split}/logs.json', encoding="utf-8") as f:
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