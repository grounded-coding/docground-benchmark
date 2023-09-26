from src.data_collector import DSTCDataCollector
import json
from tqdm import tqdm
import random
from typing import List, Dict


if __name__ == "__main__":
    dstc_collector = DSTCDataCollector(dataset="../dstc11-track5/data", dataset_split="val")

    sample_indices = dstc_collector.get_samples_with_target()
    reference_responses, turn_historys, knowledge_contexts = dstc_collector.collect_sample_contexts(sample_indices)

    print(reference_responses)
    print(turn_historys)
    print(knowledge_contexts)