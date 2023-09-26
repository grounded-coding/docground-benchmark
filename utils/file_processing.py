import json

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def convert_to_json(output_list, src_list=None, ref_list=None, context_list=None, \
            scores=None, doc_id=None, system_id=None):
    """
        Convert the data into the json format.

        output_list: a list of model output
        src_list: source input for different NLG tasks. For example, source document for summarization 
                  and dialogue history for dialogue response generation
        ref_list: human-annotated groundtruth
        context_list: the context needed to evaluate several specific dimension. For example,
                      additional factual information when evaluating engagingness and groundedness in dialogues
        scores: human scores for evaluating the model output. They can be used to calculate the correlation
                between evaluators and human judgements. The scores should be stored in a dictionary. For example,
                {'fluency': 2.0, 'coherence': 3.0} could be the human score for a sample.
        doc_id: the index of the input source. It can be used to calculate summary-level correlation for summarzation
        system_id: the index of the generation system. It can be used to calculate system-level correlation.
    """
    json_data = []
    for i in range(len(output_list)):
        cur = {}
        cur['system_output'] = output_list[i]
        if src_list is not None:
            cur['source'] = src_list[i]
        if ref_list is not None:
            cur['reference'] = ref_list[i]
        if context_list is not None:
            cur['context'] = context_list[i]
        if scores is not None:
            cur['scores'] = scores[i]
        if doc_id is not None:
            cur['doc_id'] = doc_id[i]
        if system_id is not None:
            cur['system_id'] = system_id[i]
        json_data.append(cur)
    return json_data