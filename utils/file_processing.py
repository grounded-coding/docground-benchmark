import json

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data