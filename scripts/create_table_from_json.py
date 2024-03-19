import json
import csv

# This script helps you to convert the JSON output of the evaluation script to a CSV file for easy import to 
# a sheet or a database. The script reads the JSON file and writes the data to a CSV file.

def write_evaluations_to_csv(json_file_path, csv_file_path):
    # Read the JSON data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Open a CSV file for writing
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        headers = ['response_index', 'accurate', 'appropriate', 'overall', 'accurate_expl', 'appropriate_expl']
        writer.writerow(headers)

        # Write data rows
        for entry in data:
            row=[]
            for header in headers:
                cur_entry = entry.get(header, '')
                if len(header.split('_')) > 1 and header.split('_')[1] == 'expl':
                    cur_entry = entry.get(header, '')
                    cur_entry = cur_entry.replace('Explanation: ', '').rstrip("\"").lstrip("\"")
                row.append(cur_entry)
            writer.writerow(row)

def write_prediction_to_csv(json_file_path, csv_file_path):
    # Read the JSON data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Open a CSV file for writing
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        headers = ['response_index', 'response']
        writer.writerow(headers)

        # Write data rows
        for index, entry in enumerate(data):
            if entry.get('target') is True:
                response = entry.get('response', [''])
                if isinstance(response, list):
                    response = response[0]
                writer.writerow([index, response])


# Usage example
json_file_paths = ['outputs/dstc11/val/llama-7b-ft/GPT4Eval.json',
                    'outputs/dstc11/val/bart-baseline-ft/GPT4Eval.json',
                    'outputs/dstc11/val/llama-7b-ft-filtered/GPT4Eval.json',
                    'outputs/dstc11/val/humanref/GPT4Eval.json',
                    'outputs/dstc11/train/humanref/GPT4Eval.json'
                    ]

for json_file_path in json_file_paths:
    csv_file_path = json_file_path.replace('.json', '.csv')

    write_evaluations_to_csv(json_file_path, csv_file_path)

# Usage example
pred_json_file_paths = ['datasets/dstc11-track5/pred/val/rg.llama-7b-peft-opt-0104103455.json',
                        'datasets/dstc11-track5/pred/val/rg.llama-7b-peft-opt-0131151009.json',
                        'datasets/dstc11-track5/pred/val/rg.special_tok/bart-base-baseline-0108110257.json',
                        'datasets/dstc11-track5/data/val/labels.json',
                        'datasets/dstc11-track5/data/train/labels.json'] 

for pred_json_file_path in pred_json_file_paths:
    pred_csv_file_path = pred_json_file_path.replace('.json', '.csv')

    write_prediction_to_csv(pred_json_file_path, pred_csv_file_path)