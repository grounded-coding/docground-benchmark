import json
import csv
from collections import defaultdict

# Load the data from the JSON files
with open('outputs/TopicalChat_UE/Original Ground Truth/UniEval.json', 'r') as f:
    unieval_data = json.load(f)

with open('outputs/TopicalChat_UE/Original Ground Truth/GEval.json', 'r') as f:
    lleval_data = json.load(f)

with open('../topicalchat/restructured.json', 'r') as f:
    human_data = json.load(f)

# Create a dictionary to hold the merged data
merged_data = defaultdict(dict)

# Add data from the first JSON file
for item in unieval_data:
    index = item['response_index']
    merged_data[index]['unieval_groundedness'] = item.get('groundedness')
    merged_data[index]['unieval_coherence'] = item.get('coherence')

# Add data from the second JSON file
for item in lleval_data:
    index = item['response_index']
    if index in merged_data:
        merged_data[index]['geval_appropriate'] = item.get('appropriate')
        merged_data[index]['geval_grounded'] = item.get('grounded')
        merged_data[index]['geval_grounded_expl'] = item.get('grounded_expl')

# Add data from human file
for item in human_data:
    index = int(item)
    if index in merged_data:
        merged_data[index]['ref_grounded'] = human_data[item]["Original Ground Truth"]["scores"]["groundedness"]

# Write the merged data to a CSV file
with open('merged.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sample_id', 'ref_grounded', 'unieval_groundedness', 'unieval_coherence', 'geval_appropriate', 'geval_grounded', 'geval_grounded_expl'])
    writer.writeheader()
    for index, data in merged_data.items():
        writer.writerow({'sample_id': index, **data})