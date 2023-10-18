import json

# Load the JSON file
with open('setups/topicalchat/topical_chat.json', 'r') as f:
    data = json.load(f)

# Initialize a dictionary to hold the restructured data
restructured_data = {}

# Initialize variables to keep track of the current data sample index and system
current_data_sample_index = 0
previous_system_id = data[0]['system_id']

# Number of systems
num_systems = 6

# Loop through the data
for i, entry in enumerate(data):
    # Extract the system id from the entry
    system_id = entry['system_id']

    # If the system id is the same as the first one and it's not the first entry, increment the data sample index
    if i % num_systems == 0 and i != 0:
        current_data_sample_index += 1

    # Add the entry to the appropriate data sample in the restructured data
    if current_data_sample_index not in restructured_data:
        restructured_data[current_data_sample_index] = {}
    restructured_data[current_data_sample_index][system_id] = entry

# Save the restructured data to a new JSON file
with open('restructured.json', 'w') as f:
    json.dump(restructured_data, f, indent=4)