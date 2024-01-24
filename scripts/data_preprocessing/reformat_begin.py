import pandas as pd

# Read the CSV file
df = pd.read_csv('setups/BEGIN-dataset/topicalchat/begin_dev_tc.tsv', sep='\t')

# group by same message and knowledge
group = df.groupby(['message', 'knowledge'])
# print any group that has more than one entry
for name, groups in group:
    if len(groups) > 1:
        print(name)
df['sample_index'] = group.ngroup()


# Write the DataFrame back to a new CSV file
df.to_csv('output.csv', index=False, sep='\t')