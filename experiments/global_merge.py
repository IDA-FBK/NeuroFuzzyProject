import pandas as pd
import os

# CHANGE with the correct path containing the csv
path = '/results/maternal_hr'

dataframes = []

# Read all csv
for filename in os.listdir(path):
    if filename.startswith('global_result'):
        filepath = os.path.join(path, filename)
        df = pd.read_csv(filepath)
        dataframes.append(df)

# Merge them in a single file
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the df as csv
merged_df.to_csv('merged_global_results.csv', index=False)

print("Merge completed. File saved as 'merged_global_results.csv'.")
