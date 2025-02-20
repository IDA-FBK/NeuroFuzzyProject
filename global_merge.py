import pandas as pd
import os

path = 'experiments/results/diabetes1'

dataframes = []

# Leggi tutti i file CSV nella directory
for filename in os.listdir(path):
    if filename.startswith('global_result'):
        filepath = os.path.join(path, filename)
        df = pd.read_csv(filepath)
        dataframes.append(df)

# Unisci tutti i DataFrame in uno solo
merged_df = pd.concat(dataframes, ignore_index=True)

# Salva il DataFrame unito in un nuovo file CSV
merged_df.to_csv('merged_global_results.csv', index=False)

print("Unione completata. File salvato come 'merged_global_results.csv'.")