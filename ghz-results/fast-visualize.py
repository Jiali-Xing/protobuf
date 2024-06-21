import json
import os
import pandas as pd
import subprocess
import sys
import numpy as np

def find_experiment_file(params_file, load, csv_file='ghz-results/alibaba-experiment_results.csv'):
    # Read the Bayesian optimization parameter file
    with open(params_file, 'r') as f:
        params_data = json.load(f)

    # Extract the relevant parameters
    method = params_data['method']
    parameters = params_data['parameters']

    # Load the experiment results CSV
    df = pd.read_csv(csv_file)

    # Filter the dataframe based on the parameters and load
    df_filtered = df[(df['interface'] == method) & (df['capacity'] == load)]

    for param, value in parameters.items():
        if param not in df.columns:
            continue
        # if param == 'LAZY_UPDATE' or param == 'ONLY_FRONTEND':
        if value == 'true' or value == 'false':
            continue
        if pd.api.types.is_float_dtype(df_filtered[param]):
            df_filtered = df_filtered[np.isclose(df_filtered[param], float(value), atol=1e-8)]
        else:
            df_filtered = df_filtered[df_filtered[param] == value]

    # Check if we have a single matching file
    if len(df_filtered) == 1:
        experiment_file = df_filtered.iloc[0]['filename']
        return experiment_file
    elif len(df_filtered) == 0:
        print(f"Error: Found {len(df_filtered)} matching files.")
        return None
    else:
        # randomly select one of the matching files
        experiment_file = df_filtered.sample()['filename'].values[0]
        print(f"Warning: Found {len(df_filtered)} matching files. Randomly selecting {experiment_file}.")
        return experiment_file

def run_visualize(experiment_file):
    # Run visualize.py with the found experiment file
    visualize_file = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/visualize.py')
    subprocess.run(['python', visualize_file, experiment_file])

def main():
    if len(sys.argv) != 3:
        print("Usage: python find_and_visualize.py <bopt_file.json> <Load>")
        return

    params_file = sys.argv[1]
    load = int(sys.argv[2])
    if 'S_' in params_file in params_file:
        experiment_file = find_experiment_file(params_file, load, 'ghz-results/all-alibaba-experiment_results.csv')
    elif 'motivate' in params_file:
        experiment_file = find_experiment_file(params_file, load, 'ghz-results/4-nodes-monotonic-experiment_results.csv')
    elif 'hotel' in params_file:
        experiment_file = find_experiment_file(params_file, load, 'ghz-results/hotel-experiment_results.csv')
    else:
        experiment_file = find_experiment_file(params_file, load, 'ghz-results/hotel-experiment_results.csv')

    if experiment_file:
        print(f"Found experiment file: {experiment_file}")
        # try:
        run_visualize(experiment_file)

    else:
        print("No matching experiment file found.")

if __name__ == "__main__":
    main()
