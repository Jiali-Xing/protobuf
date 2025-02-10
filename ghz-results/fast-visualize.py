import json
import os
import pandas as pd
import subprocess
import argparse
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/z/large-nsdi/results'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/z/large-nsdi/old-runs'))


def find_experiment_file(params_file, load, csv_file='alibaba-experiment_results.csv', method=None, timestamp=None, concurrent=False):
    # Read the Bayesian optimization parameter file
    with open(params_file, 'r') as f:
        params_data = json.load(f)

    # Extract the relevant parameters
    if method is None:
        method = params_data['method']
    
    if method == "all-alibaba":
        method = "S_149998854"

    parameters = params_data['parameters']

    path = os.path.expanduser('~/Sync/Git/protobuf/ghz-results')
    # Load the experiment results CSV
    df = pd.read_csv(os.path.join(path, csv_file))

    # Filter the dataframe based on the parameters and load
    # if timestamp is not None:
    if timestamp is not None:
        df_filtered = df[(df['interface'] == method) & (df['capacity'] == load) & (df['timestamp'] == timestamp)]
    else:
        df_filtered = df[(df['interface'] == method) & (df['capacity'] == load)]

    # If the --concurrent flag is set, filter by the concurrency column
    if concurrent:
        df_filtered = df_filtered[(df_filtered['concurrency'] == True) | (df_filtered['concurrency'] == 'true')]
    else:
        df_filtered = df_filtered[(df_filtered['concurrency'] == False) | (df_filtered['concurrency'] == 'false')]

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
        # return experiment_file
    elif len(df_filtered) == 0:
        print(f"Error: Found {len(df_filtered)} matching files.")
        return None
    else:
        # randomly select one of the matching files
        experiment_file = df_filtered.sample()['filename'].values[0]
        print(f"Warning: Found {len(df_filtered)} matching files. Randomly selecting {experiment_file}.")

    yield experiment_file
    # check if there are concurrent experiments
    # if two experiments have the same parameters and timestamp, then we need to return both
    # in the tuple of interfaces groups (compose, user-timeline, home-timeline) or (S_102000854, S_149998854, S_161142529) or (motivate-get, motivate-set, both-motivate) or (search-hotel, reserve-hotel)
    # first, find the group of interfaces
    timestamp = experiment_file.split('-')[-1].split('.')[0]
    if method == 'compose':
        for interface in ['user-timeline', 'home-timeline']:
            yield from find_experiment_file(params_file, load, csv_file, interface, timestamp, concurrent)
    elif method == 'search-hotel':
        for interface in ['reserve-hotel']:
            yield from find_experiment_file(params_file, load, csv_file, interface, timestamp, concurrent)
    elif method == 'S_149998854':
        for interface in ['S_102000854', 'S_161142529']:
            yield from find_experiment_file(params_file, load, csv_file, interface, timestamp, concurrent)


def find_experiment_topdown(load, csv_file='ghz-results/alibaba-experiment_results.csv', method=None, timestamp=None, concurrent=False):
    # Load the experiment results CSV
    df = pd.read_csv(csv_file)

    # Filter the dataframe based on the parameters and load
    # if timestamp is not None:
    if timestamp:
        df_filtered = df[(df['interface'] == method) & (df['capacity'] == load) & (df['timestamp'] == timestamp) & (df['control_scheme'] == 'topdown')]
    else:
        df_filtered = df[(df['interface'] == method) & (df['capacity'] == load) & (df['control_scheme'] == 'topdown')]

    # If the --concurrent flag is set, filter by the concurrency column
    if concurrent:
        df_filtered = df_filtered[(df_filtered['concurrency'] == True) | (df_filtered['concurrency'] == 'true')]
    else:
        df_filtered = df_filtered[(df_filtered['concurrency'] == False) | (df_filtered['concurrency'] == 'false')]

    # Check if we have a single matching file
    if len(df_filtered) == 1:
        experiment_file = df_filtered.iloc[0]['filename']
        # return experiment_file
    elif len(df_filtered) == 0:
        print(f"Error: Found {len(df_filtered)} matching files.")
        return None
    else:
        # randomly select one of the matching files
        experiment_file = df_filtered.sample()['filename'].values[0]
        print(f"Warning: Found {len(df_filtered)} matching files. Randomly selecting {experiment_file}.")

    yield experiment_file

    timestamp = experiment_file.split('-')[-1].split('.')[0]
    if method == 'compose':
        for interface in ['user-timeline', 'home-timeline']:
            yield from find_experiment_topdown(load, csv_file, interface, timestamp, concurrent)
    elif method == 'search-hotel':
        for interface in ['reserve-hotel']:
            yield from find_experiment_topdown(load, csv_file, interface, timestamp, concurrent)

def run_visualize(experiment_file):
    # Run visualize.py with the found experiment file
    visualize_file = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/visualize.py')
    subprocess.run(['python', visualize_file, experiment_file])

def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python find_and_visualize.py <bopt_file.json> <Load>")
    #     return

    # params_file = sys.argv[1]
    # load = int(sys.argv[2])

    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Find and visualize experiment files.")
    parser.add_argument('params_file', type=str, help="The JSON file with Bayesian optimization parameters")
    parser.add_argument('load', type=int, help="The load value for filtering the experiments")
    parser.add_argument('--concurrent', action='store_true', default=False, help="Filter by concurrent experiments")

    args = parser.parse_args()
    params_file = args.params_file
    load = args.load

    # if 'S_' in params_file in params_file:
    #     experiment_files = find_experiment_file(params_file, load, 'ghz-results/all-alibaba-experiment_results.csv', concurrent=args.concurrent)
    # # elif 'motivate' in params_file:
    # #     experiment_files = find_experiment_file(params_file, load, 'ghz-results/4-nodes-monotonic-experiment_results.csv', concurrent=args.concurrent)
    # # elif 'hotel' in params_file:

    if 'topdown' in params_file:
        # filter by the method name
        method = params_file.split('-')[1]
        # filter the csv file by the method name
        experiment_files = find_experiment_topdown(load, 'ghz-results/all-experiments.csv', method=method, concurrent=args.concurrent)
    else:
        experiment_files = find_experiment_file(params_file, load, 'all-experiments.csv', concurrent=args.concurrent)
        # experiment_files = find_experiment_file(params_file, load, 'ghz-results/hotel-experiment_results.csv')

    for experiment_files in experiment_files:
        print(f"Found experiment file: {experiment_files}")
        run_visualize(experiment_files)

    else:
        print("No matching experiment file found.")

if __name__ == "__main__":
    main()
