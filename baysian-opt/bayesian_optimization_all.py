# Jiali 
# The Bayesian Optimization is used to optimize the parameters for the overload experiments
# The parameters are optimized for the goodput - tail latency


import subprocess
from bayes_opt import BayesianOptimization

import os
import glob
import json
import sys

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime

import sklearn
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ghz-results'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/z/large-nsdi/results'))

from visualize import analyze_data
from slo import get_slo, get_sustainable_load
from utils import convert_to_dataframe, calculate_goodput_mean, calculate_goodput_from_file, read_tail_latency_from_file, roundDownParams, save_iteration_details, parse_configurations

rerun = True
quantile = 0.1


param_ranges = [(50, 500), (1000, 10000), (1, 20), ]  # (priceUpdateRate, delayTarget) 
  
configDict = {
    'rajomon': {
        'price_update_rate': 5000,  # Assuming numeric values for simplicity
        'token_update_rate': 5000,  # Assuming numeric values for simplicity
        'latency_threshold': 5000,
        'price_step': 10,
        'price_strategy': 'expgrow',
        'lazy_update': 'false',
        'rate_limiting': 'true',
        'only_frontend': 'false',
        'fast_drop': 'true',
    },
    'breakwater': {
        'breakwater_slo': 12500,
        'breakwater_a': 0.001,
        'breakwater_b': 0.02,
        'breakwater_initial_credit': 400,
        'breakwater_client_expiration': 10000,
        'only_frontend': 'true',
        'breakwater_rtt': '100us',
    },    
    'breakwaterd': {
        'breakwater_slo': 12500,
        'breakwater_a': 0.001,
        'breakwater_b': 0.02,
        'breakwater_initial_credit': 400,
        'breakwater_client_expiration': 10000,
        'only_frontend': 'false',
        'breakwaterd_slo': 12500,
        'breakwaterd_a': 0.001,
        'breakwaterd_b': 0.02,
        'breakwaterd_initial_credit': 400,
        'breakwaterd_client_expiration': 10000,
    },
    'dagor': {
        'dagor_queuing_threshold': 2000,
        'dagor_alpha': 0.05,
        'dagor_beta': 0.01,
        'dagor_admission_level_update_interval': 10000,
        'dagor_umax': 20,
    }
}

def generate_output_filename(interceptor_type, method, capacity, dir='~/Sync/Git/protobuf/ghz-results/'):
    directory = os.path.expanduser(dir)
    outputFile = f"social-{method}-control-{interceptor_type}-parallel-capacity-{capacity}-*.json.output"
    return os.path.join(directory, outputFile)


def find_method_file_sets(directory, methods, capacity):
    # Regex pattern modified to include the capacity
    regex_pattern = rf"social-(?P<method>.+)-control-.+-parallel-capacity-{capacity}-(?P<identifier>.+)\.json\.output"

    # Find all files in the directory
    all_files = glob.glob(os.path.join(directory, f"social-*.json.output"))

    # Group files by their common identifier
    file_groups = defaultdict(lambda: {method: None for method in methods})
    for file in all_files:
        match = re.match(regex_pattern, os.path.basename(file))
        if match:
            method = match.group('method')
            identifier = match.group('identifier')
            if method in methods:
                file_groups[identifier][method] = file

    # Filter groups to only include complete sets (one file for each method)
    complete_sets = [list(group.values()) for group in file_groups.values() if all(group.values())]

    return complete_sets


def check_previous_run_exists(interceptor_type, method, capacity, combined_params, existing_files=None):
    prev_run_folder = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/')
    # prev_run_folder = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/rajomon-linear-price/')
    if method == "all-methods-social":
        sub_methods = ["user-timeline", "home-timeline", "compose"]
    if method == "all-methods-hotel":
        sub_methods = ["hotels-http", "reservation-http", "user-http", "recommendations-http"]
    if method == "all-alibaba":
        sub_methods = ["S_102000854", "S_149998854", "S_161142529"]
    if method == "all-methods-social" or method == "all-methods-hotel" or method == "all-alibaba":
        # return a list of files that match the parameters
        filesets = find_method_file_sets(prev_run_folder, sub_methods, capacity)
        # append the file to the list only when all sub-methods have the same timestamp
        # if all sub-methods have the same timestamp, return the list of files
        for files in filesets:
            if existing_files is not None:
                # Extract the timestamp from the filename
                timestamp = files[0].split('-')[-1].split('.')[0]
                # Skip this file if the timestamp is in the existing_files
                if timestamp in existing_files[0]:
                    continue
            try:
                with open(files[0], 'r') as file:
                    content = file.read()

                    if 'Rajomon Configurations:' in content:
                        # Extract the configurations string from the file
                        start = content.find('Rajomon Configurations:') + len('Rajomon Configurations:')
                        end = content.find(']', start) + 1
                        file_config_str = content[start:end]
                        file_config_dict = parse_configurations(file_config_str)
                        combined_params_dict = {key: str(value) for key, value in combined_params.items()}

                        # if file_config_dict is a subset or superset of combined_params_dict
                        common_keys = set(file_config_dict.keys()) & set(combined_params_dict.keys())
                        if all(file_config_dict[key] == combined_params_dict[key] for key in common_keys):
                            return list(files)
            except Exception as e:
                print(f"Error reading file {files[0]}: {e}")

    # Generate the expected output filename
    output_filename_pattern = generate_output_filename(interceptor_type, method, capacity)
    archived_output_filename_pattern = generate_output_filename(interceptor_type, method, capacity, dir='/z/large-nsdi/results')
    
    # Search for files that match the pattern
    matching_files = glob.glob(output_filename_pattern) + glob.glob(archived_output_filename_pattern)

    # Iterate through matching files and check contents
    for filename in matching_files:
        # if existing_files is not None then skip this file if it has the same timestamp as the existing files
        if existing_files is not None:
            # Extract the timestamp from the filename
            timestamp = filename.split('-')[-1].split('.')[0]
            # Skip this file if the timestamp is in the existing_files
            if timestamp in existing_files:
                continue
        # 1. remove the .output at end of filename
        base_filename = filename.replace('.output', '')
        if not os.path.exists(base_filename):
            continue
        with open(base_filename, 'r') as file:
            # 2. read it if it has line `"load-start": 10,` in first 20 rows, then skip this file
            # find the line with "load-start": 10 without looping through the file
            content = file.read(2000)
            if '"load-start": 10,' in content:
                continue

        try:
            with open(filename, 'r') as file:
                content = file.read()

                if 'Rajomon Configurations:' in content:
                    # Extract the configurations string from the file
                    start = content.find('Rajomon Configurations:') + len('Rajomon Configurations:')
                    end = content.find(']', start) + 1
                    file_config_str = content[start:end]
                    file_config_dict = parse_configurations(file_config_str)
                    combined_params_dict = {key: str(value) for key, value in combined_params.items()}

                    # if file_config_dict is a subset or superset of combined_params_dict
                    common_keys = set(file_config_dict.keys()) & set(combined_params_dict.keys())
                    if all(file_config_dict[key] == combined_params_dict[key] for key in common_keys):
                        return filename

        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    return None

# Define the function that runs the service and client as experiments, takes the interceptor type and the parameters as arguments,
# the params are dictionary of the parameters. and the load is the capacity of the service, with a default value of load = int(capacity.split('-')[1])
def run_experiments(interceptor_type, load, previous_run, **params):
    # Map the interceptor_type to its specific parameters
    specific_params = configDict[interceptor_type]

    # Combine specific params and passed params
    # overwrite the specific params with the passed params
    combined_params = {**specific_params, **params}
    # combined_params = {**specific_params, **params}

    # round the parameters to int unless the key is in the following list
    combined_params = roundDownParams(combined_params)

    # print with 2 decimal places
    print("[Run Experiment] Combined params:", {key: round(float(value), 2) if isinstance(value, float) else value for key, value in combined_params.items()})
    # Prepare the environment variable command string
    # env_vars_command = ' '.join(f'export {key}="{value}";' for key, value in combined_params.items())

    # previous_run = 'skip'
    # Check if a previous run exists with the same parameters
    if previous_run == '' or previous_run is None:
        preRun = check_previous_run_exists(interceptor_type, method, load, combined_params)
        if preRun is not None:
            print(f"[PrevRan] A previous run with the same parameters exists: {preRun}")
            if 'all' in method:
                # preRun is a list of files, return the list of files without the .output
                return [f.split('.output')[0] for f in preRun]
            return preRun.split('.output')[0]
        else:
            print("[PrevRan] No previous run with the same parameters found. Proceed with the experiment.")
    elif previous_run == 'skip':
        print("Skip checking previous run with the same parameters.")
    else:
        print("[PrevRan] found a previous run file:", previous_run, ". Now try to find another file with the same parameters.")
        preRun = check_previous_run_exists(interceptor_type, method, load, combined_params, previous_run)
        if preRun is not None:
            print(f"[2nd PrevRan] A previous run with the same parameters exists: {preRun}")
            if 'all' in method:
                # preRun is a list of files, return the list of files without the .output
                # print([f.split('.output')[0] for f in preRun])
                return [f.split('.output')[0] for f in preRun]
            return preRun.split('.output')[0]
        else:
            print("[PrevRan] No previous run with the same parameters found. Proceed with the experiment.")

    # capacity is a random number between 1000 and 10000
    # load = np.random.randint(int(capacity.split('-')[1]), 10000)
    # Full command to source envs.sh and run the experiment
    # full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && {env_vars_command} ~/Sync/Git/service-app/cloudlab/scripts/bayesian_experiments.sh -c {capacity} -s parallel --{interceptor_type}'"
    full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c {load} -s parallel --{interceptor_type}'"

    # Set environment variables in Python
    env = os.environ.copy()
    for key, value in combined_params.items():
        env[key] = str(value)
    # add the method to env
    env['METHOD'] = method
    if method == "all-alibaba":
        env['METHOD'] = "all-alibaba"
    else:
        load_value = get_sustainable_load(method)
        warmup_load = int(load_value * 0.8)
        env['WARMUP_LOAD'] = str(warmup_load)
    # Run the experiment script
    experiment_process = subprocess.Popen(full_command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = experiment_process.communicate()
    print("[Experiment] output:", stdout.decode())
    print("[Experiment] errors:", stderr.decode())
    # if `ssh` is found in the stderr, raise an exception
    if 'ssh' in stderr.decode() or 'publickey' in stderr.decode():
        raise Exception("ssh error found in stderr, please check the stderr")
    resultf = get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), pattern=f"social-{method}-control-{interceptor_type}-parallel-capacity-{load}-*.json")

    if method == "all-methods-social":
        resultf = []
        # append all 3 files one by one
        for interface in ['compose', 'user-timeline', 'home-timeline']:
            resultf.append(get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), pattern=f"social-{interface}-control-{interceptor_type}-parallel-capacity-{load}-*.json"))
    elif method == "all-methods-hotel":
        # same for hotel, with interface in ["hotels-http" "reservation-http" "user-http" "recommendations-http"]
        resultf = []
        # append all 3 files one by one
        for interface in ["hotels-http", "reservation-http", "user-http", "recommendations-http"]:
            resultf.append(get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), pattern=f"social-{interface}-control-{interceptor_type}-parallel-capacity-{load}-*.json"))
    elif method == "all-alibaba":
        # same for alibaba, with interface in ["S_102000854", "S_149998854", "S_161142529"]
        resultf = []
        # append all 3 files one by one
        for interface in ["S_102000854", "S_149998854", "S_161142529"]:
            resultf.append(get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), pattern=f"social-{interface}-control-{interceptor_type}-parallel-capacity-{load}-*.json"))

    return resultf


def run_experiments_loop(interceptor_type, interceptor_configs, capact, methodToRun):
    # Map the interceptor_type to its specific parameters
    # specific_params = configDict[interceptor_type]

    # Combine specific params and passed params
    # interceptor_configs is a dictionary of the interceptor configurations
    # overwrite the specific params with the passed params
    # combined_params = {**specific_params, **interceptor_configs}
    combined_params = {**interceptor_configs}

    # round the parameters to int unless the key is in the following list
    combined_params = roundDownParams(combined_params)

    # print with 2 decimal places
    print("[Bayesian Opt] Combined params:", {key: round(float(value), 2) if isinstance(value, float) else value for key, value in combined_params.items()})
    # Prepare the environment variable command string
    # env_vars_command = ' '.join(f'export {key}="{value}";' for key, value in combined_params.items())

    # Full command to source envs.sh and run the experiment
    # full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && {env_vars_command} ~/Sync/Git/service-app/cloudlab/scripts/bayesian_experiments.sh -c {capacity} -s parallel --{interceptor_type}'"
    full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c {capact} -s parallel --{interceptor_type}'" if interceptor_type != 'plain' else f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c {capact} -s parallel'"

    # Set environment variables in Python
    env = os.environ.copy()
    for key, value in combined_params.items():
        env[key] = str(value)
    # add the method to env
    env['METHOD'] = methodToRun
    load_value = get_sustainable_load(methodToRun)
    warmup_load = int(load_value * 0.8)
    env['WARMUP_LOAD'] = str(warmup_load)

    # Run the experiment script
    experiment_process = subprocess.Popen(full_command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = experiment_process.communicate()
    print("Experiment output:", stdout.decode())
    print("Experiment errors:", stderr.decode())
    # if `ssh` or `publickey` is found in the stderr, raise an exception
    if 'ssh' in stderr.decode() or 'publickey' in stderr.decode():
        raise Exception("ssh error found in stderr, please check the stderr")
    return get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'))



# Define the objective function to optimize
def objective_once(interceptor_type, **params):
    global tightSLO, quantile
    hugePenalty = -999999999
    load = int(capacity.split('-')[1])
    latest_file = run_experiments(interceptor_type, load, '', **params)
    if latest_file is None:
        print("No file found for objective function")
        return hugePenalty
    goodput = calculate_goodput_from_file(latest_file, tightSLO, quantile=quantile, average=True)
    tail_latency = read_tail_latency_from_file(latest_file, percentile=99, minusSLO=True)
    print(f"[Experiment Ran] average goodput: {goodput}, and tail latency: {tail_latency} \nfrom file: {latest_file}")
    # if average_goodput == nan, return 0
    if np.isnan(goodput):
        print("No average goodput found for objective function")
        return hugePenalty
    if goodput < 5:
        print("Average goodput is too low for objective function")
        return hugePenalty
    if tail_latency is None:
        print("No tail latency found for objective function")
        return hugePenalty
    if 'gpt0' in capacity:
        obj = goodput
    elif 'gpt1' in capacity:
        obj = goodput - 10 * (tail_latency )
    elif 'tgpt' in capacity:
        obj = calculate_goodput_from_file(latest_file, tightSLO, quantile=quantile, average=False)
    return obj / maximum_goodput


def objective(interceptor_type, **params):
    global tightSLO, quantile
    obj = []
    # same as objective, but run twice with different loads, and return the lower goodput
    latest_file = ''
    for loadTest in [int(capacity.split('-')[1]), ] * 2 :
        latest_file = run_experiments(interceptor_type, loadTest, latest_file, **params)
        if latest_file is None:
            print("No file found for objective function")
            return -999999999
        goodput = calculate_goodput_from_file(latest_file, tightSLO, quantile=quantile, average=True)
        tail_goodput = calculate_goodput_from_file(latest_file, tightSLO, quantile=quantile, average=False)
        tail_latency = read_tail_latency_from_file(latest_file, percentile=99, minusSLO=True)
        print(f"[Experiment Ran] average goodput: {goodput}, {100*quantile}th percentile goodput: {tail_goodput}, and tail latency: {tail_latency} \nfrom file: {latest_file}")
        # if average_goodput == nan, return 0
        if np.isnan(goodput):
            print("No average goodput found for objective function")
            return -999999999
        if goodput < 5:
            print("Average goodput is too low for objective function")
            return -999999999
        if tail_latency is None:
            print("No tail latency found for objective function")
            return -999999999
        if 'gpt0' in capacity:
            obj.append(goodput)
        elif 'gpt1' in capacity:
            obj.append(goodput - 10 * tail_latency)
        elif 'tgpt' in capacity:
            obj.append(tail_goodput)
    # return the mean obj of the two loads
    return np.mean(obj) / maximum_goodput


def plot_opt_wrapper(priceUpdateRate, delayTarget, priceStep,):
    return plot_opt(priceUpdateRate, 0, delayTarget, -1, priceStep)


def get_latest_file(path, pattern="*.json"):
    """Return the latest file in a given directory matching the pattern."""
    list_of_files = glob.glob(os.path.join(path, pattern))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def objective_rajomon(price_update_rate, token_update_rate, latency_threshold, price_step):
    # return objective('rajomon', price_update_rate, latency_threshold, price_step)
    return objective('rajomon', price_update_rate=price_update_rate, token_update_rate=token_update_rate, latency_threshold=latency_threshold, price_step=price_step)


def objective_breakwater(breakwater_slo, breakwater_a, breakwater_b, breakwater_initial_credit, breakwater_client_expiration, breakwater_rtt):
    return objective('breakwater', breakwater_slo=breakwater_slo, breakwater_a=breakwater_a, breakwater_b=breakwater_b,
                          breakwater_initial_credit=breakwater_initial_credit, breakwater_client_expiration=breakwater_client_expiration, breakwater_rtt=breakwater_rtt)

def objective_breakwaterd(breakwater_slo, breakwater_a, breakwater_b, breakwater_initial_credit, breakwater_client_expiration, breakwater_rtt,
                          breakwaterd_slo, breakwaterd_a, breakwaterd_b, breakwaterd_initial_credit, breakwaterd_client_expiration, breakwaterd_rtt):
    return objective('breakwaterd', breakwater_slo=breakwater_slo, breakwater_a=breakwater_a, breakwater_b=breakwater_b, breakwater_initial_credit=breakwater_initial_credit, 
                          breakwater_client_expiration=breakwater_client_expiration, breakwaterd_slo=breakwaterd_slo, breakwaterd_a=breakwaterd_a, 
                          breakwaterd_b=breakwaterd_b, breakwaterd_initial_credit=breakwaterd_initial_credit, breakwaterd_client_expiration=breakwaterd_client_expiration,
                          breakwater_rtt=breakwater_rtt, breakwaterd_rtt=breakwaterd_rtt)


def objective_dagor(dagor_queuing_threshold, dagor_alpha, dagor_beta, dagor_admission_level_update_interval, dagor_umax):
    return objective('dagor', dagor_queuing_threshold=dagor_queuing_threshold, dagor_alpha=dagor_alpha, dagor_beta=dagor_beta, dagor_admission_level_update_interval=dagor_admission_level_update_interval, dagor_umax=dagor_umax)


# Define a function to read and parse the JSON file
def parse_file(file_path):
    folder = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/')
    file_path = os.path.join(folder, file_path)
    with open(file_path, 'r') as file:
        content = file.read()

    # Extracting 'METHOD' and 'INTERCEPT' parts
    method = re.search(r'METHOD: (\w+)', content)
    intercept = re.search(r'INTERCEPT: (\w+)', content)

    # Extracting Rajomon Configurations
    rajomon_config_match = re.search(r'Rajomon Configurations:\s*\[(.*?)\]', content, re.DOTALL)
    if rajomon_config_match:
        rajomon_config_str = rajomon_config_match.group(1)
        rajomon_config_pairs = rajomon_config_str.split('} {')
        rajomon_config = {item.split(' ')[0]: item.split(' ')[1] for item in rajomon_config_pairs}

    return {
        'METHOD': method.group(1) if method else None,
        'INTERCEPT': intercept.group(1) if intercept else None,
        'Rajomon_Configurations': rajomon_config
    }


def record_optimal_parameters(filename, data):
    path = os.path.expanduser('~/Sync/Git/protobuf/baysian-opt/')
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(data, f, indent=4)
    # return


def read_optimal_parameters(filename):
    path = os.path.expanduser('~/Sync/Git/protobuf/baysian-opt/')
    # if the file does not exist, raise an exception
    if not filename or not os.path.exists(os.path.join(path, filename)):
        raise Exception(f"File {filename} does not exist in the directory {path}")
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)
    return data


def load_optimal_parameters(method, control):
    """
    Loads optimal parameters from a JSON file.

    Args:
            filename (str): The name of the JSON file containing the parameters.

    Returns:
            dict: A dictionary containing the loaded parameters (or None if not found).
    """
    # Try opening the file
    filename = f'bopt_False_{control}_{method}_gpt1-best.json'
    path = os.path.expanduser('~/Sync/Git/protobuf/baysian-opt/')
    try:
        with open(os.path.join(path, filename), 'r') as f:
            # Load the JSON data
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle errors like file not found or invalid JSON data
        return None
    return None


def main():
    
    pbounds_rajomon = {
        # the range above is too large... I will use the following range based on the empirical results
        'price_update_rate': (10000, 20000),
        'token_update_rate': (80000, 120000),
        'price_step': (150, 250),
        'latency_threshold': (1, 1000),
    }
    pbounds_breakwater = {
        'breakwater_slo': (500, 5000),
        'breakwater_a': (0.001, 0.5),
        'breakwater_b': (0.02, 0.5),
        'breakwater_initial_credit': (500, 1000),
        'breakwater_client_expiration': (10, 5000),
        'breakwater_rtt': (500, 2000),
    }
    pbounds_breakwaterd = {
        'breakwater_slo': (500, 5000),
        'breakwater_a': (0.001, 0.5),
        'breakwater_b': (0.02, 0.5),
        'breakwater_initial_credit': (500, 1000),
        'breakwater_client_expiration': (0, 500),
        'breakwater_rtt': (500, 2000),
        'breakwaterd_initial_credit': (1000, 2000),
        'breakwaterd_client_expiration': (0, 1000),
        'breakwaterd_slo': (500, 5000),
        'breakwaterd_a': (1, 10),
        'breakwaterd_b': (0.02, 0.5),
        'breakwaterd_rtt': (500, 2000),
    }
    pbounds_dagor = {
        'dagor_queuing_threshold': (500, 100000),
        'dagor_alpha': (0, 1),
        'dagor_beta': (0, 4),
        'dagor_admission_level_update_interval': (1000, 40000),
        'dagor_umax': (5, 20),
    }

    if 'hotel' in method:
        #  {DAGOR_QUEUING_THRESHOLD 1600us} {DAGOR_ALPHA 0.83} {DAGOR_BETA 6} {DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL 14000us} {DAGOR_UMAX 11}
        # use the above values as the range for the parameters, surrounding the optimal values
        pbounds_dagor = {
            'dagor_queuing_threshold': (1000, 2000),
            'dagor_alpha': (0, 4),
            'dagor_beta': (0, 10),
            'dagor_admission_level_update_interval': (1000, 24000),
            'dagor_umax': (5, 15),
        }

        pbounds_rajomon['latency_threshold'] = (300, 700)
        pbounds_rajomon['token_update_rate'] = (90000, 150000)
        pbounds_rajomon['price_update_rate'] = (1000, 6000)
        pbounds_rajomon['price_step'] = (100, 500)

        # for large scale experiments, use the ranges near the optimal values below
        # "BREAKWATER_SLO": "500us",
        # "BREAKWATER_A": 0.0001,
        # "BREAKWATER_B": 5.0,
        # "BREAKWATER_INITIAL_CREDIT": 600,
        # "BREAKWATER_CLIENT_EXPIRATION": "100us",
        # "ONLY_FRONTEND": "true",
        # "BREAKWATER_RTT": "1000us"
        pbounds_breakwater = {
            'breakwater_slo': (100, 1000),
            'breakwater_a': (0.00001, 0.1),
            'breakwater_b': (0.5, 10),
            'breakwater_initial_credit': (400, 800),
            'breakwater_client_expiration': (10, 500),
            'breakwater_rtt': (500, 2000),
        }


        # pbounds_breakwater = {
        #     'breakwater_slo': (1000, 2000),  # Adjusting around 1382us
        #     'breakwater_client_expiration': (400, 800),  # Adjusting around 592us
        #     'breakwater_a': (0.0001, 0.01),  # Adjusting around 0.001
        #     'breakwater_b': (0.4, 2),  # Adjusting around 0.5
        #     'breakwater_initial_credit': (600, 700),  # Adding around 685
        #     'breakwater_rtt': (1500, 2000)  # Adjusting around 1853us
        # }
        pbounds_breakwaterd = {
            'breakwater_slo': (950, 1050),  # Marginally around 1000us
            'breakwaterd_slo': (5700, 5900),  # Marginally around 5791us
            'breakwater_client_expiration': (390, 410),  # Marginally around 400us
            'breakwaterd_client_expiration': (460, 480),  # Marginally around 471us
            'breakwater_a': (0.00009, 0.00011),  # Marginally around 0.0001
            'breakwaterd_a': (2.8, 3.0),  # Marginally around 2.9019
            'breakwater_b': (1.1, 1.2),  # Marginally around 1.1456
            'breakwaterd_b': (0.75, 0.8),  # Marginally around 0.7841
            'breakwater_rtt': (1450, 1550),  # Marginally around 1500us
            'breakwaterd_rtt': (8500, 8700),  # Marginally around 8599us
            'breakwater_initial_credit': (600, 630),  # Marginally around 615
            'breakwaterd_initial_credit': (900, 930),  # Marginally around 918
        }

        # pbounds_breakwaterd = {
        #     'breakwaterd_slo': (1000, 8000),  # Adjusting around 1382us
        #     'breakwater_slo': (1000, 2000),  # Adjusting around 1382us
        #     'breakwater_client_expiration': (400, 800),  # Adjusting around 592us
        #     'breakwaterd_client_expiration': (400, 800),  # Adjusting around 592us
        #     'breakwater_a': (0.0001, 0.01),  # Adjusting around 0.001
        #     'breakwaterd_a': (2, 5),  # Keeping same as before
        #     'breakwater_b': (0.4, 2),  # Adjusting around 0.5
        #     'breakwaterd_b': (0.1, 2),  # Keeping same as before
        #     'breakwater_rtt': (1500, 2000),  # Adjusting around 1853us
        #     'breakwaterd_rtt': (7000, 10000),  # Keeping same as before
        #     'breakwater_initial_credit': (600, 700),  # Adding around 685
        #     'breakwaterd_initial_credit': (900, 1000)  # Assuming similar range as previous credit
        # }

    if method == 'compose':
        pbounds_rajomon['latency_threshold'] = (300, 500)  # 300 to 500 due to better machine
        pbounds_rajomon['price_update_rate'] = (2000, 5000)
        pbounds_rajomon['price_step'] = (100, 200)
        pbounds_rajomon['token_update_rate'] = (80000, 120000)

        pbounds_dagor['dagor_queuing_threshold'] = (1000, 2000)
        pbounds_dagor['dagor_alpha'] = (0, 2)
        pbounds_dagor['dagor_beta'] = (1.5, 4)
        pbounds_dagor['dagor_admission_level_update_interval'] = (12000, 15000)
        pbounds_dagor['dagor_umax'] = (10, 20)

        # for large scale experiments, use the ranges near the optimal values below
        # "BREAKWATER_SLO": "250us",
        # "BREAKWATER_A": 0.001,
        # "BREAKWATER_B": 5,
        # "BREAKWATER_INITIAL_CREDIT": 600,
        # "BREAKWATER_CLIENT_EXPIRATION": "10us",
        # "ONLY_FRONTEND": "true",
        # "BREAKWATER_RTT": "1000us"
        pbounds_breakwater = {
            'breakwater_slo': (10, 500),  # Adjusting around 1382us
            'breakwater_a': (0.0001, 0.01),  # Adjusting around 0.001
            'breakwater_b': (0.5, 20),  # Adjusting around 0.5
            'breakwater_initial_credit': (400, 800),  # Adding around 685
            'breakwater_client_expiration': (10, 200),  # Adjusting around 592us
            'breakwater_rtt': (50, 2000)  # Adjusting around 1853us
        }


        # pbounds_breakwater = {
        #     'breakwater_slo': (10, 12000),  # Adjusting around 1382us
        #     'breakwater_client_expiration': (300, 1000),  # Adjusting around 592us
        #     'breakwater_a': (0.00001, 0.5),  # Adjusting around 0.001
        #     'breakwater_b': (0.1, 100),  # Adjusting around 0.5
        #     'breakwater_initial_credit': (200, 800),  # Adding around 685
        #     'breakwater_rtt': (800, 2000)  # Adjusting around 1853us
        # }
        
        # explore the following ranges for the parameters
        pbounds_breakwaterd = {
            'breakwater_slo': (1000, 11000),  # Covers 1000us to 10783us
            'breakwaterd_slo': (5000, 9000),  # Covers 5791us to 8609us
            'breakwater_client_expiration': (400, 2000),  # Covers 400us to 1848us
            'breakwaterd_client_expiration': (400, 1200),  # Covers 471us to 1121us
            'breakwater_a': (0.0001, 0.2),  # Covers 0.0001 to 0.150467
            'breakwaterd_a': (2.5, 40),  # Covers 2.9 to 35.4929
            'breakwater_b': (1, 10),  # Covers 1.15 to 9.0168
            'breakwaterd_b': (0.75, 4),  # Covers 0.78 to 3.2773
            'breakwater_rtt': (1500, 2000),  # Covers 1500us to 1757us
            'breakwaterd_rtt': (1500, 9000),  # Covers 1757us to 8599us
            'breakwater_initial_credit': (600, 1700),  # Covers 615 to 1630
            'breakwaterd_initial_credit': (800, 1000),  # Covers 867 to 918
        }


    if 'S_16' in method:
        pbounds_rajomon['latency_threshold'] = (100, 30000)
        pbounds_breakwater['breakwater_slo'] = (100, 4000)
        pbounds_breakwater['breakwater_a'] = (0, 3)
        pbounds_breakwater['breakwater_b'] = (0, 3)
        pbounds_breakwater['breakwater_rtt'] = (1000, 5000)
    if 'S_14' in method or method == 'all-alibaba':
        # pbounds_breakwater['breakwater_a'] = (0, 30)
        pbounds_rajomon['latency_threshold'] = (50, 6000)
        pbounds_rajomon['price_update_rate'] = (10, 5000)
        pbounds_rajomon['token_update_rate'] = (30000, 60000)
        pbounds_rajomon['price_step'] = (100, 700)
        
        pbounds_breakwater = {
            'breakwater_slo': (2000, 3000),  # Around 2418us
            'breakwater_a': (1.5, 2.5),  # Around 1.85
            'breakwater_b': (1.5, 2.5),  # Around 1.99
            'breakwater_initial_credit': (200, 350),  # Around 279
            'breakwater_client_expiration': (1500, 2500),  # Around 2313us
            'breakwater_rtt': (4000, 7000)  # Around 5913us
        }

        pbounds_breakwaterd = {
            'breakwater_slo': (3000, 45000),  # Around 3500us
            'breakwater_a': (0.1, 2.5),  # Around 1.8
            'breakwater_b': (0.1, 1.5),  # Around 0.9
            'breakwater_initial_credit': (50, 150),  # Around 80
            'breakwater_client_expiration': (2000, 4000),  # Around 3000us
            'breakwater_rtt': (5000, 9000),  # Around 7000us
            'breakwaterd_slo': (50000, 210000),  # Around 180000us
            'breakwaterd_a': (1, 140),  # Around 120
            'breakwaterd_b': (0.05, 0.1),  # Around 0.08
            'breakwaterd_initial_credit': (50, 150),  # Around 80
            'breakwaterd_client_expiration': (80000, 100000),  # Around 90000us
            'breakwaterd_rtt': (10000, 14000)  # Around 12000us
        }
        pbounds_dagor = {
            'dagor_queuing_threshold': (4000, 10000),  # Around 8000us
            'dagor_alpha': (0.05, 1),  # Around 0.9
            'dagor_beta': (1, 8),  # Around 2.01
            'dagor_admission_level_update_interval': (1000, 5000),  # Around 3806us
            'dagor_umax': (7, 15)  # Around 9
        }

    if 'motivate' in method:
        pbounds_breakwater = {
            'breakwater_slo': (2000, 8000),  # 圍繞5724us
            'breakwater_a': (0.01, 1.5),      # 圍繞0.869
            'breakwater_b': (0.01, 2),      # 圍繞0.506
            'breakwater_initial_credit': (200, 300),  # 圍繞255
            'breakwater_client_expiration': (300, 600),  # 圍繞484us
            'breakwater_rtt': (6000, 10000),  # 圍繞8986us
        }
        pbounds_breakwaterd = {
            'breakwater_slo': (50000, 70000),  # 圍繞60000us
            'breakwater_a': (7.0, 12.0),       # 圍繞8.1
            'breakwater_b': (7.0, 12.0),       # 圍繞9.26
            'breakwater_initial_credit': (3000, 5000),  # 圍繞4000
            'breakwater_client_expiration': (4000, 8000),  # 圍繞5385us
            'breakwater_rtt': (18000, 25000),  # 圍繞20000us
            'breakwaterd_slo': (15000, 25000),  # 圍繞20080us
            'breakwaterd_a': (9.0, 13.0),      # 圍繞10.57
            'breakwaterd_b': (5.0, 10.0),      # 圍繞6.16
            'breakwaterd_initial_credit': (1200, 1800),  # 圍繞1509
            'breakwaterd_client_expiration': (40000, 60000),  # 圍繞45801us
            'breakwaterd_rtt': (15000, 25000)  # 圍繞18000us
        }
        pbounds_dagor = {
            'dagor_queuing_threshold': (3000, 6000),  # 圍繞4051us
            'dagor_alpha': (0.5, 1.0),  # 圍繞0.75
            'dagor_beta': (1.0, 2.0),  # 圍繞1.12
            'dagor_admission_level_update_interval': (4000, 6000),  # 圍繞4949us
            'dagor_umax': (10, 20)  # 圍繞15
        }


    # run the experiments with the interceptors for all capacities
    capacity_step = 1000
    capacity_start = 2000
    capacity_end = 8000

    capacity_range = range(capacity_start, capacity_end, capacity_step)

    timestamp = datetime.datetime.now().strftime("%m-%d")
    print("Timestamp:", timestamp)
    print("method:", method)
    
    optimization_config = {
        'rajomon': (optimizeRajomon, objective_rajomon, pbounds_rajomon, 'rajomon'),
        'breakwater': (optimizeBreakwater, objective_breakwater, pbounds_breakwater, 'breakwater'),
        'breakwaterd': (optimizeBreakwaterD, objective_breakwaterd, pbounds_breakwaterd, 'breakwaterd'),
        'dagor': (optimizeDagor, objective_dagor, pbounds_dagor, 'dagor')
    }

    n_iterate = 55

    for opt_key, (optimize, objective_func, pbounds, opt_name) in optimization_config.items():
        if optimize and not skipOptimize:
            optimizer = BayesianOptimization(
                f=objective_func,
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True
            )

            # Try loading previous results
            try:
                best_past_results = load_optimal_parameters(control=opt_name, method=method)
                if 'parameters' in best_past_results:
                    initial_points = [best_past_results['parameters']]
                    print(f"[Bayesian Init] Loaded initial points from previous run for {opt_name}. Parameters:", initial_points)
                    # now add the initial points to the optimizer to start from the previous best results
                    lowerCaseParams = {key.lower(): value for key, value in initial_points[0].items()}
                    # drop the extra key of `only_frontend` if it exists
                    if 'only_frontend' in lowerCaseParams:
                        lowerCaseParams.pop('only_frontend')
                    # convert string values us to float
                    drop_keys = []
                    for key, value in lowerCaseParams.items():
                        if isinstance(value, str) and value.endswith('us'):
                            lowerCaseParams[key] = float(value[:-2])
                        elif isinstance(value, str) and value.endswith('ms'):
                            lowerCaseParams[key] = float(value[:-2]) * 1000
                        elif not isinstance(value, (int, float)):
                            # drop the key if the value is not a number
                            drop_keys.append(key)
                    for key in drop_keys:
                        lowerCaseParams.pop(key)
                    optimizer.probe(params=lowerCaseParams)
                else:
                    print(f"[Bayesian Init] No initial points found for {opt_name}.")
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"[Bayesian Opt] No previous results found for {opt_name}, starting fresh.")

            try:
                optimizer.maximize(init_points=10, n_iter=n_iterate)
            except Exception as e:
                print(f"[Bayesian Opt] Error Optimization encountered an error for {opt_name}:", e)
                # continue

            if 'params' in optimizer.max:
                # global SLO
                best_params = optimizer.max['params']
                # combine the best_params with the specific params
                best_params = {**configDict[opt_name], **best_params}
                print(f"[Bayesian Opt] Best parameters found for {opt_name}:", best_params)
                results = {
                    'target': optimizer.max['target'],
                    'parameters': roundDownParams(best_params),
                    'method': method,
                    'capacity': capacity,
                    'load': int(capacity.split('-')[1]),
                    'timestamp': timestamp,
                    # 'quantile': quantile,
                    # 'SLO': SLO,
                }
                record_optimal_parameters(f'bopt_{tightSLO}_{opt_name}_{method}_{capacity}_{timestamp}.json', results)
                # Save the iteration details at the end of each optimization
                save_iteration_details(optimizer, f'iterations_{opt_name}_{method}_{capacity}_{timestamp}.json')
            else:
                print(f"[Bayesian Opt] No successful optimization results available for {opt_name}.")

    for opt_key, (optimize, objective_func, pbounds, opt_name) in optimization_config.items():
        if optimize and not skipPostOptimize:
            # find the latest file with `bopt_{tightSLO}_{opt_name}_{method}_{capacity}_*.json`
            latest_bopt = get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/baysian-opt/'), pattern=f"bopt_{tightSLO}_{opt_name}_{method}_{capacity}_*.json")
            print(f"[Post Opt] Latest Bayesian Opt file for {opt_name}:", latest_bopt)
            bayesian_result = read_optimal_parameters(latest_bopt)
            for capact in capacity_range:
                print(f"[Post Opt] Analyzing file and run experiments in loop for {opt_name}:", bayesian_result)
                run_experiments_loop(opt_name, bayesian_result['parameters'], capact, method)
   

if __name__ == '__main__':
    # take an optional argument to specify the method
    # make method and SLO global variables

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('method', type=str, help='Specify the method')
    parser.add_argument('--tight-slo', action='store_true', default=False, help='Enable tight SLO')
    parser.add_argument('--skip-opt', action='store_true', default=False, help='Skip optimization')
    parser.add_argument('--skip-post-opt', action='store_true', default=True, help='Skip the loop after optimization')

    # add a new argument to specify the scheme to use for optimization, default is True for all optimization schemes
    parser.add_argument('--breakwater', action='store_true', default=False, help='Optimize Breakwater')
    parser.add_argument('--breakwaterd', action='store_true', default=False, help='Optimize BreakwaterD')
    parser.add_argument('--rajomon', action='store_true', default=False, help='Optimize Rajomon')
    parser.add_argument('--dagor', action='store_true', default=False, help='Optimize Dagor')
    # capacity is gpt1 by default, unless otherwise specified
    parser.add_argument('--tune', type=str, default='gpt1', help='Specify the weight for tuning')
    parser.add_argument('--tune-load', type=int, default=8000, help='Specify the load for tuning')

    args = parser.parse_args()

    global method, tightSLO, skipOptimize

    global optimizeBreakwater, optimizeBreakwaterD, optimizeRajomon, optimizeDagor
    
    # if specified, only optimize the specified scheme. If none is specified, optimize all
    optimizeBreakwater = args.breakwater
    optimizeBreakwaterD = args.breakwaterd
    optimizeRajomon = args.rajomon
    optimizeDagor = args.dagor

    if not optimizeBreakwater and not optimizeBreakwaterD and not optimizeRajomon and not optimizeDagor:
        optimizeRajomon = optimizeBreakwater = optimizeBreakwaterD = optimizeDagor = True


    method = args.method
    # map s1 s2 s3 to the actual method S_102000854 S_149998854 S_161142529
    map = {
        's1': 'S_102000854',
        's2': 'S_149998854',
        's3': 'S_161142529',
        'n4get': 'motivate-get',
        'n4set': 'motivate-set',
        'n4both': 'both-motivate',
    }
    method = map[method] if method in map else method

    # if method == 'both-motivate':
    #     # we focus on the get method for tuning
    #     method = 'motivate-get'
    #     motivate_both = True
        
    tightSLO = args.tight_slo
    skipOptimize = args.skip_opt
    skipPostOptimize = args.skip_post_opt

    if method == 'all-alibaba':
        maximum_goodput = 1000
        load_control = 10000
    else:
        load_no_control = get_sustainable_load(method)

        # load_control = 3 * load_no_control
        # load_control = 10000
        
        if 'motivate' in method:
            maximum_goodput = 30000
            load_control = 2 * load_no_control
        elif 'S_1' in method:
            maximum_goodput = 5000
            load_control = 2 * load_no_control
        elif method == 'compose' or method == 'user-timeline' or method == 'home-timeline':
            maximum_goodput = 10000
            load_control = 2 * load_no_control
        elif 'hotel' in method:
            maximum_goodput = 12000
            load_control = 2 * load_no_control
    # convert it to string
    if '-' in args.tune:
        capacity = args.tune
    else:
        load_control = str(load_control)
        capacity = args.tune + '-' + load_control


    # SLO = get_slo(method, tight=tightSLO)

    print(f"Now running the optimization for method: {method}, capacity: {capacity}, tightSLO: {tightSLO}, skipOptimize: {skipOptimize}")
    print(f"Optimizing Breakwater: {optimizeBreakwater}, BreakwaterD: {optimizeBreakwaterD}, Rajomon: {optimizeRajomon}, Dagor: {optimizeDagor}")

    main()
