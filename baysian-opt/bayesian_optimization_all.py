# In this file, we loop over 3 parameters: priceUpdateRate, delayTarget, and clientTimeout, as param1 2 3
# run the following bash command:
    # bash go run ./one-service.go A 50051 param1 param2 param3 & > ./server.output
    # cd /home/ying/Sync/Git/protobuf/ghz-client
    # go run ./main.go 2000 > ./ghz.output
    # cd /home/ying/Sync/Git/service-app/services/protobuf-grpc
    # bash kill_services.sh

# with the parameters, and calculate the average goodput as a target,
# This process is repeated and the parameters are updated using Bayesian Optimization.
 
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

# import the ~/Sync/Git/protobuf/ghz-results/visualize.py file and run its function `analyze_data` 
# with the optimal results from the Bayesian Optimization
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ghz-results'))
from visualize import analyze_data
from slo import get_slo


# Define global variables
global method
global SLO
global tightSLO
global skipOptimize


throughput_time_interval = '50ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds
# filename = '/home/ying/Sync/Git/protobuf/ghz-results/charon_stepup_nclients_1000.json'
rerun = True
# offset = 1
offset = 2 # for the all-methods-social and all-methods-hotel

quantile = 0.1

# method = 'compose'
# method = 'S_149998854'

# if method == 'compose':
#     SLO = 20 * 2 # 20ms * 2 = 40ms
#     capacity = 'random-7000'
#     maximum_goodput = 5000
# elif method == 'S_149998854':
#     SLO = 111 * 2 # 111ms * 2 = 222ms
#     maximum_goodput = 5000
#     capacity = 'random-7000'


def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["details"]


def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # in unit of milliseconds
    df.set_index('timestamp', inplace=True)
    df['latency'] = df['latency'] / 1000000
    # drop the rows if the `status` is Unavailable
    df = df[df['status'] != 'Unavailable']
    # remove the data within first second of df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]

    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df


def calculate_goodput(filename, average=True):
    # Insert your code for calculating average goodput here
    # Read the ghz.output file and calculate the average goodput
    # Return the average goodput
    # if filename is a list, average the goodput of all the files
    if isinstance(filename, list):
        goodput = 0
        for f in filename:
            goodput += calculate_goodput(f, average=average)
        goodput = goodput / len(filename)
        return goodput

    data = read_data(filename)
    df = convert_to_dataframe(data)
    if average:
        goodput = calculate_goodput_mean(df, slo=SLO)
    else:
        goodput = goodput_quantile(df, slo=SLO, quantile=quantile)
    return goodput  # Replace with your actual function


def calculate_goodput_mean(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='bfill').replace(np.nan, 0)
 
    # calculate goodput by counting the number of requests that are ok and have latency less than slo, and then divide by the time interval
    # time_interval is the time interval for calculating the goodput, last request time - first request time
    time_interval = (df.index.max() - df.index.min()).total_seconds()
    goodput = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].count() / time_interval   # take out the goodput during the last 3 seconds by index
    return goodput


def goodput_quantile(df, slo, quantile=0.1):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='bfill').replace(np.nan, 0)
    # take out the goodput during the last 3 seconds by index
    goodput = df['goodput']
    # goodput = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]['goodput']
    # return the goodput, but round it to 2 decimal places
    goodput = goodput.quantile(quantile)
    return goodput


def read_tail_latency(filename, percentile=99):
    if isinstance(filename, list):
        latency = 0
        for f in filename:
            latency += read_tail_latency(f, percentile=percentile)
        latency = latency / len(filename)
        return latency

    # with open(filename, 'r') as f:
    #     data = json.load(f)
    data = read_data(filename)        
    df = convert_to_dataframe(data)
    percentile_latency = df[(df['status'] == 'OK')]['latency'].quantile(percentile / 100)

    return percentile_latency  # Replace with your actual function
    # latency_distribution = data["latencyDistribution"]
    # if latency_distribution is None:
    #     print("[Error Lat] No latency found for file:", filename)
    #     return None
    # for item in latency_distribution:
    #     if item["percentage"] == percentile:
    #         latency_xxth = item["latency"]
    #         # convert the latency_99th from string to milliseconds
    #         latency_xxth = pd.Timedelta(latency_xxth).total_seconds() * 1000
    #         return latency_xxth
    # return None  # Return None if the 99th percentile latency is not found


# similarly to read_tail_latency, read_mean_latency returns the mean latency
def read_mean_latency(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["average"]  


def calculate_throughput(df):
    # sample throughput every time_interval
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    ok_requests_per_second = ok_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='bfill')
    return df

param_ranges = [(50, 500), (1000, 10000), (1, 20), ]  # (priceUpdateRate, delayTarget) 
  
configDict = {
    'charon': {
        'price_update_rate': 5000,  # Assuming numeric values for simplicity
        'token_update_rate': 5000,  # Assuming numeric values for simplicity
        'latency_threshold': 5000,
        'price_step': 10,
        'price_strategy': 'proportional',
        'lazy_update': 'true',
        'rate_limiting': 'true',
        'only_frontend': 'false',
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

def generate_output_filename(interceptor_type, method, capacity):
    directory = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/')
    outputFile = f"social-{method}-control-{interceptor_type}-parallel-capacity-{capacity}-*.json.output"
    if method == "all-methods-social" or method == "compose" or method == "home-timeline" or method == "user-timeline" or method == "all-methods-hotel":
        outputFile = f"social-{method}-control-{interceptor_type}-parallel-capacity-{capacity}-01*.json.output"
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


def parse_configurations(config_str):
    # Remove potential \n[ 
    config_str = config_str.replace('\n[', '[')
    # Remove surrounding brackets and split by '} {'
    config_items = config_str.strip('[]').split('} {')
    config_dict = {}

    for item in config_items:
        # Remove potential curly braces and split by space
        key_value = item.replace('{', '').replace('}', '').split(' ', 1)
        if len(key_value) == 2:
            config_dict[key_value[0]] = key_value[1]

    return config_dict

def check_previous_run_exists(interceptor_type, method, capacity, combined_params, existing_files=None):
    if method == "all-methods-social":
        sub_methods = ["user-timeline", "home-timeline", "compose"]
    if method == "all-methods-hotel":
        sub_methods = ["hotels-http", "reservation-http", "user-http", "recommendations-http"]
    if method == "all-methods-social" or method == "all-methods-hotel":
        # return a list of files that match the parameters
        filesets = find_method_file_sets(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), sub_methods, capacity)
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

                    if 'Charon Configurations:' in content:
                        # Extract the configurations string from the file
                        start = content.find('Charon Configurations:') + len('Charon Configurations:')
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
    
    # Search for files that match the pattern
    matching_files = glob.glob(output_filename_pattern)

    # Iterate through matching files and check contents
    for filename in matching_files:
        # if existing_files is not None then skip this file if it has the same timestamp as the existing files
        if existing_files is not None:
            # Extract the timestamp from the filename
            timestamp = filename.split('-')[-1].split('.')[0]
            # Skip this file if the timestamp is in the existing_files
            if timestamp in existing_files:
                continue
        try:
            with open(filename, 'r') as file:
                content = file.read()

                if 'Charon Configurations:' in content:
                    # Extract the configurations string from the file
                    start = content.find('Charon Configurations:') + len('Charon Configurations:')
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
    full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c {load} -s parallel --{interceptor_type}'"

    # Set environment variables in Python
    env = os.environ.copy()
    for key, value in combined_params.items():
        env[key] = str(value)
    # add the method to env
    env['METHOD'] = method

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
    full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c {capact} -s parallel --{interceptor_type}'" if interceptor_type != 'plain' else f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c {capact} -s parallel'"

    # Set environment variables in Python
    env = os.environ.copy()
    for key, value in combined_params.items():
        env[key] = str(value)
    # add the method to env
    env['METHOD'] = methodToRun

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
def objective(interceptor_type, **params):
    hugePenalty = -999999999
    load = int(capacity.split('-')[1])
    latest_file = run_experiments(interceptor_type, load, '', **params)
    if latest_file is None:
        print("No file found for objective function")
        return hugePenalty
    goodput = calculate_goodput(latest_file, average=True)
    tail_latency = read_tail_latency(latest_file, percentile=99)
    print("[Experiment Ran] average goodput:", goodput, "and tail latency:", tail_latency, "\nfrom file:", latest_file)
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
        obj = goodput - 10 * (tail_latency - SLO) if tail_latency > SLO else goodput
    elif 'gpt2' in capacity:
        obj = goodput - (tail_latency - SLO) ** 2 if tail_latency > SLO else goodput
    elif 'tgpt' in capacity:
        obj = calculate_goodput(latest_file, average=False)
    return obj / maximum_goodput


def objective_dual(interceptor_type, **params):
    obj = []
    # same as objective, but run twice with different loads, and return the lower goodput
    latest_file = ''
    for loadTest in [int(capacity.split('-')[1]), ] * 2 :
        latest_file = run_experiments(interceptor_type, loadTest, latest_file, **params)
        if latest_file is None:
            print("No file found for objective function")
            return -999999999
        goodput = calculate_goodput(latest_file, average=True)
        tail_goodput = calculate_goodput(latest_file, average=False)
        tail_latency = read_tail_latency(latest_file, percentile=99)
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
            obj.append(goodput - 10 * (tail_latency - SLO) if tail_latency > SLO else goodput)
        elif 'gpt2' in capacity:
            obj.append(goodput - (tail_latency - SLO) ** 2 if tail_latency > SLO else goodput)
        elif 'tgpt' in capacity:
            obj.append(tail_goodput)
    # return the mean obj of the two loads
    return np.mean(obj) / maximum_goodput


# def plot_opt(priceUpdateRate, clientTimeout, delayTarget, guidePrice, priceStep):
#     # Convert the parameters to int64
#     priceUpdateRate = int(priceUpdateRate)
#     # priceUpdateRate = 100
#     delayTarget = int(delayTarget)
#     guidePrice = int(guidePrice)
#     # clientTimeout = int(clientTimeout)
#     clientTimeout = int(clientTimeout)
#     priceStep = int(priceStep)

#     # Run the experiments
#     run_experiments(priceUpdateRate, delayTarget, priceStep, guidePrice, clientTimeout)

#     analyze_data(filename)


# # Define the objective function to optimize
# def objective_wrapper(priceUpdateRate, delayTarget, priceStep,):
#     return objective(priceUpdateRate, 0, delayTarget, -1, priceStep)


def plot_opt_wrapper(priceUpdateRate, delayTarget, priceStep,):
    return plot_opt(priceUpdateRate, 0, delayTarget, -1, priceStep)


def get_latest_file(path, pattern="*.json"):
    """Return the latest file in a given directory matching the pattern."""
    list_of_files = glob.glob(os.path.join(path, pattern))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def objective_charon(price_update_rate, token_update_rate, latency_threshold, price_step):
    # return objective('charon', price_update_rate, latency_threshold, price_step)
    return objective_dual('charon', price_update_rate=price_update_rate, token_update_rate=token_update_rate, latency_threshold=latency_threshold, price_step=price_step)


def objective_breakwater(breakwater_slo, breakwater_a, breakwater_b, breakwater_initial_credit, breakwater_client_expiration, breakwater_rtt):
    return objective_dual('breakwater', breakwater_slo=breakwater_slo, breakwater_a=breakwater_a, breakwater_b=breakwater_b,
                          breakwater_initial_credit=breakwater_initial_credit, breakwater_client_expiration=breakwater_client_expiration, breakwater_rtt=breakwater_rtt)

def objective_breakwaterd(breakwater_slo, breakwater_a, breakwater_b, breakwater_initial_credit, breakwater_client_expiration, breakwater_rtt,
                          breakwaterd_slo, breakwaterd_a, breakwaterd_b, breakwaterd_initial_credit, breakwaterd_client_expiration, breakwaterd_rtt):
    return objective_dual('breakwaterd', breakwater_slo=breakwater_slo, breakwater_a=breakwater_a, breakwater_b=breakwater_b, breakwater_initial_credit=breakwater_initial_credit, 
                          breakwater_client_expiration=breakwater_client_expiration, breakwaterd_slo=breakwaterd_slo, breakwaterd_a=breakwaterd_a, 
                          breakwaterd_b=breakwaterd_b, breakwaterd_initial_credit=breakwaterd_initial_credit, breakwaterd_client_expiration=breakwaterd_client_expiration,
                          breakwater_rtt=breakwater_rtt, breakwaterd_rtt=breakwaterd_rtt)


def objective_dagor(dagor_queuing_threshold, dagor_alpha, dagor_beta, dagor_admission_level_update_interval, dagor_umax):
    return objective_dual('dagor', dagor_queuing_threshold=dagor_queuing_threshold, dagor_alpha=dagor_alpha, dagor_beta=dagor_beta, dagor_admission_level_update_interval=dagor_admission_level_update_interval, dagor_umax=dagor_umax)


# Define a function to read and parse the JSON file
def parse_file(file_path):
    folder = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/')
    file_path = os.path.join(folder, file_path)
    with open(file_path, 'r') as file:
        content = file.read()

    # Extracting 'METHOD' and 'INTERCEPT' parts
    method = re.search(r'METHOD: (\w+)', content)
    intercept = re.search(r'INTERCEPT: (\w+)', content)

    # Extracting Charon Configurations
    charon_config_match = re.search(r'Charon Configurations:\s*\[(.*?)\]', content, re.DOTALL)
    if charon_config_match:
        charon_config_str = charon_config_match.group(1)
        charon_config_pairs = charon_config_str.split('} {')
        charon_config = {item.split(' ')[0]: item.split(' ')[1] for item in charon_config_pairs}

    return {
        'METHOD': method.group(1) if method else None,
        'INTERCEPT': intercept.group(1) if intercept else None,
        'Charon_Configurations': charon_config
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


def roundDownParams(params):
    # capitalize the params
    params = {key.upper(): value for key, value in params.items()}
    # round the parameters to int unless the key is in the following list
    for key in params:
        # for numerical values only, skip the string values
        if isinstance(params[key], str):
            continue
        if key not in ['BREAKWATER_A', 'BREAKWATER_B', 'DAGOR_ALPHA', 'DAGOR_BETA', 'BREAKWATERD_A', 'BREAKWATERD_B']:
            params[key] = int(params[key])

    # add `us` to the end of the values in combined_params if the key is in the following list
    # ['PRICE_UPDATE_RATE', 'LATENCY_THRESHOLD', 'BREAKWATER_SLO', 'BREAKWATER_CLIENT_EXPIRATION', 'DAGOR_QUEUING_THRESHOLD', 'DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL']
    for key in params:
        if key in ['PRICE_UPDATE_RATE', 'TOKEN_UPDATE_RATE', 'LATENCY_THRESHOLD', \
                   'BREAKWATER_SLO', 'BREAKWATER_CLIENT_EXPIRATION', \
                   'BREAKWATERD_SLO', 'BREAKWATERD_CLIENT_EXPIRATION', \
                   'BREAKWATER_RTT', 'BREAKWATERD_RTT', \
                   'DAGOR_QUEUING_THRESHOLD', 'DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL']:
            # if there is already a `us` or `ms` at the end of the value, don't add another `us`
            # if it is a string
            if isinstance(params[key], str):
                if params[key][-2:] not in ['us', 'ms']:
                    params[key] = str(params[key]) + 'us'
            if isinstance(params[key], int):
                params[key] = str(params[key]) + 'us'
    return params


def save_iteration_details(optimizer, file_path):
    try:
        # Convert the iteration details to a JSON-compatible format
        iteration_details = optimizer.res
        iteration_details_json = json.dumps(iteration_details, default=str)

        # Save to a file
        with open(file_path, 'w') as file:
            file.write(iteration_details_json)
        print(f"Iteration details successfully saved to {file_path}")
    except IOError as e:
        print(f"IOError encountered while saving iteration details: {e}")
    except TypeError as e:
        print(f"TypeError encountered during JSON serialization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    
    pbounds_charon = {
        # 'price_update_rate': (100, 500000),  # Example range
        # 'token_update_rate': (100, 500000),  # Example range
        # 'latency_threshold': (10, 50000),  # Example range
        # 'price_step': (1, 500)  # Example range
        # the range above is too large... I will use the following range based on the empirical results
        'price_update_rate': (1000, 30000), 
        'token_update_rate': (1000, 30000), 
        'price_step': (10, 150),
        'latency_threshold': (100, 3000),
    }
    pbounds_breakwater = {
        # 'breakwater_slo': (100, 100000),  # Example range
        # 'breakwater_a': (0.0001, 0.9),    # Example range
        # 'breakwater_b': (0.01, 0.9),      # Example range
        # 'breakwater_initial_credit': (10, 3000),      # Example range
        # 'breakwater_client_expiration': (1, 100000) # Example range
        # the range above is too large... I will use the following range based on the empirical results {BREAKWATER_SLO 749158us} {BREAKWATER_A 11.692336257688945} {BREAKWATER_B 0.004983989475762508} {B    REAKWATER_CLIENT_EXPIRATION 13317us} {BREAKWATER_INITIAL_CREDIT 59} 
        'breakwater_slo': (100, 5000),
        'breakwater_a': (0, 20),
        'breakwater_b': (0, 2),
        'breakwater_initial_credit': (1, 2000),
        'breakwater_client_expiration': (0, 5000),
        'breakwater_rtt': (0, 20000),
    }

    # pbounds_breakwaterd = {
    #     'breakwater_slo': (10000, 40000),
    #     'breakwater_a': (0, 2),
    #     'breakwater_b': (0, 2),
    #     'breakwater_initial_credit': (1, 400),
    #     'breakwater_client_expiration': (1, 3000),
    #     'breakwaterd_slo': (30000, 80000),
    #     'breakwaterd_a': (0, 10),
    #     'breakwaterd_b': (0, 30),
    #     'breakwaterd_initial_credit': (1, 1000),
    #     'breakwaterd_client_expiration': (10000, 40000),
    #     'breakwater_rtt': (1000, 20000),
    #     'breakwaterd_rtt': (1000, 20000),
    # }

    pbounds_breakwaterd = {
        'breakwater_slo': (10000, 50000),
        'breakwater_a': (0, 20),
        'breakwater_b': (0, 10),
        'breakwater_initial_credit': (1, 400),
        'breakwater_client_expiration': (1, 5000),
        'breakwaterd_slo': (10000, 80000),
        'breakwaterd_a': (0, 10),
        'breakwaterd_b': (0, 10),
        'breakwaterd_initial_credit': (1, 5000),
        'breakwaterd_client_expiration': (10000, 100000),
        'breakwater_rtt': (0, 20000),
        'breakwaterd_rtt': (1000, 20000),
    }
    pbounds_dagor = {
        'dagor_queuing_threshold': (100, 50000),  # Example range
        'dagor_alpha': (0, 1.5),              # Example range
        'dagor_beta': (0, 0.5),             # Example range
        'dagor_admission_level_update_interval': (10000, 20000),  # Example range
        'dagor_umax': (5, 20)  # Example range
    }

    if not tightSLO:
        # loosen the control target for charon and breakwater.
        pbounds_charon['latency_threshold'] = (100, 10000)
        pbounds_breakwater['breakwater_slo'] = (100, 20000)
        pbounds_breakwaterd['breakwater_a'] = (0, 30)
        pbounds_dagor['dagor_queuing_threshold'] = (100000, 500000)  # Example range
        pbounds_dagor['dagor_alpha'] = (0, 2)              # Example range
        pbounds_dagor['dagor_admission_level_update_interval'] = (100, 20000)

        
        if 'S_16' in method:
            pbounds_charon['latency_threshold'] = (100, 30000)
        if 'S_14' in method:
            # pbounds_breakwater['breakwater_a'] = (0, 30)
            pbounds_charon['price_update_rate'] = (1000, 40000)
            pbounds_charon['token_update_rate'] = (1000, 40000)
        # if 'S_10' in method:
        #     pbounds_breakwater['breakwater_a'] = (0, 30)

    # if 'all' in method:
        # change the pbounds for 4 mechanisms
        # pbounds_charon['price_step'] = (10, 80)
        # pbounds_breakwater = {
        #     'breakwater_slo': (100, 20000),
        #     'breakwater_a': (0, 2),
        #     'breakwater_b': (0, 1),
        #     'breakwater_initial_credit': (1, 1000),
        #     'breakwater_client_expiration': (10, 10000),
        # }
        # pbounds_breakwaterd = {
        #     'breakwater_slo': (20000, 50000),
        #     'breakwater_a': (0, 2),
        #     'breakwater_b': (0, 1),
        #     'breakwater_initial_credit': (1, 500),
        #     'breakwater_client_expiration': (1, 5000),
        #     'breakwaterd_slo': (30000, 80000),
        #     'breakwaterd_a': (0, 10),
        #     'breakwaterd_b': (0, 20),
        #     'breakwaterd_initial_credit': (10, 500),
        #     'breakwaterd_client_expiration': (10000, 40000),
        # }
        # pbounds_dagor = {
        #     'dagor_queuing_threshold': (100000, 500000),  # Example range
        #     'dagor_alpha': (0, 1.5),              # Example range
        #     'dagor_beta': (0, 2),             # Example range
        #     'dagor_admission_level_update_interval': (10000, 20000),  # Example range
        #     'dagor_umax': (2, 20)  # Example range
        # }
    optimizeCharon = True
    optimizeBreakwater = True
    optimizeBreakwaterD = True
    optimizeDagor = True

    # run the experiments with the interceptors for all capacities
    capacity_step = 1000
    capacity_start = 1000
    capacity_end = 17000
    # if '8000' in capacity:
    #     capacity_range = range(5000, 13500, 500) if not tightSLO else range(4000, 10500, 500)
    # else:
    #     capacity_range = range(6000, 12500, 500)

    # if 'all' in method:
    #     capacity_range = range(2000, 9500, 500)
    capacity_range = range(capacity_start, capacity_end, capacity_step)

    timestamp = datetime.datetime.now().strftime("%m-%d")
    print("Timestamp:", timestamp)
    print("method:", method)
    
    # skipOptimize = False

    # if skipOptimize:
    #     # set all the optimize to False
    #     optimizeCharon = optimizeBreakwater = optimizeBreakwaterD = optimizeDagor = False


    optimization_config = {
        'charon': (optimizeCharon, objective_charon, pbounds_charon, 'charon'),
        'breakwater': (optimizeBreakwater, objective_breakwater, pbounds_breakwater, 'breakwater'),
        'breakwaterd': (optimizeBreakwaterD, objective_breakwaterd, pbounds_breakwaterd, 'breakwaterd'),
        'dagor': (optimizeDagor, objective_dagor, pbounds_dagor, 'dagor')
    }

    for opt_key, (optimize, objective_func, pbounds, opt_name) in optimization_config.items():
        if optimize and not skipOptimize:
            optimizer = BayesianOptimization(
                f=objective_func,
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True
            )
            try:
                optimizer.maximize(init_points=10, n_iter=55)
            except Exception as e:
                print(f"[Bayesian Opt] Error Optimization encountered an error for {opt_name}:", e)
                # continue

            if 'params' in optimizer.max:
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
                    'quantile': quantile,
                    'SLO': SLO,
                }
                record_optimal_parameters(f'bopt_{tightSLO}_{opt_name}_{method}_{capacity}_{timestamp}.json', results)
                # Save the iteration details at the end of each optimization
                save_iteration_details(optimizer, f'iterations_{opt_name}_{method}_{capacity}_{timestamp}.json')
            else:
                print(f"[Bayesian Opt] No successful optimization results available for {opt_name}.")

    for opt_key, (optimize, objective_func, pbounds, opt_name) in optimization_config.items():
        if optimize:
            # find the latest file with `bopt_{tightSLO}_{opt_name}_{method}_{capacity}_*.json`
            latest_bopt = get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/baysian-opt/'), pattern=f"bopt_{tightSLO}_{opt_name}_{method}_{capacity}_*.json")
            bayesian_result = read_optimal_parameters(latest_bopt)
            for capact in capacity_range:
                print(f"[Post Opt] Analyzing file and run experiments in loop for {opt_name}:", bayesian_result)
                run_experiments_loop(opt_name, bayesian_result['parameters'], capact, method)
    
    # # run the experiments without interceptors too for all capacities
    # for capact in capacity_range:
    #     print("[Post Opt] Analyzing file and run experiments in loop for no-interceptor:")
    #     run_experiments_loop('plain', {}, capact, method)


def check_goodput(file):
    # check the goodput distribution of social-compose-control-charon-parallel-capacity-8000-1211_1804.json
    dir = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/')
    filename = os.path.join(dir, file)

    goodput = calculate_goodput(filename, average=True)
    tail_latency = read_tail_latency(filename, percentile=99)
  
    data = read_data(filename)
    df = convert_to_dataframe(data)
    goodput_mean = calculate_goodput_mean(df, slo=SLO)
    print("goodput_mean:", goodput_mean)
    goodquantile = goodput_quantile(df, slo=SLO, quantile=quantile)
    print("goodquantile:", goodquantile)
    # print(df["goodput"].describe())
    # plot the histogram of goodput
    df["goodput"].plot.hist(bins=100)
    plt.show()
    loadTested = int(re.search(r'capacity-(\d+)', file).group(1))
    if 'gpt0' in capacity:
        obj = goodput
    elif 'gpt1' in capacity:
        obj = goodput - 10 * (tail_latency - SLO) if tail_latency > SLO else goodput
    elif 'gpt2' in capacity:
        obj = goodput - (tail_latency - SLO) ** 2 if tail_latency > SLO else goodput
    print("[Experiment Ran] objecive {} with average goodput: {} and loadTested: {}".format(obj, goodput_mean, loadTested))


if __name__ == '__main__':
    # take an optional argument to specify the method
    # make method and SLO global variables

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('method', type=str, help='Specify the method')
    parser.add_argument('--tight-slo', action='store_true', default=False, help='Enable tight SLO')
    parser.add_argument('--skip-opt', action='store_true', default=False, help='Skip optimization')
    args = parser.parse_args()

    method = args.method
    tightSLO = args.tight_slo
    skipOptimize = args.skip_opt

    # if len(sys.argv) > 1:
    #     method = sys.argv[1]
    # else:
    #     raise Exception("Please specify the method")
    
    # tightSLO = False
    # if len(sys.argv) > 2:
    #     tightSLO = sys.argv[2] == 'tightSLO'
    
    capacity = 'gpt2-10000' if 'S_' in method else 'gpt2-8000'
    maximum_goodput = 10000
    # if method == 'S_102000854':
    #     capacity = 'gpt1-10000'
    if method == 'hotels-http':
        capacity = 'gpt1-2000'

    if method == 'all-methods-social':
        capacity = 'gpt1-6000'
    if method == 'all-methods-hotel':
        capacity = 'gpt1-8000'

    # if 'S_' in method:
    #     capacity = 'gpt0-10000'
    # set the SLO based on the method
    SLO = get_slo(method, tight=tightSLO)



    main()

    # for capacity in range(4000, 10000, 500), get latest file and run the check below
    # check_goodput('social-compose-control-breakwater-parallel-capacity-6807-1213_2351.json')
    # check_goodput('social-compose-control-breakwaterd-parallel-capacity-6601-1214_0006.json')
    # for load in range(4000, 10000, 500):
    #     experiment = get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), f'social-compose-control-charon-parallel-capacity-{load}-1213*json')
    #     check_goodput(experiment)
