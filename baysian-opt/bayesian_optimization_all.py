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
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime

import sklearn

# import the ~/Sync/Git/protobuf/ghz-results/visualize.py file and run its function `analyze_data` 
# with the optimal results from the Bayesian Optimization
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ghz-results'))
from visualize import analyze_data

throughput_time_interval = '50ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds
# filename = '/home/ying/Sync/Git/protobuf/ghz-results/charon_stepup_nclients_1000.json'
rerun = True
offset = 1

quantile = 0.05

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
    # take out the goodput during the last 3 seconds by index
    goodput = df['goodput']
    # goodput = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]['goodput']
    # return the goodput, but round it to 2 decimal places
    goodput = goodput.mean()
    # goodput = round(goodput, -2)
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
    with open(filename, 'r') as f:
        data = json.load(f)
        
    latency_distribution = data["latencyDistribution"]
    if latency_distribution is None:
        print("No latency found for file:", filename)
        return None
    for item in latency_distribution:
        if item["percentage"] == percentile:
            latency_xxth = item["latency"]
            # convert the latency_99th from string to milliseconds
            latency_xxth = pd.Timedelta(latency_xxth).total_seconds() * 1000
            return latency_xxth
    return None  # Return None if the 99th percentile latency is not found


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
        'PRICE_UPDATE_RATE': 5000,  # Assuming numeric values for simplicity
        'TOKEN_UPDATE_RATE': 5000,  # Assuming numeric values for simplicity
        'LATENCY_THRESHOLD': 5000,
        'PRICE_STEP': 10,
        'PRICE_STRATEGY': 'proportional',
        'LAZY_UPDATE': 'true',
        'RATE_LIMITING': 'true',
        'ONLY_FRONTEND': 'false',
    },
    'breakwater': {
        'BREAKWATER_SLO': 12500,
        'BREAKWATER_A': 0.001,
        'BREAKWATER_B': 0.02,
        'BREAKWATER_INITIAL_CREDIT': 400,
        'BREAKWATER_CLIENT_EXPIRATION': 10000,
        'ONLY_FRONTEND': 'true',
    },    
    'breakwaterd': {
        'BREAKWATER_SLO': 12500,
        'BREAKWATER_A': 0.001,
        'BREAKWATER_B': 0.02,
        'BREAKWATER_INITIAL_CREDIT': 400,
        'BREAKWATER_CLIENT_EXPIRATION': 10000,
        'ONLY_FRONTEND': 'false',
        'BREAKWATERD_SLO': 12500,
        'BREAKWATERD_A': 0.001,
        'BREAKWATERD_B': 0.02,
        'BREAKWATERD_INITIAL_CREDIT': 400,
        'BREAKWATERD_CLIENT_EXPIRATION': 10000,
    },
    'dagor': {
        'DAGOR_QUEUING_THRESHOLD': 2000,
        'DAGOR_ALPHA': 0.05,
        'DAGOR_BETA': 0.01,
        'DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL': 10000,
        'DAGOR_UMAX': 20,
    }
}


# Define the function that runs the service and client as experiments, takes the interceptor type and the parameters as arguments,
# the params are dictionary of the parameters. and the load is the capacity of the service, with a default value of load = int(capacity.split('-')[1])
def run_experiments(interceptor_type, load, **params):
    # Map the interceptor_type to its specific parameters
    specific_params = configDict[interceptor_type]

    # Combine specific params and passed params
    # overwrite the specific params with the passed params
    combined_params = {**specific_params, **params}
    # combined_params = {**specific_params, **params}

    # round the parameters to int unless the key is in the following list
    combined_params = roundDownParams(combined_params)

    # print with 2 decimal places
    print("[Bayesian Opt] Combined params:", {key: round(float(value), 2) if isinstance(value, float) else value for key, value in combined_params.items()})
    # Prepare the environment variable command string
    # env_vars_command = ' '.join(f'export {key}="{value}";' for key, value in combined_params.items())
    
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
    print("Experiment output:", stdout.decode())
    print("Experiment errors:", stderr.decode())
    # if `ssh` is found in the stderr, raise an exception
    if 'ssh' in stderr.decode() or 'publickey' in stderr.decode():
        raise Exception("ssh error found in stderr, please check the stderr")
    return get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'))


def run_experiments_loop(interceptor_type, interceptor_configs, capact, methodToRun):
    # Map the interceptor_type to its specific parameters
    specific_params = configDict[interceptor_type]

    # Combine specific params and passed params
    # interceptor_configs is a dictionary of the interceptor configurations
    # overwrite the specific params with the passed params
    combined_params = {**specific_params, **interceptor_configs}

    # combined_params = {**specific_params, **params}

    # round the parameters to int unless the key is in the following list
    combined_params = roundDownParams(combined_params)

    # print with 2 decimal places
    print("[Bayesian Opt] Combined params:", {key: round(float(value), 2) if isinstance(value, float) else value for key, value in combined_params.items()})
    # Prepare the environment variable command string
    # env_vars_command = ' '.join(f'export {key}="{value}";' for key, value in combined_params.items())

    # Full command to source envs.sh and run the experiment
    # full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && {env_vars_command} ~/Sync/Git/service-app/cloudlab/scripts/bayesian_experiments.sh -c {capacity} -s parallel --{interceptor_type}'"
    full_command = f"bash -c 'source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh && ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c {capact} -s parallel --{interceptor_type}'"

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
    latest_file = run_experiments(interceptor_type, load, **params)
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
    for loadTest in [int(capacity.split('-')[1]), ] * 2 :
        latest_file = run_experiments(interceptor_type, loadTest, **params)
        if latest_file is None:
            print("No file found for objective function")
            return -999999999
        goodput = calculate_goodput(latest_file, average=True)
        tail_latency = read_tail_latency(latest_file, percentile=99)
        print("[Experiment Ran] average goodput:", goodput, "and tail latency:", tail_latency, "\nfrom file:", latest_file)
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
            obj.append(calculate_goodput(latest_file, average=False))
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

pbounds_charon = {
    'price_update_rate': (100, 500000),  # Example range
    'token_update_rate': (100, 500000),  # Example range
    'latency_threshold': (10, 500000),  # Example range
    'price_step': (1, 500)  # Example range
}

def objective_breakwater(breakwater_slo, breakwater_a, breakwater_b, breakwater_initial_credit, breakwater_client_expiration):
    return objective_dual('breakwater', breakwater_slo=breakwater_slo, breakwater_a=breakwater_a, breakwater_b=breakwater_b, breakwater_initial_credit=breakwater_initial_credit, breakwater_client_expiration=breakwater_client_expiration)

def objective_breakwaterd(breakwater_slo, breakwater_a, breakwater_b, breakwater_initial_credit, breakwater_client_expiration, \
                          breakwaterd_slo, breakwaterd_a, breakwaterd_b, breakwaterd_initial_credit, breakwaterd_client_expiration):
    return objective_dual('breakwaterd', breakwater_slo=breakwater_slo, breakwater_a=breakwater_a, breakwater_b=breakwater_b, breakwater_initial_credit=breakwater_initial_credit, breakwater_client_expiration=breakwater_client_expiration,
                        breakwaterd_slo=breakwaterd_slo, breakwaterd_a=breakwaterd_a, breakwaterd_b=breakwaterd_b, breakwaterd_initial_credit=breakwaterd_initial_credit, breakwaterd_client_expiration=breakwaterd_client_expiration)

pbounds_breakwater = {
    'breakwater_slo': (100, 100000),  # Example range
    'breakwater_a': (0.0001, 0.9),    # Example range
    'breakwater_b': (0.01, 0.9),      # Example range
    'breakwater_initial_credit': (10, 3000),      # Example range
    'breakwater_client_expiration': (1, 100000) # Example range
}

pbounds_breakwaterd = {
    'breakwaterd_slo': (1000, 1000000),
    'breakwaterd_a': (0.0001, 50),
    'breakwaterd_b': (0.001, 0.03),
    'breakwaterd_initial_credit': (1, 20000),
    'breakwaterd_client_expiration': (500, 25000),
}

pbounds_breakwaterd = {
    **pbounds_breakwater,
    **pbounds_breakwaterd,
}


def objective_dagor(dagor_queuing_threshold, dagor_alpha, dagor_beta, dagor_admission_level_update_interval, dagor_umax):
    return objective_dual('dagor', dagor_queuing_threshold=dagor_queuing_threshold, dagor_alpha=dagor_alpha, dagor_beta=dagor_beta, dagor_admission_level_update_interval=dagor_admission_level_update_interval, dagor_umax=dagor_umax)

pbounds_dagor = {
    'dagor_queuing_threshold': (1000, 1000000),  # Example range
    'dagor_alpha': (0.1, 0.95),              # Example range
    'dagor_beta': (0.001, 0.5),             # Example range
    'dagor_admission_level_update_interval': (1000, 1000000),  # Example range
    'dagor_umax': (2, 20)  # Example range
}


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
                   'DAGOR_QUEUING_THRESHOLD', 'DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL']:
            # if there is already a `us` or `ms` at the end of the value, don't add another `us`
            # if it is a string
            if isinstance(params[key], str):
                if params[key][-2:] not in ['us', 'ms']:
                    params[key] = str(params[key]) + 'us'
            if isinstance(params[key], int):
                params[key] = str(params[key]) + 'us'
    return params


def main():
    optimizeCharon = True
    optimizeBreakwater = True
    optimizeBreakwaterD = True
    optimizeDagor = True

    capacity_range = range(4000, 10500, 500)

    timestamp = datetime.datetime.now().strftime("%m-%d")
    print("Timestamp:", timestamp)
    print("method:", method)
    
    skipOptimize = False

    if skipOptimize:
        # set all the optimize to False
        optimizeCharon, optimizeBreakwater, optimizeBreakwaterD, optimizeDagor = False, False, False, False

    if optimizeCharon:
        # Optimize for Charon
        optimizer_charon = BayesianOptimization(
            f=objective_charon,
            pbounds=pbounds_charon,
            random_state=1,
            allow_duplicate_points=True,
        )
        # optimizer_charon.set_gp_params(alpha=1e-1,
        #                                normalize_y=True, kernel=sklearn.gaussian_process.kernels.Matern(length_scale=10),
        #                                n_restarts_optimizer=5)
        try:
            optimizer_charon.maximize(init_points=10, n_iter=35)
        except Exception as e:
            print("Optimization encountered an error:", e)

        # Attempt to access the best parameters found
        if 'params' in optimizer_charon.max:
            best_params_charon = optimizer_charon.max['params']
            print("Best parameters found:", best_params_charon)
        else:
            print("No successful optimization results available.")
    
        # optimizer_charon.maximize(init_points=15, n_iter=35)
        # best_params_charon = optimizer_charon.max['params']
        # print("Best for Charon has Target:", optimizer_charon.max['target'], "with parameters:", best_params_charon)
        # ... after running optimizer_charon.maximize(...)
        results_charon = {
            'target': optimizer_charon.max['target'],
            'parameters': roundDownParams(best_params_charon),
        }
        record_optimal_parameters(f'gpbayes_charon_{method}_{capacity}_{timestamp}.json', results_charon)

    bayesian_result = read_optimal_parameters(f'gpbayes_charon_{method}_{capacity}_{timestamp}.json') 
    for capact in capacity_range:
        print("Analyzing file and run experiments in loop:", bayesian_result)
        run_experiments_loop('charon', bayesian_result['parameters'], capact, method)

        
    if optimizeBreakwater:
        # Optimize for Breakwater
        optimizer_breakwater = BayesianOptimization(
            f=objective_breakwater,
            pbounds=pbounds_breakwater,
            random_state=1,
            allow_duplicate_points=True
        )
        # optimizer_breakwater.set_gp_params(alpha=1e-3,
        #                                    normalize_y=True, # kernel=sklearn.gaussian_process.kernels.Matern(length_scale=10),
        #                                    n_restarts_optimizer=5)
        # optimizer_breakwater.maximize(init_points=10, n_iter=35)
        # best_params_breakwater = optimizer_breakwater.max['params']
        try :
            optimizer_breakwater.maximize(init_points=10, n_iter=35)
        except Exception as e:
            print("Optimization encountered an error:", e)
        if 'params' in optimizer_breakwater.max:
            best_params_breakwater = optimizer_breakwater.max['params']
            print("Best parameters found:", best_params_breakwater)
        else:
            print("No successful optimization results available.")
        # print("Best for Breakwater has Target:", optimizer_breakwater.max['target'], "with parameters:", best_params_breakwater)

        results_breakwater = {
            'target': optimizer_breakwater.max['target'],
            'parameters': roundDownParams(best_params_breakwater),
        }
        record_optimal_parameters(f'gpbayes_breakwater_{method}_{capacity}_{timestamp}.json', results_breakwater)
    
    bayesian_result = read_optimal_parameters(f'gpbayes_breakwater_{method}_{capacity}_{timestamp}.json')
    for capact in capacity_range:
        print("Analyzing file and run experiments in loop:", bayesian_result)
        run_experiments_loop('breakwater', bayesian_result['parameters'], capact, method)


    if optimizeBreakwaterD:
        # Optimize for Breakwater
        optimizer_breakwaterd = BayesianOptimization(
            f=objective_breakwaterd,
            pbounds=pbounds_breakwaterd,
            random_state=1,
            allow_duplicate_points=True
        )
        # optimizer_breakwaterd.set_gp_params(alpha=1e-3,
        #                                     normalize_y=True, # kernel=sklearn.gaussian_process.kernels.Matern(length_scale=10),
        #                                     n_restarts_optimizer=5)
        # optimizer_breakwaterd.maximize(init_points=10, n_iter=35)
        # best_params_breakwaterd = optimizer_breakwaterd.max['params']
        # print("Best for BreakwaterD has target:", optimizer_breakwaterd.max['target'], "with parameters:", best_params_breakwaterd)
        try:
            optimizer_breakwaterd.maximize(init_points=10, n_iter=35)
        except Exception as e:
            print("Optimization encountered an error:", e)
        if 'params' in optimizer_breakwaterd.max:
            best_params_breakwaterd = optimizer_breakwaterd.max['params']
            print("Best parameters found:", best_params_breakwaterd)
        else:
            print("No successful optimization results available.")

        results_breakwaterd = {
            'target': optimizer_breakwaterd.max['target'],
            'parameters': roundDownParams(best_params_breakwaterd),
        }
        record_optimal_parameters(f'gpbayes_breakwaterd_{method}_{capacity}_{timestamp}.json', results_breakwaterd)

    bayesian_result = read_optimal_parameters(f'gpbayes_breakwaterd_{method}_{capacity}_{timestamp}.json')
    for capact in capacity_range:
        print("Analyzing file and run experiments in loop:", bayesian_result)
        run_experiments_loop('breakwaterd', bayesian_result['parameters'], capact, method)


    if optimizeDagor:
        # Optimize for Dagor
        optimizer_dagor = BayesianOptimization(
            f=objective_dagor,
            pbounds=pbounds_dagor,
            random_state=1,
            allow_duplicate_points=True
        )
        # optimizer_dagor.set_gp_params(alpha=1e-3,
        #                               normalize_y=True, # kernel=sklearn.gaussian_process.kernels.Matern(length_scale=10),
        #                               n_restarts_optimizer=5)
        # optimizer_dagor.maximize(init_points=10, n_iter=35)
        # best_params_dagor = optimizer_dagor.max['params']
        # print("Best for Dagor has target:", optimizer_dagor.max['target'], "with parameters:", best_params_dagor)
        try:
            optimizer_dagor.maximize(init_points=10, n_iter=35)
        except Exception as e:
            print("Optimization encountered an error:", e)
        if 'params' in optimizer_dagor.max:
            best_params_dagor = optimizer_dagor.max['params']
            print("Best parameters found:", best_params_dagor)
        else:
            print("No successful optimization results available.")

        results_dagor = {
            'target': optimizer_dagor.max['target'],
            'parameters': roundDownParams(best_params_dagor),
        }
        record_optimal_parameters(f'gpbayes_dagor_{method}_{capacity}_{timestamp}.json', results_dagor)
    
    bayesian_result = read_optimal_parameters(f'gpbayes_dagor_{method}_{capacity}_{timestamp}.json')
    for capact in capacity_range:
        print("Analyzing file and run experiments in loop:", bayesian_result)
        run_experiments_loop('dagor', bayesian_result['parameters'], capact, method)


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
    if len(sys.argv) > 1:
        method = sys.argv[1]
    else:
        raise Exception("Please specify the method")
    if method == 'compose':
        SLO = 20 * 2 # 20ms * 2 = 40ms
    elif method == 'S_149998854':
        SLO = 111 * 2 # 111ms * 2 = 222ms
    elif method == 'S_102000854':
        SLO = 102 * 2
    maximum_goodput = 5000
    capacity = 'tgpt-8000'
    main()
    # for capacity in range(4000, 10000, 500), get latest file and run the check below
    # check_goodput('social-compose-control-breakwater-parallel-capacity-6807-1213_2351.json')
    # check_goodput('social-compose-control-breakwaterd-parallel-capacity-6601-1214_0006.json')
    # for load in range(4000, 10000, 500):
    #     experiment = get_latest_file(os.path.expanduser('~/Sync/Git/protobuf/ghz-results/'), f'social-compose-control-charon-parallel-capacity-{load}-1213*json')
    #     check_goodput(experiment)