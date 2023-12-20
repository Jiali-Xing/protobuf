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

import json
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# import the /home/ying/Sync/Git/protobuf/ghz-results/visualize.py file and run its function `analyze_data` 
# with the optimal results from the Bayesian Optimization
sys.path.append('/home/ying/Sync/Git/protobuf/ghz-results')
from visualize import analyze_data

throughput_time_interval = '50ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds
filename = '/home/ying/Sync/Git/protobuf/ghz-results/charon_stepup_nclients_1000.json'
rerun = True


def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["details"]


def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # print(df['timestamp'].min())
    df.set_index('timestamp', inplace=True)
    df['latency'] = df['latency'] / 1000000
    # drop the rows if the `status` is Unavailable
    df = df[df['status'] != 'Unavailable']
    # remove the data within first second of df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=1)]

    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df


def calculate_average_goodput(filename):
    # Insert your code for calculating average goodput here
    # Read the ghz.output file and calculate the average goodput
    # Return the average goodput
    data = read_data(filename)
    df = convert_to_dataframe(data)
    # print(df.head())
    # df = calculate_throughput(df)
    goodput = calculate_goodput(df, 20)
    return goodput  # Replace with your actual function


def calculate_goodput(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='ffill')
    # take out the goodput during the last 3 seconds by index
    goodput = df[df.index > df.index[0] + pd.Timedelta(seconds=4)]['goodput']
    # return the goodput, but round it to 2 decimal places
    goodput = goodput.mean()
    goodput = round(goodput, -2)
    return goodput


def calculate_throughput(df):
    # sample throughput every time_interval
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    ok_requests_per_second = ok_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='ffill')
    return df


# Define the parameter ranges
# param_ranges = [(10, 200), (1, 50), (100, 20000)]  # (priceUpdateRate, delayTarget, clientTimeout)
# param_ranges = [(10, 200), (1, 50), (5, 50)]  # (priceUpdateRate, delayTarget, clientTimeout)
# param_ranges = [(-5, 5), (-2000, 2000), (1, 75)]  # (priceUpdateRate, delayTarget, clientTimeout)
# param_ranges = [(1, 200), (1, 10000), (1, 10)]  # (priceUpdateRate, delayTarget) 
# param_ranges = [(1, 200), (1, 40), (1, 10), (5, 50)]  # (priceUpdateRate, delayTarget) 
param_ranges = [(50, 500), (1000, 10000), (1, 20), ]  # (priceUpdateRate, delayTarget) 
    
# Define the function that runs the service and client as experiments
def run_experiments(priceUpdateRate, delayTarget, priceStep, guidePrice, clientTimeout):
    output_file_path = '/home/ying/Sync/Git/service-app/services/protobuf-grpc/server.output'
    # Open the file in write mode
    with open(output_file_path, 'w') as output_file:
        process1 = subprocess.Popen(
            [
                "go", "run", "/home/ying/Sync/Git/protobuf/baysian-opt/one-service.go", "A", "50051",
                str(priceUpdateRate), str(delayTarget), str(priceStep), str(guidePrice)
            ],
            stdout=output_file,  # Save stdout to the file
            stderr=subprocess.PIPE
        )

    # Set the working directory
    working_dir = "/home/ying/Sync/Git/protobuf/ghz-client"

    # save the stdout to working_dir/ghz.output
    with open(working_dir + '/ghz.output', 'w') as output_file:
        process2 = subprocess.run([
            "go", "run", "/home/ying/Sync/Git/protobuf/ghz-client/main.go", str(clientTimeout)
        ], cwd=working_dir, stdout=output_file, stderr=subprocess.PIPE)

    # Retrieve the stdout and stderr outputs
    # stderr_output = process2.stderr.decode('utf-8')

    # Wait for process2 to finish before proceeding
    # process2.wait()

    process3 = subprocess.run([
        "bash", "/home/ying/Sync/Git/protobuf/baysian-opt/kill_services.sh"
    ])

    return

# Define the objective function to optimize
def objective(priceUpdateRate, clientTimeout, delayTarget, guidePrice, priceStep):
    # Convert the parameters to int64
    priceUpdateRate = int(priceUpdateRate)
    # priceUpdateRate = 100
    delayTarget = int(delayTarget)
    guidePrice = int(guidePrice)
    # clientTimeout = int(clientTimeout)
    clientTimeout = int(clientTimeout)
    priceStep = int(priceStep)

    # Run the experiments
    # run_experiments(priceUpdateRate, delayTarget, guidePrice, clientTimeout)
    run_experiments(priceUpdateRate, delayTarget, priceStep, guidePrice, clientTimeout)

    # Perform the calculations for average goodput
    # Insert your code for calculating average goodput here
    average_goodput = calculate_average_goodput(filename)  

    return average_goodput  # Minimize the negative average goodput


def plot_opt(priceUpdateRate, clientTimeout, delayTarget, guidePrice, priceStep):
    # Convert the parameters to int64
    priceUpdateRate = int(priceUpdateRate)
    # priceUpdateRate = 100
    delayTarget = int(delayTarget)
    guidePrice = int(guidePrice)
    # clientTimeout = int(clientTimeout)
    clientTimeout = int(clientTimeout)
    priceStep = int(priceStep)

    # Run the experiments
    run_experiments(priceUpdateRate, delayTarget, priceStep, guidePrice, clientTimeout)

    analyze_data(filename)


# Define the objective function to optimize
def objective_wrapper(priceUpdateRate, delayTarget, priceStep,):
    return objective(priceUpdateRate, 0, delayTarget, -1, priceStep)


def plot_opt_wrapper(priceUpdateRate, delayTarget, priceStep,):
    return plot_opt(priceUpdateRate, 0, delayTarget, -1, priceStep)

if __name__ == '__main__':
    if rerun == True:
        # Create the optimizer
        optimizer = BayesianOptimization(
            f=objective_wrapper,
            # pbounds={'guidePrice': param_ranges[2]},  # Only optimize the guidePrice parameter
            pbounds=dict(zip(['priceUpdateRate', 'delayTarget', 'priceStep'], param_ranges)),
            random_state=1,
        )

        # Perform the optimization
        optimizer.maximize(
            init_points=5,  # Number of initial random points
            n_iter=25,  # Number of optimization iterations
            # add initial values of priceUpdateRate, delayTarget, guidePrice as 50, -1, 7
            # init_points=[{'clientTimeout': 0, 'delayTarget': 0}]
        )

        # Print the best parameters and objective value found
        best_params = optimizer.max['params']
        # best_guide_price = optimizer.max['params']['guidePrice']
        best_objective = optimizer.max['target']  # Convert back to positive value
        print("Best Parameters:", best_params)
        # print("Best Guide Price:", best_guide_price)
        print("Best Average Goodput:", best_objective)

        # save the best parameters to a file, pickle
        with open('optimizer.pkl', 'wb') as f:
            pickle.dump(optimizer, f)
    else:
        # read the best parameters from the file
        optimizer = pickle.load(open('optimizer.pkl', 'rb'))
        best_params = optimizer.max['params']
        best_objective = optimizer.max['target']  # Convert back to positive value
        print("Best Parameters:", best_params)
        print("Best Average Goodput:", best_objective)

    # plot_opt(**best_params)
    plot_opt_wrapper(**best_params)
    # print: the best obj is objective_wrapper(**best_params)
    print("Best Average Goodput:", objective_wrapper(**best_params))
