import json
import sys
import re
import os
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
from slo import get_slo
from utils import calculate_goodput_ave_var, calculate_tail_latency_dynamic, read_load_info_from_json, read_data, read_tail_latency, calculate_goodput_dynamic, calculate_goodput_ave_var, calculate_throughput_dynamic, calculate_loadshedded, calculate_ratelimited, read_mean_latency

# import the function calculate_average_goodput from /home/ying/Sync/Git/protobuf/baysian-opt/bayesian_opt.py

throughput_time_interval = '100ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds
offset = 0  # Define the offset as 50 milliseconds
oneNode = False
# remote = True
cloudlab = True

# SLO = 20 if oneNode else 80
capacity = 24000 if oneNode else 5000
# computationTime = 10 if oneNode else 70

computationTime = 0
# alibaba = False
# SLO = 2 * 111 if alibaba else 25 * 2

INTERCEPTOR = os.environ.get('INTERCEPT', 'plain').lower()

# cloudlabOutput = r"grpc-(service-\d+)"
# match `deathstar_xxx.output` where xxx is the deathstar social network service names
cloudlabOutput = r"deathstar_([\w-]+)\.output"

# read CONSTANT_LOAD as bool from env
CONSTANT_LOAD = False
if 'CONSTANT_LOAD' in os.environ:
    CONSTANT_LOAD = os.environ['CONSTANT_LOAD'].lower() == 'true'
    print("CONSTANT_LOAD is set to", CONSTANT_LOAD)

# if capacity is given as an environment variable, use it
if 'CAPACITY' in os.environ:
    capacity = int(os.environ['CAPACITY'])
    print("Capacity is set to", capacity)


def convert_to_dataframe(data, init=False):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    summary = df.groupby('status')['status'].count().reset_index(name='count')
    print("Summary of the status", summary)
    # print(df['timestamp'].min())

    df.set_index('timestamp', inplace=True)
    df['latency'] = df['latency'] / 1000000
    # drop the rows if the `status` is Unavailable
    df = df[df['status'] != 'Unavailable']
    # remove the data within first second of df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]

    min_timestamp = df.index.min()
    # record the start time of the first request and the end time of the last request as global variables
    # only if the global variables are not set yet
    # if 'start_time' not in globals():
    if init:
        global start_time, end_time
        start_time = min_timestamp.astimezone(pytz.UTC)
        end_time = df.index.max().astimezone(pytz.UTC)

    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df

def plot_latency_pdf_cdf(df, filename):
    latency = df['latency']
    pdf, bins = np.histogram(latency, bins=50, density=True)
    pdf_x = (bins[:-1] + bins[1:]) / 2
    cdf_x = np.sort(latency)
    cdf_y = np.arange(1, len(cdf_x) + 1) / len(cdf_x)

    fig, ax1 = plt.subplots()
    ax1.plot(pdf_x, pdf, label='PDF')
    ax1.set_xlabel('Latency (millisecond)')
    ax1.set_ylabel('PDF')

    ax2 = ax1.twinx()
    ax2.plot(cdf_x, cdf_y, color='orange', label='CDF')
    ax2.set_ylabel('CDF')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
    ax2.legend().remove()

    plt.savefig(filename + '.latency.png')
    plt.show()


def calculate_average_var_goodput(filename):
    # Insert your code for calculating average goodput here
    # Read the ghz.output file and calculate the average goodput
    # Return the average goodput
    data = read_data(filename)
    df = convert_to_dataframe(data)
    # print(df.head())
    # df = calculate_throughput(df)
    goodput, goodputVar = calculate_goodput_ave_var(df, SLO)
    return goodput, goodputVar


def plot_timeseries(df, filename):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, df['latency_ma'], color='orange', linestyle='--', label='Average Latency (e2e)')
    ax1.plot(df.index, df['tail_latency'], color='green', linestyle='-.', label='99% Tail Latency (e2e)')
    ax1.set_ylim(0, 2000)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Throughput (req/s)', color='tab:blue')
    ax2.plot(df.index, df['throughput'], color='tab:blue', label='Throughput')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 2500)
    ax2.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
    ax2.legend().remove()

    concurrent_clients = re.findall(r"\d+", filename)[0]
    start_index = filename.rfind("/") + 1 if "/" in filename else 0
    end_index = filename.index("_") if "_" in filename else len(filename)
    mechanism = filename[start_index:end_index]
    plt.title(f"Mechanism: {mechanism}. Number of Concurrent Clients: {concurrent_clients}")

    plt.savefig(mechanism + '.timeseries.png')
    plt.show()


def extract_waiting_times(file_path):
    # Define the regular expression patterns for extracting timestamp and waiting time
    timestamp_pattern = r"LOG: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+[-+]\d{2}:\d{2})"
    waiting_time_patterns = {
        # "Cumulative Waiting Time Median": r"\[Cumulative Waiting Time Median\]:\s+(\d+\.\d+) ms.",
        # "Incremental Waiting Time 90-tile": r"\[Incremental Waiting Time 90-tile\]:\s+(\d+\.\d+) ms.",
        # "Incremental Waiting Time Median": r"\[Incremental Waiting Time Median\]:\s+(\d+\.\d+) ms.",
        "Incremental Waiting Time Maximum": r"\[Incremental Waiting Time Maximum\]:\s+(\d+\.\d+) ms."
    }

    data = {pattern: [] for pattern in waiting_time_patterns}
    timestamps = []

    with open(file_path, "r") as file:
        for line in file:
            if "Incremental Waiting Time Maximum" in line:
                match_timestamp = re.search(timestamp_pattern, line)
                if match_timestamp:
                    timestamps.append(match_timestamp.group(1))

            for key, pattern in waiting_time_patterns.items():
                match = re.search(pattern, line)
                if match:
                    data[key].append(float(match.group(1)))
                elif key not in data:
                    data[key].append(None)

    # print(data)
    # Create a DataFrame with timestamp as one column with waiting times as columns
    df = pd.DataFrame(data)

    # if df is empty, return empty df
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(timestamps)
    # adjust the index of df_queuing_delay to match the index of df, time wise
    # calculate the second last mininum timestamp of df
    # df['timestamp'] = df['timestamp'] - min_timestamp + pd.Timestamp('2000-01-01')
    # print(df['timestamp'].min())
    df.set_index('timestamp', inplace=True)

    # sort the df by timestamp
    df.sort_index(inplace=True)

    '''
    # remove the data within first second of df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]
    '''

    # keep only the data of df within the time range of [start_time, end_time]
    df = df[(df.index >= start_time) & (df.index <= end_time)]
    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df


# similar to extract_waiting_times, we extract the ownPrice update:
def extract_ownPrice_update(file_path):
    # Define the regular expression patterns for extracting timestamp and waiting time
    timestamp_pattern = r"LOG: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+[-+]\d{2}:\d{2})"
    price_update_patterns = r"Own price updated to (\d+)"

    data = []
    timestamps = []

    with open(file_path, "r") as file:
        for line in file:
            # if "Update OwnPrice" or "Update Price by Queue Delay" in line:
            if "Own price updated to" in line:
                match = re.search(price_update_patterns, line)
                if match:
                    data.append(int(match.group(1)))

                    match_timestamp = re.search(timestamp_pattern, line)
                    if match_timestamp:
                        timestamps.append(match_timestamp.group(1))

    # print(data)
    # Create a DataFrame with timestamp as one column with waiting times as columns
    df = pd.DataFrame({"ownPrice": data})
    df["timestamp"] = pd.to_datetime(timestamps)
    # adjust the index of df_queuing_delay to match the index of df, time wise
    # calculate the second last mininum timestamp of df
    # df['timestamp'] = df['timestamp'] - min_timestamp + pd.Timestamp('2000-01-01')
    # print(df['timestamp'].min())
    df.set_index('timestamp', inplace=True)

    # sort the df by timestamp
    df.sort_index(inplace=True)
    # # remove the data within first second of df
    # df = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]
     
    # keep only the data of df within the time range of [start_time, end_time]
    df = df[(df.index >= start_time) & (df.index <= end_time)]

    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df


def extract_ownPrices(file_pattern):
    # Define the regular expression patterns for extracting timestamp and own price update
    timestamp_pattern = r"LOG: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+[-+]\d{2}:\d{2})"
    price_update_patterns = r"Own price updated to (\d+)"

    data_dict = {}
   
    # Provide the full path to the directory containing the files
    directory_path = os.path.expanduser('~/Sync/Git/protobuf/priceOutput/')

    # Get a list of files that match the given pattern
    files = [f for f in os.listdir(directory_path) if re.match(file_pattern, f)]

    # append the directory path to each file name
    files = [directory_path + f for f in files]

    for file_path in files:
        # print(file_path)
        data = []
        timestamps = []

        # Extract the service name from the file name (assuming it follows the pattern "grpc-service-x.output")
        match_service_name = re.search(cloudlabOutput, file_path)
        if match_service_name:
            service_name = match_service_name.group(1)
        else:
            # If the regular expression does not find a match, skip this file
            continue

        with open(file_path, "r") as file:
            for line in file:
                if "Own price updated to" in line:
                    match = re.search(price_update_patterns, line)
                    if match:
                        data.append(int(match.group(1)))

                        match_timestamp = re.search(timestamp_pattern, line)
                        if match_timestamp:
                            timestamps.append(match_timestamp.group(1))

        # Create a DataFrame with timestamp as one column and own price data as another
        df = pd.DataFrame({service_name: data})
        # if df is empty, return
        if df.empty:
            continue

        df["timestamp"] = pd.to_datetime(timestamps)
        df.set_index('timestamp', inplace=True)
        # sort the df by timestamp
        df.sort_index(inplace=True)
        # keep only the data of df within the time range of [start_time, end_time]
        df = df[(df.index >= start_time) & (df.index <= end_time)]

        min_timestamp = df.index.min()
        df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')

        # Add the DataFrame to the data_dict with the service name as the key
        data_dict[service_name] = df

    return data_dict


def extract_waiting_times_all(file_pattern):
    # Define the regular expression patterns for extracting timestamp and own price update
    # timestamp_pattern = r"LOG: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+[-+]\d{2}:\d{2})"
    # price_update_patterns = r"Own price updated to (\d+)"

    data_dict = {}
   
    # Provide the full path to the directory containing the files
    directory_path = os.path.expanduser('~/Sync/Git/protobuf/ghz-results/')
    # Get a list of files that match the given pattern
    files = [f for f in os.listdir(directory_path) if re.match(file_pattern, f)]

    # append the directory path to each file name
    files = [directory_path + f for f in files]

    for file_path in files:
        # print(file_path)
        # data = []
        # timestamps = []

        # Extract the service name from the file name (assuming it follows the pattern "grpc-service-x.output")
        match_service_name = re.search(cloudlabOutput, file_path)
        if match_service_name:
            service_name = match_service_name.group(1)
        else:
            # If the regular expression does not find a match, skip this file
            continue

        with open(file_path, "r") as file:
            df = extract_waiting_times(file_path)
        
        # rename the column name "Incremental Waiting Time Maximum" in df to be the service name
        df.rename(columns={"Incremental Waiting Time Maximum": service_name}, inplace=True)

        # Add the DataFrame to the data_dict with the service name as the key
        data_dict[service_name] = df
    return data_dict


def plot_timeseries_ok(df, filename):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, df['latency_ma'], linestyle='-.', label='Average Latency (e2e)')
    ax1.plot(df.index, df['tail_latency'], linestyle='-', label='99% Tail Latency (e2e)')
    # ax1.set_ylim(70, 2000)
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Throughput (req/s)', color='tab:blue')

    ax2.plot(df.index, df['throughput'], 'r-.', )
    ax2.plot(df.index, df['goodput'], color='green', linestyle='--')
    ax2.plot(df.index, df['dropped'].fillna(0)+df['throughput'], color='tab:blue', linestyle='-', label='Throughput')
    ax2.fill_between(df.index, 0, df['goodput'], color='green', alpha=0.1, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color='red', alpha=0.1, label='SLO Violation')
    ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color='c', alpha=0.1, label='Dropped Req')

    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.set_ylim(0, 3000)
    ax2.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # move the legend to be above the plot
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax2.legend().remove()

    concurrent_clients = re.findall(r"\d+", filename)[0]
    start_index = filename.rfind("/") + 1 if "/" in filename else 0
    end_index = filename.index("_") if "_" in filename else len(filename)
    mechanism = filename[start_index:end_index]
    # move the title to be above the plot and above the legend
    plt.title(f"Mechanism: {mechanism}. Number of Concurrent Clients: {concurrent_clients}", y=1.3)

    plt.savefig(mechanism + '.latency-throughput.png')
    plt.show()

# plot_timeseries_lat is same as the above function,
# but in ax2 we add 4 more line from the waiting time dataframe
def plot_timeseries_lat(df, filename, computation_time=0):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # bound the latency to be above 0.001

    # if computation_time > 0, label is "Average Latency (e2e) \n minus computation time", otherwise, label is "Average Latency (e2e)"
    ax1.plot(df.index, np.maximum(0.001, df['latency_ma']-computation_time), linestyle='--',
             label='Average Latency (e2e)' if computation_time == 0 else 'Average Latency (e2e) \nminus computation time')
    ax1.plot(df.index, np.maximum(0.001, df['tail_latency']-computation_time), linestyle='-.',
             label='99% Tail Latency (e2e)' if computation_time == 0 else '99% Tail Latency (e2e) \nminus computation time')
    # set the y axis to be above the 0.001
    ax1.set_ylim(0.01, np.max(df['tail_latency'])*1.1)
    ax1.set_yscale('log')

    # # ax2 = ax1.twinx()
    # ax2.set_ylabel('Throughput (req/s)', color='tab:blue')

    # ax2.plot(df.index, df['throughput'], 'r-.', )
    # ax2.plot(df.index, df['goodput'], color='green', linestyle='--')
    # ax2.plot(df.index, df['dropped'].fillna(0)+df['throughput'], color='tab:blue', linestyle='-', label='Total Req')
    # ax2.fill_between(df.index, 0, df['goodput'], color='green', alpha=0.2, label='Goodput')
    # ax2.fill_between(df.index, df['goodput'], df['throughput'], color='red', alpha=0.3, label='SLO Violated Req')
    # ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color='c', alpha=0.3, label='Dropped Req')

    df_queuing_delay = extract_waiting_times("~/Sync/Git/service-app/services/protobuf-grpc/server.output")
    # plot the four waiting time patterns above in ax2, with for loop over the column of df_queuing_delay
    for waiting_time in df_queuing_delay.columns:
        # before plotting, we need to calculate the moving average of the waiting time
        mean_queuing_delay = df_queuing_delay[waiting_time].rolling(latency_window_size).mean()
        ax1.plot(df_queuing_delay.index, mean_queuing_delay, label=waiting_time)

    # print(df_queuing_delay.head())
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.set_ylim(0, 3000)
    # ax2.grid(True)

    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
    ax1.legend()

    concurrent_clients = re.findall(r"\d+", filename)[0]
    start_index = filename.rfind("/") + 1 if "/" in filename else 0
    end_index = filename.index("_") if "_" in filename else len(filename)
    mechanism = filename[start_index:end_index]
    plt.title(f"Mechanism: {mechanism}. Number of Concurrent Clients: {concurrent_clients}")

    plt.savefig(mechanism + '.queuing-delay.png')
    plt.show()

# plot 2 charon experiments in the same plot with function similar to plot_timeseries_split
def plot_timeseries_split_2(df1, df2, filename):
    # similar to plot_timeseries_split, only difference is that we have 2 dataframes and 2 columns each row.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 4), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Define colors that are clearly readible in grayscale
    colors = ['#34a853', '#ea4335', '#4285f4', '#fbbc05']
    # plot the ax1 and ax2 for df1 first in the first column
    ax1, ax2 = axes[:, 0]
    df = df1
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, np.maximum(0.001, df['latency_ma']), linestyle='--',
             label='Average Latency (e2e)')
    ax1.plot(df.index, np.maximum(0.001, df['tail_latency']), linestyle='-.',
             label='99% Tail Latency (e2e)')
    ax1.set_ylim(10, 200)
    ax1.set_yscale('log')
    # ax1 add legend on top of the plot 

    SLO = get_slo(method, tight=True, all_methods=False)
    ax1.axhline(y=SLO, color='c', linestyle='-.', label='SLO')
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.1), ncol=2, frameon=False)
    
    # add .fillna(0) to all the columns of df, so that we can plot the throughput and goodput
    df = df.fillna(0)
    # create a list of 4 colors clearly readible in grayscale

    ax2.set_ylabel('Throughput (req/s)', color='tab:blue')
    ax2.plot(df.index, df['throughput'], 'r-.', alpha=0.2)
    ax2.plot(df.index, df['goodput'], color='green', linestyle='--', alpha=0.2)
    ax2.plot(df.index, df['dropped']+df['throughput'], color='tab:blue', linestyle='-', label='Req Sent', alpha=0.2)
    # plot dropped requests + rate limit requests + throughput = total demand

    load_info = read_load_info_from_json(filename)
    capacity = load_info['load-end']
    df['total_demand'] = load_info['load-start']
    # Define the time as 00:00:00 + load-step-duration - offset
    mid_start_time = pd.Timestamp('2000-01-01 00:00:00') + pd.Timedelta(load_info['load-step-duration']) - pd.Timedelta(offset, unit='s')
    # Create a new column 'new_column' and fill it with 100 for rows within the time range
    df.loc[mid_start_time:, 'total_demand'] = capacity  # Set the value to 100 for the specified time range

    ax2.fill_between(df.index, 0, df['goodput'], color=colors[0], alpha=0.8, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], alpha=0.8, label='SLO\nViolation')
    ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], alpha=0.8, label='Dropped')
    ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color=colors[3], alpha=0.8, label='Rate\nLimited')

    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 12000)
    ax2.set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in ax2.get_yticks()/1000])
    # add legend to ax2 inside the plot upper left


    # plot the ax3 and ax4 for df2 in the second column
    ax3, ax4 = axes[:, 1]
    # legend for ax3 and ax4 above the plot
    df = df2
    ax3.set_ylabel('Latencies (ms)', color='tab:red')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.plot(df.index, np.maximum(0.001, df['latency_ma']), linestyle='--',
             label='Average Latency (e2e)')
    ax3.plot(df.index, np.maximum(0.001, df['tail_latency']), linestyle='-.',
             label='99% Tail Latency (e2e)')
    ax3.set_ylim(10, 200)
    ax3.set_yscale('log')
    SLO = get_slo(method, tight=False, all_methods=False)
    ax3.axhline(y=SLO, color='c', linestyle='-.', label='SLO')

    # add .fillna(0) to all the columns of df, so that we can plot the throughput and goodput
    df = df.fillna(0)
    # create a list of 4 colors clearly readible in grayscale

    ax4.set_ylabel('Throughput (req/s)', color='tab:blue')
    ax4.plot(df.index, df['throughput'], 'r-.', alpha=0.2)
    ax4.plot(df.index, df['goodput'], color='green', linestyle='--', alpha=0.2)
    ax4.plot(df.index, df['dropped']+df['throughput'], color='tab:blue', linestyle='-', label='Req Sent', alpha=0.2)
    # plot dropped requests + rate limit requests + throughput = total demand
    
    load_info = read_load_info_from_json(filename)
    capacity = load_info['load-end']
    df['total_demand'] = load_info['load-start']
    # Define the time as 00:00:00 + load-step-duration - offset
    mid_start_time = pd.Timestamp('2000-01-01 00:00:00') + pd.Timedelta(load_info['load-step-duration']) - pd.Timedelta(offset, unit='s')
    # Create a new column 'new_column' and fill it with 100 for rows within the time range
    df.loc[mid_start_time:, 'total_demand'] = capacity  # Set the value to 100 for the specified time range

    # if df['limited'].sum() > 0, then plot the rate limited requests
    ax4.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color=colors[3], alpha=0.8, label='Rate\nLimited')
    ax4.fill_between(df.index, 0, df['goodput'], color=colors[0], alpha=0.8, label='Goodput')
    ax4.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], alpha=0.8, label='SLO\nViolation')
    ax4.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], alpha=0.8, label='Dropped')

    ax4.tick_params(axis='y', labelcolor='tab:blue')
    ax4.set_ylim(0, 12000)


    concurrent_clients = re.findall(r"\d+", filename)[0]
    start_index = filename.rfind("/") + 1 if "/" in filename else 0
    end_index = filename.index("_") if "_" in filename else len(filename)
    mechanism = filename[start_index:end_index]
    # move the title to be above the plot and above the legend
    # for 2nd column, remove the y label and y ticks but keep the grid
    ax3.set_ylabel('')
    ax4.set_ylabel('')
    ax3.tick_params(axis='y', labelcolor='tab:red', labelleft=False)
    ax4.tick_params(axis='y', labelcolor='tab:blue', labelleft=False)

    # add grid to all 4 axes
    for ax in [ax1, ax2, ax3, ax4]:
        # ax.grid for both x and y axis
        ax.grid(True)


    # make the legend for ax3 outside the plot on the right
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=False)
    # compact the layout
    plt.tight_layout()
    # remove space between the subplots
    plt.subplots_adjust(wspace=0.1)


    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%S'))
    plt.xlabel('Time (second)')
    plt.savefig(mechanism + '.latency-throughput2.pdf')
    plt.show()


def plot_timeseries_split(df, filename, computation_time=0):
    # mechanism is the word after `control-` in the filename
    # e.g., breakwater in social-compose-control-breakwater-parallel-capacity-8000-1209_1620.json
    mechanism = re.findall(r"control-(\w+)-", filename)[0]
    
    servicePrice = False
    narrow = False
    width = 3 if narrow else 6
    if servicePrice:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, 4), sharex=True, height_ratios=[1, 3])

    # make ax1 shorter
    
    # Define colors that are clearly readible in grayscale
    colors = ['#34a853', '#ea4335', '#4285f4', '#fbbc05']
    
    # add to all 3 axes vertical x grid lines for each second, x axis are datetime objects
    for ax in [ax1, ax2]:
        # x axis are datetime objects, so we want to add vertical grid lines for each second
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        #  Set the x-axis locator to show grid lines for each second
        # ax.xaxis.set_major_locator(mdates.SecondLocator())
        # also, set the x limit to the end of the last timestamp of the 3 dataframes
        # ax.set_xlim(0, df.index[-1])


    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, np.maximum(0.001, df['latency_ma']-computation_time), linestyle='--',
             label='Average Latency (e2e)' if computation_time == 0 else 'Average Latency (e2e) \nminus computation time')
    ax1.plot(df.index, np.maximum(0.001, df['tail_latency']-computation_time), linestyle='-.',
             label='99% Tail Latency (e2e)' if computation_time == 0 else '99% Tail Latency (e2e) \nminus computation time')
    # if alibaba:
        # ax1.set_ylim(100, 500)
    if not alibaba:
        ax1.set_ylim(10, 200)
    ax1.set_yscale('log')

    # add .fillna(0) to all the columns of df, so that we can plot the throughput and goodput
    df = df.fillna(0)
    # create a list of 4 colors clearly readible in grayscale

    ax2.set_ylabel('Throughput (req/s)', color='tab:blue')
    ax2.plot(df.index, df['throughput'], 'r-.', alpha=0.2)
    ax2.plot(df.index, df['goodput'], color='green', linestyle='--', alpha=0.2)
    ax2.plot(df.index, df['dropped']+df['throughput'], color='tab:blue', linestyle='-', label='Req Sent', alpha=0.2)
    # plot dropped requests + rate limit requests + throughput = total demand
    # if df['limited'].sum() > 0, then plot the rate limited requests
    load_info = read_load_info_from_json(filename)
    capacity = load_info['load-end']
    
    if df['limited'].sum() > 0:
        df['total_demand'] = df['dropped']+df['throughput']+df['limited']
    elif CONSTANT_LOAD:
        df['total_demand'] = capacity
    else:
        # otherwise, plot the a total demand that is half of capacity for the first 2 seconds, and then 100% capacity for next 2 seconds
        # and then 150% capacity for the rest of the time
        # add a new column to df, called total_demand, fill in with half of capacity for the first 2 seconds, and then 100% capacity for next 2 seconds

        df['total_demand'] = load_info['load-start']
        # Define the time as 00:00:00 + load-step-duration - offset
        mid_start_time = pd.Timestamp('2000-01-01 00:00:00') + pd.Timedelta(load_info['load-step-duration']) - pd.Timedelta(offset, unit='s')
        # Create a new column 'new_column' and fill it with 100 for rows within the time range
        df.loc[mid_start_time:, 'total_demand'] = capacity  # Set the value to 100 for the specified time range

    if mechanism != 'baseline':
        ax2.plot(df.index, df['total_demand'], color='c', linestyle='-.', label='Demand')


    ax2.fill_between(df.index, 0, df['goodput'], color=colors[0], alpha=0.8, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], alpha=0.8, label='SLO Violation')
    ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], alpha=0.8, label='Dropped')
    # if mechanism in ['charon', 'breakwater', 'breakwaterd']:
    #     ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], color='tab:blue', alpha=0.3, label='Rate Limited Req')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 15000)

    # Apply the custom formatter to the x-axis
    # use second locator to show grid lines for each second, not `09` but `9`
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%S'))
    plt.xlabel('Time (second)')

    if not cloudlab:
        df_queuing_delay = extract_waiting_times("~/Sync/Git/service-app/services/protobuf-grpc/server.output")
        # assert that df_queuing_delay.index is not empty
        assert len(df_queuing_delay.index) > 0
        # only keep the data when the index is smaller than the last timestamp of the df.index
        df_queuing_delay = df_queuing_delay[df_queuing_delay.index < df.index[-1]]
        # use the difference between the second timestamp and the first timestamp as the latency window size
        df_latency_window_size = (df_queuing_delay.index[1] - df_queuing_delay.index[0]).total_seconds() 
        for waiting_time in df_queuing_delay.columns:
            # compare the df_latency_window_size with the latency_window_size (string), use the smaller one
            # convert the latency_window_size (string) to milliseconds
            latency_window_size_ms = pd.Timedelta(latency_window_size).total_seconds() 
            if df_latency_window_size < latency_window_size_ms:
                mean_queuing_delay = df_queuing_delay[waiting_time].rolling(latency_window_size).mean()
            else:
                mean_queuing_delay = df_queuing_delay[waiting_time]
            ax1.plot(df_queuing_delay.index, mean_queuing_delay, label=waiting_time)
    else:
        # loop over /home/ying/Sync/Git/protobuf/ghz-results/grpc-service-*.output to get the df queuing delay
        dict_queuing_delay = extract_waiting_times_all(cloudlabOutput)
        for service_name, df_queuing_delay in dict_queuing_delay.items():
            # assert len(df_queuing_delay.index) > 0
            if len(df_queuing_delay.index) == 0:
                continue
            # only keep the data when the index is smaller than the last timestamp of the df.index
            df_queuing_delay = df_queuing_delay[df_queuing_delay.index < df.index[-1]]
            # use the difference between the second timestamp and the first timestamp as the latency window size
            df_latency_window_size = (df_queuing_delay.index[1] - df_queuing_delay.index[0]).total_seconds()
            for waiting_time in df_queuing_delay.columns:
                # compare the df_latency_window_size with the latency_window_size (string), use the smaller one
                # convert the latency_window_size (string) to milliseconds
                latency_window_size_ms = pd.Timedelta(latency_window_size).total_seconds()
                if df_latency_window_size < latency_window_size_ms:
                    mean_queuing_delay = df_queuing_delay[waiting_time].rolling(latency_window_size).mean()
                else:
                    mean_queuing_delay = df_queuing_delay[waiting_time]
                ax1.plot(mean_queuing_delay.index, mean_queuing_delay, label=service_name)
    # add a horizontal line at y=SLO
    ax1.axhline(y=SLO - computation_time, color='c', linestyle='-.', label='SLO')
    # find the line `export LATENCY_THRESHOLD=???us` in `/home/ying/Sync/Git/service-app/cloudlab/scripts/cloudlab_run_and_fetch.sh`
    # and add a horizontal line at y=???us
    latency_threshold = os.environ.get('LATENCY_THRESHOLD')
    latency_ms = pd.Timedelta(latency_threshold).total_seconds() * 1000
    
    # convert the latency_threshold from 5000us to 5ms
    # ax1.axhline(y=latency_ms, color='g', linestyle='-.', label='Latency Threshold')

    # with open ("/home/ying/Sync/Git/service-app/cloudlab/scripts/cloudlab_run_and_fetch.sh", "r") as file:
    #     for line in file:
    #         if "export LATENCY_THRESHOLD=" in line:
    #             latency_threshold = int(re.findall(r"\d+", line)[0])
    #             # convert the latency_threshold from us to ms
    #             latency_threshold = latency_threshold / 1000
    #             ax1.axhline(y=latency_threshold, color='g', linestyle='-.', label='Latency Threshold')
    #             break
    

    # put legend on the top left corner
    # for ax1, don't show the box border of the legend
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.1), ncol=1 if narrow else 2, frameon=False)

    if cloudlab:
        max_price = 0
        df_price_dict = extract_ownPrices(cloudlabOutput)
        for service_name, df_price in df_price_dict.items():
            # only keep the data when the index is smaller than the last timestamp of the df.index
            df_price = df_price[df_price.index < df.index[-1]]
            moving_average_price = df_price[service_name].rolling(latency_window_size).mean()
            if servicePrice:
                ax3.plot(df_price.index, moving_average_price, label=service_name)
                ax3.legend(loc='upper left', ncol=2, frameon=False)
            max_price = max(moving_average_price.max(), max_price)
    else:
        df_price = extract_ownPrice_update("~/Sync/Git/service-app/services/protobuf-grpc/server.output")
        # only keep the data when the index is smaller than the last timestamp of the df.index
        df_price = df_price[df_price.index < df.index[-1]]
        moving_average_price = df_price['ownPrice'].rolling(latency_window_size).mean()
        if servicePrice:
            ax3.plot(df_price.index, moving_average_price, label='Service Price')
        max_price = moving_average_price.max()
    if servicePrice:
        ax3.set_ylabel('Service Price')
        ax3.set_xlabel('Time')

 
    # fill between total demand and throughput + dropped requests, only if total demand is larger than throughput + dropped requests
    if mechanism != 'baseline' and mechanism != 'dagor':
        ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color='tab:blue', alpha=0.3, label='Rate Limited')
    
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2 if narrow else 3, frameon=True)
    # concurrent_clients = re.findall(r"\d+", filename)[0]
    # plt.suptitle(f"Mechanism: {mechanism}. Number of Concurrent Clients: {concurrent_clients}")

    # add the ax2 a title, saying that `The Average Goodput Under Overload is: ` + calculate_average_goodput(filename)
    goodputAve, goodputStd = calculate_average_var_goodput(filename)
    if not narrow:
        ax2.set_title(f"The Goodput has Mean: {goodputAve} and Std: {goodputStd}")

    latency_99th = read_tail_latency(filename)
    # print("99th percentile latency:", latency_99th)
    # convert the latency_99th from string to milliseconds
    # latency_99th = pd.Timedelta(latency_99th).total_seconds() * 1000
    # round the latency_99th to 2 decimal places
    average_99th = df['tail_latency'].mean()
    lat95 = df[(df['status'] == 'OK')]['latency'].quantile(95 / 100)
    if not narrow:
        ax1.set_title(f"95-tile Latency over Time: {round(lat95, 2)} ms")

    filetosave = mechanism + '-' + method + '-' + str(capacity) + timestamp + '.pdf' 
    plt.savefig(filetosave, dpi=300, bbox_inches='tight')
    print(f"Saved the plot to {filetosave}")
    if not noPlot:
        plt.show()
    plt.close()


def plot_latencies(df, filename, computation_time=0):
    start_index = filename.rfind("/") + 1 if "/" in filename else 0
    end_index = filename.index("_") if "_" in filename else len(filename)
    mechanism = filename[start_index:end_index]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, np.maximum(0.001, df['latency_ma']-computation_time), linestyle='--',
             label='Average Latency (e2e)' if computation_time == 0 else 'Average Latency (e2e) \nminus computation time')
    ax1.plot(df.index, np.maximum(0.001, df['tail_latency']-computation_time), linestyle='-.',
             label='99% Tail Latency (e2e)' if computation_time == 0 else '99% Tail Latency (e2e) \nminus computation time')
    # ax1.set_ylim(0.01, np.max(df['tail_latency'])*1.1)
    ax1.set_yscale('log')
    ax1.set_xlabel('Time')
    # add a vertical grind line per second on x-axis
    ax1.grid(True, which='both', axis='x', linestyle='--')

    # add a horizontal line at y=SLO - computation_time
    ax1.axhline(y=SLO - computation_time, color='c', linestyle='-.', label='SLO - computation time')
    # find the line `export LATENCY_THRESHOLD=???us` in `/home/ying/Sync/Git/service-app/cloudlab/scripts/cloudlab_run_and_fetch.sh`
    # and add a horizontal line at y=???us
    # with open ("/home/ying/Sync/Git/service-app/cloudlab/scripts/cloudlab_run_and_fetch.sh", "r") as file:
    #     for line in file:
    #         if "export LATENCY_THRESHOLD=" in line:
    #             latency_threshold = int(re.findall(r"\d+", line)[0])
    #             # convert the latency_threshold from us to ms
    #             latency_threshold = latency_threshold / 1000
    #             ax1.axhline(y=latency_threshold, color='g', linestyle='-.', label='Latency Threshold')
    #             break
    # read the latency_threshold from the environment variable, it looks like 5000us
    # latency_threshold read as time delta
    latency_threshold = os.environ.get('LATENCY_THRESHOLD')
    latency_ms = pd.Timedelta(latency_threshold).total_seconds() * 1000
    # convert the latency_threshold from 5000us to 5ms
    ax1.axhline(y=latency_ms, color='g', linestyle='-.', label='Latency Threshold')

    if not cloudlab:
        df_queuing_delay = extract_waiting_times("~/Sync/Git/service-app/services/protobuf-grpc/server.output")
        # assert that df_queuing_delay.index is not empty
        assert len(df_queuing_delay.index) > 0
        # only keep the data when the index is smaller than the last timestamp of the df.index
        df_queuing_delay = df_queuing_delay[df_queuing_delay.index < df.index[-1]]
        # use the difference between the second timestamp and the first timestamp as the latency window size
        df_latency_window_size = (df_queuing_delay.index[1] - df_queuing_delay.index[0]).total_seconds() 
        for waiting_time in df_queuing_delay.columns:
            # compare the df_latency_window_size with the latency_window_size (string), use the smaller one
            # convert the latency_window_size (string) to milliseconds
            latency_window_size_ms = pd.Timedelta(latency_window_size).total_seconds() 
            if df_latency_window_size < latency_window_size_ms:
                mean_queuing_delay = df_queuing_delay[waiting_time].rolling(latency_window_size).mean()
            else:
                mean_queuing_delay = df_queuing_delay[waiting_time]
            ax1.plot(df_queuing_delay.index, mean_queuing_delay, label=waiting_time)
    else:
        # loop over /home/ying/Sync/Git/protobuf/ghz-results/grpc-service-*.output to get the df queuing delay
        dict_queuing_delay = extract_waiting_times_all(cloudlabOutput)

        # also, plot the sum of the queuing delay of all services by adding up the queuing delay of each service
        # first create an empty dataframe 
        df_queuing_delay_sum = pd.DataFrame()

        for service_name, df_queuing_delay in dict_queuing_delay.items():
            if len(df_queuing_delay.index) == 0:
                continue
            # only keep the data when the index is smaller than the last timestamp of the df.index
            df_queuing_delay = df_queuing_delay[df_queuing_delay.index < df.index[-1]]
            # use the difference between the second timestamp and the first timestamp as the latency window size
            df_latency_window_size = (df_queuing_delay.index[1] - df_queuing_delay.index[0]).total_seconds()
            for waiting_time in df_queuing_delay.columns:
                # compare the df_latency_window_size with the latency_window_size (string), use the smaller one
                # convert the latency_window_size (string) to milliseconds
                latency_window_size_ms = pd.Timedelta(latency_window_size).total_seconds()
                if df_latency_window_size < latency_window_size_ms:
                    mean_queuing_delay = df_queuing_delay[waiting_time].resample(latency_window_size).mean()
                else:
                    mean_queuing_delay = df_queuing_delay[waiting_time]

                ax1.plot(mean_queuing_delay.index, mean_queuing_delay, label=service_name)

            # add the queuing delay `df_queuing_delay[waiting_time]` of each service to the sum dataframe
            # also add the index of the sum dataframe to the master_index, join the two dataframes on the index
            df_queuing_delay_sum = df_queuing_delay_sum.join(mean_queuing_delay, how='outer')

            # print the average queuing delay of each service
            print("average queuing delay of service", service_name, ":", mean_queuing_delay.mean())    
        
        # if the charon is turned on, plot the sum of the queuing delay of all services
        # if INTERCPTOR is true 
        if INTERCEPTOR == 'true':
            # Sum the queuing delay of all services, fill the NaN with 0
            df_queuing_delay_sum = df_queuing_delay_sum.sum(axis=1).fillna(method='bfill')
            # Plot the sum of mean_queuing_delay for all services
            if df_latency_window_size < latency_window_size_ms:
                sum_queuing_delay = df_queuing_delay_sum.resample(latency_window_size).mean()
            else:
                sum_queuing_delay = df_queuing_delay_sum
            ax1.plot(sum_queuing_delay.index, sum_queuing_delay, label="sum of queuing delay")

            # print the average queuing delay of all services 
            print("average queuing delay of all services:", sum_queuing_delay.mean())

    # put legend on the top left corner
    # for ax1, don't show the box border of the legend
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=3, frameon=False)

    # df_queuing_delay = extract_waiting_times("/home/ying/Sync/Git/service-app/services/protobuf-grpc/server.output")
    # assert len(df_queuing_delay.index) > 0
    # df_queuing_delay = df_queuing_delay[df_queuing_delay.index < df.index[-1]]
    # df_latency_window_size = (df_queuing_delay.index[1] - df_queuing_delay.index[0]).total_seconds() 
    # for waiting_time in df_queuing_delay.columns:
    #     latency_window_size_ms = pd.Timedelta(latency_window_size).total_seconds() 
    #     if df_latency_window_size < latency_window_size_ms:
    #         mean_queuing_delay = df_queuing_delay[waiting_time].rolling(latency_window_size).mean()
    #     else:
    #         mean_queuing_delay = df_queuing_delay[waiting_time]
    #     ax1.plot(df_queuing_delay.index, mean_queuing_delay, label=waiting_time)

    # ax1.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=1, frameon=False)

    goodputAve, goodputStd = calculate_average_var_goodput(filename)
    ax1.set_title(f"The Goodput has Mean: {goodputAve} and Std: {goodputStd}")

    latency_99th = read_tail_latency(filename)
    # latency_99th = pd.Timedelta(latency_99th).total_seconds() * 1000
    latency_median = read_tail_latency(filename, percentile=50)
    # latency_median = pd.Timedelta(latency_median).total_seconds() * 1000
    latency_mean = read_mean_latency(filename)
    # latency_mean = pd.Timedelta(latency_mean).total_seconds() * 1000

    ax1.set_title(f"Mean, Median, 99-tile Latency: {round(latency_mean, 2)} ms, {round(latency_median, 2)} ms, {round(latency_99th, 2)} ms")

    plt.savefig(mechanism + '.1000c.latency.png', format='png', dpi=300, bbox_inches='tight')
    if not noPlot:
        plt.show()
    plt.close()


def analyze_data(filename):
    # ./social-compose-control-charon-parallel-capacity-7000-1023_1521.json
    # if filename contains control, read the string after it as INTERCEPTOR
    global INTERCEPTOR, capacity 
    if "control" in filename:
        match = re.search(r'control-(\w+)-', filename)
        if match:
            INTERCEPTOR =  match.group(1)
            print("Interceptor:", INTERCEPTOR)
    
    if "capacity" in filename:
        match = re.search(r'capacity-(\w+)-', filename)
        if match:
            capacity =  match.group(1)
            # convert the capacity from string to int
            capacity = int(capacity)
            print("Capacity:", capacity)

    data = read_data(filename)
    df = convert_to_dataframe(data, init=True)
    # print(df.head())
    # plot_latency_pdf_cdf(df, filename)
    df = calculate_throughput_dynamic(df)
    df = calculate_goodput_dynamic(df, SLO)
    df = calculate_loadshedded(df)
    df = calculate_ratelimited(df)
    df = calculate_tail_latency_dynamic(df)
    # plot_timeseries_ok(df, filename)
    # plot_timeseries_lat(df, filename, 10)
    plot_timeseries_split(df, filename, computationTime)
    plot_latencies(df, filename, computationTime)
    # summary = df.groupby('error')['error'].count().reset_index(name='count')
    # print(summary)

# function below generalize the analyze_data function to 2 dataframes from 2 files
def analyze_data_2(filename1, filename2):
    global INTERCEPTOR, capacity 
    if "control" in filename1:
        match = re.search(r'control-(\w+)-', filename1)
        if match:
            INTERCEPTOR =  match.group(1)
            print("Interceptor:", INTERCEPTOR)
    
    if "capacity" in filename1:
        match = re.search(r'capacity-(\w+)-', filename1)
        if match:
            capacity =  match.group(1)
            # convert the capacity from string to int
            capacity = int(capacity)
            print("Capacity:", capacity)

    data1 = read_data(filename1)
    df1 = convert_to_dataframe(data1, init=True)
    # print(df.head())
    # plot_latency_pdf_cdf(df, filename)
    df1 = calculate_throughput_dynamic(df1)
    SLO = get_slo(method, tight=True, all_methods=False)
    df1 = calculate_goodput_dynamic(df1, SLO)
    df1 = calculate_loadshedded(df1)
    df1 = calculate_ratelimited(df1)
    df1 = calculate_tail_latency_dynamic(df1)

    if "control" in filename2:
        match = re.search(r'control-(\w+)-', filename2)
        if match:
            INTERCEPTOR =  match.group(1)
            print("Interceptor:", INTERCEPTOR)
    
    if "capacity" in filename2:
        match = re.search(r'capacity-(\w+)-', filename2)
        if match:
            capacity =  match.group(1)
            # convert the capacity from string to int
            capacity = int(capacity)
            print("Capacity:", capacity)

    data2 = read_data(filename2)
    df2 = convert_to_dataframe(data2, init=True)
    # print(df.head())
    # plot_latency_pdf_cdf(df, filename)
    df2 = calculate_throughput_dynamic(df2)
    SLO = get_slo(method, tight=False, all_methods=False)
    df2 = calculate_goodput_dynamic(df2, SLO)
    df2 = calculate_loadshedded(df2)
    df2 = calculate_ratelimited(df2)
    df2 = calculate_tail_latency_dynamic(df2)
    plot_timeseries_split_2(df1, df2, filename1)

if __name__ == '__main__':
    # handle multiple files
    if len(sys.argv) > 2:
        alibaba = False
        method = "compose"
        filename1 = sys.argv[1]
        filename2 = sys.argv[2]
        analyze_data_2(filename1, filename2)
        sys.exit(0)

    filename = sys.argv[1]

    # read from the filename, if `S_` is in the filename, then alibaba is true, otherwise, alibaba is false
    alibaba = "S_" in filename
    # the method is the word between `social-` and `-control` in the filename
    method = re.findall(r"social-(.*?)-control", filename)[0]
    timestamp = re.findall(r"-\d+_\d+", filename)[0]
    SLO = get_slo(method, tight=False, all_methods=('S_' not in method))

    # if there is a second argument, it is the handle of showing the plot
    if len(sys.argv) > 2:
        noPlot = sys.argv[2]
        # convert it to boolean
        noPlot = noPlot.lower() == 'no-plot'
    else:
        noPlot = False
    analyze_data(filename)
