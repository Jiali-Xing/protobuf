import json
import sys
import re
import os
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ghz-results'))
from slo import get_slo

# import the function calculate_average_goodput from /home/ying/Sync/Git/protobuf/baysian-opt/bayesian_opt.py

throughput_time_interval = '100ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds
offset = 2.0  # Define the offset as 50 milliseconds
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

def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["details"]


def read_tail_latency(filename, percentile=99):
    with open(filename, 'r') as f:
        data = json.load(f)
        
    latency_distribution = data["latencyDistribution"]
    for item in latency_distribution:
        if item["percentage"] == percentile:
            return item["latency"]
    return None  # Return None if the 99th percentile latency is not found


# similarly to read_tail_latency, read_mean_latency returns the mean latency
def read_mean_latency(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["average"]  


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


def calculate_throughput(df):
    # sample throughput every time_interval
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    ok_requests_per_second = ok_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='bfill')
    return df


def calculate_goodput(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    # fill in zeros for the missing goodput
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='bfill')
    return df


def calculate_average_goodput(filename):
    # Insert your code for calculating average goodput here
    # Read the ghz.output file and calculate the average goodput
    # Return the average goodput
    data = read_data(filename)
    df = convert_to_dataframe(data)
    # print(df.head())
    # df = calculate_throughput(df)
    goodput, goodputVar = calculate_goodput_ave_var(df, SLO)
    return goodput, goodputVar


def calculate_goodput_ave_var(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='bfill')
    # take out the goodput during the last 3 seconds by index
    # goodput = df[df.index > df.index[-1] - pd.Timedelta(seconds=3)]['goodput']
    goodput = df['goodput']
    # return the goodput, but round it to 2 decimal places
    goodputAve = goodput.mean()
    # also calculate the standard deviation of the goodput
    goodputStd = goodput.std()
    goodputAve = round(goodputAve, 1)
    goodputStd = round(goodputStd, 1)
    return goodputAve, goodputStd


# when a load is shed, item looks like this:
    # {
    #   "timestamp": "2023-07-03T01:24:49.080597187-04:00",
    #   "latency": 535957,
    #   "error": "rpc error: code = ResourceExhausted desc = 17 token for 90 price. req dropped, try again later",
    #   "status": "ResourceExhausted"
    # },

def calculate_loadshedded(df):
    # extract the dropped requests by status == 'ResourceExhausted' and error message contains 'req dropped'
    dropped = df[(df['status'] == 'ResourceExhausted')]
    dropped_requests_per_second = dropped['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    dropped_requests_per_second = dropped_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['dropped'] = dropped_requests_per_second.reindex(df.index, method='ffill')
    return df

# when the rate is limited, item looks like this:
    # {
    #   "timestamp": "2023-07-03T01:24:58.847045036-04:00",
    #   "latency": 13831,
    #   "error": "rpc error: code = ResourceExhausted desc = trying to send message larger than max (131 vs. 0)",
    #   "status": "ResourceExhausted"
    # },
def calculate_ratelimited(df):
    # extract the dropped requests by status == 'ResourceExhausted' and error message contains 'trying to send message larger than max'
    limited = df[(df['status'] == 'ResourceExhausted') & (df['error'].str.contains('trying to send message larger than max'))]
    limited_requests_per_second = limited['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    limited_requests_per_second = limited_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['limited'] = limited_requests_per_second.reindex(df.index, method='bfill')
    return df


def calculate_tail_latency(df):
    # Only cound the latency of successful requests, i.e., status == 'OK'
    df.sort_index(inplace=True)
    # Assuming your DataFrame is named 'df' and the column to calculate the moving average is 'data'
    tail_latency = df['latency'].rolling(latency_window_size).quantile(0.99)
    df['tail_latency'] = tail_latency
    # Calculate moving average of latency
    df['latency_ma'] = df['latency'].rolling(latency_window_size).mean()

    # calculate the average tail latency of each second 
    # df['tail_latency_ave'] = df['tail_latency'].resample('1s').mean()
    # print('[Average Tail Latency] ', df['tail_latency'].mean())
    
    return df


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
        print(file_path)
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
    directory_path = os.path.expanduser('~/Sync/Git/protobuf/priceOutput/')
    # Get a list of files that match the given pattern
    files = [f for f in os.listdir(directory_path) if re.match(file_pattern, f)]

    # append the directory path to each file name
    files = [directory_path + f for f in files]

    for file_path in files:
        print(file_path)
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



def plot_timeseries_split(df, filename, computation_time=0):
    # mechanism is the word after `control-` in the filename
    # e.g., breakwater in social-compose-control-breakwater-parallel-capacity-8000-1209_1620.json
    mechanism = re.findall(r"control-(\w+)-", filename)[0]
    
    servicePrice = True
    if servicePrice:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True, height_ratios=[1, 3])

    # make ax1 shorter
    
    
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
        ax1.set_ylim(16, 200)
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
    if df['limited'].sum() > 0:
        df['total_demand'] = df['dropped']+df['throughput']+df['limited']
    elif CONSTANT_LOAD:
        df['total_demand'] = capacity
    else:
        # otherwise, plot the a total demand that is half of capacity for the first 2 seconds, and then 100% capacity for next 2 seconds
        # and then 150% capacity for the rest of the time
        # add a new column to df, called total_demand, fill in with half of capacity for the first 2 seconds, and then 100% capacity for next 2 seconds
        df['total_demand'] = capacity/2

        # Define the time range from 2nd to 4th second
        mid_start_time = pd.Timestamp('2000-01-01 00:00:02')
        mid_end_time = pd.Timestamp('2000-01-01 00:00:04')

        # Create a new column 'new_column' and fill it with 100 for rows within the time range
        df.loc[mid_start_time:, 'total_demand'] = capacity  # Set the value to 100 for the specified time range

        # from 2nd second to 4th second, total demand is 100% capacity
        # df.loc[mid_end_time:, 'total_demand'] = capacity *3/2
    

    if mechanism != 'baseline':
        ax2.plot(df.index, df['total_demand'], color='c', linestyle='-.', label='Demand')


    ax2.fill_between(df.index, 0, df['goodput'], color='green', alpha=0.2, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color='red', alpha=0.3, label='SLO Violation')
    ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color='c', alpha=0.3, label='Dropped Req')
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
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.1), ncol=2, frameon=False)

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
        ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color='tab:blue', alpha=0.3, label='Rate Limited Req')
    
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, frameon=True)
    # concurrent_clients = re.findall(r"\d+", filename)[0]
    # plt.suptitle(f"Mechanism: {mechanism}. Number of Concurrent Clients: {concurrent_clients}")

    # add the ax2 a title, saying that `The Average Goodput Under Overload is: ` + calculate_average_goodput(filename)
    goodputAve, goodputStd = calculate_average_goodput(filename)
    ax2.set_title(f"The Goodput has Mean: {goodputAve} and Std: {goodputStd}")

    latency_99th = read_tail_latency(filename)
    # print("99th percentile latency:", latency_99th)
    # convert the latency_99th from string to milliseconds
    latency_99th = pd.Timedelta(latency_99th).total_seconds() * 1000
    # round the latency_99th to 2 decimal places
    average_99th = df['tail_latency'].mean()
    lat95 = df[(df['status'] == 'OK')]['latency'].quantile(95 / 100)
    ax1.set_title(f"95-tile Latency over Time: {round(lat95, 2)} ms")

    filetosave = mechanism + '-' + method + '-' + str(capacity) + timestamp + '.pdf' 
    plt.savefig(filetosave, dpi=300, bbox_inches='tight')
    # plt.show()
    if not noPlot:
        plt.show()
    plt.close()


def plot_all_interfaces(dfall):
    # Extract mechanism from the filename
    # mechanism_match = re.search(r"control-(\w+)-", filename)
    # if mechanism_match:
    #     mechanism = mechanism_match.group(1)
    # else:
    #     raise ValueError("Mechanism not found in filename.")

    # Define colors that are clearly readible in grayscale
    colors = ['#34a853', '#ea4335', '#4285f4', '#fbbc05']

 
    # Create 2 rows x 3 columns subplots with shared x-axis
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 6.5), sharex=True)
    # axs = axs.ravel()  # Flatten the 2D array of axes

    # Define interfaces
    interfaces = ['compose', 'user-timeline', 'home-timeline']

    # Plot data for each interface in the first column
    for i, interface in enumerate(interfaces):
        # ax = axs[i]
        ax = axs[i, 0]
        df = dfall[interface]
        # Fill NaN values in DataFrame
        df = df.fillna(0)
        ax.plot(df.index, df['throughput'], color='tab:blue', linestyle='-', label='Req Sent', alpha=0.7)
        ax.plot(df.index, df['goodput'], color='green', linestyle='--', label='Goodput', alpha=0.7)
        
        ax.fill_between(df.index, 0, df['goodput'], color=colors[0], alpha=0.8)
        ax.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], alpha=0.8, label='SLO Violation')
        ax.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], alpha=0.8, label='Dropped Req')

        df['total_demand'] = capacity/2

        # Define the time range from 2nd to 4th second
        mid_start_time = pd.Timestamp('2000-01-01 00:00:01')
        # Create a new column 'new_column' and fill it with 100 for rows within the time range
        df.loc[mid_start_time:, 'total_demand'] = capacity  # Set the value to 100 for the specified time range

        ax.plot(df.index, df['total_demand'], color='c', linestyle='-.', label='Demand')
        ax.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color=colors[3], alpha=0.8, label='Rate Limited Req')

        axs[i, 0].set_ylim(0, 8000)
        ax.set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in ax.get_yticks()/1000])
        ax.set_title(f"{interface}")
        ax.set_ylabel('Req/s')

    # share y axis for the first column
    # axs[1, 0].set_ylim(0, 8000)
    # axs[2, 0].set_ylim(0, 8000)
        
    # first row, share the same legend above the plot
    axs[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.5), frameon=False, ncol=3)

    # remove the x & y tick labels for the first row 2nd and 3rd subplots
    axs[1, 0].set_xticklabels([])
    axs[2, 0].set_xticklabels([])
    # axs[1, 0].set_yticklabels([])
    # axs[2, 0].set_yticklabels([])

    for ax in axs.flat:
        # x axis are datetime objects, so we want to add vertical grid lines for each second
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
    # Customize x-axis to display seconds
    # for ax in axs:
    #     ax.xaxis.set_major_formatter(md.DateFormatter('%S'))

    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%S'))
    # # Plot additional metrics for the second row
    # for ax 0, 1
    ax3 = axs[1, 1]
    max_price = 0
    df_price_dict = extract_ownPrices(cloudlabOutput)
    # create 5 different line styles
    linestyles = ['-', '--', '-.', ':', '-']
    selected_services = []
    for service_name, df_price in df_price_dict.items():
        print_name = service_name.replace('-service', '')
        print_name = print_name.replace('-', '\n')
        # only keep the data when the index is smaller than the last timestamp of the df.index
        df_price = df_price[df_price.index < df.index[-1]]
        moving_average_price = df_price[service_name].rolling(latency_window_size).mean()
        # only plot the service price if it is not alway 0
        if moving_average_price.max() < 100:
            continue
        else:
            selected_services.append(service_name)
        # use the line style from the list for different services
        ax3.plot(df_price.index, moving_average_price, label=print_name, linestyle=linestyles.pop(0))
        # put the legend on the right side of the plot
        ax3.legend(ncol=1, frameon=False, bbox_to_anchor=(1.0, 1), loc='upper left')
        # replace y tick of 1000 with 1k, 2000 with 2k, etc
        ax3.set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in ax3.get_yticks()/1000])
        max_price = max(moving_average_price.max(), max_price)
    ax3.set_title('Price per Service')
    # ax3.set_xlabel('Time')

    # delete the df_price in df_price_dict when its max price is 0 
    keys_to_delete = []
    for service_name, df_price in df_price_dict.items():
        if df_price[service_name].rolling(latency_window_size).mean().max() < 1:
            keys_to_delete.append(service_name)

    # Now delete the keys outside the loop
    for key in keys_to_delete:
        del df_price_dict[key]

    # # Concatenate all DataFrames along the columns
    # for service_name, df_price in df_price_dict.items():
    #     df = pd.merge(df, df_price, left_index=True, right_index=True, how='left')

    
    # add all the service price to df with the c    # ax4 = axs[1, 1], this subplot plot the price per interface rather than per service.
    interface_service = {'compose': ['nginx-web-server', 'compose-post-service', 'social-graph-service', 'post-storage-service', 'home-timeline-service', 'user-timeline-service', 'user-service', 'text-service', 'media-service', 'unique-id-service', 'user-mention-memcached', 'url-shorten-mongo', 'post-storage-service', 'user-timeline-redis', 'user-timeline-mongo', 'social-graph-redis', 'compose-post-service', 'social-graph-service', 'post-storage-service', 'home-timeline-service', 'user-timeline-service', 'user-service', 'text-service', 'media-service', 'unique-id-service', 'user-mention-memcached', 'url-shorten-mongo', 'post-storage-service', 'user-timeline-redis', 'user-timeline-mongo', 'social-graph-redis', 'post-storage-service', 'home-timeline-redis', 'social-graph-service', 'post-storage-mongo', 'url-shorten-service', 'user-mention-service'], 
                         'home-timeline': ['nginx-web-server', 'home-timeline-service', 'post-storage-mongo', 'post-storage-memcached', 'home-timeline-service', 'post-storage-mongo', 'post-storage-memcached', 'post-storage-service', 'home-timeline-redis'], 
                         'user-timeline': ['nginx-web-server', 'user-timeline-service', 'post-storage-mongo', 'post-storage-memcached', 'post-storage-service', 'user-timeline-redis', 'user-timeline-mongo', 'user-timeline-service', 'post-storage-mongo', 'post-storage-memcached', 'post-storage-service', 'user-timeline-redis', 'user-timeline-mongo']}
    # create a new column in df, named with the interface name, and fill in with 0
    ax4 = axs[2, 1]
    # for each interface, select the services that is left in df_price_dict
    linestyles = ['-', '--', '-.', ':', '-']
    for interface, services in interface_service.items():
        relevant_services = [service for service in services if service in df_price_dict]
        # print name is the interface name
        print_name = interface
        print_name = print_name.replace('-', '\n')
        if relevant_services:
            # Concatenate the price DataFrames of the relevant services
            combined_prices_df = pd.concat([df_price_dict[service] for service in relevant_services], axis=1)

            # smooth the dataframe by interpolating the missing values
            combined_prices_df = combined_prices_df.interpolate(method='linear', axis=0).ffill().bfill()

            # Calculate the maximum price for the interface
            max_prices = combined_prices_df.max(axis=1)

        ax4.plot(max_prices, label=print_name, linestyle=linestyles.pop(0))
        ax4.legend(ncol=1, frameon=False, bbox_to_anchor=(1.0, 1), loc='upper left')
        # replace y tick of 1000 with 1k, 2000 with 2k, etc
        ax4.set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in ax4.get_yticks()/1000])
        ax4.set_title('Price per Interface')
        # ax4.set_xlabel('Time')

    # last subplot, 0, 1, plot the per service queueing delay measured by the golang
    ax5 = axs[0, 1]
    dict_queuing_delay = extract_waiting_times_all(cloudlabOutput)

    # also, plot the sum of the queuing delay of all services by adding up the queuing delay of each service
    # first create an empty dataframe 
    df_queuing_delay_sum = pd.DataFrame()

    linestyles = ['-', '--', '-.', ':', '-']
    for service_name, df_queuing_delay in dict_queuing_delay.items():
        print_name = service_name.replace('-service', '')
        print_name = print_name.replace('-', '\n')
        if len(df_queuing_delay.index) == 0:
            continue
        # if service_name is not in ax3, then continue
        if service_name not in selected_services:
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

            # smooth the mean_queuing_delay with moving average of the neighbouring 3 data points
            mean_queuing_delay = mean_queuing_delay.rolling(3).mean()
            ax5.plot(mean_queuing_delay.index, mean_queuing_delay, label=print_name, linestyle=linestyles.pop(0))
    ax5.set_yscale('log')
    ax5.set_title('Queueing Delay per Service')
    ax5.legend(ncol=1, frameon=False, bbox_to_anchor=(1.0, 1), loc='upper left')



    plt.xlabel('Time (second)')
    # plt.tight_layout()
    plt.savefig('all-interfaces.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# # function to plot the goodput per interface (3 in total) on the top row, and the 1. queuing delay per node 2. price per node 3. total price per interface on the 2nd row
# def plot_all_interfaces(df, filename, computation_time=0):
#     # mechanism is the word after `control-` in the filename
#     # e.g., breakwater in social-compose-control-breakwater-parallel-capacity-8000-1209_1620.json
#     mechanism = re.findall(r"control-(\w+)-", filename)[0]

#     # create a list of 4 colors clearly readible in grayscale
#     colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

#     # create a list of 6 subplots, 2 rows and 3 columns. with shared x axis and y axis
#     fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(6, 5), sharex=True, )

#     # add to all 3 axes vertical x grid lines for each second, x axis are datetime objects


#     # add .fillna(0) to all the columns of df, so that we can plot the throughput and goodput
#     df = df.fillna(0)
#     # for axes in first row, plot the throughput and goodput and fill between them
#     interfaces = ['compose', 'user-timeline', 'home-timeline']
#     first_row_axes = [ax1, ax2, ax3]
#     for i in range(len(interfaces)):
#         interface = interfaces[i]
#         ax = first_row_axes[i]

#         mask = df['interface'] == interface
#         ax.set_ylabel('Throughput (req/s)', color='tab:blue')
#         ax.plot(df[mask].index, df[mask]['throughput'], color='tab:blue', linestyle='-', label='Req Sent', alpha=0.2)
#         ax.plot(df[mask].index, df[mask]['goodput'], color='green', linestyle='--', alpha=0.2)
#         ax.plot(df[mask].index, df[mask]['dropped']+df[mask]['throughput'], color='tab:blue', linestyle='-', alpha=0.2, label='Req Sent')
#         ax.fill_between(df[mask].index, 0, df[mask]['goodput'], color='green', alpha=0.2, label='Goodput')
#         ax.fill_between(df[mask].index, df[mask]['goodput'], df[mask]['throughput'], color='red', alpha=0.3, label='SLO Violation')
#         ax.fill_between(df[mask].index, df[mask]['throughput'], df[mask]['throughput'] + df[mask]['dropped'], color='c', alpha=0.3, label='Dropped Req')
#         # if mechanism in ['charon', 'breakwater', 'breakwaterd']:
#         #     ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], color='tab:blue', alpha=0.3, label='Rate Limited Req')
#         ax.tick_params(axis='y', labelcolor='tab:blue')
#         # ax.set_ylim(0, 15000)
#         ax.set_title(f"Interface: {interface}")
#         ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, frameon=True)

#     plt.gca().xaxis.set_major_formatter(md.DateFormatter('%S'))
#     plt.xlabel('Time (second)')
#     # plot and show the figure
#     plt.show()


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

    goodputAve, goodputStd = calculate_average_goodput(filename)
    ax1.set_title(f"The Goodput has Mean: {goodputAve} and Std: {goodputStd}")

    latency_99th = read_tail_latency(filename)
    latency_99th = pd.Timedelta(latency_99th).total_seconds() * 1000
    latency_median = read_tail_latency(filename, percentile=50)
    latency_median = pd.Timedelta(latency_median).total_seconds() * 1000
    latency_mean = read_mean_latency(filename)
    latency_mean = pd.Timedelta(latency_mean).total_seconds() * 1000

    ax1.set_title(f"Mean, Median, 99-tile Latency: {round(latency_mean, 2)} ms, {round(latency_median, 2)} ms, {round(latency_99th, 2)} ms")

    plt.savefig(mechanism + '.1000c.latency.png', format='png', dpi=300, bbox_inches='tight')
    if not noPlot:
        plt.show()
    plt.close()


def analyze_data_all(filenames):
    # ./social-compose-control-charon-parallel-capacity-7000-1023_1521.json
    # if filename contains control, read the string after it as INTERCEPTOR
    global INTERCEPTOR, capacity 
    filename = filenames[0]
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
    # interface is between `social-` and `-control` in the filename
    # e.g., compose in social-compose-control-charon-parallel-capacity-7000-1023_1521.json
    # loop and concatenate all the dataframes
    data_dict = {}
    for filename in filenames:
        interface = re.findall(r"social-(.*?)-control", filename)[0]
        data = read_data(filename)
        df = convert_to_dataframe(data, init=True)
        df = calculate_throughput(df)
        df = calculate_goodput(df, SLO)
        df = calculate_loadshedded(df)
        df = calculate_ratelimited(df)
        df = calculate_tail_latency(df)
        # df["interface"] = interface
        data_dict[interface] = df

    # concatenate all the dataframes in data_dict
    # dfall = pd.concat(data_dict.values(), ignore_index=True)

    plot_all_interfaces(data_dict)
    

if __name__ == '__main__':
    # take multiple files as input (e.g., 3 files)
    files = sys.argv[1:]
    # read from the filename, if `S_` is in the filename, then alibaba is true, otherwise, alibaba is false
    alibaba = False
    # the method is the word between `social-` and `-control` in the filename
    method = re.findall(r"social-(.*?)-control", files[0])[0]
    timestamp = re.findall(r"-\d+_\d+", files[0])[0]

    SLO = get_slo(method, tight=False, all_methods=('S_' not in method))

    analyze_data_all(files)
