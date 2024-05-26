import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Constants used throughout the module
throughput_time_interval = '50ms'
latency_window_size = '200ms'
offset = 2.5

def medianL(lst):
    """
    Calculate the median of a list of numbers.

    Parameters:
    lst (list): List of numerical values.

    Returns:
    float: Median value of the list. None if the list is empty.
    """
    n = len(lst)
    if n < 1:
        return None
    if n % 2 == 1:
        return sorted(lst)[n//2]
    else:
        return sum(sorted(lst)[n//2-1:n//2+1])/2.0

def read_tail_latency(filename, percentile=99):
    """
    Read the tail latency from a JSON file.

    Parameters:
    filename (str): Path to the JSON file.
    percentile (int): Percentile of latency to read (default is 99).

    Returns:
    float: Tail latency in milliseconds. None if the latency distribution is empty or percentile not found.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    latency_distribution = data["latencyDistribution"]
    if not latency_distribution:
        return None
    for item in latency_distribution:
        if item["percentage"] == percentile:
            return item["latency"] / 1000000
    return None

def read_mean_latency(filename):
    """
    Read the mean latency from a JSON file.

    Parameters:
    filename (str): Path to the JSON file.

    Returns:
    float: Mean latency in milliseconds.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["average"] / 1000000

def read_data(filename):
    """
    Read detailed data from a JSON file.

    Parameters:
    filename (str): Path to the JSON file.

    Returns:
    dict: Detailed data from the JSON file. None if there is an error reading the file.
    """
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
            return data["details"]
        except:
            print(f"Error reading file {filename}")
            return None

def convert_to_dataframe(data):
    """
    Convert detailed data into a pandas DataFrame and preprocess it.

    Parameters:
    data (dict): Detailed data.

    Returns:
    pandas.DataFrame: Preprocessed DataFrame. Empty DataFrame if the input data is empty.
    """
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    summary = df.groupby('status')['status'].count().reset_index(name='count')
    df.set_index('timestamp', inplace=True)
    df['latency'] = df['latency'] / 1000000
    df = df[df['status'] != 'Unavailable']
    df = df[df['status'] != 'Canceled']
    if len(df) == 0:
        return df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]
    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df

def calculate_tail_latency(filename, percentile=99):
    """
    Calculate the tail latency from a JSON file.

    Parameters:
    filename (str): Path to the JSON file.
    percentile (int): Percentile of latency to calculate (default is 99).

    Returns:
    float: Tail latency in milliseconds. None if the DataFrame is empty.
    """
    data = read_data(filename)
    df = convert_to_dataframe(data)
    if df.empty:
        print("DataFrame is empty for ", filename)
        return None
    latency_percentile = df[df['status'] == 'OK']['latency'].quantile(percentile / 100)
    return latency_percentile

def calculate_throughput(filename):
    """
    Calculate the throughput of requests from a JSON file.

    Parameters:
    filename (str): Path to the JSON file.

    Returns:
    float: Throughput of OK requests in requests per second. None if the DataFrame is empty.
    """
    data = read_data(filename)
    df = convert_to_dataframe(data)
    if df.empty:
        print("DataFrame is empty for ", filename)
        return None
    throughput = df['status'][df['status'] == 'OK'].count() / (df.index.max() - df.index.min()).total_seconds()
    return throughput

def calculate_average_goodput(filename, slo):
    """
    Calculate the average goodput from a JSON file.

    Parameters:
    filename (str): Path to the JSON file.
    slo (float): Service Level Objective (SLO) for goodput calculation.

    Returns:
    float: Average goodput in requests per second.
    """
    data = read_data(filename)
    df = convert_to_dataframe(data)
    goodput = calculate_goodput(df, slo=slo)
    return goodput

def calculate_goodput(df, slo):
    """
    Calculate the goodput from a DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing request data.
    slo (float): Service Level Objective (SLO) for goodput calculation.

    Returns:
    float: Goodput in requests per second.
    """
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='ffill')
    time_interval = (df.index.max() - df.index.min()).total_seconds()
    goodput = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].count() / time_interval
    return goodput

def is_within_duration(file_timestamp_str, start_time_str, end_time_str):
    """
    Check if a file's timestamp is within a given duration.

    Parameters:
    file_timestamp_str (str): Timestamp of the file in the format MMDD_HHMM.
    start_time_str (str): Start time of the duration in the format MMDD_HHMM.
    end_time_str (str): End time of the duration in the format MMDD_HHMM.

    Returns:
    bool: True if the file's timestamp is within the duration, False otherwise.
    """
    start_time = datetime.strptime(start_time_str, "%m%d_%H%M")
    end_time = datetime.strptime(end_time_str, "%m%d_%H%M")
    file_time = datetime.strptime(file_timestamp_str, "%m%d_%H%M")
    return start_time <= file_time <= end_time

def is_within_any_duration(file_timestamp_str, time_ranges):
    """
    Check if a file's timestamp is within any given duration.

    Parameters:
    file_timestamp_str (str): Timestamp of the file in the format MMDD_HHMM.
    time_ranges (list of tuple): List of start and end time tuples in the format (MMDD_HHMM, MMDD_HHMM).

    Returns:
    bool: True if the file's timestamp is within any of the given durations, False otherwise.
    """
    file_time = datetime.strptime(file_timestamp_str, "%m%d_%H%M")
    for start_time_str, end_time_str in time_ranges:
        start_time = datetime.strptime(start_time_str, "%m%d_%H%M")
        end_time = datetime.strptime(end_time_str, "%m%d_%H%M")
        if start_time <= file_time <= end_time:
            return True
    return False

def find_latest_files(selected_files):
    """
    Find the latest files for each unique configuration from a list of selected files.

    Parameters:
    selected_files (list of str): List of file paths.

    Returns:
    list of str: List of latest file paths for each unique configuration.
    """
    latest_files = {}
    for filename in selected_files:
        parts = os.path.basename(filename).split('-')
        if len(parts) >= 8:
            config_key = tuple(parts[3:7])
            if config_key not in latest_files or os.path.getmtime(filename) > os.path.getmtime(latest_files[config_key]):
                latest_files[config_key] = filename
    return list(latest_files.values())


def load_data(method, list_of_tuples_of_experiment_timestamps, slo, request_count_cutoff=None, tail_latency_cutoff=None):
    """
    Load data from JSON files for a given method and list of experiment timestamps.

    Parameters:
    method (str): Method name.
    list_of_tuples_of_experiment_timestamps (list of tuple): List of tuples containing start and end time strings.
    request_count_cutoff (int): Minimum request count for a file to be considered valid (default is None).
    tail_latency_cutoff (float): Maximum tail latency in milliseconds for a file to be considered valid (default is None).

    Returns:
    DataFrame: DataFrame containing the loaded data.
    """

    # A dictionary to hold intermediate results, indexed by (overload_control, method_subcall, capacity)
    results = {}

    # For every file in the directory
    selected_files = []
    # For every file in the directory and another directory `~/Sync/Git/charon-experiments/json`
    for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{method}-control-*-parallel-capacity-*.json')) \
        + glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/charon-experiments-results/'), f'social-{method}-control-*-parallel-capacity-*.json')):
        # Extract the date and time part from the filename
        timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
        # check if the file's timestamp is given format
        # claim 
        if len(timestamp_str) != 9:
            print("File ", filename, " is not valid")
            continue
        
        if is_within_any_duration(timestamp_str, list_of_tuples_of_experiment_timestamps):
            selected_files.append(filename)

    for filename in selected_files:
        # Extract the metadata from the filename
        overload_control, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[3:8]
        capacity = int(capacity_str)


        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid, no OK")
            os.remove(filename)
            continue

        if request_count_cutoff is not None:
            # if the Count: xxx is less than 20000, remove the file
            with open(filename, 'r') as f:
                data = json.load(f)
                if data['count'] < request_count_cutoff:
                    print("File ", filename, " is not valid, count is less than", request_count_cutoff)
                    continue

        # Calculate latencies and throughput
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)
        goodput = calculate_average_goodput(filename, slo)

        # If valid latency data
        if latency_99 is not None:
            if tail_latency_cutoff is not None and latency_95 > tail_latency_cutoff:
                print("File ", filename, f" is not valid, 95th percentile is greater than {tail_latency_cutoff}ms")
                continue

            key = (overload_control, method_subcall, capacity)
            if key not in results:
                results[key] = {
                    'Load': capacity,
                    'Throughput': [throughput],
                    'Goodput': [goodput],
                    '99th_percentile': [latency_99],
                    '95th_percentile': [latency_95],
                    'Median Latency': [latency_median],
                    'File': [filename]
                }
            else:
                results[key]['Throughput'].append(throughput)
                results[key]['99th_percentile'].append(latency_99)
                results[key]['95th_percentile'].append(latency_95)
                results[key]['Median Latency'].append(latency_median)
                results[key]['Goodput'].append(goodput) 
                results[key]['File'].append(filename)
 
    # check if the results is empty
    if not results:
        print(f"[ERROR] No results extracted from the files for {method}")
        return None
    rows = []
    for (overload_control, method_subcall, capacity), data in results.items():
        row = {
            'Load': capacity,
            'Throughput': np.nanmean(data['Throughput']),
            'Throughput std': np.nanstd(data['Throughput']),
            'Goodput': np.nanmean(data['Goodput']),
            'Goodput std': np.nanstd(data['Goodput']),
            '99th_percentile': np.nanmean(data['99th_percentile']),
            '99th_percentile std': np.nanstd(data['99th_percentile']),
            '95th_percentile': np.nanmean(data['95th_percentile']),
            '95th_percentile std': np.nanstd(data['95th_percentile']),
            'Median Latency': np.nanmean(data['Median Latency']),
            'method_subcall': method_subcall,
            'overload_control': overload_control,
            'file_count': len(data['File'])
        }

        rows.append(row)
        print(f'Control: {overload_control}, \t Method: {method_subcall}, Load: {capacity}, Average 99th Lat: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, # Files: {row["file_count"]}, Throughput: {row["Throughput"]:.2f}, Goodput: {row["Goodput"]:.2f}')

    df = pd.DataFrame(rows)
    df.sort_values('Load', inplace=True)

    return df
