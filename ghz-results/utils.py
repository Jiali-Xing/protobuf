import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Constants used throughout the module
throughput_time_interval = '200ms'
latency_window_size = '200ms'
offset = 5 # an offset of 5 seconds to omit pre-spike metrics

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

def convert_to_dataframe(data, include_warmup=False):
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
    if include_warmup:
        df = df[df.index < df.index[-1] - pd.Timedelta(seconds=offset)]
    else:
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

            # # debug:
            # # if load is 8000 and app is `S_161142529` and method is `breakwater`, print the filename
            # if '8000' in filename and 'S_161142529' in filename and 'breakwater' in filename:
            #     print(filename) 

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


def calculate_goodput_from_file(filename, slo, quantile, average=True):
    # Read the ghz.output file and calculate the average goodput
    # Return the average goodput
    # if filename is a list, average the goodput of all the files
    # this is for bayesian optimization, thus, we will include warm up period too!
    if isinstance(filename, list):
        goodput = 0
        for f in filename:
            goodput += calculate_goodput_from_file(f, slo, quantile, average)
        goodput = goodput / len(filename)
        return goodput

    data = read_data(filename)
    df = convert_to_dataframe(data, include_warmup=True)
    if average:
        goodput = calculate_goodput_mean(df, slo=slo)
    else:
        goodput = goodput_quantile(df, slo=slo, quantile=quantile)
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


def read_tail_latency_from_file(filename, percentile=99):
    if isinstance(filename, list):
        latency = 0
        for f in filename:
            latency += read_tail_latency_from_file(f, percentile=percentile)
        latency = latency / len(filename)
        return latency

    # with open(filename, 'r') as f:
    #     data = json.load(f)
    data = read_data(filename)        
    df = convert_to_dataframe(data)
    percentile_latency = df[(df['status'] == 'OK')]['latency'].quantile(percentile / 100)

    return percentile_latency  # Replace with your actual function


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


def calculate_tail_latency_dynamic(df):
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

    #  remove outliers of the tail latency (those super small values)

    
    return df



def calculate_throughput_dynamic(df):
    # sample throughput every time_interval
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    ok_requests_per_second = ok_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='bfill')
    return df


def calculate_goodput_dynamic(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    # fill in zeros for the missing goodput
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='bfill')
    return df


def read_load_info_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    options = data.get('options', {})
    
    load_info = {
        "load-start": options.get("load-start"),
        "load-end": options.get("load-end"),
        "load-step": options.get("load-step"),
        "load-step-duration": options.get("load-step-duration")
    }
    
    return load_info


def save_iteration_details(optimizer, file_path):
    try:
        # Convert the iteration details to a JSON-compatible format
        iteration_details = optimizer.res
        iteration_details_json = json.dumps(iteration_details, default=str, indent=4)

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