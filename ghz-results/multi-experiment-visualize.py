import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime
import re
from collections import defaultdict
import numpy as np

from slo import get_slo

throughput_time_interval = '50ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds

offset = 2.5  # Define the offset as 2.5 seconds to remove the initial warm-up period

# use the mediam of all data rows rather than the average
# write a function to calculate the median of the data in the list
def medianL(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0

def read_tail_latency(filename, percentile=99):
    with open(filename, 'r') as f:
        data = json.load(f)
    latency_distribution = data["latencyDistribution"]
    # if latency_distribution is empty
    if not latency_distribution:
        return None
    for item in latency_distribution:
        if item["percentage"] == percentile:
            # return item["latency"] convert to ms
            return item["latency"] / 1000000
    return None

def read_mean_latency(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["average"] / 1000000


def read_data(filename):
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
            return data["details"]
        except:
            print(f"Error reading file {filename}")
            return None


def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    summary = df.groupby('status')['status'].count().reset_index(name='count')
    # print("Summary of the status", summary)
    # print(df['timestamp'].min())

    df.set_index('timestamp', inplace=True)
    df['latency'] = df['latency'] / 1000000
    # drop the rows if the `status` is Unavailable
    df = df[df['status'] != 'Unavailable']
    df = df[df['status'] != 'Canceled']
    # if df is empty, return empty df and print the message
    if len(df) == 0:
        return df
    # remove the data within first couple seconds of df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]
    # df = df[df.index < df.index[-1] - pd.Timedelta(seconds=offset)]

    min_timestamp = df.index.min()
    # # record the start time of the first request and the end time of the last request as global variables
    # # only if the global variables are not set yet
    # # if 'start_time' not in globals():
    # if init:
    #     global start_time, end_time
    #     start_time = min_timestamp.astimezone(pytz.UTC)
    #     end_time = df.index.max().astimezone(pytz.UTC)

    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df


def calculate_tail_latency(filename, percentile=99):
    data = read_data(filename)
    df = convert_to_dataframe(data)
    
    if df.empty:
        print("DataFrame is empty for ", filename)
        return None
    
    # Compute the percentile latency
    latency_percentile = df[df['status'] == 'OK']['latency'].quantile(percentile / 100)
    
    return latency_percentile


# calculate_throughput calculates the throughput of the requests
def calculate_throughput(filename):
    data = read_data(filename)
    df = convert_to_dataframe(data)
    if df.empty:
        print("DataFrame is empty for ", filename)
        return None
    # Compute the throughput
    # throughput = df['latency'].count() / (df.index.max() - df.index.min()).total_seconds()
    # throughput should be the status of OK requests 
    throughput = df['status'][df['status'] == 'OK'].count() / (df.index.max() - df.index.min()).total_seconds()
    return throughput


def calculate_average_goodput(filename):
    # Insert your code for calculating average goodput here
    # Read the ghz.output file and calculate the average goodput
    # Return the average goodput
    data = read_data(filename)
    df = convert_to_dataframe(data)
    # slo = 50
    goodput = calculate_goodput(df, slo=SLO)
    return goodput  # Replace with your actual function


def calculate_goodput(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='ffill')

    # calculate goodput by counting the number of requests that are ok and have latency less than slo, and then divide by the time interval
    # time_interval is the time interval for calculating the goodput, last request time - first request time
    time_interval = (df.index.max() - df.index.min()).total_seconds()
    goodput = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].count() / time_interval
    return goodput


# Function to check if a file's timestamp is within the given duration
def is_within_duration(file_timestamp_str, start_time_str, end_time_str):
    # Convert the time strings to datetime objects
    start_time = datetime.strptime(start_time_str, "%m%d_%H%M")
    end_time = datetime.strptime(end_time_str, "%m%d_%H%M")
    file_time = datetime.strptime(file_timestamp_str, "%m%d_%H%M")

    # Check if the file's time is within the start and end times
    return start_time <= file_time <= end_time


# Function to check if a file's timestamp is within any of the given durations
def is_within_any_duration(file_timestamp_str, time_ranges):
    # Convert the file timestamp string to a datetime object
    file_time = datetime.strptime(file_timestamp_str, "%m%d_%H%M")
    
    # Check if the file's time is within any of the start and end times in the ranges
    for start_time_str, end_time_str in time_ranges:
        start_time = datetime.strptime(start_time_str, "%m%d_%H%M")
        end_time = datetime.strptime(end_time_str, "%m%d_%H%M")
        if start_time <= file_time <= end_time:
            return True
    return False


def find_latest_files(selected_files):
    # Dictionary to store the latest file for each unique configuration
    latest_files = {}

    for filename in selected_files:
        # Extract configuration components from the filename
        parts = os.path.basename(filename).split('-')
        if len(parts) >= 8:  # Ensure there are enough parts in the filename
            config_key = tuple(parts[3:7])  # Create a tuple of configuration parts
            # If this config is not in latest_files or the current file is newer, update it
            if config_key not in latest_files or os.path.getmtime(filename) > os.path.getmtime(latest_files[config_key]):
                latest_files[config_key] = filename

    return list(latest_files.values())


def load_data():
    # A dictionary to hold intermediate results, indexed by (overload_control, method_subcall, capacity)
    results = {}

    # For every file in the directory
    selected_files = []
    # For every file in the directory and another directory `~/Sync/Git/rajomon-experiments/json`
    for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{method}-control-*-parallel-capacity-*.json')) \
        + glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/rajomon-experiments-results/'), f'social-{method}-control-*-parallel-capacity-*.json')):
        # Extract the date and time part from the filename
        timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
        # check if the file's timestamp is given format
        # claim 
        if len(timestamp_str) != 9:
            print("File ", filename, " is not valid")
            continue

    # for filename in find_latest_files(selected_files):
    for filename in selected_files:
        # Extract the metadata from the filename
        overload_control, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[3:8]
        capacity = int(capacity_str)

        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid, no OK")
            os.remove(filename)
            continue

        # if the Count: xxx is less than 20000, remove the file
        with open(filename, 'r') as f:
            data = json.load(f)
            countCut = 12000 if 'compose' in method else 10000
            countCut = 2000 if 'S_149998854' in method else countCut
            if data['count'] < countCut:
                print("File ", filename, " is not valid, count is less than", countCut)
                continue

        # Calculate latencies and throughput
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)
        goodput = calculate_average_goodput(filename)

        # If valid latency data
        if latency_99 is not None:
            # if the 99th percentile is greater than 900ms
            tailCut = 900 if 'S_' in method else 500
            if latency_95 > 500:
                print("File ", filename, f" is not valid, 95th percentile is greater than {tailCut}ms")
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
            'Goodput': np.nanmean(data['Goodput']),
            '99th_percentile': np.nanmean(data['99th_percentile']),
            '95th_percentile': np.nanmean(data['95th_percentile']),
            'Median Latency': np.nanmean(data['Median Latency']),
            'method_subcall': method_subcall,
            'overload_control': overload_control,
            'file_count': len(data['File'])
        }

        rows.append(row)
        print(f'Control: {overload_control}, Method: {method_subcall}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')

    df = pd.DataFrame(rows)
    df.sort_values('Load', inplace=True)

    return df


def count_files_by_capacity(selected_files, capacity_pattern):
    capacity_counts = defaultdict(int)

    for filename in selected_files:
        match = re.search(capacity_pattern, filename)
        if match:
            capacity = match.group(1)
            capacity_counts[capacity] += 1

    return capacity_counts


def create_capacity_file_mapping(selected_files, capacity_pattern):
    capacity_file_mapping = defaultdict(list)

    for filename in selected_files:
        if 'motivation-aqm-control-plain' in filename or 'motivation-rl-control-plain' in filename:
            continue
        match = re.search(capacity_pattern, filename)
        if match:
            capacity = match.group(1)
            capacity_file_mapping[capacity].append(filename)

    # Sorting files within each capacity based on the timestamp
    for capacity in capacity_file_mapping:
        capacity_file_mapping[capacity].sort(key=lambda x: re.search(r'(\d{4}_\d{4})\.json', x).group(1))

    return capacity_file_mapping

def categorize_files(capacity_file_mapping, scenarios):
    file_categories = {}

    for capacity, files in capacity_file_mapping.items():
        for i, filename in enumerate(files):
            scenario_index = i % len(scenarios)
            file_categories[filename] = scenarios[scenario_index]

    return file_categories


# Function to extract control type and timestamp from filename
def extract_info(filename):
    parts = filename.split('-')
    control_type = parts[3]
    timestamp = parts[-1].split('.')[0]
    return control_type, timestamp


def extract_info_rl(file_list):
    # Group files by control type and sort by timestamp
    file_groups = defaultdict(list)
    for file in file_list:
        control_type, timestamp = extract_info(file)
        file_groups[control_type].append((timestamp, file))

    for control_type in file_groups:
        file_groups[control_type].sort()

    # Assign depths to each file
    file_depths = {}
    for control_type, files in file_groups.items():
        for i, (_, filename) in enumerate(files):
            file_depths[filename] = (i % 5) + 1
    return file_depths


# load motivation data is similar to load_data, it loads social-motivation-aqm-control-rajomon-parallel-capacity-8000-1222_xxxx.json 
# select them with a time frame and then rank them by timpstamp
def load_motivation_data():
    # A dictionary to hold intermediate results, indexed by (overload_control, method_subcall, capacity)
    results = {}

    # For every file in the directory
    selected_files = []
    # For every file in the directory
    for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{method}-control-*-parallel-capacity-*.json')):
        # Extract the date and time part from the filename
        timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
        # check if the file's timestamp is given format
        if len(timestamp_str) != 9:
            print("File ", filename, " is not valid")
            continue

    for filename in selected_files:
        # Extract the metadata from the filename
        interceptor, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[3:9]
        
        capacity = int(capacity_str)
        # let's skip if the capacity is not factor of 1000
        if capacity > 10000:
            continue
        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid, no OK")
            os.remove(filename)
            continue

        # skip breakwaterd with timestamp earlier than 0302_1912
        if 'breakwaterd' in interceptor and timestamp < '0302_1912':
            continue

        # if the Count: xxx is less than 20000, remove the file
        with open(filename, 'r') as f:
            data = json.load(f)
            countCut = 2000
            if data['count'] < countCut:
                print("File ", filename, " is not valid, count is less than", countCut)
                continue

        # Calculate latencies and throughput
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)
        goodput = calculate_average_goodput(filename)
        
        # Categorize files
        # scenario = scenarioDict[filename]
        scenario = interceptor

        print(f"File {filename} is categorized as {scenario}")
        # If valid latency data
        if latency_99 is not None:
            # if the 99th percentile is greater than 900ms
            if latency_95 > 900 and 'rl' not in method:
                print("File ", filename, " is not valid, 99th percentile is greater than 900ms")
                continue

            key = (scenario, capacity) if 'rl' not in method else (scenario, capacity, interceptor)
            if key not in results:
                results[key] = {
                    'Load': capacity,
                    'Throughput': [throughput],
                    'Goodput': [goodput],
                    '99th_percentile': [latency_99],
                    '95th_percentile': [latency_95],
                    'Median Latency': [latency_median],
                    'File': [filename],
                    'Timestamp': [timestamp],
                    'Scenario': [scenario],
                    'Scheme': [interceptor],
                }
            else:
                results[key]['Throughput'].append(throughput)
                results[key]['99th_percentile'].append(latency_99)
                results[key]['95th_percentile'].append(latency_95)
                results[key]['Median Latency'].append(latency_median)
                results[key]['Goodput'].append(goodput)
                results[key]['File'].append(filename)
                results[key]['Timestamp'].append(timestamp)
                results[key]['Scenario'].append(scenario)
                results[key]['Scheme'].append(interceptor)
 
    # check if the results is empty
    if not results:
        print("[ERROR] No results extracted from the files for motivation")
        return None
    rows = []
    if 'rl' in method:
        for (scenario, capacity, interceptor), data in results.items():
            # use the mediam of all data rows rather than the average
            # write a function to calculate the median of the data in the list

            row = {
                'Load': capacity,
                'Throughput': np.nanmean(data['Throughput']),
                'Goodput': np.nanmean(data['Goodput']),
                '99th_percentile': np.nanmean(data['99th_percentile']),
                '95th_percentile': np.nanmean(data['95th_percentile']),
                'Median Latency': np.nanmean(data['Median Latency']),
                'Service Depth': scenario,
                'timestamp': data['Timestamp'][0],
                'scheme': interceptor,
            }    
            
            rows.append(row)
            print(f'Control: {scenario}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')
    else:
        for (scenario, capacity), data in results.items():

            row = {
                'Load': capacity,
                'Throughput': medianL(data['Throughput']),
                'Goodput': medianL(data['Goodput']),
                '99th_percentile': medianL(data['99th_percentile']),
                '95th_percentile': medianL(data['95th_percentile']),
                'Median Latency': medianL(data['Median Latency']),
                'overload_control': scenario,
                'timestamp': data['Timestamp'][0],
            }    
            
            rows.append(row)
            print(f'Control: {scenario}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')

    df = pd.DataFrame(rows)
    df.sort_values('Load', inplace=True)
    return df


def load_more_data(interfaces):
    # this function is similar to load_data, it loads the json files for all control mechanisms
    # for all 3 methods, and combine them into one dataframe
    results = {}
    # For every file in the directory
    selected_files = []
    # interfaces = ["compose", "user-timeline", "home-timeline"] if app == 'social' else ["hotels-http", "reservation-http", "user-http", "recommendations-http"]
    for interface in interfaces:
        for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{interface}-control-*-parallel-capacity-*.json')):
            # Extract the date and time part from the filename
            timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
            # check if the file's timestamp is given format
            if len(timestamp_str) != 9:
                print("File ", filename, " is not valid")
                continue

    for filename in selected_files:
        # Extract the metadata from the filename
        # overload_control, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[3:8]
        # find the control type, interface, capacity, and timestamp
        # between social and control, is the interface
        interface = re.search(r'social-(.*)-control', filename).group(1)
        # between control and parallel, is the control type
        overload_control = re.search(r'control-(.*)-parallel', filename).group(1)
        # at the end, is the timestamp
        timestamp = re.search(r'(\d{4}_\d{4})\.json', filename).group(1)
        # between parallel and timestamp, is the capacity
        capacity = re.search(r'parallel-capacity-(\d+)', filename).group(1)
        capacity = int(capacity)
        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid, no OK")
            os.remove(filename)
            continue

        if 'all' in method:
            if capacity >= 7000:
                continue

        if 'breakwater' in overload_control and timestamp < '0127_1819':
            continue

        # if the Count: xxx is less than 20000, remove the file
        with open(filename, 'r') as f:
            data = json.load(f)
            if data['count'] < capacity:
                print("File ", filename, " is not valid, count is less than load")
                continue

        # Calculate latencies and throughput
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)
        goodput = calculate_average_goodput(filename)

        # # If valid latency data
        if latency_99 is not None:
            # if the 99th percentile is greater than 900ms
            tailCut = 500 if 'social' in method else 900
            if latency_95 > tailCut:
                print("File ", filename, f" is not valid, 95th percentile is greater than {tailCut}ms")
                continue

            key = (overload_control, interface, capacity)
            if key not in results:
                results[key] = {
                    'Load': capacity,
                    'Request': interface,
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
        
    if not results:
        print("[ERROR] No results extracted from the files for social")
        return None
    rows = []
    for (overload_control, interface, capacity), data in results.items():
        row = {
            'Load': capacity,
            'Request': interface,
            'Throughput': np.nanmean(data['Throughput']),
            'Goodput': np.nanmean(data['Goodput']),
            '99th_percentile': np.nanmean(data['99th_percentile']),
            '95th_percentile': np.nanmean(data['95th_percentile']),
            'Median Latency': np.nanmean(data['Median Latency']),
            'overload_control': overload_control,
        }
        rows.append(row)
        print(f'Control: {overload_control}, Request: {interface}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')

    df = pd.DataFrame(rows)
    # fill in missing goodput and tail latency values 

    # Interpolate to fill NaN values
    df['95th_percentile'] = df['95th_percentile'].interpolate()

    df.sort_values('Load', inplace=True)
    return df

# load sensitivity data is same as load_more_data
def load_sensitivity_data():
    results = {}
    # For every file in the directory
    selected_files = []
    interfaces = ["compose", "S_149998854", "S_102000854", "S_161142529"]
    for interface in interfaces:
        for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{interface}-control-*-parallel-capacity-*.json')):
            # Extract the date and time part from the filename
            timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
            # check if the file's timestamp is given format
            if len(timestamp_str) != 9:
                print("File ", filename, " is not valid")
                continue

            # Check if the file's timestamp is within the given duration
            robust_rajomon = [
                ("0202_0104", "0202_0249"),
            ]
            if is_within_any_duration(timestamp_str, robust_rajomon):
                selected_files.append(filename)
 
    for filename in selected_files:
        # Extract the metadata from the filename
        interface = re.search(r'social-(.*)-control', filename).group(1)
        overload_control = re.search(r'control-(.*)-parallel', filename).group(1)
        timestamp = re.search(r'(\d{4}_\d{4})\.json', filename).group(1)
        capacity = re.search(r'parallel-capacity-(\d+)', filename).group(1)
        capacity = int(capacity)
        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid, no OK")
            os.remove(filename)
            continue

        # if the Count: xxx is less than 20000, remove the file
        with open(filename, 'r') as f:
            data = json.load(f)
            if data['count'] < capacity:
                print("File ", filename, " is not valid, count is less than load")
                continue

        # Calculate latencies and throughput
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)
        goodput = calculate_average_goodput(filename)

        # # If valid latency data
        if latency_99 is not None:
            tailCut = 500 if 'social' in method else 900
            if latency_95 > tailCut:
                print("File ", filename, f" is not valid, 95th percentile is greater than {tailCut}ms")
                continue

            key = (overload_control, interface, capacity)
            if key not in results:
                results[key] = {
                    'Load': capacity,
                    'Request': interface,
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
        
    if not results:
        print("[ERROR] No results extracted from the files for social")
        return None
    rows = []
    for (overload_control, interface, capacity), data in results.items():
        row = {
            'Load': capacity,
            'Request': interface,
            'Throughput': np.nanmean(data['Throughput']),
            'Goodput': np.nanmean(data['Goodput']),
            '99th_percentile': np.nanmean(data['99th_percentile']),
            '95th_percentile': np.nanmean(data['95th_percentile']),
            'Median Latency': np.nanmean(data['Median Latency']),
            'overload_control': overload_control,
        }
        rows.append(row)
        print(f'Control: {overload_control}, Request: {interface}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')

    df = pd.DataFrame(rows)
    # Interpolate to fill NaN values
    df['95th_percentile'] = df['95th_percentile'].interpolate()

    df.sort_values('Load', inplace=True)
    return df
   

def main():
    global method
    global SLO
    global tightSLO 

    method = os.getenv('METHOD', 'compose')
    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'
    SLO = get_slo(method=method, tight=tightSLO, all_methods=False)

    motivation = multipleFiles = sensitivity = alibaba_combined = False
    alibaba_combined = False
    if alibaba_combined:
        # init a dataframe to merge the 3 alibaba interfaces
        alibaba_df = pd.DataFrame()
        # loop through the 3 alibaba interfaces and load the data and combine them into one dataframe
        interfaces = ["S_102000854", "S_149998854", "S_161142529"]
        for interface in interfaces:
            method = interface
            SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
            df = load_data()
            df['Request'] = interface
            alibaba_df = pd.concat([alibaba_df, df])
        df = alibaba_df
    elif sensitivity:
        df = load_sensitivity_data()
    elif os.getenv('MOTIVATION_BOTH', 'False').lower() == 'true':
        motivation = True
        multipleFiles = False
        df = load_motivation_data()
    elif method == 'all-methods-social':
        multipleFiles = True
        interfaces = ["compose", "user-timeline", "home-timeline"]
        df = load_more_data(interfaces)
    elif method == 'all-methods-hotel':
        multipleFiles = True
        interfaces = ["hotels-http", "reservation-http", "user-http", "recommendations-http"]
        df = load_more_data(interfaces)
    else:
        df = load_data()
    print(f"Method: {method}, Tight SLO: {tightSLO}, Motivation: {motivation}, Multiple Files: {multipleFiles}, Sensitivity: {sensitivity}, Alibaba Combined: {alibaba_combined}")
    print(df)
    if df is None:
        return
    # Extract data for each method and latency percentile
    # Define the methods you want to plot
    # methods = ['parallel']
    # methods = ['sequential', 'parallel']
    # control_mechanisms = ['dagor', 'breakwater', 'rajomon', 'breakwaterd'] if not motivation else ['nginx-web-server', 'service-6', 'all']
    if not motivation:
        control_mechanisms = ['breakwaterd', 'breakwater', 'rajomon', 'dagor', ]
    elif method == 'motivation-aqm':
        control_mechanisms = ['nginx-web-server', 'service-6', 'all', 'plain']
    else:
        control_mechanisms = ['breakwater', 'breakwaterd', 'dagorf', 'dagor']


    # Define markers for each method
    # markers = ['o', 's', 'x']
    # whatLatency = ['Median Latency']
    whatLatency = ['95th_percentile']
    # whatLatency = ['95th_percentile', '99th_percentile',]

    # Map control mechanisms to colors, user material design colors
    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        # 'breakwaterd': a darker version of breakwater
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'dagorf': '#1B5E20',
        'rajomon': '#FF9800',
        'nginx-web-server': '#9C27B0',  # Example color for nginx-web-server
        'service-6': '#3F51B5',  # Example color for service-6
        'all': '#009688',  # Example color for all
    }

    # Map latency metrics to markers
    markers = {
        '99th_percentile': 'o',
        'Median Latency': 's',
        '95th_percentile': 'x',
    }

    lineStyles = {
        'plain': '-',
        'breakwater': '--',
        'breakwaterd': '-.',
        'dagor': ':',
        'dagorf': '-',
        'rajomon': '-',
        'nginx-web-server': '--',  # Example line style for nginx-web-server
        'service-6': ':',  # Example line style for service-6
        'all': '-.',  # Example line style for all
    }

    labelDict = {
        'plain': 'No Control',
        'breakwater': 'breakwater',
        'breakwaterd': 'breakwaterd',
        'dagor': 'dagor',
        'dagorf': 'dagor frontend',
        'rajomon': 'our model',
        'nginx-web-server': 'Frontend',
        'service-6': 'Backend',
        'all': 'Coordinated',
    } if not motivation else {
        'plain': 'No Control',
        'breakwater': 'Rate\nLimiting\nFrontend',
        'dagorf': 'AQM\nFrontend',
        'breakwaterd': 'Rate\nLimiting',
        'dagor': 'AQM',
        'rajomon': 'our model',
    }

    if alibaba_combined:
        # similar to the multipleFiles, with rotate = True, but now we have 3 columns x 2 rows, each column for a alibaba interface S_10, S_14, S_16
        # Plot the ax1 and ax2, with for all control mechanisms the latency and throughput. but now we have multiple interfaces,
        # each row of subplots are for one interface
        # interfaces = ["S_102000854", "S_149998854", "S_161142529"]
        ali_dict = {
            "S_102000854": "S1",
            "S_149998854": "S2",
            "S_161142529": "S3",
        }
        rotate = True
        if rotate:
            fig, axs = plt.subplots(2, len(interfaces), figsize=(6, 3.5))
        else:
            fig, axs = plt.subplots(len(interfaces), 2, figsize=(6, 7/4*len(interfaces)), sharex=True, sharey='row')

        for i, interface in enumerate(interfaces):
            ax1, ax2 = axs[i] if not rotate else axs[:, i]
            for control in control_mechanisms:
                # apply moving average on the Goodput of subset
                mask = (df['overload_control'] == control) & (df['Request'] == interface)
                subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
                # Plot each latency metric with a different marker, but the same color for the control mechanism
                for latency in whatLatency:
                    # subset_filtered = subset[subset[latency] < 550]  # Filter out any extreme latency values if necessary
                    subset_filtered = subset
                    # use marker for the 99th percentile and 95th percentile
                    ax1.plot(subset_filtered['Load'], subset_filtered[latency],
                            color=colors[control], linestyle=lineStyles[control],
                            label=labelDict[control] if latency == '95th_percentile' else None,
                            linewidth=2,
                            marker=markers[latency] if latency == '99th_percentile' else None,
                            )
                
                # Plot throughput on ax2 using the same color for each control mechanism
                ax2.plot(subset['Load'], subset['Goodput'],
                        label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2)
                
                iname = ali_dict[interface]

                ax1.set_title(f'{iname}')
                if rotate:
                    # set y label for the first column only
                    if i == 0:
                        ax1.set_ylabel('95th Tail\nLatency (ms)')
                        ax2.set_ylabel('Goodput (RPS)')
                    else:
                        # remove y ticks 
                        ax1.set_yticklabels([])
                        ax2.set_yticklabels([])
                else:
                    ax1.set_ylabel('95th Tail Latency (ms)')
                    ax2.set_ylabel('Goodput (RPS)')

            if not rotate:
                ax1.grid(True)
                ax2.grid(True)

        # Configure ax2 (throughput)
        # axs second row first column add legend
        if not rotate:
            axs[0][0].legend(frameon=False)
        ax2.set_xlabel('Load (RPS)')

        if rotate:
            # skip the x label for the first row
            for ax in axs[0]:
                ax.set_xlabel('')
                # ax.set_xticklabels([])
            # plt.subplots_adjust(wspace=0, hspace=0)
            # add grid to the all subplots and align the grid lines across the subplots
            # set the ylimit for each row of subplots to be same
            # first row
            for ax in axs.flatten()[0:len(interfaces)]:
                ax.grid(True)
                ax.set_ylim(40, 250)
            # second row
            for ax in axs.flatten()[len(interfaces):]:
                ax.grid(True)
                ax.set_ylim(0, 4200)

            # replace the y tick of second row from 4000 to 4k
            # for ax in axs.flatten()[4]:
            axs.flatten()[len(interfaces)].set_yticklabels(['0', '1k', '2k', '3k', '4k'])
            # remove the spaces between the columns and rows

            # add legend to the first row outside the plot, on the top
            axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=4)

            # share the x axis for each column
            for i in range(len(interfaces)):
                axs[0][i].get_shared_x_axes().join(axs[0][i], axs[1][i])


        # Save and display the plot
        # plt.tight_layout()
        plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime("%m%d")}.pdf'))
        plt.show()
        return


    # Create 1x2 subplots for latencies and throughput
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
    ax1, ax2 = axs  # ax1 for latencies, ax2 for throughput

    for control in control_mechanisms:
        mask = (df['overload_control'] == control)
        subset = df[mask]

        # rename the control mechanism breakwaterd to breakwaterd
        if control == 'breakwaters':
            control = 'breakwaterd'
        # Plot each latency metric with a different marker, but the same color for the control mechanism
        for latency in whatLatency:
            # subset_filtered = subset[subset[latency] < 550]  # Filter out any extreme latency values if necessary
            subset_filtered = subset
            # use marker for the 99th percentile and 95th percentile
            ax1.plot(subset_filtered['Load'], subset_filtered[latency],
                     color=colors[control], linestyle=lineStyles[control],
                     label=labelDict[control] if latency == '95th_percentile' else None,
                     linewidth=2,
                     marker=markers[latency] if latency == '99th_percentile' else None,
                     )  
        plotGoodput = True if not motivation else False

        if plotGoodput:
            ax2.plot(subset['Load'], subset['Goodput'],
                    label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2)
                    # marker=markers[next(iter(markers))])
        else:
            # Plot throughput on ax2 using the same color for each control mechanism
            ax2.plot(subset['Load'], subset['Throughput'],
                    label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2) # marker=markers[next(iter(markers))])  # Use any marker for throughput

    # Configure ax1 (latencies)
    ax1.set_xlabel('Load (RPS)')
    ax1.set_ylabel('95th Tail Latency (ms)')
    ax1.set_title('Load vs Tail Latency')
    ax1.grid(True)

    # Configure ax2 (throughput)
    if plotGoodput:
        # put legend outside the plot on the right
        if 'S_' not in method:
            ax1.legend(frameon=False)
            ax1.set_ylim(20, 190)
        # else:
            # add legend to the first row outside the plot, on the top
            # ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=4)
        ax2.set_ylabel('Goodput (RPS)')
        ax2.set_title('Load vs Goodput')
    else:
        ax2.legend(title='Subcalls and Throughput')
        ax2.set_ylabel('Throughput (RPS)')
        ax2.set_title('Load vs Throughput')
    ax2.set_xlabel('Load (RPS)')
    ax2.grid(True)

    # add legend outside the plot, on the top legend(frameon=False, loc='upper center', bbox_to_anchor=(1, 1.5), ncol=2)
    # for both ax1 and ax2, they share the same legend
    # ax1.legend(frameon=False, loc='upper center', ncol=2)
    ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime("%m%d")}.pdf'))
    plt.show()

if __name__ == "__main__":
    main()
