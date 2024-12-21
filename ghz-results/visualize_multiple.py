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

# window_size = 3  # You can adjust this to change the amount of smoothing
window_size = 1  # Set window size to 1 for no smoothing

offset = 2.5  # Define the offset as 2.5 seconds to remove the initial warm-up period

# if 'rl' in method:
#     offset = 2

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

        # Check if the file's timestamp is within the given duration
        # for compose method, the first experiment is from 1207_0018 to 1207_2210
        first_compose_experiment = ("1207_0018", "1207_0404")
        second_compose_experiment = ("1208_0101", "1208_0142")
        third_compose_experiment = ("1208_1112", "1208_1230")
        fourth_compose_experiment = ("1208_1941", "1208_2001")
        fiveth_compose_experiment = ("1209_2112", "1209_2231")

        tail_goodput_experiment = ("1213_0256", "1213_0415")
        random_experiment = ("1213_2021", "1213_2144")
        # random_experiment_avegoodput5000 = ("1214_1822", "1214_2010")
        random_experiment_avegoodput7000 = ("1215_0016", "1215_0150")
        death_gpt0_7000 = ("1215_1433", "1215_1600")
        # death_gpt1_8000 = ("1216_0021", "1216_0450")
        # death_gpt1_8000 = ("1216_1938", "1216_2130")
        # death_gpt1_8000 = ("1216_2304", "1217_0210")
        # death_gpt1_8000 = ("1217_0330", "1217_0900")
        # death_gpt1_8000_rajomon = ("1217_2121", "1217_2138")
        # # death_gpt1_8000_bw = ("1217_2307", "1217_2324")
        # death_gpt1_8000_bwd = ("1218_0056", "1218_0115")
        # death_gpt1_9000_rajomon = ("1218_1904", "1218_1921")
        death_gpt1_10000_rajomon = ("1219_2100", "1219_2117")
        death_tgpt_8000_rajomon = ("1220_0008", "1220_0030")
        death_tgpt_8000 = ("1220_0314", "1220_0855")
        death_gpt1_8000 = ("1221_0016", "1221_0609")
        compose_gpt1_8000 = [("1221_2126", "1221_2142"), ("1221_2018", "1221_2038"), ("1221_1904", "1221_1921"), ("1221_1801", "1221_1817")]
        compose_gpt1_8000_2 = ("1223_1656", "1223_1813")
        compose_gpt1_8000_3 = ("1225_0123", "1225_0241")
        compose_gpt1_8000_4 = ("1226_0437", "1226_0555")

        # compose_bw_8000 = [("1226_0455", "1226_0515"), ("1225_0141", "1225_0201")]
        compose_bw_8000 = [("1226_0455", "1226_0515"), ("1225_0141", "1225_0202"), ("1225_2053", "1225_2113")]
        compose_bwd_8000 = [("1226_0516", "1226_0537"),("1220_0649", "1220_0708")]
        compose_rajomon_8000 = [("1226_0436", "1226_0453"), ("1221_1800", "1221_1817"), ("1225_0123", "1225_0140")]
        compose_dg_8000 = [("1225_0225", "1225_0241"), ("1226_0538", "1226_0555")]

        # Now you can call this function with a list of ranges
        time_ranges_compose = [
            # first_compose_experiment,
            # second_compose_experiment,
            # third_compose_experiment,
            # fourth_compose_experiment,
            # fiveth_compose_experiment,
            # *compose_gpt1_8000,
            # compose_gpt1_8000_4,
            # compose_gpt1_8000_2,
            # compose_gpt1_8000_3,
            # *compose_bw_8000,
            # *compose_bwd_8000,
            # *compose_rajomon_8000,
            # *compose_dg_8000,
            # below are new compose experiments based on new profile
            # ("0114_1323", "0114_1455"),
            # ("0115_0251", "0115_0421"),
            # ("0115_1315", "0115_1445"),
            # this one below is under tight slo
            # ("0115_2224", "0115_2355")
            # these are non-tight slo
            ("0116_1653", "0116_2052")
        ]

        new_compose_looseSLO = [
            ("0116_1653", "0116_1714"), # these two are rajomon tuned for loose SLO, but with 50 ms buffer.
            ("0116_1855", "0116_1916"), # these two are rajomon tuned for loose SLO, but with 50 ms buffer.
            ("0117_0437", "0117_0637"),
            ("0117_1418", "0117_1618"),
            ("0117_1939", "0117_2139"),
            ("0118_0225", "0118_0425"),
            ("0127_1819", "0127_2129"),
            ("0128_1710", "0128_1800"),
            ("0129_2133", "0129_2240"),
        ]

        new_compose_tightSLO = [
            ("0117_1707", "0117_1839"),
            ("0117_2230", "0118_0002"),
            # below is the new compose with rtt param for breakwater
            # ("0128_2145", "0128_2300"),
            ("0128_0223", "0128_0257"),
            ("0129_1814", "0129_1928"),
            ("0130_1335", "0130_1355"),
        ]

        new_compose = new_compose_tightSLO if tightSLO else new_compose_looseSLO

        first_alibaba_experiment = ("1207_0404", "1207_0600")
        second_alibaba_experiment = ("1207_1458", "1207_1519")
        third_alibaba_experiment = ("1207_1953", "1207_2039")
        fourth_alibaba_experiment = ("1210_1956", "1210_2131")
        fiveth_alibaba_experiment = ("1215_0016", "1215_0150")
        alib_gpt2_7000 = ("1215_0838", "1215_1350")
        alib_gpt0_7000 = ("1215_1538", "1215_1600")
        alib_gpt1_7000 = ("1215_1818", "1215_1900")
        alib_gpt1_8000 = ("1217_0409", "1217_1140")
        S_14_gpt1_8000 = ("1224_0010", "1224_0153")
        S_14_gpt1_8000_2 = ("1226_2018", "1226_2200")
        S_14_gpt1_8000_3 = ("1221_2028", "1221_2052")

        time_ranges_S14 = [
            # first_alibaba_experiment,
            # second_alibaba_experiment,
            # third_alibaba_experiment,
            # fourth_alibaba_experiment
            # fiveth_alibaba_experiment
            # alib_gpt2_7000
            # # alib_gpt1_8000
            # ("1228_0259", "1228_0445"),
            ("1228_1702", "1228_1844"),
            ("1228_2356", "1229_0203"),
            ("1229_0141", "1229_0203"), # this is plain no overload control.
            # S_14_gpt1_8000,
            # S_14_gpt1_8000_2,
            # S_14_gpt1_8000_3,
            ("1230_2124", "1230_2333"),
            ("1231_2244", "0101_0027"),  # newest result
            # below is the exper with rtt param for breakwater
            ("0128_0842", "0128_0902"),
            ("0128_1543", "0128_1640"),
        ]

        relaxed_S14 = [
            # ("0120_1440", "0120_1730"),
            # ("0121_0115", "0121_0230"),
            # ("0121_1058", "0121_1347"),
            ("0121_2049", "0121_2307"),
        ]

        relaxed_S16 = [
            ("")
        ]

        # before 0121_0230, all S_ experiments are with 4x SLO. After that, all S_ experiments are with 250 bounded SLO.

        time_ranges_S10 = [
            # ("1227_0434", "1227_0600"),
            # ("1228_0110", "1228_0237"),
            # ("1228_0511", "1228_0643"),
            # ("1228_2129", "1228_2256"),
            ("1229_0301", "1229_0448"), # this includes plain no overload control. 4000-10000
            ("1231_2055", "1231_2241"), # this is for 6000-12000 load 
            ("1231_1747", "1231_1950"), # this is also for 6000-12000 load
            ("0129_0049", "0129_0138")
        ]

        time_ranges_S16 = [
            ("1230_0611", "1230_0754"),
            ("1231_0042", "1231_0225"),  # this is new
            ("0101_0127", "0101_0251"),
            # below is the exper with rtt param for breakwater
            ("0129_1654", "0129_1742"),
        ]


        if method == 'compose':
            # print(f"Timestamp: {timestamp_str}")
            if is_within_any_duration(timestamp_str, new_compose):
            # if is_within_any_duration(timestamp_str, new_compose_tightSLO):
                selected_files.append(filename)
        elif method == 'S_149998854':
            # if is_within_duration(timestamp_str, *first_alibaba_experiment) \
            # or is_within_duration(timestamp_str, *second_alibaba_experiment) \
            # or is_within_duration(timestamp_str, *third_alibaba_experiment):
            if is_within_any_duration(timestamp_str, time_ranges_S14):
                selected_files.append(filename)
        elif method == 'S_102000854':
            if is_within_any_duration(timestamp_str, time_ranges_S10):
                selected_files.append(filename)
        elif method == 'S_161142529':
            if is_within_any_duration(timestamp_str, time_ranges_S16):
                selected_files.append(filename)

    # for filename in find_latest_files(selected_files):
    for filename in selected_files:
        # Extract the metadata from the filename
        overload_control, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[3:8]
        capacity = int(capacity_str)
        # let's skip if the capacity is not factor of 1000
        # if capacity % 1000 != 0 and 'compose' in method:
        #     continue

        if capacity == 8000 and '120' in timestamp:
            continue
        if capacity == 8000 and '0101_0' in timestamp and 'S_16' in method and 'breakwater-' in filename:
            continue

        # if timestamp is earlier than 0127_1819 and control is breakwater, skip the file
        # due to the rtt param added 
        if 'breakwater' in overload_control and timestamp < '0127_1819':
            continue

        if capacity > 12000 and 'compose' in method:
            continue
        if capacity <= 6000 and 'compose' in method:
            continue

        if capacity == 10500 and 'S_10' in method:
            continue

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

        # row = {
        #     'Load': capacity,
        #     'Throughput': medianL(data['Throughput']),
        #     'Goodput': medianL(data['Goodput']),
        #     '99th_percentile': medianL(data['99th_percentile']),
        #     '95th_percentile': medianL(data['95th_percentile']),
        #     'Median Latency': medianL(data['Median Latency']),
        #     'method_subcall': method_subcall,
        #     'overload_control': overload_control,
        # }

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
    # social-compose-control-breakwaterd-parallel-capacity-8000-1207_0018.json
    for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{method}-control-*-parallel-capacity-*.json')):
        # Extract the date and time part from the filename
        timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
        # check if the file's timestamp is given format
        if len(timestamp_str) != 9:
            print("File ", filename, " is not valid")
            continue

        # Check if the file's timestamp is within the given duration
        if 'aqm' in method:
            aqm_experiment = [
                ("1222_2154", "1222_2212"),
            #     # ("1223_1858", "1223_1931")
                ("1223_1858", "1223_1933"),
                ("1223_2328", "1223_2350"),
                ("1225_2310", "1225_2315"),
                # ("1224_1343", "1224_1400")
                # ("1224_1701", "1224_1814"),
            ]
            if is_within_any_duration(timestamp_str, aqm_experiment):
                selected_files.append(filename)
        elif 'rl' in method:
            # rl_experiment = [
            #     ("1227_2138", "1227_2148"), # is when breakwaterd is on both frontend and backend.
            #     ("1227_2046", "1227_2056"), # is when breakwater is only on the frontend.
            #     ("1227_2314", "1227_2326"),
            #     ("1228_0025", "1228_0037"),
            #     ("1228_0043", "1228_0059"),
            #     ("1231_1732", "1231_1737"),
            # ]
            rl_experiment = [
                # ("0108_1328", "0108_1427"),
                # below is with new computation time
                ("0119_1832", "0119_1843"),
            ]
                
            if is_within_any_duration(timestamp_str, rl_experiment):
                selected_files.append(filename)
        else:
            new_experiment = [
                ("0227_0451", "0228_0324"),
                ("0302_1912", "0302_1953"), # this is the new experiment for breakwaterd
                ("0302_2333", "0303_0015"), # this is the new experiment for breakwaterd
            ]
            if is_within_any_duration(timestamp_str, new_experiment):
                selected_files.append(filename)

    '''
    if 'aqm' in method:
        scenarios = ['nginx-web-server', 'service-6', 'all']
        capacity_pattern = r"capacity-(\d+)-\d{4}_\d{4}"
        # Create a mapping of each capacity to its files
        capacity_file_mapping = create_capacity_file_mapping(selected_files, capacity_pattern)
        file_categories = categorize_files(capacity_file_mapping, scenarios)

        for filename in selected_files:
            if 'motivation-aqm-control-plain' in filename:
                file_categories[filename] = 'plain'
    
    # sort the plain files by timestamp and assign the file_categories from 1 to 5
    if 'rl' in method:
        file_categories = extract_info_rl(selected_files)

    scenarioDict = file_categories
    print(f"[Scenario count] {scenarioDict}")
    '''

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
            # row = {
            #     'Load': capacity,
            #     'Throughput': sum(data['Throughput']) / len(data['Throughput']),
            #     'Goodput': sum(data['Goodput']) / len(data['Goodput']),
            #     '99th_percentile': sum(data['99th_percentile']) / len(data['99th_percentile']),
            #     '95th_percentile': sum(data['95th_percentile']) / len(data['95th_percentile']),
            #     'Median Latency': sum(data['Median Latency']) / len(data['Median Latency']),
            #     # 'method_subcall': method_subcall,
            #     'overload_control': scenario,
            # }
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

            # Check if the file's timestamp is within the given duration
            all_method_experiment = [
                # ("0106_1545", "0106_1626")
                ("0107_1646", "0107_1841"),
                ("0108_0021", "0108_0316"),
                # below are new compose experiments based on new profile
                # ("0114_2120", "0114_2309"),
                # ("0115_2224", "0115_2359")
            ] if 'social' in method else [
                ("0106_0709", "0106_0836"),
                ("0107_1328", "0107_1456"),
                ("0107_1830", "0107_2003")
            ]

            relaxed_hotel = [
                # # ("0122_1203", "0122_1336"),
                # # # ("0122_1714", "0122_1847"),
                # ("0124_1444", "0124_1627"),
                # above is with 250x #interfaces clients, below is with 1000x #interfaces clients
                # ("0126_0518", "0126_0651"),
                # ("0126_1907", "0126_2308"),
                # below is with 500x #interfaces clients
                ("0127_1247", "0127_1435"),
                # below is with new breakwater profile, rtt as param
                ("0129_1548", "0129_1709"),
                # below is try to fix dagor
                ("0130_1337", "0130_1359"),
                ("0201_0031", "0201_0052"),
                ("0201_1112", "0201_1134"),
                ("0201_2123", "0201_2244"),
            ]

            tight_hotel_social = [
                ("0130_0339", "0130_0459"),
                ("0201_0205", "0201_0444"),
            ]

            relaxed_social = [
                # ("0122_0024", "0122_0216")
                # below is with 1000x #interfaces clients
                ("0125_1630", "0125_2028"),
                # below is with rtt param for breakwater
                ("0128_2137", "0128_2300"),
                # ("0131_0545", "0131_0719"),
                ("0131_1015", "0131_1151"),
                ("0131_1322", "0131_1457"),
                ("0131_1616", "0131_1750"),
                ("0131_1851", "0131_2000"),
            ]

            # before 0131_0100, I ran some tight slo experiments, after that, I ran relaxed slo experiments.

            if 'social' in method:
                relaxed_json = relaxed_social
            else:
                relaxed_json = relaxed_hotel

            if is_within_any_duration(timestamp_str, relaxed_json):
                selected_files.append(filename)

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

        # if 'dagor' in overload_control and (capacity == 6000):
        #     continue

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

            # if capacity >= 8000 or capacity <= 2000:
            #     if 'social' in method:
            #         continue

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

    if multipleFiles:
        # Plot the ax1 and ax2, with for all control mechanisms the latency and throughput. but now we have multiple interfaces,
        # each row of subplots are for one interface
        rotate = True
        if rotate:
            fig, axs = plt.subplots(2, len(interfaces), figsize=(6, 3.5), sharex=True)
        else:
            fig, axs = plt.subplots(len(interfaces), 2, figsize=(6, 7/4*len(interfaces)), sharex=True, sharey='row')

        for i, interface in enumerate(interfaces):
            ax1, ax2 = axs[i] if not rotate else axs[:, i]
            for control in control_mechanisms:
                # apply moving average on the Goodput of subset
                mask = (df['overload_control'] == control) & (df['Request'] == interface)
                # Apply rolling window for 'Goodput' and assign to 'Smoothed_Goodput' in place
                df.loc[mask, 'Smoothed_Goodput'] = df.loc[mask, 'Goodput'].rolling(window=window_size, min_periods=1).mean()
                # Apply rolling window for '95th_percentile' and assign in place
                df.loc[mask, '95th_percentile'] = df.loc[mask, '95th_percentile'].rolling(window=window_size, min_periods=1).mean()

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
                ax2.plot(subset['Load'], subset['Smoothed_Goodput'],
                        label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2)
                
                if interface == 'compose':
                    iname = 'POST'
                else:
                    # interface upper the letters while replace the - with space
                    iname = interface.upper().replace('-', ' ')
                ax1.set_title(f'{iname}'.replace('HTTP', ''))
                # ax2.set_title(f'{interface} Request')
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

                # set xlim to 2500 to 6500
                # ax1.set_xlim(2500, 6500)
                # ax2.set_xlim(2500, 6500)
                # add more x ticks to the ax1 and ax2, from 2500 to 7500, with step 1000
                # ax1.set_xticks([2500, 3500, 4500, 5500, 6500, 7500])
                # ax2.set_xticks([2500, 3500, 4500, 5500, 6500, 7500])
            if not rotate:
                ax1.grid(True)
                ax2.grid(True)

            # ax1.set_xlabel('Load (RPS)')
        # ax1.set_ylim(0, 500)

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
            # remove the spaces between the columns and rows
            # plt.subplots_adjust(wspace=0, hspace=0)
            # add grid to the all subplots and align the grid lines across the subplots
            # set the ylimit for each row of subplots to be same
            # first row
            for ax in axs.flatten()[0:len(interfaces)]:
                ax.grid(True)
                ax.set_ylim(0, 120)
            # second row
            for ax in axs.flatten()[len(interfaces):]:
                ax.grid(True)
                ax.set_ylim(0, 4000)

            # replace the y tick of second row from 4000 to 4k
            # for ax in axs.flatten()[4]:
            axs.flatten()[len(interfaces)].set_yticklabels(['0', '1k', '2k', '3k', '4k'])
            # remove the spaces between the columns and rows
            # plt.subplots_adjust(wspace=0, hspace=0)

            # add legend to the first row outside the plot, on the top
            axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=4)


        # Save and display the plot
        # plt.tight_layout()
        plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime("%m%d")}.pdf'))
        plt.show()
        return

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
                # Apply rolling window for 'Goodput' and assign to 'Smoothed_Goodput' in place
                df.loc[mask, 'Smoothed_Goodput'] = df.loc[mask, 'Goodput'].rolling(window=window_size, min_periods=1).mean()
                # Apply rolling window for '95th_percentile' and assign in place
                df.loc[mask, '95th_percentile'] = df.loc[mask, '95th_percentile'].rolling(window=window_size, min_periods=1).mean()

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
                ax2.plot(subset['Load'], subset['Smoothed_Goodput'],
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

    if 'rl' in method:
        # Plot the ax1 and ax2, with x axis as the number of tiers and y axis as the latency and throughput, plot separately the breakwater and breakwaterd and plain
        plotGoodput = True
        for latency in whatLatency:
            # use marker for the 99th percentile and 95th percentile
            for scheme in ['plain', 'breakwater', 'breakwaterd']:
                # subset = df[df['scheme'] == scheme] and when Load is 15000
                subset = df[(df['scheme'] == scheme) & (df['Load'] == 25000)]
                # sort the subset by the service depth
                subset.sort_values('Service Depth', inplace=True)
                ax1.plot(subset['Service Depth'], subset[latency],
                        color=colors[scheme], linestyle=lineStyles[scheme],
                        linewidth=2,
                        marker=markers[latency] if latency == '99th_percentile' else None,
                        label=labelDict[scheme],
                        )
                
                if plotGoodput:
                    ax2.plot(subset['Service Depth'], subset['Goodput'],
                            label=labelDict[scheme], color=colors[scheme], linestyle=lineStyles[scheme], linewidth=2)
                            # marker=markers[next(iter(markers))])
                        
        # Configure ax1 (latencies)
        ax1.set_xlabel('Depth of Service Graph')
        ax1.set_ylabel('95th Tail Latency (ms)')
        ax1.set_title('Depth vs Tail Latency')
        # ax1 use log scale
        ax1.set_yscale('log')
        ax1.grid(True)
        # put legend outside the plot on the right
        ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Configure ax2 (throughput)
        if plotGoodput:
            ax2.set_ylabel('Goodput (RPS)')
            ax2.set_title('Depth vs Goodput')
        else:
            ax2.legend(title='Subcalls and Throughput')
            ax2.set_ylabel('Throughput (RPS)')
            ax2.set_title('Depth vs Throughput')
        ax2.set_xlabel('Depth of Service Graph')
        ax2.grid(True)

        # Save and display the plot
        plt.tight_layout()
        plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime("%m%d")}.pdf'))
        plt.show()
        return

    for control in control_mechanisms:
        # Apply rolling window for 'Goodput' and assign to 'Smoothed_Goodput' in place
        mask = (df['overload_control'] == control)
        df.loc[mask, 'Smoothed_Goodput'] = df.loc[mask, 'Goodput'].rolling(window=window_size, min_periods=1).mean()
        df.loc[mask, 'Smoothed_Throughput'] = df.loc[mask, 'Throughput'].rolling(window=window_size, min_periods=1).mean()
        # Apply rolling window for '95th_percentile' and assign in place
        df.loc[mask, '95th_percentile'] = df.loc[mask, '95th_percentile'].rolling(window=window_size, min_periods=1).mean()
        subset = df[df['overload_control'] == control]

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
            ax2.plot(subset['Load'], subset['Smoothed_Goodput'],
                    label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2)
                    # marker=markers[next(iter(markers))])
        else:
            # Plot throughput on ax2 using the same color for each control mechanism
            ax2.plot(subset['Load'], subset['Smoothed_Throughput'],
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
