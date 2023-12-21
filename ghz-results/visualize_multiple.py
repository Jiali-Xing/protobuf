import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime


throughput_time_interval = '50ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds

method = 'compose'
# method = 'S_149998854'


if method == 'compose':
    SLO = 20 * 2 # 20ms * 2 = 40ms
    # capacity = 8000
elif method == 'S_149998854':
    SLO = 111 * 2 # 111ms * 2 = 222ms
    # capacity = 6000

offset = 1  # Define the offset as 50 milliseconds


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
            print("File ", filename, " is not valid")
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
    latency_percentile = df['latency'].quantile(percentile / 100)
    
    return latency_percentile


# calculate_throughput calculates the throughput of the requests
def calculate_throughput(filename):
    data = read_data(filename)
    df = convert_to_dataframe(data)
    if df.empty:
        print("DataFrame is empty for ", filename)
        return None
    # Compute the throughput
    throughput = df['latency'].count() / (df.index.max() - df.index.min()).total_seconds()
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
    # take out the goodput during the last 3 seconds by index
    goodput = df['goodput']
    # goodput = df[df.index > df.index[0] + pd.Timedelta(seconds=offset)]['goodput']
    # return the goodput, but round it to 2 decimal places
    goodput = goodput.mean()
    # goodput = round(goodput, -2)
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
    # For every file in the directory
    # social-compose-control-breakwaterd-parallel-capacity-8000-1207_0018.json
    for filename in glob.glob(os.path.join(os.path.expanduser('~/Sync/Git/protobuf/ghz-results'), f'social-{method}-control-*-parallel-capacity-*12*.json')):
        # Extract the date and time part from the filename
        timestamp_str = os.path.basename(filename).split('-')[-1].rstrip('.json')
        # check if the file's timestamp is given format
        # claim 
        if len(timestamp_str) != 9:
            print("File ", filename, " is not valid")
            continue

        # Check if the file's timestamp is within the given duration
        # for compose method, the first experiment is from 1207_0018 to 1207_2210
        frist_compose_experiment = ("1207_0018", "1207_0404")
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
        # death_gpt1_8000_charon = ("1217_2121", "1217_2138")
        # # death_gpt1_8000_bw = ("1217_2307", "1217_2324")
        # death_gpt1_8000_bwd = ("1218_0056", "1218_0115")
        # death_gpt1_9000_charon = ("1218_1904", "1218_1921")
        death_gpt1_10000_charon = ("1219_2100", "1219_2117")
        death_tgpt_8000_charon = ("1220_0008", "1220_0030")
        death_tgpt_8000 = ("1220_0314", "1220_0855")
        death_gpt1_8000 = ("1221_0016", "1221_0609")

        # Now you can call this function with a list of ranges
        time_ranges_compose = [
            # frist_compose_experiment,
            # second_compose_experiment,
            # third_compose_experiment,
            # fourth_compose_experiment,
            # fiveth_compose_experiment,
            # tail_goodput_experiment
            # random_experiment
            # random_experiment_avegoodput7000
            death_gpt1_8000,
            # death_gpt1_8000_bw,
            # death_gpt1_8000_bwd,
        ]

        first_alibaba_experiment = ("1207_0404", "1207_0600")
        second_alibaba_experiment = ("1207_1458", "1207_1519")
        third_alibaba_experiment = ("1207_1953", "1207_2039")
        fourth_alibaba_experiment = ("1210_1956", "1210_2131")
        fiveth_alibaba_experiment = ("1215_0016", "1215_0150")
        alib_gpt2_7000 = ("1215_0838", "1215_1350")
        alib_gpt0_7000 = ("1215_1538", "1215_1600")
        alib_gpt1_7000 = ("1215_1818", "1215_1900")
        alib_gpt1_8000 = ("1217_0409", "1217_1140")

        time_ranges_alibaba = [
            # first_alibaba_experiment,
            # second_alibaba_experiment,
            # third_alibaba_experiment,
            # fourth_alibaba_experiment
            # fiveth_alibaba_experiment
            # alib_gpt2_7000
            alib_gpt1_8000
        ]

        if method == 'compose':
        #     if is_within_duration(timestamp_str, *frist_compose_experiment) \
        #     or is_within_duration(timestamp_str, *second_compose_experiment) \
        #     or is_within_duration(timestamp_str, *third_compose_experiment) \
        #     or is_within_duration(timestamp_str, *fourth_compose_experiment):
            if is_within_any_duration(timestamp_str, time_ranges_compose):
                selected_files.append(filename)
        elif method == 'S_149998854':
            # if is_within_duration(timestamp_str, *first_alibaba_experiment) \
            # or is_within_duration(timestamp_str, *second_alibaba_experiment) \
            # or is_within_duration(timestamp_str, *third_alibaba_experiment):
            if is_within_any_duration(timestamp_str, time_ranges_alibaba):
                selected_files.append(filename)

    for filename in find_latest_files(selected_files):
    # for filename in selected_files:
        # Extract the metadata from the filename
        overload_control, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[3:8]
        capacity = int(capacity_str)
        # let's skip if the capacity is not factor of 1000
        if capacity % 500 != 0:
            continue

        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid, no OK")
            os.remove(filename)
            continue

        # if the Count: xxx is less than 20000, remove the file
        with open(filename, 'r') as f:
            data = json.load(f)
            if data['count'] < 20000:
                print("File ", filename, " is not valid, count is less than 20000")
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
            if latency_99 > 900:
                print("File ", filename, " is not valid, 99th percentile is greater than 900ms")
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
        print("No valid data")
        return None
    rows = []
    for (overload_control, method_subcall, capacity), data in results.items():
        row = {
            'Load': capacity,
            'Throughput': sum(data['Throughput']) / len(data['Throughput']),
            'Goodput': sum(data['Goodput']) / len(data['Goodput']),
            '99th_percentile': sum(data['99th_percentile']) / len(data['99th_percentile']),
            '95th_percentile': sum(data['95th_percentile']) / len(data['95th_percentile']),
            'Median Latency': sum(data['Median Latency']) / len(data['Median Latency']),
            'method_subcall': method_subcall,
            'overload_control': overload_control,
        }
        rows.append(row)
        print(f'Control: {overload_control}, Method: {method_subcall}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')

    df = pd.DataFrame(rows)
    df.sort_values('Load', inplace=True)

    return df


def main():
    df = load_data()
    print(df)
    if df is None:
        return
    # Extract data for each method and latency percentile
    # Define the methods you want to plot
    # methods = ['parallel']
    # methods = ['sequential', 'parallel']
    control_mechanisms = ['dagor', 'breakwater', 'charon', 'breakwaterd']

    # Define markers for each method
    # markers = ['o', 's', 'x']
    # whatLatency = ['Median Latency']
    # whatLatency = ['99th_percentile', 'Median Latency']
    whatLatency = ['95th_percentile', '99th_percentile',]

    # Map control mechanisms to colors, user material design colors
    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        # 'breakwaterd': a darker version of breakwater
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'charon': '#FF9800',
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
        'charon': '-',
    }

    # Create a plot for each method and latency percentile
    # plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Create 1x2 subplots for latencies and throughput
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax1, ax2 = axs  # ax1 for latencies, ax2 for throughput

    for control in control_mechanisms:
        subset = df[df['overload_control'] == control]
        
        # rename the control mechanism breakwaterd to breakwaterd
        if control == 'breakwaterd':
            control = 'breakwaterd'
        # Plot each latency metric with a different marker, but the same color for the control mechanism
        for latency in whatLatency:
            # subset_filtered = subset[subset[latency] < 550]  # Filter out any extreme latency values if necessary
            subset_filtered = subset
            ax1.plot(subset_filtered['Load'], subset_filtered[latency],
                     color=colors[control], linestyle=lineStyles[control],
                    #  marker=markers[latency], # legend=f'{control} and the first 2 charactors of latency'
                     label=control if latency == '99th_percentile' else None,
                    #  thinkness of the line
                     linewidth=2,
                    #  label=f'{control} {latency}'.split('_')[0]
                     )  
        plotGoodput = True

        if plotGoodput:
            ax2.plot(subset['Load'], subset['Goodput'],
                    label=f'{control}', color=colors[control], linestyle=lineStyles[control], linewidth=2)
                    # marker=markers[next(iter(markers))])
        else:
            # Plot throughput on ax2 using the same color for each control mechanism
            ax2.plot(subset['Load'], subset['Throughput'],
                    label=f'{control}', color=colors[control], linestyle=lineStyles[control], linewidth=2) # marker=markers[next(iter(markers))])  # Use any marker for throughput

    # Configure ax1 (latencies)
    ax1.legend(title='Tail Latency')
    ax1.set_xlabel('Load (rps)')
    ax1.set_ylabel('Tail Latency (ms)')
    ax1.set_title('Load vs Tail Latency')
    ax1.grid(True)

    # Configure ax2 (throughput)
    if plotGoodput:
        ax2.legend(title='Goodput')
        ax2.set_ylabel('Goodput (RPS)')
        ax2.set_title('Load vs Goodput')
    else:
        ax2.legend(title='Subcalls and Throughput')
        ax2.set_ylabel('Throughput (RPS)')
        ax2.set_title('Load vs Throughput')
    ax2.set_xlabel('Load (RPS)')
    ax2.grid(True)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime("%m%d")}.png'))
    plt.show()

if __name__ == "__main__":
    main()
