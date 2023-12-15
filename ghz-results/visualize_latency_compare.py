import glob
import json, csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from datetime import datetime


latency_window_size = '100ms'  # Define the window size as 100 milliseconds
throughput_time_interval = '150ms'
offset = 1  # Define the offset as 50 milliseconds

def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["details"]


def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # summary = df.groupby('status')['status'].count().reset_index(name='count')
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
    df = df[df.index < df.index[-1] - pd.Timedelta(seconds=offset)]

    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    df = calculate_throughput(df)
    df = calculate_tail_latency(df)
    return df



def plot_combined(dfs, labels):
    assert len(dfs) == len(labels), "Length of DataFrames list and labels list must match."
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 18))
    ax1, ax2, ax3, ax4 = axs
    # Find the global min and max latency values across all dfs
    global_min = 0
    global_max = 300

    # Define the bin edges based on the global range and desired number of bins
    # The np.histogram_bin_edges function can help with this
    num_bins = 250  # or whatever desired number of bins you want
    bin_edges = np.linspace(global_min, global_max, num_bins)
    # Histograms on ax1
    for df, label in zip(dfs, labels):
        ax1.hist(df['latency'], alpha=0.5, label=label, bins=bin_edges)
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Latencies (ms)')
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, 200])
    ax1.set_yscale('log')

    # Throughput on ax2
    for df, label in zip(dfs, labels):
        ax2.plot(df.index, df['throughput'], label=label)
    ax2.set_ylabel('Throughput (req/s)')
    ax2.set_title('Throughput Over Time')
    ax2.legend(loc='upper right')

    # 99th percentile time-series on ax3
    for df, label in zip(dfs, labels):
        ax3.plot(df.index, df['tail_latency'], label=f"{label} - 99th percentile")
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('99th Percentile Latencies Over Time')
    ax3.legend(loc='upper right')

    # Average latency on ax4
    # for df, label in zip(dfs, labels):
    #     ax4.plot(df.index, df['latency_ma'], label=f"{label} - avg percentile")
    # ax4.set_ylabel('Latency (ms)')
    # ax4.set_title('Average Latency Over Time')
    # ax4.legend(loc='upper right')

    for df, label in zip(dfs, labels):
        ax4.plot(df.index, df['latency_median'], label=f"{label} - median")
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Median Latency Over Time')
    ax4.legend(loc='upper right')

    # Share x-axis for ax2, ax3, ax4
    ax2.get_shared_x_axes().join(ax2, ax3, ax4)

    # Add grid to all subplots
    for ax in axs:
        ax.grid(True)

    # for 3 and 4, use log scale
    ax3.set_yscale('log')
    ax4.set_yscale('log')

    # set y limit for 3 and 4 to be 10 to 100
    ax3.set_ylim([10, 100])
    ax4.set_ylim([10, 100])
    
    # Adjust spacing
    plt.tight_layout(h_pad=4)
    plt.savefig('overload_comparison.png')
    plt.show()

    # # Quantitative comparisons
    # for df, label in zip(dfs, labels):
    #     print(f"Average latency of {label}: {df['latency'].mean()}")
    #     print(f"99th percentile latency of {label}: {df['latency'].quantile(0.99)}")
    #     print(f"Throughput of {label}: {df['throughput'].mean()}")

    # Print table header
    print(f"{'Label':<30}{'Average Latency':<20}{'Median Latency':<20}{'99th %ile Latency':<20}{'Throughput':<20}")

    # Separator
    print("=" * 110)

    # Quantitative comparisons
    for df, label in zip(dfs, labels):
        avg_latency = df['latency'].mean()
        median_latency = df['latency'].median()
        percentile_latency = df['latency'].quantile(0.99)
        throughput = df['throughput'].mean()
        
        print(f"{label:<30}{avg_latency:<20.2f}{median_latency:<20.2f}{percentile_latency:<20.2f}{throughput:<20.2f}")
    
    # Create and write the CSV file
    with open('latency_throughput_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['Label', 'Average Latency', 'Median Latency', '99th %ile Latency', 'Throughput']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for df, label in zip(dfs, labels):
            avg_latency = df['latency'].mean()
            median_latency = df['latency'].median()
            percentile_latency = df['latency'].quantile(0.99)
            throughput = df['throughput'].mean()
            
            writer.writerow({
                'Label': label,
                'Average Latency': f"{avg_latency:.2f}",
                'Median Latency': f"{median_latency:.2f}",
                '99th %ile Latency': f"{percentile_latency:.2f}",
                'Throughput': f"{throughput:.2f}"
            })

'''
def plot_combined(df1, df2, df3, labels):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18))
    # Plot histograms on ax1
    ax1.hist(df1['latency'], alpha=0.5, label=labels[0], bins='auto')
    ax1.hist(df2['latency'], alpha=0.5, label=labels[1], bins='auto')
    ax1.hist(df3['latency'], alpha=0.5, label=labels[2], bins='auto')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Latencies (ms)')
    ax1.legend(loc='upper right')
    # ax1 set x range to 0, 500
    ax1.set_xlim([0, 500])

    # Plot throughput on ax2
    ax2.plot(df1.index, df1['throughput'], label=labels[0])
    ax2.plot(df2.index, df2['throughput'], label=labels[1])
    ax2.plot(df3.index, df3['throughput'], label=labels[2])
    ax2.set_ylabel('Throughput (req/s)')
    ax2.set_title('Throughput Over Time')
    ax2.legend(loc='upper right')
    
    # Plot 99th percentile time-series on ax3
    for df, label in zip([df1, df2, df3], labels):
        # latency_99th = df['latency'].rolling(window='1s').apply(lambda x: np.percentile(x, 99), raw=True)
        # latency_50th = df['latency'].rolling(window='1s').apply(lambda x: np.percentile(x, 50), raw=True)
        
        ax3.plot(df.index, df['tail_latency'], label=f"{label} - 99th percentile")
        # ax3.plot(df.index, df['latency_ma'], label=f"{label} - avg percentile")
        
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('99th Percentile Latencies Over Time')
    ax3.legend(loc='upper right')

    # ax2 and 3 don't show x labels
    # ax2.set_xlabel('')
    # ax3.set_xlabel('')

    # Share x-axis for ax2, ax3, ax4
    ax2.get_shared_x_axes().join(ax2, ax3, ax4)
    
    # plot average latency on ax4
    for df, label in zip([df1, df2, df3], labels):
        ax4.plot(df.index, df['latency_ma'], label=f"{label} - avg percentile")
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Average Latency Over Time')
    ax4.legend(loc='upper right')
 
    # add grid to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)
    
    # Adjust spacing
    plt.tight_layout(h_pad=4)

    # plt.xlabel('Time')
    # plt.tight_layout()
    plt.savefig('Overhead_Study_when_no_control.png')
    plt.show()

    # also, quantitatively compare the average, 99th latency and throughput of all files
    for df, label in zip([df1, df2, df3], labels):
        print(f"Average latency of {label}: {df['latency'].mean()}")
        print(f"99th percentile latency of {label}: {df['latency'].quantile(0.99)}")
        print(f"Throughput of {label}: {df['throughput'].mean()}")
'''


def find_latest_files(capacity, directory):
    latencyThreshold = '8000'
    if latencyThreshold == 'inf':
        start_time_str = "1108_1000"
        end_time_str = "1108_2210"
    elif latencyThreshold == '8000':
        start_time_str = "1108_2210"
        # end_time_str = "1109_0008"
        end_time_str = "1110_0000"
    elif latencyThreshold == '12000':
        start_time_str = "1109_0008"
        end_time_str = "1109_2300"

    # List of all charon files for the given capacity
    charon_files = glob.glob(f"{directory}social-compose-control-charon-parallel-capacity-{capacity}-*.json")
    plain_files = glob.glob(f"{directory}social-compose-control-plain-parallel-capacity-{capacity}-*.json")
    breakwater_files = glob.glob(f"{directory}social-compose-control-breakwater-parallel-capacity-{capacity}-*.json")
    
    # Now filter the files using the is_within_duration function
    charon_files = [f for f in charon_files if is_within_duration(f.split('-')[-1].rstrip('.json'), start_time_str, end_time_str)]
    plain_files = [f for f in plain_files if is_within_duration(f.split('-')[-1].rstrip('.json'), start_time_str, end_time_str)]
    breakwater_files = [f for f in breakwater_files if is_within_duration(f.split('-')[-1].rstrip('.json'), start_time_str, end_time_str)]

    # Sort the files by their modification time to get the latest file
    charon_files.sort(key=os.path.getmtime, reverse=True)
    plain_files.sort(key=os.path.getmtime, reverse=True)
    breakwater_files.sort(key=os.path.getmtime, reverse=True)

    # Now you can read the latest files (if they exist)
    if charon_files:
        charon_file = charon_files[0]  # this will be the latest file
    else:
        print("No charon files found for the given capacity!")

    if plain_files:
        plain_file = plain_files[0]
    else:
        print("No plain files found for the given capacity!")

    if breakwater_files:
        breakwater_file = breakwater_files[0]
    else:
        print("No breakwater files found for the given capacity!")

    files = [charon_file, plain_file, breakwater_file] if charon_files and plain_files and breakwater_files else []

    print(files)
    return files


# Function to check if a file's timestamp is within the given duration
def is_within_duration(file_timestamp_str, start_time_str, end_time_str):
    # Convert the time strings to datetime objects
    start_time = datetime.strptime(start_time_str, "%m%d_%H%M")
    end_time = datetime.strptime(end_time_str, "%m%d_%H%M")
    file_time = datetime.strptime(file_timestamp_str, "%m%d_%H%M")

    # Check if the file's time is within the start and end times
    return start_time <= file_time <= end_time


def main():
    capacity = 8500
    directory = '/home/ying/Sync/Git/protobuf/ghz-results/'

    # If two arguments are provided, use them as the file names
    if len(sys.argv) == 3:
        charon_file = sys.argv[1]
        plain_file = sys.argv[2]
        files = [charon_file, plain_file]
        print(files)
    else:
        # Otherwise, find the latest files for the given capacity
        files = find_latest_files(capacity, directory)


    # files = [f'/home/ying/Sync/Git/protobuf/ghz-results/social-compose-charon-parallel-capacity-{capacity}.json',
    files = [
        '/home/ying/Sync/Git/protobuf/ghz-results/social-compose-control-plain-parallel-capacity-8000-1209_1522.json',
        '/home/ying/Sync/Git/protobuf/ghz-results/social-compose-control-charon-parallel-capacity-8000-1209_1907.json',
        '/home/ying/Sync/Git/protobuf/ghz-results/social-compose-control-breakwater-parallel-capacity-8000-1209_1329.json',
        '/home/ying/Sync/Git/protobuf/ghz-results/social-compose-control-breakwaterd-parallel-capacity-8000-1209_1439.json',
        '/home/ying/Sync/Git/protobuf/ghz-results/social-compose-control-dagor-parallel-capacity-8000-1209_2106.json'
    ]
    # colors = [ a list of 4 material colors ]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    labels = ['Plain', 'Charon', 'Breakwater', 'Breakwaterd', 'Dagor']
    # labels = ['Charon Lazy', 'Charon Full', 'Charon w/o Log', 'Plain']
    # labels = ['Charon Lazy', 'Plain', 'Breakwater']
    dfs = []

    for i, filename in enumerate(files):
        data = read_data(filename)
        df = convert_to_dataframe(data)
        df = df.fillna(0)
        dfs.append(df)

    # df1, df2, df3, df4 = dfs    
    # plot_combined([df1, df2, df3, df4], labels)
    plot_combined(dfs, labels)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # for i, filename in enumerate(files):
    #     data = read_data(filename)
    #     df = convert_to_dataframe(data)
    #     df = df.fillna(0)

    #     if len(df) > 0:
    #         sns.histplot(df['latency'], kde=False, color=colors[i], alpha=0.5, label=labels[i], ax=ax1)
            
    #         # Calculate throughput and plot it
    #         df = calculate_throughput(df)
    #         ax2.plot(df.index, df['throughput'], label=f"{labels[i]} Throughput", color=colors[i], alpha=0.5)

    # ax1.legend(title='File Type')
    # ax1.set_xlabel('Latency (ms)')
    # ax1.set_ylabel('Frequency')
    # ax1.set_title('Latency Distribution')
    # ax1.grid(True)

    # ax2.legend(title='Throughput')
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Throughput (req/s)')
    # ax2.set_title('Throughput over Time')
    # ax2.grid(True)

    # plt.tight_layout()
    # plt.show()


def calculate_tail_latency(df):
    # Only cound the latency of successful requests, i.e., status == 'OK'
    df.sort_index(inplace=True)
    # Assuming your DataFrame is named 'df' and the column to calculate the moving average is 'data'
    tail_latency = df['latency'].rolling(latency_window_size).quantile(0.99)
    df['tail_latency'] = tail_latency
    # Calculate moving average of latency
    df['latency_ma'] = df['latency'].rolling(latency_window_size).mean()
    # and the median latency
    df['latency_median'] = df['latency'].rolling(latency_window_size).median()
    return df


def calculate_throughput(df):
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    ok_requests_per_second *= (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='bfill')
    return df

if __name__ == '__main__':
    main()

