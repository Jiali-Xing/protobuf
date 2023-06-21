import json
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


throughput_time_interval = '100ms'
latency_window_size = '200ms'  # Define the window size as 100 milliseconds

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
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='ffill')
    return df


def calculate_goodput(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='ffill')
    return df


def calculate_loadshedded(df):
    dropped_requests_per_second = df[df['status'] == 'ResourceExhausted']['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    dropped_requests_per_second = dropped_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    df['dropped'] = dropped_requests_per_second.reindex(df.index, method='ffill')
    print(df['dropped'].describe())
    print(df['dropped'])
    return df


def calculate_tail_latency(df):
    # Only cound the latency of successful requests, i.e., status == 'OK'
    df.sort_index(inplace=True)
    # Assuming your DataFrame is named 'df' and the column to calculate the moving average is 'data'
    tail_latency = df['latency'].rolling(latency_window_size).quantile(0.99)
    df['tail_latency'] = tail_latency
    # Calculate moving average of latency
    df['latency_ma'] = df['latency'].rolling(latency_window_size).mean()
    return df


def plot_timeseries(df, filename):
    fig, ax1 = plt.subplots(figsize=(12, 4))
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
        "Incremental Waiting Time 90-tile": r"\[Incremental Waiting Time 90-tile\]:\s+(\d+\.\d+) ms.",
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
    df["timestamp"] = pd.to_datetime(timestamps)
    # adjust the index of df_queuing_delay to match the index of df, time wise
    # calculate the second last mininum timestamp of df 
    # df['timestamp'] = df['timestamp'] - min_timestamp + pd.Timestamp('2000-01-01')
    # print(df['timestamp'].min())
    df.set_index('timestamp', inplace=True)
    
    # sort the df by timestamp
    df.sort_index(inplace=True)

    # remove the data within first second of df
    df = df[df.index > df.index[0] + pd.Timedelta(seconds=1)]

    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df


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
    ax2.plot(df.index, df['dropped'].fillna(0)+df['throughput'], color='tab:blue', linestyle='-', label='Total Req')
    ax2.fill_between(df.index, 0, df['goodput'], color='green', alpha=0.1, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color='red', alpha=0.1, label='SLO Violated Req')
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

    # if computation_time > 0, label is "Average Latency (e2e) minus computation time", otherwise, label is "Average Latency (e2e)"
    ax1.plot(df.index, np.maximum(0.001, df['latency_ma']-computation_time), linestyle='--', 
             label='Average Latency (e2e)' if computation_time == 0 else 'Average Latency (e2e) minus computation time')
    ax1.plot(df.index, np.maximum(0.001, df['tail_latency']-computation_time), linestyle='-.', 
             label='99% Tail Latency (e2e)' if computation_time == 0 else '99% Tail Latency (e2e) minus computation time')
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

    df_queuing_delay = extract_waiting_times("/home/ying/Sync/Git/service-app/services/protobuf-grpc/server.output")
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
    


def analyze_data(filename):
    data = read_data(filename)
    df = convert_to_dataframe(data)
    # print(df.head())
    plot_latency_pdf_cdf(df, filename)
    df = calculate_throughput(df)
    df = calculate_goodput(df, 20)
    df = calculate_loadshedded(df)
    df = calculate_tail_latency(df)
    plot_timeseries_ok(df, filename)
    plot_timeseries_lat(df, filename, 10)
    summary = df.groupby('status')['status'].count().reset_index(name='count')
    print(summary)
    # summary = df.groupby('error')['error'].count().reset_index(name='count')
    # print(summary)


if __name__ == '__main__':
    filename = sys.argv[1]
    analyze_data(filename)
