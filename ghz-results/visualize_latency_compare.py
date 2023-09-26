import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


latency_window_size = '100ms'  # Define the window size as 100 milliseconds
throughput_time_interval = '50ms'
offset = 0  # Define the offset as 50 milliseconds

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
    # df = df[df.index < df.index[-1] - pd.Timedelta(seconds=offset)]

    min_timestamp = df.index.min()
    df.index = df.index - min_timestamp + pd.Timestamp('2000-01-01')
    df = calculate_throughput(df)
    df = calculate_tail_latency(df)
    return df


def plot_combined(df1, df2, df3, labels):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot histograms on ax1
    ax1.hist(df1['latency'], alpha=0.5, label=labels[0], bins='auto')
    ax1.hist(df2['latency'], alpha=0.5, label=labels[1], bins='auto')
    ax1.hist(df3['latency'], alpha=0.5, label=labels[2], bins='auto')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Latencies')
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
    
    # Plot 99th and 50th percentile time-series on ax3
    for df, label in zip([df1, df2, df3], labels):
        # latency_99th = df['latency'].rolling(window='1s').apply(lambda x: np.percentile(x, 99), raw=True)
        # latency_50th = df['latency'].rolling(window='1s').apply(lambda x: np.percentile(x, 50), raw=True)
        
        ax3.plot(df.index, df['latency'], label=f"{label} - 99th percentile")
        # ax3.plot(df.index, latency_50th, label=f"{label} - 50th percentile")
        
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('99th and 50th Percentile Latencies Over Time')
    ax3.legend(loc='upper right')
    
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()


def main():
    files = ['/home/ying/Sync/Git/protobuf/ghz-results/lazy-social-compose-charon-parallel-capacity-6000.json',
             '/home/ying/Sync/Git/protobuf/ghz-results/nolazy-social-compose-charon-parallel-capacity-6000.json',
             '/home/ying/Sync/Git/protobuf/ghz-results/social-compose-plain-parallel-capacity-6000.json']
    colors = ['red', 'blue', 'green']
    labels = ['Charon Lazy', 'Charon Full', 'Plain']
    dfs = []

    for i, filename in enumerate(files):
        data = read_data(filename)
        df = convert_to_dataframe(data)
        df = df.fillna(0)
        dfs.append(df)

    df1, df2, df3 = dfs    
    plot_combined(df1, df2, df3, labels)

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
    # plt.savefig('Overhead_Study_when_no_control.png')
    # plt.show()

def calculate_tail_latency(df):
    # Only cound the latency of successful requests, i.e., status == 'OK'
    df.sort_index(inplace=True)
    # Assuming your DataFrame is named 'df' and the column to calculate the moving average is 'data'
    tail_latency = df['latency'].rolling(latency_window_size).quantile(0.99)
    df['tail_latency'] = tail_latency
    # Calculate moving average of latency
    df['latency_ma'] = df['latency'].rolling(latency_window_size).mean()
    return df

def calculate_throughput(df):
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    ok_requests_per_second *= (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='bfill')
    return df

if __name__ == '__main__':
    main()

