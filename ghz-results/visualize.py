import json
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["details"]


def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    min_timestamp = df['timestamp'].min()
    df['timestamp'] = df['timestamp'] - min_timestamp + pd.Timestamp('2000-01-01')
    df.set_index('timestamp', inplace=True)
    df['latency'] = df['latency'] / 1000000
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
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample('1S').count()
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='ffill')
    return df


def calculate_goodput(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample('1S').count()
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='ffill')
    return df


def calculate_loadshedded(df):
    dropped_requests_per_second = df[df['status'] == 'ResourceExhausted']['status'].resample('1S').count()
    df['dropped'] = dropped_requests_per_second.reindex(df.index, method='ffill')
    return df


def calculate_tail_latency(df):
    tail_latency = df['latency'].rolling(window=500).quantile(0.99)
    df['tail_latency'] = tail_latency
    # Calculate moving average of latency
    df['latency_ma'] = df['latency'].rolling(window=500).mean()
    return df


def plot_timeseries(df, filename):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, df['latency_ma'], color='orange', linestyle='--', label='Latency')
    ax1.plot(df.index, df['tail_latency'], color='green', linestyle='-.', label='99% Tail Latency')
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


def plot_timeseries_ok(df, filename):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Latencies (ms)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.plot(df.index, df['latency_ma'], color='orange', linestyle='--', label='Latency')
    ax1.plot(df.index, df['tail_latency'], color='c', linestyle='-.', label='99% Tail Latency')
    ax1.set_ylim(2, 2000)
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Throughput (req/s)', color='tab:blue')

    ax2.plot(df.index, df['throughput'], 'r-.', )
    ax2.plot(df.index, df['goodput'], color='green', linestyle='--')
    ax2.plot(df.index, df['dropped'].fillna(0)+df['throughput'], color='tab:blue', linestyle='-', label='Total Req')
    ax2.fill_between(df.index, 0, df['goodput'], color='green', alpha=0.2, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color='red', alpha=0.3, label='SLO Violated Req')
    ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color='c', alpha=0.3, label='Dropped Req')

    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 3000)
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


def analyze_data(filename):
    data = read_data(filename)
    df = convert_to_dataframe(data)
    plot_latency_pdf_cdf(df, filename)
    df = calculate_throughput(df)
    df = calculate_goodput(df, 20)
    df = calculate_loadshedded(df)
    df = calculate_tail_latency(df)
    plot_timeseries_ok(df, filename)
    summary = df.groupby('status')['status'].count().reset_index(name='count')
    print(summary)
    summary = df.groupby('error')['error'].count().reset_index(name='count')
    print(summary)



if __name__ == '__main__':
    filename = sys.argv[1]
    analyze_data(filename)
