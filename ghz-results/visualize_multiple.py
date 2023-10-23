import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


offset = 2  # Define the offset as 50 milliseconds

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
        data = json.load(f)
    return data["details"]


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


def load_data():
    # A dictionary to hold intermediate results, indexed by (overload_control, method_subcall, capacity)
    results = {}

    # For every file in the directory
    for filename in glob.glob('/home/ying/Sync/Git/protobuf/ghz-results/social-compose-*-*-capacity-*.json'):
        # Extract the metadata from the filename
        overload_control, method_subcall, _, capacity_str, timestamp = os.path.basename(filename).split('-')[2:7]
        capacity = int(capacity_str)

        # if there's no `OK` in the file, remove the file
        if 'OK' not in open(filename).read():
            print("File ", filename, " is not valid")
            os.remove(filename)
            continue

        # Calculate latencies and throughput
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)

        # If valid latency data
        if latency_99 is not None:
            key = (overload_control, method_subcall, capacity)
            if key not in results:
                results[key] = {
                    'Load': capacity,
                    'Throughput': [throughput],
                    '99th_percentile': [latency_99],
                    '95th_percentile': [latency_95],
                    'Median Latency': [latency_median]
                }
            else:
                results[key]['Throughput'].append(throughput)
                results[key]['99th_percentile'].append(latency_99)
                results[key]['95th_percentile'].append(latency_95)
                results[key]['Median Latency'].append(latency_median)

    # Averaging and preparing rows for dataframe
    rows = []
    for (overload_control, method_subcall, capacity), data in results.items():
        row = {
            'Load': capacity,
            'Throughput': sum(data['Throughput']) / len(data['Throughput']),
            '99th_percentile': sum(data['99th_percentile']) / len(data['99th_percentile']),
            '95th_percentile': sum(data['95th_percentile']) / len(data['95th_percentile']),
            'Median Latency': sum(data['Median Latency']) / len(data['Median Latency']),
            'method_subcall': method_subcall,
            'overload_control': overload_control
        }
        rows.append(row)
        print(f'Control: {overload_control}, Method: {method_subcall}, Load: {capacity}, 99th: {row["99th_percentile"]:.2f}, 95th: {row["95th_percentile"]:.2f}, Median: {row["Median Latency"]:.2f}')

    df = pd.DataFrame(rows)
    df.sort_values('Load', inplace=True)

    return df


def main():
    '''
    rows = []
    # for filename in glob.glob('/home/ying/Sync/Git/protobuf/ghz-results/social-compose-plain-*-capacity-*.json'):
    for filename in glob.glob('/home/ying/Sync/Git/protobuf/ghz-results/social-compose-*-*-capacity-*.json'):
        # for a file name like social-compose-plain-sequential-capacity-6000.json
        # method_subcall = sequential
        # capacity = 6000
        overload_control, method_subcall, _, capacity_str = os.path.basename(filename).split('-')[2:6]
        capacity = int(capacity_str.split('.')[0])
        
        latency_99 = calculate_tail_latency(filename)
        latency_95 = calculate_tail_latency(filename, 95)
        latency_median = calculate_tail_latency(filename, 50)
        throughput = calculate_throughput(filename)
        
        if latency_99 is not None:
            rows.append({
                'Load': capacity,
                'Throughput': throughput,
                '99th_percentile': latency_99,
                '95th_percentile': latency_95,
                'Median Latency': latency_median,
                'method_subcall': method_subcall,
                'overload_control': overload_control
            })
            # log the latency_95 and latency_99 for each capacity and method_subcall, print the row with 2 decimal places
            print(f'Control: {overload_control}, Method: {method_subcall}, Load: {capacity}, 99th: {latency_99:.2f}, 95th: {latency_95:.2f}, Median: {latency_median:.2f}')
    
    df = pd.DataFrame(rows)
    df.sort_values('Load', inplace=True)
    '''
    df = load_data()
    print(df)
    # Extract data for each method and latency percentile
    # Define the methods you want to plot
    methods = ['parallel']
    # methods = ['sequential', 'parallel']
    control_mechanisms = ['plain', 'charon']

    # Define markers for each method
    markers = ['o', 's']
    whatLatency = ['Median Latency']
    # whatLatency = ['99th_percentile', 'Median Latency']
    # whatLatency = ['95th_percentile']

    # Create a plot for each method and latency percentile
    # plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Create 1x2 subplots for latencies and throughput
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    ax1, ax2 = axs  # ax1 for latencies, ax2 for throughput

    for i, method in enumerate(methods):
        for j, control in enumerate(control_mechanisms):
            subset = df[(df['method_subcall'] == method) & (df['overload_control'] == control)]
            
            for latency in whatLatency:
                subset_filtered = subset[subset[latency] < 200]
                ax1.plot(subset_filtered['Load'], subset_filtered[latency],
                         label=f'{control} {method} {latency}', marker=markers[i])

            # Plot throughput on ax2
            ax2.plot(subset['Load'], subset['Throughput'],
                     label=f'{control} {method} Throughput', marker=markers[i])

    # Configure ax1 (latencies)
    ax1.legend(title='Subcalls and Percentile')
    ax1.set_xlabel('Load (rps)')
    ax1.set_ylabel('Tail Latency (ms)')
    ax1.set_title('Load vs Tail Latency')
    ax1.grid(True)

    # Configure ax2 (throughput)
    ax2.legend(title='Subcalls and Throughput')
    ax2.set_xlabel('Load (RPS)')
    ax2.set_ylabel('Throughput (RPS)')
    ax2.set_title('Load vs Throughput')
    ax2.grid(True)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('combined_capacity_vs_metrics.png')
    plt.show()


    # for i, method in enumerate(methods):
    #     for j, control in enumerate(control_mechanisms):
    #         subset = df[(df['method_subcall'] == method) & (df['overload_control'] == control)]
            
    #         # Extract capacity values (assuming they are the same for all data points)
    #         # capacity = subset['Load'].unique()

    #         for latency in whatLatency:
    #             subset = subset[subset[latency] < 200]  
    #             plt.plot(subset['Load'], subset[latency], 
    #                      label=f'{control} {method} {latency}', marker=markers[i])
    #         # remove the data within if the latency is < 150ms
    #         # subset = subset[subset['99th_percentile'] < 200]
    #         # subset = subset[subset['95th_percentile'] < 200]
                            

    #         # Plot the data for this method
    #         # plt.plot(subset['Load'], subset['99th_percentile'], 
    #         #          label=f'{control} {method} 99th Percentile', marker='o')
                     
    #         # plt.plot(subset['Load'], subset['95th_percentile'], 
    #         #          label=f'{control} {method} 95th Percentile', marker='s')


    # plt.legend(title='Subcalls and Percentile')
    # plt.xlabel('Load (rps)')
    # plt.ylabel('Tail Latency (ms)')
    # plt.title('Load vs Tail Latency')
    # plt.grid(True)
    # # filename includes the latency percentile
    # # filename = 'capacity_vs_tail_latency_10msUpdate_99th_95th.png' if len(whatLatency) == 2 else 'capacity_vs_tail_latency_10msUpdate_99th.png'
    # if len(whatLatency) == 2:
    #     plt.savefig('capacity_vs_tail_latency_10msUpdate_99th_95th.png')
    #     plt.savefig('capacity_vs_tail_latency_10msUpdate_99th_95th.pdf')
    # elif whatLatency[0] == '99th_percentile':
    #     plt.savefig('capacity_vs_tail_latency_10msUpdate_99th.png')
    #     plt.savefig('capacity_vs_tail_latency_10msUpdate_99th.pdf')
    # else:
    #     plt.savefig('capacity_vs_tail_latency_10msUpdate_95th.png')
    #     plt.savefig('capacity_vs_tail_latency_10msUpdate_95th.pdf')
    # # plt.savefig('capacity_vs_tail_latency_10msUpdate.png')
    # # plt.savefig('capacity_vs_tail_latency_10msUpdate.pdf')
    # plt.show()

if __name__ == "__main__":
    main()
