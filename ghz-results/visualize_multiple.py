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


def main():
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
        
        if latency_99 is not None:
            rows.append({
                'Capacity': capacity,
                '99th_percentile': latency_99,
                '95th_percentile': latency_95,
                'method_subcall': method_subcall,
                'overload_control': overload_control
            })
            # log the latency_95 and latency_99 for each capacity and method_subcall, print the row with 2 decimal places
            print(f'Control: {overload_control}, Method: {method_subcall}, Capacity: {capacity}, 99th: {latency_99:.2f}, 95th: {latency_95:.2f}')
    
    df = pd.DataFrame(rows)
    df.sort_values('Capacity', inplace=True)

    # Extract data for each method and latency percentile
    # Define the methods you want to plot
    methods = ['sequential', 'parallel']
    control_mechanisms = ['plain', 'charon']

    # Define markers for each method
    markers = ['o', 's']
    whatLatency = ['99th_percentile', '95th_percentile']
    # whatLatency = ['95th_percentile']

    # Create a plot for each method and latency percentile
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    for i, method in enumerate(methods):
        for j, control in enumerate(control_mechanisms):
            subset = df[(df['method_subcall'] == method) & (df['overload_control'] == control)]
            
            # Extract capacity values (assuming they are the same for all data points)
            # capacity = subset['Capacity'].unique()

            for latency in whatLatency:
                subset = subset[subset[latency] < 200]  
                plt.plot(subset['Capacity'], subset[latency], 
                         label=f'{control} {method} {latency}', marker=markers[i])
            # remove the data within if the latency is < 150ms
            # subset = subset[subset['99th_percentile'] < 200]
            # subset = subset[subset['95th_percentile'] < 200]
                            

            # Plot the data for this method
            # plt.plot(subset['Capacity'], subset['99th_percentile'], 
            #          label=f'{control} {method} 99th Percentile', marker='o')
                     
            # plt.plot(subset['Capacity'], subset['95th_percentile'], 
            #          label=f'{control} {method} 95th Percentile', marker='s')


    plt.legend(title='Subcalls and Percentile')
    plt.xlabel('Capacity (rps)')
    plt.ylabel('Tail Latency (ms)')
    plt.title('Capacity vs Tail Latency')
    plt.grid(True)
    # filename includes the latency percentile
    # filename = 'capacity_vs_tail_latency_10msUpdate_99th_95th.png' if len(whatLatency) == 2 else 'capacity_vs_tail_latency_10msUpdate_99th.png'
    if len(whatLatency) == 2:
        plt.savefig('capacity_vs_tail_latency_10msUpdate_99th_95th.png')
        plt.savefig('capacity_vs_tail_latency_10msUpdate_99th_95th.pdf')
    elif whatLatency[0] == '99th_percentile':
        plt.savefig('capacity_vs_tail_latency_10msUpdate_99th.png')
        plt.savefig('capacity_vs_tail_latency_10msUpdate_99th.pdf')
    else:
        plt.savefig('capacity_vs_tail_latency_10msUpdate_95th.png')
        plt.savefig('capacity_vs_tail_latency_10msUpdate_95th.pdf')
    # plt.savefig('capacity_vs_tail_latency_10msUpdate.png')
    # plt.savefig('capacity_vs_tail_latency_10msUpdate.pdf')
    plt.show()

if __name__ == "__main__":
    main()
