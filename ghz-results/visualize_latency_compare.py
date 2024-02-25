import glob
import json, csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from datetime import datetime
import matplotlib.dates as md
import matplotlib.ticker as mticker


latency_window_size = '150ms'  # Define the window size as 100 milliseconds
throughput_time_interval = '150ms'
offset = 1.1  # Define the offset as 50 milliseconds

def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["details"]


def convert_to_dataframe(data, SLO):
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
    df = calculate_throughput(df, SLO)
    df = calculate_tail_latency(df)
    df = calculate_goodput(df, SLO)
    return df


def calculate_goodput(df, slo):
    goodput_requests_per_second = df[(df['status'] == 'OK') & (df['latency'] < slo)]['status'].resample(throughput_time_interval).count()
    # scale the throughput to requests per second
    goodput_requests_per_second = goodput_requests_per_second * (1000 / int(throughput_time_interval[:-2]))
    # fill in zeros for the missing goodput
    df['goodput'] = goodput_requests_per_second.reindex(df.index, method='bfill')
    return df

def plot_combined(dfs, labels):
    assert len(dfs) == len(labels), "Length of DataFrames list and labels list must match."
    
    colors = {
        'Plain': '#F44336',
        'Breakwater': '#2196F3',
        'Breakwaterd': '#0D47A1',
        'Dagor': '#4CAF50',
        'Our model': '#FF9800',
        'nginx-web-server': '#9C27B0',  # Example color for nginx-web-server
        'service-6': '#3F51B5',  # Example color for service-6
        'all': '#009688',  # Example color for all
    }

    lineStyles = {
        'Plain': '-',
        'Breakwater': '--',
        'Breakwaterd': '-.',
        'Dagor': ':',
        'Our model': '-'
    }

    # give each label a different hatch pattern
    hatches = {
        'Plain': '-',
        'Breakwater': '--',
        'Breakwaterd': '-.',
        'Dagor': ':',
        'Our model': '-',
    }

    # fig, axs = plt.subplots(4, 1, figsize=(6, 8))
    
    fig = plt.figure(figsize=(3, 8), constrained_layout=True)

    # Create ax1 as a subplot in the figure
    ax1 = fig.add_subplot(4, 1, 1)  # 4 rows, 1 column, 1st subplot

    # Create ax2, ax3, ax4 as subplots in the figure, with shared x-axis
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax2)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax2)

    # ax1, ax2, ax3, ax4 = axs
    axs = [ax1, ax2, ax3, ax4]
    
    # Find the global min and max latency values across all dfs
    global_min = 0
    global_max = 300

    # Define the bin edges based on the global range and desired number of bins
    # The np.histogram_bin_edges function can help with this
    num_bins = 250  # or whatever desired number of bins you want
    bin_edges = np.linspace(global_min, global_max, num_bins)
    # # Histograms on ax1
    # for df, label in zip(dfs, labels):
    #     ax1.hist(df['latency'], alpha=0.5, label=label, bins=bin_edges, color=colors[label], hatch=hatches[label])
    # ax1.set_ylabel('Frequency')
    # ax1.set_title('Histogram of Latencies (ms)')
    # ax1.legend(loc='upper right')
    # ax1.set_xlim([0, 200])
    # ax1.set_yscale('log')

    # Box plots on ax1, with whiskers at extremes
    # ax1.boxplot([df['latency'] for df in dfs], labels=labels, showfliers=False, whis=[0, 100]), but only the latency values where status == 'OK'
    ax1.boxplot([df[df['status'] == 'OK']['latency'] for df in dfs], labels=labels, showfliers=False, whis=[0, 95])
    ax1.set_ylabel('Latency Distribution (ms)')
    ax1.set_yscale('log')
    # show more y ticks with numbers
    ax1.set_yticks([20, 40, 60, 80, 100])
    # show the y numbers in log scale
    ax1.set_yticklabels([20, 40, 60, 80, 100])
    # Share x-axis for ax2, ax3, ax4
    # ax2.get_shared_x_axes().join(ax2, ax3, ax4)

    # Goodput on ax2
    for df, label in zip(dfs, labels):
        ax2.plot(df.index, df['goodput'], label=label, color=colors[label], linestyle=lineStyles[label])
    ax2.set_ylabel('Goodput (RPS)')
    ax2.set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in ax2.get_yticks()/1000])
    #     ax2.plot(df.index, df['throughput'], label=label, color=colors[label], linestyle=lineStyles[label])
    # ax2.set_ylabel('Throughput (RPS)') 
    # ax2.set_title('Throughput Over Time')

    # 99th percentile time-series on ax3
    for df, label in zip(dfs, labels):
        # ax3.plot(df.index, df['tail_latency'], label=f"{label} - 99th percentile", color=colors[label], linestyle=lineStyles[label])
    # ax3.set_ylabel('99th Tail Latency (ms)')
        ax3.plot(df.index, df['95tail'], label=f"{label}", color=colors[label], linestyle=lineStyles[label])
    ax3.set_ylabel('95th Tail Latency (ms)')

    ax1.axhline(y=SLO, color='c', linestyle='-.', label='SLO')
    ax3.axhline(y=SLO, color='c', linestyle='-.', label='SLO')
    # ax3.set_title('99th Percentile Latencies Over Time')
    # ax3.legend(loc='upper right')

    # Average latency on ax4
    # for df, label in zip(dfs, labels):
    #     ax4.plot(df.index, df['latency_ma'], label=f"{label} - avg percentile")
    # ax4.set_ylabel('Latency (ms)')
    # ax4.set_title('Average Latency Over Time')
    # ax4.legend(loc='upper right')

    for df, label in zip(dfs, labels):
        ax4.plot(df.index, df['slo_violation'], label=f"{label}", color=colors[label], linestyle=lineStyles[label])
    ax4.set_ylabel('SLO Violation (RPS)')
    # ax4.set_title('Median Latency Over Time')

    # ax4.legend without box
    ax4.legend(loc='upper left', frameon=False)

    # Add grid to all subplots
    for ax in axs:
        ax.grid(True)

    # for 3 and 4, use log scale
    ax3.set_yscale('log')
    # ax4.set_yscale('log')

    # set y limit for 3 and 4 to be 10 to 100
    ax3.set_ylim([20, 200])
    ax4.set_ylim([0, 10000])
    
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%S'))

    plt.subplots_adjust(hspace=0.01)
    # Adjust spacing
    timeStamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(f'overload_comparison_{timeStamp}.pdf')
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

    # # If two arguments are provided, use them as the file names
    # if len(sys.argv) == 3:
    #     charon_file = sys.argv[1]
    #     plain_file = sys.argv[2]
    #     files = [charon_file, plain_file]
    #     print(files)
    # else:
    #     # Otherwise, find the latest files for the given capacity
    #     files = find_latest_files(capacity, directory)


    # files = [
    #     'social-compose-control-plain-parallel-capacity-8000-1209_1522.json',
    #     'social-compose-control-charon-parallel-capacity-8000-1209_1907.json',
    #     'social-compose-control-breakwater-parallel-capacity-8000-1209_1329.json',
    #     'social-compose-control-breakwaters-parallel-capacity-8000-1209_1439.json',
    #     'social-compose-control-dagor-parallel-capacity-8000-1209_2106.json'
    # ] these are old files for 8000 capacity

    files = [
        'social-compose-control-charon-parallel-capacity-10000-1226_0453.json',
        'social-compose-control-breakwater-parallel-capacity-10000-1225_0201.json',
        'social-compose-control-breakwaterd-parallel-capacity-10000-1225_0223.json',
        'social-compose-control-dagor-parallel-capacity-10000-1225_0241.json',
    ]

    filesLooseSLO = [
        # 'social-compose-control-breakwater-parallel-capacity-10000-0117_2017.json',
        'social-compose-control-breakwater-parallel-capacity-10000-0127_2053.json',
        # 'social-compose-control-breakwaterd-parallel-capacity-10000-0117_2044.json',
        'social-compose-control-breakwaterd-parallel-capacity-10000-0127_2119.json',
        'social-compose-control-charon-parallel-capacity-10000-0116_1908.json',
        'social-compose-control-dagor-parallel-capacity-10000-0118_0355.json',
    ]
    filesTightSLO = [
        # 'social-compose-control-breakwater-parallel-capacity-8000-0117_1737.json',
        'social-compose-control-breakwater-parallel-capacity-8000-0128_0236.json',
        # 'social-compose-control-breakwaterd-parallel-capacity-8000-0117_2321.json',
        'social-compose-control-breakwaterd-parallel-capacity-8000-0130_1348.json',
        'social-compose-control-charon-parallel-capacity-8000-0117_1718.json',
        'social-compose-control-dagor-parallel-capacity-8000-0117_2339.json',
    ]



    # base_dir = '/home/ying/Sync/Git/charon-experiments/old_social/'
    base_dir = '/home/ying/Sync/Git/protobuf/ghz-results/'
    lossefiles = [base_dir + f for f in filesLooseSLO] 

    labels = ['Breakwater', 'Breakwaterd', 'Our model', 'Dagor']
    dfsl = []

    for i, filename in enumerate(lossefiles):
        data = read_data(filename)
        df = convert_to_dataframe(data, loose_slo_value)
        df = df.fillna(0)
        dfsl.append(df)

    tightfiles = [base_dir + f for f in filesTightSLO]
    dfst = []
    for i, filename in enumerate(tightfiles):
        data = read_data(filename)
        df = convert_to_dataframe(data, tight_slo_value)
        df = df.fillna(0)
        dfst.append(df)

    # plot_combined(dfs, labels)
    plot_tight_loose(dfst, dfsl, labels)

def plot_tight_loose(dfs_tight, dfs_loose, labels):
    assert len(dfs_tight) == len(labels) and len(dfs_loose) == len(labels), "Length of DataFrames lists and labels list must match."

    # Define colors, line styles, and hatches as before
    colors = {
        'Plain': '#F44336',
        'Breakwater': '#2196F3',
        'Breakwaterd': '#0D47A1',
        'Dagor': '#4CAF50',
        'Our model': '#FF9800',
        'nginx-web-server': '#9C27B0',  # Example color for nginx-web-server
        'service-6': '#3F51B5',  # Example color for service-6
        'all': '#009688',  # Example color for all
    }

    lineStyles = {
        'Plain': '-',
        'Breakwater': '--',
        'Breakwaterd': '-.',
        'Dagor': ':',
        'Our model': '-'
    }

    # give each label a different hatch pattern
    hatches = {
        'Plain': '-',
        'Breakwater': '--',
        'Breakwaterd': '-.',
        'Dagor': ':',
        'Our model': '-',
    }

    # gridspec_kw={'height_ratios': [1, 2, 2, 1]}
    fig, axs = plt.subplots(4, 2, figsize=(6, 6.4), constrained_layout=True, gridspec_kw={'height_ratios': [2, 3, 3, 3]})

    # # Sharing y-axis for each row
    # for i in range(4):
    #     axs[i, 1].sharey(axs[i, 0])

    axs[0, 0].set_yticks([20, 40, 60, 80, 100])
    # Function to plot each subplot
    def plot_subplot(dfs, ax_row, SLO):
        # Box plots for latency distribution on the first row
        ax_row[0].boxplot([df[df['status'] == 'OK']['latency'] for df in dfs], labels=labels, showfliers=False, whis=[0, 95])
        ax_row[0].set_ylabel('Latency\nDistribution (ms)')
        ax_row[0].set_yscale('log')
        ax_row[0].set_yticks([20, 40, 80, 120])
        ax_row[0].set_yticklabels([20, 40, 80, 120])
        # ax_row[0].grid(True)
        ax_row[0].axhline(y=SLO, color='c', linestyle='-.', label='SLO')

        for i, df in enumerate(dfs):
            # Goodput on the second row
            ax_row[1].plot(df.index, df['goodput'], label=labels[i], color=colors[labels[i]], linestyle=lineStyles[labels[i]])
            ax_row[1].set_ylabel('Goodput (RPS)')
            ax_row[1].grid(True)

            # 95th percentile time-series on the third row
            ax_row[2].plot(df.index, df['95tail'], label=labels[i], color=colors[labels[i]], linestyle=lineStyles[labels[i]])
            ax_row[2].set_ylabel('95th Tail Latency (ms)')
            ax_row[2].grid(True)
            # ax_row[2].set_yscale('log')
            # ax_row[2].set_ylim([20, 200])

            # SLO violations on the fourth row
            ax_row[3].plot(df.index, df['slo_violation'], label=labels[i], color=colors[labels[i]], linestyle=lineStyles[labels[i]])
            ax_row[3].set_ylabel('SLO Violation (RPS)')
            ax_row[3].grid(True)
        # if its the first column, add legend
        # if ax_row[0] == axs[0, 0]:
        #     ax_row[3].legend(loc='upper left', frameon=False)
        ax_row[2].axhline(y=SLO, color='c', linestyle='-.', label='SLO')
    

#     # Plot for tight SLO
    plot_subplot(dfs_tight, axs[:, 0], tight_slo_value)

    # Plot for loose SLO
    plot_subplot(dfs_loose, axs[:, 1], loose_slo_value)

    axs[0, 0].set_title('Tight SLO')
    axs[0, 1].set_title('Relaxed SLO')

    # add legend to the ax 2, 0
    axs[3, 0].legend(loc='upper left', frameon=False)

    # for the first row, rotate the x-axis labels
    for i in range(2):
        axs[0, i].tick_params(axis='x', labelrotation=15)

    axs[2, 0].set_yscale('log')
    axs[2, 1].set_yscale('log')

    for i in range(1, 4):
        # axs[i, 0] set xaxis with set_major_formatter(md.DateFormatter('%S'))
        axs[i, 0].xaxis.set_major_formatter(md.DateFormatter('%S'))
        axs[i, 1].xaxis.set_major_formatter(md.DateFormatter('%S'))

    # set ylimit for 4 rows
    for j in range(2):
        # axs[2, j].set_yscale('log')
        axs[0, j].set_ylim([0, 200])
        # axs[0, j].set_yticks([20, 40, 60, 80, 100])
        axs[1, j].set_ylim([0, 10000])
        axs[2, j].set_ylim([20, 200])
        axs[3, j].set_ylim([0, 10000])
        # Apply log scale to specific rows if needed
        # axs[2, j].set_yscale('log')
    # # Set y-axis labels for the first column and remove them for the second column
    for i in range(4):
        axs[i, 1].set_ylabel('')  # Remove y-axis label for the second column
        # Remove y-tick labels only for the second column
        axs[i, 1].set_yticklabels([])
        axs[i, 1].tick_params(axis='y', which='both', left=False)  # Remove y-ticks for the second column

    axs[2, 1].set_yticklabels([])
    axs[2, 1].minorticks_off()

    axs[3, 0].set_xlabel('Time (seconds)')
    axs[3, 1].set_xlabel('Time (seconds)')

    # for ax in first column, set yticks to be x k:
    axs[1, 0].set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in axs[1, 0].get_yticks()/1000])
    axs[3, 0].set_yticklabels(['{:,}'.format(int(x)) + 'k' for x in axs[3, 0].get_yticks()/1000])

    # for 2, 0, set yticks to not use scientific notation
    axs[2, 0].set_yticklabels(['{:,}'.format(int(x)) for x in axs[2, 0].get_yticks()])

    plt.savefig(f'overload_comparison_{datetime.now().strftime("%m%d_%H%M")}.pdf')
    plt.show()


def calculate_tail_latency(df):
    # Only cound the latency of successful requests, i.e., status == 'OK'
    df.sort_index(inplace=True)
    # Assuming your DataFrame is named 'df' and the column to calculate the moving average is 'data'
    tail_latency = df['latency'].rolling(latency_window_size).quantile(0.99)
    df['tail_latency'] = tail_latency
    df['95tail'] = df['latency'].rolling(latency_window_size).quantile(0.95)
    # Calculate moving average of latency
    df['latency_ma'] = df['latency'].rolling(latency_window_size).mean()
    # and the median latency
    df['latency_median'] = df['latency'].rolling(latency_window_size).median()
    return df


def calculate_throughput(df, SLO):
    ok_requests_per_second = df[df['status'] == 'OK']['status'].resample(throughput_time_interval).count()
    ok_requests_per_second *= (1000 / int(throughput_time_interval[:-2]))
    df['throughput'] = ok_requests_per_second.reindex(df.index, method='bfill')

    slo_violation = df[(df['latency'] > SLO) & (df['status'] == 'OK')]['latency'].resample(throughput_time_interval).count()
    slo_violation *= (1000 / int(throughput_time_interval[:-2]))
    df['slo_violation'] = slo_violation.reindex(df.index, method='ffill')
    return df

if __name__ == '__main__':
    # global SLO
    # global tightSLO
    # tightSLO = True
    # SLO = 40 if tightSLO else 120
    
    tight_slo_value = 40
    loose_slo_value = 120
    main()