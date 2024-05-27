import glob, os, sys, re, json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

from slo import get_slo
from utils import (
    medianL,
    read_tail_latency,
    read_mean_latency,
    read_data,
    convert_to_dataframe,
    calculate_tail_latency,
    calculate_throughput,
    calculate_average_goodput,
    calculate_goodput,
    is_within_duration,
    is_within_any_duration,
    find_latest_files,
    load_data,
)

throughput_time_interval = '50ms'
latency_window_size = '200ms' 
offset = 2.5




def plot_alibaba_eval(df, interfaces):
    # if interfaces is of size 1, then plot the individual interface
    # if interfaces is of size 3, then plot the combined alibaba interfaces
    alibaba_combined = len(interfaces) > 1
    method = interfaces[0]

    if df is None:
        return

    # Define control mechanisms based on the `motivation` flag
    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'charon']

    # Define latency metrics to plot
    whatLatency = ['95th_percentile']

    # Map control mechanisms to colors
    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
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

    # Map control mechanisms to line styles
    lineStyles = {
        'plain': '-',
        'breakwater': '--',
        'breakwaterd': '-.',
        'dagor': ':',
        'charon': '-',
    }

    # Define markers for each control mechanism
    control_markers = {
        'plain': 'o',
        'breakwater': 's',
        'breakwaterd': '^',
        'dagor': 'v',
        'charon': 'x',
        # Add more mappings if necessary
    }


    # Map control mechanisms to labels
    labelDict = {
        'plain': 'No Control',
        'breakwater': 'Breakwater',
        'breakwaterd': 'Breakwaterd',
        'dagor': 'Dagor',
        'charon': 'Rajomon',
    }

    if alibaba_combined:
        # interfaces = ['S_102000854', 'S_149998854', 'S_161142529']
        # Plotting logic for combined Alibaba interfaces
        ali_dict = {
            "S_102000854": "S1",
            "S_149998854": "S2",
            "S_161142529": "S3",
        }
        fig, axs = plt.subplots(2, len(interfaces), figsize=(12, 5))

        for i, interface in enumerate(interfaces):
            ax1, ax2 = axs[:, i]
            for control in control_mechanisms:
                mask = (df['overload_control'] == control) & (df['Request'] == interface)
                subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]

                for latency in whatLatency:
                    subset_filtered = subset
                    # ax1.plot(subset_filtered['Load'], subset_filtered[latency],
                    #         color=colors[control], linestyle=lineStyles[control],
                    #         label=labelDict[control] if latency == '95th_percentile' else None,
                    #         linewidth=2,
                    #         marker=markers[latency] if latency == '99th_percentile' else None,
                    #         )

                    ax1.errorbar(subset_filtered['Load'], subset_filtered[latency], yerr=subset_filtered[latency + ' std'], fmt=control_markers[control],
                                color=colors[control], linestyle=lineStyles[control], label=labelDict[control],
                                linewidth=2, capsize=5)

                # ax2.plot(subset['Load'], subset['Goodput'],
                #         label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2)

                ax2.errorbar(subset['Load'], subset['Goodput'], yerr=subset['Goodput std'], fmt=control_markers[control],
                            label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2, capsize=5)


            iname = ali_dict[interface]
            ax1.set_title(f'{iname}')
            if i == 0:
                ax1.set_ylabel('95th Tail\nLatency (ms)')
                ax2.set_ylabel('Goodput (RPS)')
            else:
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])

        max_latency = max(df['95th_percentile'])
        maximum_goodput = max(df['Goodput'])
        for ax in axs.flatten()[0:len(interfaces)]:
            ax.grid(True)
            # ax.set_ylim(40, max_latency + 20)
        for ax in axs.flatten()[len(interfaces):]:
            ax.grid(True)
            # ax.set_ylim(0, maximum_goodput + 100)
        axs.flatten()[len(interfaces)].set_yticklabels(['0', '1k', '2k', '3k', '4k'])
        axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
        for i in range(len(interfaces)):
            axs[0][i].get_shared_x_axes().join(axs[0][i], axs[1][i])

        plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/all-alibaba-{datetime.now().strftime("%m%d")}.pdf'))
        plt.show()
        print(f"Saved plot to ~/Sync/Git/protobuf/ghz-results/all-alibaba-{datetime.now().strftime('%m%d')}.pdf")
    else:
        # Plotting logic for individual interface
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
        ax1, ax2 = axs

        for control in control_mechanisms:
            mask = (df['overload_control'] == control)
            subset = df[df['overload_control'] == control]

            for latency in whatLatency:
                subset_filtered = subset
                # ax1.plot(subset_filtered['Load'], subset_filtered[latency],
                #         color=colors[control], linestyle=lineStyles[control],
                #         label=labelDict[control] if latency == '95th_percentile' else None,
                #         linewidth=2,
                #         marker=markers[latency] if latency == '99th_percentile' else None,
                #         )

                ax1.errorbar(subset_filtered['Load'], subset_filtered[latency], yerr=subset_filtered[latency + ' std'], fmt=control_markers[control],
                            color=colors[control], linestyle=lineStyles[control], label=labelDict[control],
                            linewidth=2, capsize=5)

            # ax2.plot(subset['Load'], subset['Goodput'],
            #         label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2)

            ax2.errorbar(subset['Load'], subset['Goodput'], yerr=subset['Goodput std'], fmt=control_markers[control],
                        label=labelDict[control], color=colors[control], linestyle=lineStyles[control], linewidth=2, capsize=5)

        ax1.set_xlabel('Load (RPS)')
        ax1.set_ylabel('95th Tail Latency (ms)')
        ax1.set_title('Load vs Tail Latency')
        ax1.grid(True)

        ax2.set_ylabel('Goodput (RPS)')
        ax2.set_title('Load vs Goodput')
        ax2.set_xlabel('Load (RPS)')
        ax2.grid(True)
        ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime("%m%d")}.pdf'))
        plt.show()
        print(f"Saved plot to ~/Sync/Git/protobuf/ghz-results/{method}-{datetime.now().strftime('%m%d')}.pdf")


def main():
    global method
    global tightSLO
    global SLO
    
    old_data_sigcomm = False

    method = os.getenv('METHOD', 'ALL')
    alibaba_combined = method == 'ALL'

    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'

    print(f"Method: {os.getenv('METHOD', 'ALL')}, Tight SLO: {tightSLO}, Alibaba Combined: {alibaba_combined}")

    # experiment time ranges
    time_ranges = {
        # 'S_102000854': [('0525_1000', '0525_2150')], these are tuned with goodput - 10x tail latency
        'S_102000854': [
            ('0526_0409', '0526_0627'), # run from bento
            ('0526_1634', '0526_1832'), # run from laptop
            ], # these are tuned with goodput - squared tail latency
        'S_149998854': [('0501_0000', '0520_0000')],
        'S_161142529': [('0501_0000', '0520_0000')],
    }

    old_time_ranges = {
        'S_102000854': [
            ("1229_0301", "1229_0448"), # this includes plain no overload control. 4000-10000
            ("1231_2055", "1231_2241"), # this is for 6000-12000 load 
            ("1231_1747", "1231_1950"), # this is also for 6000-12000 load
            ("0129_0049", "0129_0138")
            ],
        'S_149998854': [
            ("1228_1702", "1228_1844"),
            ("1228_2356", "1229_0203"),
            ("1229_0141", "1229_0203"), # this is plain no overload control.
            ("1230_2124", "1230_2333"),
            ("1231_2244", "0101_0027"),  # newest result
            ("0128_0842", "0128_0902"),
            ("0128_1543", "0128_1640"),            
            ],
        'S_161142529': [
            ("1230_0611", "1230_0754"),
            ("1231_0042", "1231_0225"),  # this is new
            # ("0101_0127", "0101_0251"),
            ("0129_1654", "0129_1742"),
        ],
    }

    time_ranges = old_time_ranges if old_data_sigcomm else time_ranges
    tightSLO = True if old_data_sigcomm else tightSLO

    # Load data
    if alibaba_combined:
        # init a dataframe to merge the 3 alibaba interfaces
        alibaba_df = pd.DataFrame()
        # loop through the 3 alibaba interfaces and load the data and combine them into one dataframe
        interfaces = ["S_102000854", "S_149998854", "S_161142529"]
        for interface in interfaces:
            method = interface
            SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
            df = load_data(method=method, list_of_tuples_of_experiment_timestamps=time_ranges[interface], slo=SLO)
            df['Request'] = interface
            alibaba_df = pd.concat([alibaba_df, df])
        df = alibaba_df
        # print some stats of the combined dataframe, like sizes of each interface and load 
    else:
        SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
        df = load_data(method=method, list_of_tuples_of_experiment_timestamps=time_ranges[method], slo=SLO)
        interfaces = [method]

    plot_alibaba_eval(df, interfaces)


if __name__ == '__main__':
    main()