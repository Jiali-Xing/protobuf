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
    calculate_tail_latency_dynamic,
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
offset = 3  # an offset of 3 seconds to omit pre-spike metrics



def plot_error_bars(ax, subset, metric, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False):
    ax.errorbar(subset['Load'], subset[metric], yerr=subset[metric + ' std'], fmt=control_markers[control],
                color=colors[control], linestyle=lineStyles[control], label=labelDict[control],
                linewidth=2, capsize=5, alpha=0.6)
    if add_hline:
        ax.axhline(y=SLO, color='c', linestyle='-.', label='SLO')

def setup_axes(axs, interfaces, ali_dict, alibaba_combined):
    if alibaba_combined:
        for i, interface in enumerate(interfaces):
            iname = ali_dict[interface]
            ax1, ax2 = axs[:, i]
            ax1.set_title(f'{iname}')
            if i == 0:
                ax1.set_ylabel('95th Tail\nLatency (ms)')
                ax2.set_ylabel('Goodput (RPS)')
            else:
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
        axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
    else:
        ax1, ax2 = axs
        ax1.set_xlabel('Load (RPS)')
        ax1.set_ylabel('95th Tail Latency (ms)')
        ax1.set_title('Load vs Tail Latency')
        ax1.grid(True)
        ax2.set_ylabel('Goodput (RPS)')
        ax2.set_title('Load vs Goodput')
        ax2.set_xlabel('Load (RPS)')
        ax2.grid(True)
        # ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)
        ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency):
    ax1, ax2 = axs
    for control in control_mechanisms:
        subset = df[df['overload_control'] == control]
        SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
        for latency in whatLatency:
            plot_error_bars(ax1, subset, latency, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=True)
        plot_error_bars(ax2, subset, 'Goodput', control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False)

def plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency, ali_dict):
    for i, interface in enumerate(interfaces):
        ax1, ax2 = axs[:, i]
        SLO = get_slo(method=interface, tight=tightSLO, all_methods=False)
        for control in control_mechanisms:
            subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
            for latency in whatLatency:
                plot_error_bars(ax1, subset, latency, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=True)
            plot_error_bars(ax2, subset, 'Goodput', control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False)


def plot_alibaba_eval(df, interfaces):
    if df is None:
        return

    alibaba_combined = len(interfaces) > 1

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'charon']
    whatLatency = ['95th_percentile']

    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'charon': '#FF9800',
    }

    control_markers = {
        'plain': 'o',
        'breakwater': 's',
        'breakwaterd': '^',
        'dagor': 'v',
        'charon': 'x',
    }

    lineStyles = {
        'plain': '-',
        'breakwater': '--',
        'breakwaterd': '-.',
        'dagor': ':',
        'charon': '-',
    }

    labelDict = {
        'plain': 'No Control',
        'breakwater': 'Breakwater',
        'breakwaterd': 'Breakwaterd',
        'dagor': 'Dagor',
        'charon': 'Rajomon',
    }

    ali_dict = {
        "S_102000854": "S1",
        "S_149998854": "S2",
        "S_161142529": "S3",
    }

    if alibaba_combined:
        fig, axs = plt.subplots(2, len(interfaces), figsize=(12, 5), sharex=True, sharey='row')
        plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency, ali_dict)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)

    setup_axes(axs, interfaces, ali_dict, alibaba_combined)

    save_path = os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/')
    save_name = 'all-alibaba' if alibaba_combined else interfaces[0]
    plt.savefig(f'{save_path}{save_name}-{datetime.now().strftime("%m%d")}.pdf')
    plt.show()
    print(f"Saved plot to {save_path}{save_name}-{datetime.now().strftime('%m%d')}.pdf")


def main():
    global method
    global tightSLO
    # global SLO
    
    old_data_sigcomm = True

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
            ('0526_2318', '0527_0109'), # run from bento
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