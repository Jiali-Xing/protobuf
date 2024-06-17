import glob, os, sys, re, json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

from slo import get_slo
from utils import (
    # load_data,
    load_data_from_csv,
)

# throughput_time_interval = '50ms'
# latency_window_size = '200ms' 
# offset = 3  # an offset of 3 seconds to omit pre-spike metrics



def plot_error_bars(ax, subset, metric, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False, hline_label=True, max_x_range=None):
    ax.errorbar(subset['Load'], subset[metric], yerr=subset[metric + ' std'], fmt=control_markers[control],
                color=colors[control], linestyle=lineStyles[control], label=labelDict[control],
                linewidth=2, capsize=5, alpha=0.6)
    if max_x_range:
        ax.set_xlim(9000, max_x_range+1000)
    if add_hline:
        if hline_label:
            ax.axhline(y=SLO, color='c', linestyle='-.', label='SLO')
        else:
            ax.axhline(y=SLO, color='c', linestyle='-.')

def setup_axes(axs, interfaces, ali_dict, alibaba_combined):
    if alibaba_combined:
        for i, interface in enumerate(interfaces):
            iname = ali_dict[interface]
            ax1, ax2 = axs[:, i]
            ax1.set_title(f'{iname}')
            if i == 0:
                ax1.set_ylabel('95th Tail\nLatency (ms)')
                ax2.set_ylabel('Goodput (RPS)')
                # # Set y-ticks for the leftmost plots
                # ax1.set_yticks(np.arange(0, 1000, 100))
                # ax1.set_yticklabels(np.arange(0, 1000, 100))
                # ax2.set_yticks(np.arange(0, 9000, 1000))
                # ax2.set_yticklabels(np.arange(0, 9000, 1000))
            # else:
            #     ax1.set_yticklabels([])
            #     ax2.set_yticklabels([])
        axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.02, 1.3), ncol=4)

        # make the gap between the subplots smaller
        plt.subplots_adjust(wspace=0.02)
        plt.subplots_adjust(hspace=0.03)
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

def plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency):
    slo_label = True
    for i, interface in enumerate(interfaces):
        ax1, ax2 = axs[:, i]
        SLO = get_slo(method=interface, tight=tightSLO, all_methods=False)
        for control in control_mechanisms:
            subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
            for latency in whatLatency:
                plot_error_bars(ax1, subset, latency, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=True, hline_label=slo_label)
                slo_label = False
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
        plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)

    setup_axes(axs, interfaces, ali_dict, alibaba_combined)

    save_path = os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/')
    save_name = 'all-alibaba' if alibaba_combined else interfaces[0]
    plt.savefig(f'{save_path}{save_name}-{datetime.now().strftime("%m%d")}.pdf')
    plt.show()
    print(f"Saved plot to {save_path}{save_name}-{datetime.now().strftime('%m%d')}.pdf")


def plot_4nodes(df, interfaces):
    if df is None:
        return

    motivate_combined = len(interfaces) > 1

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
        "motivate-set": "Set",
        "motivate-get": "Get",
    }

    if motivate_combined:
        fig, axs = plt.subplots(2, len(interfaces), figsize=(6, 5), sharex=True, sharey='row')
        plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)

    setup_axes(axs, interfaces, ali_dict, motivate_combined)

    save_path = os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/')
    save_name = 'both-4nodes' if motivate_combined else '4nodes'
    plt.savefig(f'{save_path}{save_name}-{datetime.now().strftime("%m%d")}.pdf')
    plt.show()
    print(f"Saved plot to {save_path}{save_name}-{datetime.now().strftime('%m%d')}.pdf")


def load_plot_4nodes():
    global method
    global tightSLO

    motivate_combined = 'both' in method

    method = os.getenv('METHOD', 'motivate-set')
    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'

    print(f"Method: {os.getenv('METHOD', 'ali3')}, Tight SLO: {tightSLO}")

    # experiment time ranges
    time_ranges = {
        'motivate-set': [
            ('0607_0236', '0607_1300'),
        ],
        'motivate-get': [
            ('0607_0236', '0607_1300'),
        ],
    }

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'charon']

    parameter_files = {
        'motivate-set': {
            control: f'bopt_False_{control}_motivate-set_gpt1-30000_06-10.json' for control in control_mechanisms
        }, 
    }
        # motivate-get uses the same parameter files as motivate-set
    parameter_files['motivate-get'] = {
        k: v for k, v in parameter_files['motivate-set'].items()
    }

    csv_file = '~/Sync/Git/protobuf/ghz-results/grouped_4n_monotonic.csv'
    if motivate_combined:
        # init a dataframe to merge the 2 motivate interfaces for `motivate-set` and `motivate-get`
        motivate_df = pd.DataFrame()
        # loop through the 2 motivate interfaces and load the data and combine them into one dataframe
        interfaces = ["motivate-set", "motivate-get"]
        for method in interfaces:
            df = load_data_from_csv(csv_file, method=method, list_of_tuples_of_experiment_timestamps=time_ranges[method], given_parameter=parameter_files)
            assert df is not None, f"Dataframe for {method} is None"
            df['Request'] = method
            motivate_df = pd.concat([motivate_df, df])
        df = motivate_df
        # report the file_count column for each interface and capacity
        df['file_count'] = df['file_count'].astype(int)
        for interface in interfaces:
            for control in control_mechanisms:
                subset = df[(df['Request'] == interface) & (df['overload_control'] == control)]
                print(f"{interface} {control} file_count: {subset['file_count'].values}")
    else:
        # Load data
        df = load_data_from_csv(csv_file, method=method, list_of_tuples_of_experiment_timestamps=time_ranges[method], given_parameter=parameter_files)
        interfaces = [method]

    # remove the rows with Load > 24000
    df = df[df['Load'] > 9000]
    plot_4nodes(df, interfaces)

def load_plot_alibaba():
    global method
    global tightSLO
    # global SLO
    
    old_data_sigcomm = False

    method = os.getenv('METHOD', 'ali3')
    alibaba_combined = (method == 'ali3' or method == 'all-alibaba')

    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'

    print(f"Method: {os.getenv('METHOD', 'ali3')}, Tight SLO: {tightSLO}, Alibaba Combined: {alibaba_combined}")

    # # experiment time ranges
    # time_ranges = {
    #     # 'S_102000854': [('0525_1000', '0525_2150')], these are tuned with goodput - 10x tail latency
    #     'S_102000854': [
    #         ('0526_0409', '0526_0627'), # run from bento
    #         ('0526_1634', '0526_1832'), # run from laptop
    #         ('0526_2318', '0527_0109'), # run from bento
    #         ], # these are tuned with goodput - squared tail latency
    #     'S_149998854': [('0501_0000', '0520_0000')],
    #     'S_161142529': [('0501_0000', '0520_0000')],
    # }
    
    # New time ranges are for the 15 second experiment, 5 second warmup, and fixed start load (80% of sustainable load)
    new_time_ranges = {
        'S_102000854': [
            ('0528_0000', '0531_0000'), # this is tuned with only goodput
        ],
        'S_149998854': [
            # # ('0528_1009', '0528_1216'), # this is tuned with square tail latency
            # ('0528_1658', '0528_1852'), # this is tuned with 10x tail latency
            # ('0528_2304', '0528_2341'), # this is tuned with only goodput
            # above are tuned without warmup. below are tuned with warmup counted.
            ('0528_0000', '0531_0000'), # this is tuned with only goodput
        ],
        'S_161142529': [
            ('0528_0000', '0531_0000'), # this is tuned with only goodput
        ],
    }

    # parameter_files = {
    #     'S_102000854': {
    #         'breakwaterd': 'bopt_False_breakwaterd_S_102000854_gpt1-10000_05-29.json',  
    #         'breakwater': 'bopt_False_breakwater_S_102000854_gpt1-10000_05-28.json',
    #         # 'breakwater': 'bopt_False_breakwater_S_102000854_gpt1-10000_05-29.json',  
    #         'dagor': 'bopt_False_dagor_S_102000854_gpt1-10000_05-29.json',
    #         'charon': 'bopt_False_charon_S_102000854_gpt1-10000_05-29.json',
    #     },
    #     'S_149998854': {
    #         'breakwaterd': 'bopt_False_breakwaterd_S_149998854_gpt1-10000_05-29.json',
    #         'breakwater': 'bopt_False_breakwater_S_149998854_gpt1-10000_05-29.json',
    #         'dagor': 'bopt_False_dagor_S_149998854_gpt1-10000_05-29.json',
    #         'charon': 'bopt_False_charon_S_149998854_gpt1-10000_05-29.json',
    #     },
    #     'S_161142529': {
    #         'breakwaterd': 'bopt_False_breakwaterd_S_161142529_gpt1-10000_05-29.json',
    #         'breakwater': 'bopt_False_breakwater_S_161142529_gpt1-10000_05-29.json',
    #         'dagor': 'bopt_False_dagor_S_161142529_gpt1-10000_05-29.json',
    #         'charon': 'bopt_False_charon_S_161142529_gpt1-10000_05-29.json',
    #     },
    # }

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'charon']

    parameter_files = {
        'S_102000854': {
            control: f'bopt_False_{control}_S_102000854_gpt1-10000_06-06.json' for control in control_mechanisms
        },
        'S_149998854': {
            control: f'bopt_False_{control}_S_149998854_gpt1-10000_06-06.json' for control in control_mechanisms
        },
        'S_161142529': {
            control: f'bopt_False_{control}_S_161142529_gpt1-10000_06-06.json' for control in control_mechanisms
        },
    }
    # # replace S_14 charon with the new json
    parameter_files['S_149998854']['charon'] = 'bopt_False_charon_S_149998854_gpt1-10000_05-29.json'
    parameter_files['S_161142529']['charon'] = 'bopt_False_charon_S_161142529_gpt1-10000_06-05.json'

    different_spike = {
        'S_102000854': {
            # control: f'bopt_False_{control}_S_102000854_gpt1-12000_05-30.json' for control in control_mechanisms
            control: f'bopt_False_{control}_S_102000854_gpt1-12000_06-10.json' for control in control_mechanisms
        },
        'S_149998854': {
            # control: f'bopt_False_{control}_S_149998854_gpt1-9000_06-03.json' for control in control_mechanisms
            control: f'bopt_False_{control}_S_149998854_gpt1-10000_06-10.json' for control in control_mechanisms
        },
        'S_161142529': {
            # control: f'bopt_False_{control}_S_161142529_gpt1-12000_05-30.json' for control in control_mechanisms
            control: f'bopt_False_{control}_S_161142529_gpt1-12000_06-10.json' for control in control_mechanisms
        },
    }
    parameter_files = different_spike
    
    # add bopt_False_dagor_S_*_gpt1-10000_05-29.json to the parameter_files
    # dagor_files = {
    #     interface: {
    #         'dagor': f'bopt_False_dagor_{interface}_gpt1-10000_05-29.json'
    #     } for interface in ['S_102000854', 'S_149998854', 'S_161142529']
    # }
    # merge the two dictionaries at second level
    # parameter_files = {k: {**parameter_files[k], **dagor_files[k]} for k in parameter_files}


    # old_time_ranges = {
    #     'S_102000854': [
    #         ("1229_0301", "1229_0448"), # this includes plain no overload control. 4000-10000
    #         ("1231_2055", "1231_2241"), # this is for 6000-12000 load 
    #         ("1231_1747", "1231_1950"), # this is also for 6000-12000 load
    #         ("0129_0049", "0129_0138")
    #         ],
    #     'S_149998854': [
    #         ("1228_1702", "1228_1844"),
    #         ("1228_2356", "1229_0203"),
    #         ("1229_0141", "1229_0203"), # this is plain no overload control.
    #         ("1230_2124", "1230_2333"),
    #         ("1231_2244", "0101_0027"),  # newest result
    #         ("0128_0842", "0128_0902"),
    #         ("0128_1543", "0128_1640"),            
    #         ],
    #     'S_161142529': [
    #         ("1230_0611", "1230_0754"),
    #         ("1231_0042", "1231_0225"),  # this is new
    #         # ("0101_0127", "0101_0251"),
    #         ("0129_1654", "0129_1742"),
    #     ],
    # }

    # time_ranges = old_time_ranges if old_data_sigcomm else time_ranges
    time_ranges = new_time_ranges 
    # tightSLO = True if old_data_sigcomm else tightSLO

    # Load data
    if alibaba_combined:
        # init a dataframe to merge the 3 alibaba interfaces
        alibaba_df = pd.DataFrame()
        # loop through the 3 alibaba interfaces and load the data and combine them into one dataframe
        interfaces = ["S_102000854", "S_149998854", "S_161142529"]
        for interface in interfaces:
            method = interface
            SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
            df = None
            if method == 'all-alibaba':
                parameter_files = {
                    api: {
                        control: f'bopt_False_{control}_{api}_gpt1-12000_06-10.json.json' for control in control_mechanisms
                    } for api in interfaces
                }
                df = load_data_from_csv(f'~/Sync/Git/protobuf/ghz-results/grouped_all_ali.csv', method=method, list_of_tuples_of_experiment_timestamps=time_ranges[method], given_parameter=parameter_files)
            elif method == 'ali3':
                df = load_data_from_csv(f'~/Sync/Git/protobuf/ghz-results/grouped_ali.csv', method=method, list_of_tuples_of_experiment_timestamps=time_ranges[interface], given_parameter=parameter_files)
            assert df is not None, f"Dataframe for {interface} is None"
            df['Request'] = interface
            alibaba_df = pd.concat([alibaba_df, df])
        df = alibaba_df
        # print some stats of the combined dataframe, like sizes of each interface and load 
    else:
        SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
        # df = load_data(method=method, list_of_tuples_of_experiment_timestamps=time_ranges[method], slo=SLO, given_parameter=parameter_files)
        df = load_data_from_csv(f'~/Sync/Git/protobuf/ghz-results/grouped_ali.csv', method=method, list_of_tuples_of_experiment_timestamps=time_ranges[method], given_parameter=parameter_files)
        interfaces = [method]

    # summarize the dataframe, specifically the file_count column
    # report the file_count column for each interface and capacity
    df['file_count'] = df['file_count'].astype(int)
    for interface in interfaces:
        for control in control_mechanisms:
            subset = df[(df['Request'] == interface) & (df['overload_control'] == control)]
            print(f"{interface} {control} file_count: {subset['file_count'].values}")
    plot_alibaba_eval(df, interfaces)


if __name__ == '__main__':
    method = os.getenv('METHOD', 'both-motivate')
    if 'motivate' in method:
        load_plot_4nodes()
    elif method == 'all-alibaba' or method == 'ali3':
        load_plot_alibaba()