import json, re, datetime
import sys
import re
import os
import pytz
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.dates as md
from slo import get_slo
from utils import (
    calculate_goodput_ave_var,
    calculate_tail_latency_dynamic,
    read_load_info_from_json,
    read_data,
    calculate_goodput_dynamic,
    calculate_throughput_dynamic,
    calculate_loadshedded,
    calculate_ratelimited,
    read_mean_latency
)
from visualize import convert_to_dataframe, calculate_average_var_goodput, read_tail_latency, format_ticks



throughput_time_interval = '200ms'
latency_window_size = '500ms'
offset = 5
oneNode = False
hotel = True

capacity = 24000 if oneNode else 5000
computationTime = 0
INTERCEPTOR = os.environ.get('INTERCEPT', 'plain').lower()

cloudlabOutput = r"deathstar_([\w-]+)\.output"

CONSTANT_LOAD = os.environ.get('CONSTANT_LOAD', 'false').lower() == 'true'
if 'CAPACITY' in os.environ:
    capacity = int(os.environ['CAPACITY'])
    print("Capacity is set to", capacity)

# # analyze_plot_goodput() is the combination of analyze_data() and plot_timeseries_split() but only for goodput (ax2)
# def analyze_plot_goodput(filename, ax, is_first_col):
#     global computationTime
#     if "control" in filename:
#         match = re.search(r'control-(\w+)-', filename)
#         if match:
#             INTERCEPTOR = match.group(1)
#             print("Interceptor:", INTERCEPTOR)
    
#     if "capacity" in filename:
#         match = re.search(r'capacity-(\w+)-', filename)
#         if match:
#             capacity = int(match.group(1))
#             print("Capacity:", capacity)

#     try:
#         data = read_data(filename)
#     except Exception as e:
#         print(f"Error: {e}")
#         filename = filename.replace("ghz-results", "archived_results")
#         print(f"Trying with archived_results: {filename}")
#         data = read_data(filename)

#     method = re.findall(r"social-(.*?)-control", filename)[0]
#     SLO = get_slo(method, tight=False, all_methods=False)

#     df = convert_to_dataframe(data, init=True)
#     df = calculate_throughput_dynamic(df)
#     df = calculate_goodput_dynamic(df, SLO)
#     df = calculate_loadshedded(df)
#     df = calculate_ratelimited(df)
#     plot_timeseries_split(df, filename, computationTime, ax, None, is_first_col)

def analyze_data(filename, ax1, ax2, is_first_col):
    global INTERCEPTOR, capacity
    if "control" in filename:
        match = re.search(r'control-(\w+)-', filename)
        if match:
            INTERCEPTOR = match.group(1)
            print("Interceptor:", INTERCEPTOR)
    
    if "capacity" in filename:
        match = re.search(r'capacity-(\w+)-', filename)
        if match:
            capacity = int(match.group(1))
            print("Capacity:", capacity)

    try:
        data = read_data(filename)
    except Exception as e:
        print(f"Error: {e}")
        filename = filename.replace("ghz-results", "archived_results")
        print(f"Trying with archived_results: {filename}")
        data = read_data(filename)

    method = re.findall(r"social-(.*?)-control", filename)[0]
    SLO = get_slo(method, tight=False, all_methods=False)

    df = convert_to_dataframe(data, init=True)
    df = calculate_throughput_dynamic(df)
    df = calculate_goodput_dynamic(df, SLO)
    df = calculate_loadshedded(df)
    df = calculate_ratelimited(df)
    df = calculate_tail_latency_dynamic(df, window_size=latency_window_size)
    plot_timeseries_split(df, filename, computationTime, ax1, ax2, is_first_col)

def plot_timeseries_split(df, filename, computation_time, ax1, ax2, is_first_col):
    mechanism = re.findall(r"control-(\w+)-", filename)[0]
    
    # colors = ['#34a853', '#ea4335', '#4285f4', '#fbbc05']
    # colors = ['#666666', '#000000', '#999999', '#cccccc']  # Different shades of gray
    colors = ['#66c2a5', '#e41a1c', '#ffcc33', '#4c99ff']  # Green, Red, Yellow-Orange, Blue
    
    # If ax1 is None, this is a 1xN plot; adjust plotting for throughput/goodput only
    if ax1 is None:
        for ax in [ax2]:
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
            
        # Set y-axis label only for the first subplot
        if is_first_col:
            ax2.set_ylabel('Throughput\n(kRPS)', color='tab:blue')
        
        # Plot throughput and goodput
        ax2.plot(df.index, df['throughput'], 'r-.')
        ax2.plot(df.index, df['goodput'], color='green', linestyle='--')

        load_info = read_load_info_from_json(filename)
        capacity = load_info['load-end']
        
        if df['limited'].sum() > 0:
            df['total_demand'] = df['dropped'] + df['throughput'] + df['limited']
        elif CONSTANT_LOAD:
            df['total_demand'] = capacity
        else:
            df['total_demand'] = load_info['load-start']
            mid_start_time = pd.Timestamp('2000-01-01 00:00:00') + pd.Timedelta(load_info['load-step-duration']) - pd.Timedelta(offset, unit='s')
            df.loc[mid_start_time:, 'total_demand'] = capacity
        
        # Plot goodput, dropped, and rate limited areas
        ax2.fill_between(df.index, 0, df['goodput'], color=colors[0], label='Goodput')
        ax2.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], label='SLO Violation')
        ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], label='Dropped')
        
        ax2.plot(df.index, df['throughput'], 'b-', label='Throughput')
        ax2.plot(df.index, df['total_demand'], color='c', linestyle='-.', label='Total Demand')

        # Format the x-axis and y-axis
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 6000)
        ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax2.xaxis.set_major_formatter(md.DateFormatter('%S'))
        ax2.set_xlabel('Time (second)')
        
    else:
        for ax in [ax1, ax2]:
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)

        if is_first_col:
            ax1.set_ylabel('Latencies\n(ms)', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.plot(df.index, np.maximum(0.001, df['latency_ma'] - computation_time), linestyle='--', label='Average Latency (e2e)')
        ax1.plot(df.index, np.maximum(0.001, df['tail_latency'] - computation_time), linestyle='-.', label='95% Tail Latency (e2e)')
        ax1.set_yscale('log')
        ax1.set_ylim(1, 500)
        ax1.axhline(y=SLO - computation_time, color='c', linestyle='-.', label='SLO')

        df = df.fillna(0)

        if is_first_col:
            ax2.set_ylabel('Throughput\n(kRPS)', color='tab:blue')
        ax2.plot(df.index, df['throughput'], 'r-.')
        ax2.plot(df.index, df['goodput'], color='green', linestyle='--')
        
        load_info = read_load_info_from_json(filename)
        capacity = load_info['load-end']
        
        if df['limited'].sum() > 0:
            df['total_demand'] = df['dropped'] + df['throughput'] + df['limited']
        elif CONSTANT_LOAD:
            df['total_demand'] = capacity
        else:
            df['total_demand'] = load_info['load-start']
            mid_start_time = pd.Timestamp('2000-01-01 00:00:00') + pd.Timedelta(load_info['load-step-duration']) - pd.Timedelta(offset, unit='s')
            df.loc[mid_start_time:, 'total_demand'] = capacity

        if mechanism != 'baseline':
            ax2.plot(df.index, df['total_demand'], color='c', linestyle='-.')

        ax2.fill_between(df.index, 0, df['goodput'], color=colors[0], label='Goodput')
        ax2.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], label='SLO Violation')
        ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], label='Dropped')
        
        if mechanism != 'baseline' and mechanism != 'dagor' and mechanism != 'topdown':
            ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color='tab:blue', label='Rate Limited', alpha=0.2)
        
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 9000)
        if hotel:
            ax2.set_ylim(0, 11000)
        ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax2.xaxis.set_major_formatter(md.DateFormatter('%S'))
        # if hotel:
        ax2.set_xlabel('Time (second)')

        goodputAve, goodputStd = calculate_average_var_goodput(filename, SLO)
        # ax2.set_title(f"The Goodput has Mean: {goodputAve} and Std: {goodputStd}")

        latency_99th = read_tail_latency(filename)
        average_99th = df['tail_latency'].mean()
        lat95 = df[(df['status'] == 'OK')]['latency'].quantile(0.95)
        # ax1.set_title(f"95-tile Latency over Time: {round(lat95, 2)} ms")


def motivate_plot():

    # Define the filenames and control mechanisms specific to the "motivate" flag
    filenames = [
        "social-compose-control-dagor-parallel-capacity-5000-0813_0209.json",
        "social-compose-control-breakwater-parallel-capacity-5000-0814_1938.json",
        "social-compose-control-topdown-parallel-capacity-5000-0901_1052.json"
    ]
    
    # append /z/rajomon-nsdi/ to the filename
    filenames = [f"/z/rajomon-nsdi/ghz-results/{filename}" for filename in filenames]
    
    control_mechanisms = ["Dagor", "Breakwater", "TopFull"]

    # Create a 1x3 grid for plotting goodput/throughput figures
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5, 3), sharex=True)

    for i, filename in enumerate(filenames):
        ax = axs[i]
        is_first_col = (i == 0)
        
        # Analyze and plot data for each file
        analyze_data(filename, None, ax, is_first_col)
        
        # Remove left ticks and labels for the second and third columns
        if i > 0:
            ax.tick_params(left=False, labelleft=False)
        
        # Set the column title below each subplot
        ax.set_title(control_mechanisms[i], y=-0.6)

    # Add legend for the second row (goodput/throughput)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.1)
    timestamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d")
    filetosave = f'combined_motivate_{timestamp}.pdf'
    plt.savefig(filetosave, dpi=300, bbox_inches='tight')
    print(f"Saved the plot to {filetosave}")

    plt.show()
    plt.close()


if __name__ == '__main__':
    motivate = True
    if motivate:
        motivate_plot()
        exit(0)

    if hotel:
        filenames = [
            "social-search-hotel-control-rajomon-parallel-capacity-10000-0824_2253.json",
            "social-search-hotel-control-breakwater-parallel-capacity-10000-0826_2030.json",
            # "social-search-hotel-control-breakwaterd-parallel-capacity-10000-0822_1857.json",
            "social-search-hotel-control-breakwaterd-parallel-capacity-10000-0822_1854.json",
            "social-search-hotel-control-dagor-parallel-capacity-10000-0822_1712.json",
            "social-search-hotel-control-topdown-parallel-capacity-10000-0903_2224.json"
        ]
        control_mechanisms = ["Rajomon", "Breakwater", "Breakwaterd", "Dagor", "TopFull"]
    else:
        filenames = [
            f"social-{api}-control-rajomon-parallel-capacity-8000-0825_1055.json" for api in ["compose", "user-timeline", "home-timeline"]
        ]
        control_mechanisms = ["Compose", "User Timeline", "Home Timeline"]

    if hotel:
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [2, 3]})
    else: 
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 3.8), sharex=True, gridspec_kw={'height_ratios': [2, 3]})

    for i, filename in enumerate(filenames):
        alibaba = "S_" in filename
        method = re.findall(r"social-(.*?)-control", filename)[0]
        timestamp = re.findall(r"-\d+_\d+", filename)[0]
        SLO = get_slo(method, tight=False, all_methods=False)
        print(f'[INFO] SLO for {method} is {SLO} ms')
        noPlot = len(sys.argv) > 5 and sys.argv[5].lower() == 'no-plot'
        ax1, ax2 = axs[0, i], axs[1, i]
        is_first_col = (i == 0)
        analyze_data(filename, ax1, ax2, is_first_col)

        if i > 0:
            ax1.tick_params(left=False, labelleft=False)
            ax2.tick_params(left=False, labelleft=False)

        # Set the column title below the 2nd row
        ax2.set_title(control_mechanisms[i], y=-0.6)

    # Legends for the first and second rows
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.66), ncol=5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.35)
    # timestamp is today's date
    timestamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d")
    filetosave = f'combined_experiments_{timestamp}.pdf'
    plt.savefig(filetosave, dpi=300, bbox_inches='tight')
    print(f"Saved the plot to {filetosave}")
    if not noPlot:
        plt.show()
    plt.close()

