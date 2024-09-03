import json, re, datetime
import sys
import re
import os
import pytz
import pandas as pd
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
latency_window_size = '200ms'
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

    df = convert_to_dataframe(data, init=True)
    df = calculate_throughput_dynamic(df)
    df = calculate_goodput_dynamic(df, SLO)
    df = calculate_loadshedded(df)
    df = calculate_ratelimited(df)
    df = calculate_tail_latency_dynamic(df)
    plot_timeseries_split(df, filename, computationTime, ax1, ax2, is_first_col)

def plot_timeseries_split(df, filename, computation_time, ax1, ax2, is_first_col):
    mechanism = re.findall(r"control-(\w+)-", filename)[0]
    
    colors = ['#34a853', '#ea4335', '#4285f4', '#fbbc05']
    
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
        ax2.set_ylabel('Throughput\n(RPS)', color='tab:blue')
    ax2.plot(df.index, df['throughput'], 'r-.', alpha=0.2)
    ax2.plot(df.index, df['goodput'], color='green', linestyle='--', alpha=0.2)
    
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

    ax2.fill_between(df.index, 0, df['goodput'], color=colors[0], alpha=0.8, label='Goodput')
    ax2.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], alpha=0.8, label='SLO Violation')
    ax2.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[2], alpha=0.8, label='Dropped')
    
    if mechanism != 'baseline' and mechanism != 'dagor':
        ax2.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], where=df['total_demand'] > df['throughput'] + df['dropped'], color='tab:blue', alpha=0.3, label='Rate Limited')
    
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 12000)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax2.xaxis.set_major_formatter(md.DateFormatter('%S'))
    if hotel:
        ax2.set_xlabel('Time (second)')

    goodputAve, goodputStd = calculate_average_var_goodput(filename, SLO)
    # ax2.set_title(f"The Goodput has Mean: {goodputAve} and Std: {goodputStd}")

    latency_99th = read_tail_latency(filename)
    average_99th = df['tail_latency'].mean()
    lat95 = df[(df['status'] == 'OK')]['latency'].quantile(0.95)
    # ax1.set_title(f"95-tile Latency over Time: {round(lat95, 2)} ms")

if __name__ == '__main__':
    if hotel:
        # filenames = [
        #     "social-search-hotel-control-charon-parallel-capacity-10000-0703_2022.json",
        #     # "social-search-hotel-control-breakwater-parallel-capacity-10000-0705_1959.json",
        #     "social-search-hotel-control-breakwater-parallel-capacity-10000-0708_2218.json",
        #     # "social-search-hotel-control-breakwaterd-parallel-capacity-10000-0701_0522.json",
        #     "social-search-hotel-control-breakwaterd-parallel-capacity-10000-0708_2123.json",
        #     "social-search-hotel-control-dagor-parallel-capacity-10000-0622_2318.json"
        # ]
        filenames = [
            "social-search-hotel-control-charon-parallel-capacity-10000-0824_2252.json",
            "social-search-hotel-control-breakwater-parallel-capacity-10000-0826_2030.json",
            "social-search-hotel-control-breakwaterd-parallel-capacity-10000-0822_1857.json",
            "social-search-hotel-control-dagor-parallel-capacity-10000-0822_1712.json",
            "social-search-hotel-control-topdown-parallel-capacity-10000-0901_0412.json"
        ]
        control_mechanisms = ["Rajomon", "Breakwater", "Breakwaterd", "Dagor"]
    else:
        filenames = [
            'social-compose-control-charon-parallel-capacity-10000-0711_1656.json',
            'social-user-timeline-control-charon-parallel-capacity-10000-0711_1656.json',
            'social-home-timeline-control-charon-parallel-capacity-10000-0711_1656.json',
        ]
        control_mechanisms = ["Compose", "User Timeline", "Home Timeline"]

    if hotel:
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(8, 4), sharex=True, gridspec_kw={'height_ratios': [2, 3]})
    else: 
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 3), sharex=True, gridspec_kw={'height_ratios': [2, 3]})

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
        ax2.set_title(control_mechanisms[i], y=-0.5)

    # Legends for the first and second rows
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.64), ncol=5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.35)
    # timestamp is today's date
    timestamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d")
    filetosave = f'combined_experiments_{timestamp}.png'
    plt.savefig(filetosave, dpi=300, bbox_inches='tight')
    print(f"Saved the plot to {filetosave}")
    if not noPlot:
        plt.show()
    plt.close()
