import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.dates as md
from slo import get_slo
from utils import calculate_goodput_ave_var, calculate_tail_latency_dynamic, read_load_info_from_json, read_data, read_tail_latency, calculate_goodput_dynamic, calculate_throughput_dynamic, calculate_loadshedded, calculate_ratelimited, read_mean_latency
from visualize import convert_to_dataframe, format_ticks, extract_ownPrices

throughput_time_interval = '100ms'
latency_window_size = '200ms'
offset = 5
oneNode = False
cloudlab = True
capacity = 24000 if oneNode else 5000
computationTime = 0

INTERCEPTOR = os.environ.get('INTERCEPT', 'plain').lower()
cloudlabOutput = r"deathstar_([\w-]+)\.output"
CONSTANT_LOAD = os.environ.get('CONSTANT_LOAD', 'false').lower() == 'true'

if 'CAPACITY' in os.environ:
    capacity = int(os.environ['CAPACITY'])

def format_ticks_k(value, tick_number):
    # first, divide the value by 1000, then
    # reture 1 digit after the decimal point (float)
    # tick = f'{value/1000:.1f}K'
    tick = rf"$\frac{{1}}{{2}}K$"
    # if tick is actually an integer, return it as an integer
    if value % 1000 == 0:
        return f'{int(value/1000)}K'
    return tick

def plot_interfaces(filenames, output_file):
    print("Starting plot_interfaces...")
    # the ratio of the row heights is 1:2
    fig, axs = plt.subplots(2, 3, figsize=(5, 2.5), sharey='row', gridspec_kw={'hspace': 0.5}, 
                            height_ratios=[3, 4])

    for i, filename in enumerate(filenames):
        print(f"Processing file {i+1}/{len(filenames)}: {filename}")
        df = process_data(filename)
        if df is not None:
            print(f"DataFrame for {filename} loaded successfully.")
            plot_subplot(axs[0, i], df, 'latency', filename, remove_y_labels=i > 0, show_x_labels=False)

            # show legend for both subplots but only once for the first subplot in each row
            if i == 0:
                axs[0, i].legend(loc='upper left', ncol=3, frameon=False, prop={'style':'italic'}, bbox_to_anchor=(-0.6, 1.5), fontsize=5)
                
            plot_subplot(axs[1, i], df, 'throughput', filename, remove_y_labels=i > 0, share_x=True)
            # show legend for both subplots but only once for the first subplot in each row
            if i == 0:
                axs[1, i].legend(loc='upper left', ncol=4, frameon=False, prop={'style':'italic'}, bbox_to_anchor=(-0.7, 1.5), fontsize=4)
        else:
            print(f"Failed to load data for {filename}")
    
    # add vspacing between subplots rows

    # Set column titles (one per column, apply only to the first row)
    column_titles = ["User Timeline", "Home Timeline", "Compose"]
    for i, title in enumerate(column_titles):
        # use fig.text to add text to the figure underneath the subplots
        fig.text(0.25 + i * 0.28, -0.1, title, fontsize=8, ha='center')



    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f"Saving interface plot as {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Interface plot saved as {output_file}")

def plot_prices(filenames, output_file):
    df = process_data(filenames[0])
    print("Starting plot_prices...")
    fig, axs = plt.subplots(1, 2, figsize=(5, 1.5), sharey='row')
    
    print(f"Processing price plot for cloudlab environment...")
    df_price_dict = extract_ownPrices(cloudlabOutput)

    # data_dict is a deep copy
    max_dict = {service_name: df.max().values[0] for service_name, df in df_price_dict.items()}
    # Then, we sort the max_dict by the max value
    max_dict = dict(sorted(max_dict.items(), key=lambda item: item[1], reverse=True))
    # Then, we only keep the top 5 services
    data_dict = {service_name: df for service_name, df in df_price_dict.items() if service_name in list(max_dict.keys())[:3]}

    line_styles = ['--', '-.', '-']
    marker_styles = ['o', 's', 'D']  # Circle, Square, Diamond

    # for service_name, df_price in data_dict.items():
    for i, (service_name, df_price) in enumerate(data_dict.items()):
        df_price = df_price[df_price.index < df.index[-1]]
        moving_average_price = df_price[service_name].rolling(latency_window_size).mean()

        # Only show markers at every second point
        marker_positions = range(0, len(df_price), 10)

        axs[0].plot(
            df_price.index, 
            moving_average_price, 
            label=service_name, 
            linestyle=line_styles[i % len(line_styles)], 
            alpha=0.8, 
            marker=marker_styles[i % len(marker_styles)],
            # markersize=3,
            markevery=marker_positions,  # Show markers at every second data point
            )
    axs[0].set_ylabel('Rajomon Price Per Service')
    axs[0].set_xlabel('Time (second)')
    # put the legend outside the plot on top
    axs[0].legend(loc='upper left', ncol=1, frameon=False, bbox_to_anchor=(0.05, 1.6))
    axs[0].xaxis.set_major_formatter(md.DateFormatter('%S'))
    axs[0].yaxis.set_major_formatter(FuncFormatter(format_ticks_k))
    axs[0].grid(True)

    plot_final_endpoint_prices(axs[1], df_price_dict, df.index[-1])

    plt.subplots_adjust(wspace=0.15)
    print(f"Saving price plot as {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Price plot saved as {output_file}")

def process_data(filename):
    print(f"Reading data from {filename}...")
    try:
        data = read_data(filename)
        df = convert_to_dataframe(data, init=True)
        df = calculate_throughput_dynamic(df)
        method = re.search(r'social-(.*)-control', filename).group(1)
        SLO = get_slo(method, tight=False, all_methods=False)
        df = calculate_goodput_dynamic(df, SLO)
        df = calculate_loadshedded(df)
        df = calculate_ratelimited(df)
        df = calculate_tail_latency_dynamic(df)
        print(f"Data processing complete for {filename}.")
        return df
    except Exception as e:
        print(f"Error processing data for {filename}: {e}")
        return None

def plot_subplot(ax, df, plot_type, filename, remove_y_labels=False, share_x=False, show_x_labels=True):
    print(f"Plotting {plot_type}...")
    method = re.search(r'social-(.*)-control', filename).group(1)
    SLO = get_slo(method, tight=False, all_methods=False)
    
    if plot_type == 'latency':
        if not remove_y_labels:
            ax.set_ylabel('Latencies\n(ms)', color='tab:red')
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax.plot(df.index, np.maximum(0.001, df['latency_ma'] - computationTime), linestyle='--',
                label='Average Latency' if computationTime == 0 else 'Average Latency\nminus computation time')
        ax.plot(df.index, np.maximum(0.001, df['tail_latency'] - computationTime), linestyle='-.',
                label='95% Tail Latency' if computationTime == 0 else '95% Tail Latency\nminus computation time')
        ax.axhline(y=SLO, color='c', linestyle='-.', label='SLO')
        ax.set_ylim(1, 500)
        ax.set_yscale('log')

        ax.set_xticklabels([])

    elif plot_type == 'throughput':
        if not remove_y_labels:
            ax.set_ylabel('Throughput\n(kRPS)', color='tab:blue')
        ax.plot(df.index, df['throughput'], 'r-.', alpha=0.8)
        ax.plot(df.index, df['goodput'], color='green', linestyle='--', alpha=0.8)
        
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

        ax.plot(df.index, df['total_demand'], color='c', linestyle='-.')

        colors = ['#34a853', '#ea4335', '#4285f4', '#fbbc05']
        ax.fill_between(df.index, 0, df['goodput'], color=colors[0], alpha=0.8, label='Goodput')
        ax.fill_between(df.index, df['goodput'], df['throughput'], color=colors[1], alpha=1, label='SLO\nViolation')
        ax.fill_between(df.index, df['throughput'], df['throughput'] + df['dropped'], color=colors[3], alpha=0.8, label='Dropped')

        # fill the area between the dropped and total demand (8000)
        ax.fill_between(df.index, df['throughput'] + df['dropped'], df['total_demand'], color="tab:blue", alpha=0.3, label='Rate\nLimited')

        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_ylim(0, 12000)
        ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax.xaxis.set_major_formatter(md.DateFormatter('%S'))

    if share_x:
        ax.set_xlabel('Time (second)')
    
    ax.grid(True)
    print(f"Completed plotting {plot_type}.")

def plot_final_endpoint_prices(ax, df_price_dict, last_index):
    services = [
        'composepost',
        'hometimeline',
        'usertimeline',
        'socialgraph',
        'poststorage',
        'nginx'
    ]

    topology = {
        'nginx': ['composepost', 'hometimeline', 'usertimeline'],
        'composepost': ['poststorage', 'usertimeline', 'hometimeline'],
        'hometimeline': ['poststorage', 'socialgraph'],
        'usertimeline': ['poststorage'],
        'socialgraph': [],
        'poststorage': []
    }

    final_endpoints = {
        'compose': ['nginx', 'composepost', 'poststorage', 'usertimeline', 'hometimeline'],
        'home-timeline': ['nginx', 'hometimeline', 'poststorage', 'socialgraph'],
        'user-timeline': ['nginx', 'usertimeline', 'poststorage']
    }
    
    line_styles = ['--', '-.', '-', '.']

    print("Plotting final endpoint prices...")
    print(df_price_dict.keys())

    final_prices = {}
    for endpoint, involved_services in final_endpoints.items():
        max_price_series = None
        for service in involved_services:
            if max_price_series is None:
                max_price_series = df_price_dict[service].rolling(latency_window_size).mean()
            else:
                rolling_mean = df_price_dict[service].rolling(latency_window_size).mean()

                # Ensure both have the same shape by aligning lengths
                min_length = min(len(max_price_series), len(rolling_mean))
                max_price_series = np.maximum(
                    max_price_series.iloc[-min_length:], 
                    rolling_mean.iloc[-min_length:]
                )

        final_prices[endpoint] = max_price_series[max_price_series.index < last_index]

    marker_styles = ['o', 's', 'D']  # Circle, Square, Diamond

    # for endpoint, price_series in final_prices.items():
    for i, (endpoint, price_series) in enumerate(final_prices.items()):
        ax.plot(price_series.index, price_series, label=f'{endpoint}', alpha=0.7, linestyle=line_styles[i % len(line_styles)],
                marker=marker_styles[i % len(marker_styles)], 
                # markersize=3, 
                markevery=range(0, len(price_series), 10))
    
    ax.set_ylabel('Final Endpoint Price')
    ax.set_xlabel('Time (second)')
    ax.xaxis.set_major_formatter(md.DateFormatter('%S'))
    ax.legend(loc='upper left', ncol=1, frameon=False, fancybox=True, title_fontsize='medium', prop={'style':'italic'}, bbox_to_anchor=(0.05, 1.6))
    ax.grid(True)
    # ax.yaxis.set_ticks([])
    # ax.yaxis.set_tick_params(length=0)
    # ax.tick_params(axis='y', which='both', length=0)


if __name__ == '__main__':
    filenames = [
        '/z/large-nsdi/results/social-user-timeline-control-rajomon-parallel-capacity-8000-0201_0206.json',
        '/z/large-nsdi/results/social-home-timeline-control-rajomon-parallel-capacity-8000-0201_0206.json',
        '/z/large-nsdi/results/social-compose-control-rajomon-parallel-capacity-8000-0201_0206.json',
        # 'social-compose-control-rajomon-parallel-capacity-8000-0919_1828.json',
        # 'social-user-timeline-control-rajomon-parallel-capacity-8000-0919_1828.json',
        # 'social-home-timeline-control-rajomon-parallel-capacity-8000-0919_1828.json',
        # 'social-compose-control-rajomon-parallel-capacity-10000-0711_1656.json',
        # 'social-user-timeline-control-rajomon-parallel-capacity-10000-0711_1656.json',
        # 'social-home-timeline-control-rajomon-parallel-capacity-10000-0711_1656.json',
    ]
    plot_interfaces(filenames, 'interface_plot.pdf')
    plot_prices(filenames, 'new-price_plot.pdf')
