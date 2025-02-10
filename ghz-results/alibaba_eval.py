import glob, os, sys, re, json
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from collections import defaultdict
import numpy as np

from slo import get_slo
from utils import (
    # load_data,
    load_data_from_csv,
)

# ToDo: should change the 95th_percentile to the max of the warmup period and experiment period 

parameter_files = {
    'search-hotel': {
        'dagor': 'bopt_False_dagor_search-hotel_gpt1-best.json',  # works, no need to re-tune
        # 'breakwater': 'bopt_False_breakwater_search-hotel_gpt1-bestl.json',  # works, but should be replaced with gpt0-10000_01-02
        # 'breakwater': 'bopt_False_breakwater_search-hotel_gpt0-10000_01-02.json',  # seems to work, but need to re-tune
        'breakwater': 'bopt_False_breakwater_search-hotel_gpt0-10000_01-05.json',  # works.
        # 'breakwaterd': 'bopt_False_breakwaterd_search-hotel_gpt1-bestl.json',  # not working, need to re-tune
        # 'breakwaterd': 'bopt_False_breakwaterd_search-hotel_gpt0-10000_01-03.json',  # works for single interface, need to re-tune for all
        'breakwaterd': 'bopt_False_breakwaterd_search-hotel_gpt0-10000_01-08.json',  # works for single interface, need to re-tune for all
        'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-bestl.json',
        # 'rajomon': 'bopt_False_rajomon_search-hotel_decay.json',
    },
    'compose': {
        # 'dagor': 'bopt_False_dagor_compose_gpt1-bestl.json',  # not working, need to re-tune
        'dagor': 'bopt_False_dagor_compose_gpt0-6000_01-04.json',
        # 'breakwater': 'bopt_False_breakwater_compose_gpt1-bestl.json',  # works, but should be replaced with gpt0-6000_01-02
        # 'breakwater': 'bopt_False_breakwater_compose_gpt0-6000_01-04.json',
        'breakwater': 'bopt_False_breakwater_compose_gpt0-6000_01-05.json',
        # 'breakwaterd': 'bopt_False_breakwaterd_compose_gpt1-bestl.json',  # not working, need to re-tune
        'breakwaterd': 'bopt_False_breakwaterd_compose_gpt0-6000_01-02.json',  # working for compose but not all, need to re-tune
        'rajomon': 'bopt_False_rajomon_compose_gpt1-bestl.json',
    },
    # 'search-hotel': {
    #     # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-12000_12-22.json',
    #     # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-16000_12-26.json',
    #     # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-16000_12-25.json',
    #     'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-best-4.json',
    #     # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-14000_12-26.json',
    #     # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-12000_12-24.json',
    #     # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-12000_12-28.json',
    #     'dagor': 'bopt_False_dagor_search-hotel_gpt1-12000_12-24.json',
    #     'breakwater': 'bopt_False_breakwater_search-hotel_gpt1-12000_12-24.json',
    #     'breakwaterd': 'bopt_False_breakwaterd_search-hotel_gpt1-12000_12-24.json',
    # },
    # 'compose': {
    #     'rajomon': 'bopt_False_rajomon_compose_gpt1-6000_12-21.json',
    #     'dagor': 'bopt_False_dagor_compose_gpt1-6000_12-22.json',
    #     'breakwater': 'bopt_False_breakwater_compose_gpt1-6000_12-29.json',
    #     'breakwaterd': 'bopt_False_breakwaterd_compose_gpt1-6000_12-28.json',
    # },
    'alibaba': {
        'dagor': 'bopt_False_dagor_all-alibaba_gpt1-10000_09-13.json',
        'breakwater': 'bopt_False_breakwater_all-alibaba_gpt1-10000_09-12.json',
        'breakwaterd': 'bopt_False_breakwaterd_all-alibaba_gpt1-10000_09-12.json',
        'rajomon': 'bopt_False_rajomon_all-alibaba_gpt1-10000_09-12.json',
    },
}

# for reserve-hotel, the parameter files are the same as search-hotel, for user-timeline and home-timeline, the parameter files are the same as compose
parameter_files['reserve-hotel'] = parameter_files['search-hotel']
parameter_files['user-timeline'] = parameter_files['compose']
parameter_files['home-timeline'] = parameter_files['compose']

for api in ['S_102000854', 'S_149998854', 'S_161142529']:
    parameter_files[api] = parameter_files['alibaba']


def plot_error_bars(ax, subset, metric, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False, error_bar=True, max_x_range=None):
    try:
        if error_bar:
            # fill missing y values with 0 for x values that exist in the dataframe
            ax.errorbar(subset['Load'], subset[metric], yerr=subset[metric + ' std'], fmt=control_markers[control],
                        color=colors[control], linestyle=lineStyles[control], label=labelDict[control],
                        linewidth=2, capsize=5)
        else:
            ax.plot(subset['Load'], subset[metric], label=labelDict[control], color=colors[control], linestyle=lineStyles[control], marker=control_markers[control], linewidth=2)
    except:
        print(f"Error plotting {metric} for {control}, error message: {sys.exc_info()[0]}")
    if max_x_range:
        ax.set_xlim(9000, max_x_range+1000)
    # if add_hline:
    #     if hline_label:
    #     else:
    #         ax.axhline(y=SLO, color='c', linestyle='-.')

def format_ticks(value, tick_number):
    # first, divide the value by 1000, then
    # reture 1 digit after the decimal point (float)
    tick = f'{value/1000:.1f}'
    # if tick is actually an integer, return it as an integer
    if value % 1000 == 0:
        return f'{int(value/1000)}'
    return tick

def setup_axes(axs, interfaces, ali_dict, alibaba_combined, df, in_plot_legend=False, fig=None):
    if alibaba_combined:
        for i, interface in enumerate(interfaces):
            iname = ali_dict[interface]
            ax1, ax2 = axs[:, i]
            ax1.set_title(f'{iname}')

            # **Change 1**: Set x-ticks to match the data points
            xticks = sorted(df['Load'].unique())  # New line added
            ax1.set_xticks(xticks)  # New line added
            ax2.set_xticks(xticks)  # New line added

            # **Change 2**: Use MaxNLocator to reduce the density of grid lines
            nbins = 4 if 'hotel' in method else 5
            if 'alibaba' in method:
                nbins = 4
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=nbins))  # Reduces x-axis grid density to 5 ticks
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins))

            ax1.xaxis.set_major_formatter(FuncFormatter(format_ticks))
            ax2.xaxis.set_major_formatter(FuncFormatter(format_ticks))
            ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))

            # set y limits for the latency subplots to be 0 to 800ms
            
            # if method is all alibaba, set the y limits for ax1 to be 0 - 800
            if method == 'all-alibaba':
                ax1.set_ylim(0, 1000)
                ax2.set_ylim(0, 6000)
            # else:
            #     ax1.set_ylim(1, 800)
            #     ax2.set_ylim(0, 5000)

            if method == 'all-social':
                ax1.set_ylim(1, 900)
                ax2.set_ylim(0, 10000)
            if method == 'both-hotel':
                ax1.set_ylim(1, 900)
                ax2.set_ylim(0, 6000)


            ax1.grid(True)
            ax2.grid(True)
            # make latency subplot y-axis log scale
            # ax1.set_yscale('log')
            ax2.set_xlabel('Load (kRPS)')
            # if first column, set the y label
            if i == 0:
                ax1.set_ylabel('95th Tail\nLatency (ms)')
                ax2.set_ylabel('Goodput (kRPS)')
                if method == 'all-alibaba':
                    ax1.set_ylabel('95th Tail\nLatency (ms)')
                    # ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
                    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}'))

        if in_plot_legend:
            # add the legend to the first subplot
            axs[0][1].legend(frameon=False, loc='upper left')
        else:

            handles, labels = axs[0][1].get_legend_handles_labels()

            # if len(interfaces) > 2:
            #     axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.15, 1.6), ncol=5)
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3)
            # else:
            #     axs[0][1].legend(frameon=False, loc='upper center', bbox_to_anchor=(-0.1, 1.8), ncol=3)
        
        # axs[0][1].legend(frameon=False, loc='upper left')  # Ensure the SLO legend is added to the first subplot
        # make the gap between the subplots smaller

        # remove y ticks for the second column of subplots
        for i in range(1, len(interfaces)):
            axs[0][i].tick_params(labelleft=False)
            axs[1][i].tick_params(labelleft=False)
        
        # remove the space between the subplots
        plt.subplots_adjust(hspace=0.01)
        plt.subplots_adjust(wspace=0.01) 
    else:
        ax1, ax2 = axs
        ax1.set_xlabel('Load (kRPS)')
        # ax1.set_title('Load vs Tail Latency')
        ax1.grid(True)
        ax1.set_ylabel('95th Tail Latency (ms)')
        ax2.set_ylabel('Goodput (kRPS)')
        # ax2.set_title('Load vs Goodput')
        ax2.set_xlabel('Load (kRPS)')
        ax2.grid(True)

        # **Change 1**: Set x-ticks to match the data points
        xticks = sorted(df['Load'].unique())  # New line added
        ax1.set_xticks(xticks)  # New line added
        ax2.set_xticks(xticks)  # New line added

        ax1.xaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
        # ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        # ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.1, 1.4), ncol=5)
        # ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Get handles and labels from the first subplot
        handles, labels = ax1.get_legend_handles_labels()

        # Create a single legend
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3)

        # fig.subplots_adjust(top=0.5)
        fig.subplots_adjust(wspace=0.25)
        # ax2.yaxis.set_label_position("right")


def setup_axes_single_row(axs, interfaces, ali_dict):

    for i, interface in enumerate(interfaces):
        iname = ali_dict[interface]
        ax = axs[i]
        ax.set_title(f'{iname}')
        if i == 0:
            ax.set_ylabel('95th Tail\nLatency (ms)')
        else:
            ax.tick_params(labelleft=False)  # Remove y-ticks for 2nd and 3rd subplots
        ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax.set_ylim(0, 2400)
        if i > 0:  # Share y-axis with the first plot
            ax.sharey(axs[0])
 
    # Setup for the aggregated goodput plot
    ax_agg_goodput = axs[len(interfaces)]
    ax_agg_goodput.set_title('Aggregated Goodput')
    ax_agg_goodput.set_ylabel('Goodput (RPS)')
    ax_agg_goodput.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax_agg_goodput.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    axs[0].legend(frameon=False, loc='upper left')  # Ensure the SLO legend is added to the first subplot
    plt.subplots_adjust(wspace=0.2)  # Adjust space between subplots
    plt.subplots_adjust(hspace=0.03)


def plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency):
    ax1, ax2 = axs
    for control in control_mechanisms:
        subset = df[df['overload_control'] == control]
        SLO = get_slo(method=method, tight=tightSLO, all_methods=False)
        for latency in whatLatency:
            plot_error_bars(ax1, subset, latency, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=True)
        plot_error_bars(ax2, subset, 'Goodput', control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False)
    ax1.axhline(y=SLO, color='c', linestyle='-.', label='SLO')


def plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency):
    error_bar = ('motivate' not in interfaces[0])
    for i, interface in enumerate(interfaces):
        ax1, ax2 = axs[:, i]
        SLO = get_slo(method=interface, tight=tightSLO, all_methods=False)
        for latency in whatLatency:
            for control in control_mechanisms:
                subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
                plot_error_bars(ax1, subset, latency, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=True, error_bar=error_bar)
        ax1.axhline(y=SLO, color='c', linestyle='-.', label='SLO')
        for control in control_mechanisms:
            subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
            plot_error_bars(ax2, subset, 'Goodput', control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False, error_bar=error_bar)


def plot_single_row_combined(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency):
    slo_label = True
    for i, interface in enumerate(interfaces):
        ax = axs[i]
        SLO = get_slo(method=interface, tight=tightSLO, all_methods=False)
        for control in control_mechanisms:
            subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
            for latency in whatLatency:
                plot_error_bars(ax, subset, latency, control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=slo_label)
                if slo_label:
                    slo_label = False
            ax.set_title(f'Latency - {interface}')
        ax.set_xlabel('Load')

    # Plot the aggregated goodput in the fourth plot
    ax_agg_goodput = axs[len(interfaces)]
    for control in control_mechanisms:
        # agg_goodput = df[df['overload_control'] == control].groupby('Load')['Goodput'].sum()
        # agg_goodput = df[df['overload_control'] == control].groupby('Load')['Goodput'].sum()
        # ax_agg_goodput.plot(agg_goodput.index, agg_goodput.values, label=labelDict[control], color=colors[control], linestyle=lineStyles[control], marker=control_markers[control])
        # for interface in interfaces:
        interface = interfaces[0]
        subset = df[(df['overload_control'] == control) & (df['Request'] == interface)]
        plot_error_bars(ax_agg_goodput, subset, 'Total Goodput', control, colors, lineStyles, labelDict, control_markers, SLO, add_hline=False)
    ax_agg_goodput.set_title('Aggregated Goodput')
    ax_agg_goodput.set_ylabel('Goodput')
    ax_agg_goodput.set_xlabel('Load')
    # Remove legend for the last subplot
    ax_agg_goodput.legend().remove()


def plot_alibaba_eval(df, interfaces):
    if df is None:
        return

    merge_goodput = False
    alibaba_combined = len(interfaces) > 1

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'rajomon']
    whatLatency = ['95th_percentile']

    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'rajomon': '#FF9800',
    }

    control_markers = {
        'plain': 'o',
        'breakwater': 's',
        'breakwaterd': '^',
        'dagor': 'v',
        'rajomon': 'x',
    }

    lineStyles = {
        'plain': '-',
        'breakwater': '--',
        'breakwaterd': '-.',
        'dagor': ':',
        'rajomon': '-',
    }

    labelDict = {
        'plain': 'No Control',
        'breakwater': 'Breakwater',
        'breakwaterd': 'Breakwaterd',
        'dagor': 'Dagor',
        'rajomon': 'Rajomon',
    }

    ali_dict = {
        "S_102000854": "S1",
        "S_149998854": "S2",
        "S_161142529": "S3",
    }

    if alibaba_combined:
        if merge_goodput:
            fig, axs = plt.subplots(1, len(interfaces) + 1, figsize=(10, 3.5), sharex=True, gridspec_kw={'width_ratios': [1]*len(interfaces) + [1.2]}, constrained_layout=True)
            plot_single_row_combined(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)
        else:
            fig, axs = plt.subplots(2, len(interfaces), figsize=(5, 2.5), sharex=True, sharey='row')
            plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)

    if alibaba_combined and merge_goodput:
        setup_axes_single_row(axs, interfaces, ali_dict)
    else:
        setup_axes(axs, interfaces, ali_dict, alibaba_combined, df, in_plot_legend=False, fig=fig)

    save_path = os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/')
    save_name = 'all-alibaba' if alibaba_combined else interfaces[0]
    plt.savefig(f'{save_path}{save_name}-{datetime.now().strftime("%m%d")}.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {save_path}{save_name}-{datetime.now().strftime('%m%d')}.pdf")


def plot_4nodes(df, interfaces, control_mechanisms):
    if df is None:
        return

    motivate_combined = len(interfaces) > 1

    # control_mechanisms = ['dagor', 'breakwater', 'breakwaterd']
    whatLatency = ['95th_percentile']

    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'topdown': '#F44336',
        'rajomon': '#FF9800',
    }

    control_markers = {
        'plain': 'o',
        'breakwater': 's',
        'breakwaterd': '^',
        'dagor': 'v',
        'topdown': '.',
        'rajomon': 'x',
    }

    lineStyles = {
        'plain': '-',
        'breakwater': '--',
        'breakwaterd': '-.',
        'dagor': ':',
        'topdown': '-.',
        'rajomon': '-',
    }

    labelDict = {
        'plain': 'No Control',
        'breakwater': 'Breakwater',
        'breakwaterd': 'Breakwaterd',
        'dagor': 'Dagor',
        'topdown': 'TopFull',
        'rajomon': 'Rajomon',
    }

    ali_dict = {
        "motivate-set": "Set",
        "motivate-get": "Get",
        "search-hotel": "Search Hotel",
        "reserve-hotel": "Reserve Hotel",
        "compose": "Compose Post",
        "user-timeline": "Read UserTimeline",
        "home-timeline": "Read HomeTimeline",
        "S_102000854": "S1",
        "S_149998854": "S2",
        "S_161142529": "S3",
    }

    if motivate_combined:
        fig, axs = plt.subplots(2, len(interfaces), figsize=(2*len(interfaces), 3), sharex=True, constrained_layout=True)
        if 'alibaba' in method:
            fig, axs = plt.subplots(2, len(interfaces), figsize=(2*len(interfaces)-2, 2.8), sharex=True, constrained_layout=True)
        # Share the y-axis for the second row
        for i in range(1, len(interfaces)):
            axs[1, i].sharey(axs[1, 0])
        plot_combined_interfaces(axs, df, interfaces, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(5, 1.8))
        plot_individual_interface(axs, df, control_mechanisms, colors, lineStyles, labelDict, control_markers, whatLatency)

    setup_axes(axs, interfaces, ali_dict, motivate_combined, df, in_plot_legend=False, fig=fig)

    save_path = os.path.expanduser(f'~/Sync/Git/protobuf/ghz-results/')
    save_name = method

    # Adjust margins manually
    # if len(interfaces) > 2:
    #     plt.subplots_adjust(left=0.12, top=0.8, right=0.98)
    # elif len(interfaces) == 2:
    #     plt.subplots_adjust(left=0.15, top=0.77, right=0.98)

    # plt.tight_layout()
    plt.savefig(f'{save_path}{save_name}-{datetime.now().strftime("%m%d")}.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {save_path}{save_name}-{datetime.now().strftime('%m%d')}.pdf")


def load_plot_single_requests():
    global method
    global tightSLO

    method = os.getenv('METHOD', 'both-hotel')
    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'

    print(f"Method: {method}, Tight SLO: {tightSLO}")

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'topdown', 'rajomon']

    csv_file = '~/Sync/Git/protobuf/ghz-results/grouped_single.csv'
    # Load data
    hotel_df = pd.DataFrame()
    interfaces = [method]

    for interface in interfaces:
        df = load_data_from_csv(csv_file, method=interface, given_parameter=parameter_files)
        assert df is not None, f"Dataframe for {interface} is None"
        df['Request'] = interface
        # report the file_count column for each interface and capacity
        df['file_count'] = df['file_count'].astype(int)
        for control in control_mechanisms:
            subset = df[(df['Request'] == interface) & (df['overload_control'] == control)]
            print(f"{interface} {control} file_count: {subset['file_count'].values}")
        hotel_df = pd.concat([hotel_df, df])

    df = hotel_df
    # remove the rows with Load > 24000
    df = df[df['Load'] < 15000] if method == 'compose' else df[(df['Load'] < 19000) & (df['Load'] > 4000)]
    # df = df[df['Load'] > 1500]
    # and only factors of 2000
    df = df[df['Load'] % 2000 == 0] if 'hotel' in method else df[df['Load'] % 2000 == 1000]

    # df = df[df['Load'] > 2000]
    plot_4nodes(df, interfaces, control_mechanisms)

    summarize_oc_performance(df, control_mechanisms)


def load_plot_hotel(social=False, alibaba=False):
    global method
    global tightSLO

    method = os.getenv('METHOD', 'both-hotel')
    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'

    print(f"Method: {method}, Tight SLO: {tightSLO}")

    # experiment time ranges
    time_ranges = {
        'search-hotel': [
            ('0613_1200', '0617_0000'),
        ],
        'reserve-hotel': [
            ('0613_1200', '0617_0000'),
        ],
    }

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'topdown', 'rajomon']

    csv_file = '~/Sync/Git/protobuf/ghz-results/grouped_hotel.csv'
    # Load data
    hotel_df = pd.DataFrame()
    # interfaces = ['search-hotel', 'reserve-hotel'] 
    # interfaces = ['compose', 'user-timeline', 'home-timeline'] if social else ['search-hotel', 'reserve-hotel']
    if social:
        interfaces = ['compose', 'user-timeline', 'home-timeline']
    elif alibaba:
        interfaces = ['S_102000854', 'S_149998854', 'S_161142529']
    else:
        interfaces = ['search-hotel', 'reserve-hotel']

    for interface in interfaces:
        df = load_data_from_csv(csv_file, method=interface, given_parameter=parameter_files)
        assert df is not None, f"Dataframe for {interface} is None"
        # keep only RAJOMON_TRACK_PRICE false
        # df = df[df['RAJOMON_TRACK_PRICE'] == False]
        df['Request'] = interface
        # report the file_count column for each interface and capacity
        df['file_count'] = df['file_count'].astype(int)
        for control in control_mechanisms:
            subset = df[(df['Request'] == interface) & (df['overload_control'] == control)]
            print(f"{interface} {control} file_count: {subset['file_count'].values}")
        hotel_df = pd.concat([hotel_df, df])

    df = hotel_df
    # remove the rows with Load > 24000
    if social:
        df = df[df['Load'] <= 10000] 
    elif alibaba:
        df = df[df['Load'] <= 10000]
    else:
        df = df[df['Load'] < 17000]
    df = df[df['Load'] > 1500]
    # and only factors of 2000
    df = df[df['Load'] % 2000 == 0] # if not social else df[df['Load'] % 1000 == 0]
    if 'alibaba' in method:
        df = df[df['Load'] >= 4000]
    plot_4nodes(df, interfaces, control_mechanisms)

    summarize_oc_performance(df, control_mechanisms)

def load_plot_4nodes():
    global method
    global tightSLO

    motivate_combined = 'both' in method

    method = os.getenv('METHOD', 'motivate-set')
    tightSLO = os.getenv('TIGHTSLO', 'False').lower() == 'true'

    print(f"Method: {method}, Tight SLO: {tightSLO}")

    # experiment time ranges
    time_ranges = {
        'motivate-set': [
            ('0607_0236', '0607_1300'),
        ],
        'motivate-get': [
            ('0607_0236', '0607_1300'),
        ],
    }

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd']

    parameter_files = {
        'motivate-set': {
            control: f'bopt_False_{control}_motivate-set_gpt1-30000_06-10.json' for control in control_mechanisms
        }, 
    } if 'monotonic' in method else {
        'motivate-set': {
            control: f'bopt_False_{control}_motivate-set_gpt1-30000_06-07.json' for control in control_mechanisms
        },
    }
        # motivate-get uses the same parameter files as motivate-set
    parameter_files['motivate-get'] = {
        k: v for k, v in parameter_files['motivate-set'].items()
    }

    # csv_file = '~/Sync/Git/protobuf/ghz-results/grouped_4n_monotonic.csv' if 'monotonic' in method else '~/Sync/Git/protobuf/ghz-results/grouped_4n.csv'
    csv_file = '~/Sync/Git/protobuf/ghz-results/grouped_hotel.csv'

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
    plot_4nodes(df, interfaces, control_mechanisms)


def load_plot_alibaba():
    global method
    global tightSLO
    # global SLO

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
    #         'rajomon': 'bopt_False_rajomon_S_102000854_gpt1-10000_05-29.json',
    #     },
    #     'S_149998854': {
    #         'breakwaterd': 'bopt_False_breakwaterd_S_149998854_gpt1-10000_05-29.json',
    #         'breakwater': 'bopt_False_breakwater_S_149998854_gpt1-10000_05-29.json',
    #         'dagor': 'bopt_False_dagor_S_149998854_gpt1-10000_05-29.json',
    #         'rajomon': 'bopt_False_rajomon_S_149998854_gpt1-10000_05-29.json',
    #     },
    #     'S_161142529': {
    #         'breakwaterd': 'bopt_False_breakwaterd_S_161142529_gpt1-10000_05-29.json',
    #         'breakwater': 'bopt_False_breakwater_S_161142529_gpt1-10000_05-29.json',
    #         'dagor': 'bopt_False_dagor_S_161142529_gpt1-10000_05-29.json',
    #         'rajomon': 'bopt_False_rajomon_S_161142529_gpt1-10000_05-29.json',
    #     },
    # }

    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'rajomon']

    # parameter_files = {
    #     'S_102000854': {
    #         control: f'bopt_False_{control}_S_102000854_gpt1-10000_06-06.json' for control in control_mechanisms
    #     },
    #     'S_149998854': {
    #         control: f'bopt_False_{control}_S_149998854_gpt1-10000_06-06.json' for control in control_mechanisms
    #     },
    #     'S_161142529': {
    #         control: f'bopt_False_{control}_S_161142529_gpt1-10000_06-06.json' for control in control_mechanisms
    #     },
    # }
    # # # replace S_14 rajomon with the new json
    # parameter_files['S_149998854']['rajomon'] = 'bopt_False_rajomon_S_149998854_gpt1-10000_05-29.json'
    # parameter_files['S_161142529']['rajomon'] = 'bopt_False_rajomon_S_161142529_gpt1-10000_06-05.json'

    # different_spike = {
    #     'S_102000854': {
    #         # control: f'bopt_False_{control}_S_102000854_gpt1-12000_05-30.json' for control in control_mechanisms
    #         control: f'bopt_False_{control}_S_102000854_gpt1-12000_06-10.json' for control in control_mechanisms
    #     },
    #     'S_149998854': {
    #         # control: f'bopt_False_{control}_S_149998854_gpt1-9000_06-03.json' for control in control_mechanisms
    #         control: f'bopt_False_{control}_S_149998854_gpt1-10000_06-10.json' for control in control_mechanisms
    #     },
    #     'S_161142529': {
    #         # control: f'bopt_False_{control}_S_161142529_gpt1-12000_05-30.json' for control in control_mechanisms
    #         control: f'bopt_False_{control}_S_161142529_gpt1-12000_06-10.json' for control in control_mechanisms
    #     },
    # }
    # parameter_files = different_spike

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
            # method = interface
            SLO = get_slo(method=interface, tight=tightSLO, all_methods=False)
            df = None
            if method == 'all-alibaba':
                parameter_files = {
                    api: {
                        control: f'bopt_False_{control}_S_149998854_gpt1-10000_06-10.json' for control in control_mechanisms
                    } for api in interfaces
                }
                # df = load_data_from_csv(f'~/Sync/Git/protobuf/ghz-results/grouped_all_ali.csv', method=interface, given_parameter=parameter_files)
                df = load_data_from_csv(f'~/Sync/Git/protobuf/ghz-results/grouped_all_ali.csv', method=interface, given_parameter=parameter_files)
            elif method == 'ali3':
                df = load_data_from_csv(f'~/Sync/Git/protobuf/ghz-results/grouped_ali.csv', method=interface, given_parameter=parameter_files)
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
    df = df[df['Load'] < 12000]
    plot_alibaba_eval(df, interfaces)

def summarize_oc_performance(df, control_mechanisms):
    """
    Summarizes the goodput and latency for each OC framework by calculating the 
    average goodput and latency for rows where Load is greater than the median Load.
    Additionally, computes the average goodput gain and latency reduction of Rajomon
    over all other baselines.
    """
    median_load = df['Load'].median()

    # Filter df to only consider rows with Load greater than the median
    df_filtered = df[df['Load'] > median_load]

    print(f"\nSummary for Loads > {median_load}:\n")

    summary_dict = {}
    
    for control in control_mechanisms:
        # Filter the DataFrame for the current OC framework
        subset = df_filtered[df_filtered['overload_control'] == control]

        if subset.empty:
            print(f"No data for {control} in this load range.")
            continue

        # Calculate average goodput and latency
        avg_goodput = subset['Goodput'].mean()
        avg_latency = subset['95th_percentile'].mean()

        # Store in dictionary
        summary_dict[control] = {
            'average_goodput': avg_goodput,
            'average_latency': avg_latency
        }

        print(f"{control} - Average Goodput: {avg_goodput:.2f}, Average 95th Percentile Latency: {avg_latency:.2f} ms")

    # Now calculate the percentage gain of Rajomon over other baselines
    rajomon_goodput = summary_dict.get('rajomon', {}).get('average_goodput', 0)
    rajomon_latency = summary_dict.get('rajomon', {}).get('average_latency', 0)

    total_goodput_gain = 0
    total_latency_reduction = 0
    baseline_count = 0

    if rajomon_goodput and rajomon_latency:
        for control in control_mechanisms:
            if control != 'rajomon' and control in summary_dict:
                baseline_goodput = summary_dict[control]['average_goodput']
                baseline_latency = summary_dict[control]['average_latency']

                goodput_gain = ((rajomon_goodput - baseline_goodput) / baseline_goodput) * 100 if baseline_goodput else 0
                latency_reduction = ((baseline_latency - rajomon_latency) / baseline_latency) * 100 if baseline_latency else 0

                total_goodput_gain += goodput_gain
                total_latency_reduction += latency_reduction
                baseline_count += 1

                print(f"\nRajomon vs {control}:")
                print(f"Goodput Gain: {goodput_gain:.2f}%")
                print(f"Latency Reduction: {latency_reduction:.2f}%")

        # Calculate average gains over the 4 baselines
        avg_goodput_gain = total_goodput_gain / baseline_count if baseline_count > 0 else 0
        avg_latency_reduction = total_latency_reduction / baseline_count if baseline_count > 0 else 0

        print(f"\nAverage Goodput Gain over baselines: {avg_goodput_gain:.2f}%")
        print(f"Average Latency Reduction over baselines: {avg_latency_reduction:.2f}%")


if __name__ == '__main__':
    method = os.getenv('METHOD', 'both-motivate')
    if 'motivate' in method:
        load_plot_4nodes()
    elif method == 'all-alibaba' or method == 'ali3':
        load_plot_hotel(alibaba=True)
    elif method == 'both-hotel':
        load_plot_hotel()
    elif method == 'all-social':
        load_plot_hotel(social=True)
    elif method == 'compose' or method == 'search-hotel':
        load_plot_single_requests()
