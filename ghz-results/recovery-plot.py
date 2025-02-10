import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import numpy as np
# import functions from file alibaba-eval.py
sys.path.append('.')
from utils import load_data_from_csv

# Set Seaborn style for plotting
sns.set(style="whitegrid")

def format_ticks(value, tick_number):
    # first, divide the value by 1000, then
    # reture 1 digit after the decimal point (float)
    tick = f'{value/1000:.1f}'
    # if tick is actually an integer, return it as an integer
    if value % 1000 == 0:
        return f'{int(value/1000)}'
    return tick

parameter_files = {
    'search-hotel': {
        'dagor': 'bopt_False_dagor_search-hotel_gpt0-10000_08-23.json',
        'breakwater': 'bopt_False_breakwater_search-hotel_gpt1-10000_08-22.json',
        'breakwaterd': 'bopt_False_breakwaterd_search-hotel_gpt1-10000_08-22.json',
        'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-10000_08-24.json',
        # 'rajomon': 'bopt_False_rajomon_search-hotel_gpt0-no-lazy-10000_08-24.json',
    },
    'compose': {
        'dagor': 'bopt_False_dagor_compose_gpt1-5000_08-15.json',
        'breakwater': 'bopt_False_breakwater_compose_gpt0-5000_08-27.json',
        'breakwaterd': 'bopt_False_breakwaterd_compose_gpt0-5000_08-25.json',
        'rajomon': 'bopt_False_rajomon_compose_gpt1-5000_08-25.json',
    },
    'alibaba': {
        'dagor': 'bopt_False_dagor_S_149998854_gpt1-20000_09-09.json',
        'breakwater': 'bopt_False_breakwater_S_149998854_gpt1-20000_09-09.json',
        'breakwaterd': 'bopt_False_breakwaterd_S_149998854_gpt1-20000_09-09.json',
        # 'rajomon': 'bopt_False_rajomon_S_149998854_gpt1-20000_09-10.json',
        # 'rajomon': 'bopt_False_rajomon_S_149998854_gpt1-10000_09-12.json',
        'rajomon': 'bopt_False_rajomon_all-alibaba_gpt1-10000_09-12.json',
    },
}

# for reserve-hotel, the parameter files are the same as search-hotel, for user-timeline and home-timeline, the parameter files are the same as compose
parameter_files['reserve-hotel'] = parameter_files['search-hotel']
parameter_files['user-timeline'] = parameter_files['compose']
parameter_files['home-timeline'] = parameter_files['compose']

for api in ['S_102000854', 'S_149998854', 'S_161142529']:
    parameter_files[api] = parameter_files['alibaba']


def load_data(input_file):
    """
    Load the experiment recovery data from the given CSV file.
    """
    try:
        df = pd.read_csv(input_file)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_and_group_data(file_path):
    """
    Load experiment data from a CSV file, group by the specified columns,
    and calculate the mean and standard deviation of recovery time and other metrics.
    """
    
    # Load the experiment results from the CSV file
    df = pd.read_csv(file_path)
    
    # Define primary and secondary index columns
    primary_index_columns = ['interface', 'control_scheme', 'capacity', 'concurrency']
    secondary_index_columns = [
        'BREAKWATERD_B', 'RATE_LIMITING', 'BREAKWATERD_SLO', 'DAGOR_QUEUING_THRESHOLD', 
        'PRICE_UPDATE_RATE', 'LAZY_UPDATE', 'DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL', 
        'TOKEN_UPDATE_RATE', 'LOAD_SHEDDING', 'PRICE_STEP', 'BREAKWATER_A', 
        'BREAKWATER_LOAD_SHEDDING', 'BREAKWATERD_INITIAL_CREDIT', 'LATENCY_THRESHOLD', 
        'BREAKWATER_CLIENT_EXPIRATION', 'DAGOR_UMAX', 'BREAKWATERD_A', 
        'BREAKWATERD_LOAD_SHEDDING', 'PRICE_STRATEGY', 'BREAKWATERD_RTT', 
        'BREAKWATER_INITIAL_CREDIT', 'BREAKWATER_B', 'DAGOR_ALPHA', 'INTERCEPT', 
        'BREAKWATERD_CLIENT_EXPIRATION', 'DAGOR_BETA', 'RAJOMON_TRACK_PRICE', 
        'BREAKWATER_SLO', 'BREAKWATER_RTT'
    ]
    
    # Define the metric columns for which we want to calculate the mean and variance
    metric_columns = ['recovery_time']

    # Check for recovery time column and add it to the metric columns
    if 'recovery_time' not in df.columns:
        print("Recovery time column not found in the dataframe.")
        raise ValueError("Recovery time column not found in the dataframe.")
    
    # Filter the dataframe to include only relevant columns
    all_columns = primary_index_columns + secondary_index_columns + metric_columns
    df = df[all_columns]
    
    # Remove rows with missing values in the metric columns before calculating mean/std
    df = df.dropna(subset=metric_columns)

    # Step 1: Convert 'recovery_time' to timedelta if it's a string
    df['recovery_time'] = pd.to_timedelta(df['recovery_time'], errors='coerce')
    # Convert recovery_time (timedelta) to milliseconds
    df['recovery_ms'] = df['recovery_time'].apply(lambda x: x.total_seconds() * 1000 if pd.notnull(x) else None)

    # Debugging: Print to verify if recovery_ms is being correctly computed
    print(df[['recovery_time', 'recovery_ms']])

    # print 10 rows of the grouped dataframe
    print(df.head(10))


    # Group by the primary and secondary index columns
    grouped_df = df.groupby(primary_index_columns + secondary_index_columns)
    
    # Calculate the mean and standard deviation for the relevant metrics
    mean_df = grouped_df['recovery_ms'].mean().reset_index()
    std_df = grouped_df['recovery_ms'].std().reset_index()
    
    # Combine mean and std into a single dataframe for reporting
    result_df = pd.merge(mean_df, std_df, on=primary_index_columns + secondary_index_columns, suffixes=('_mean', '_std'))
    
    return result_df


def group_experiments(df, parameter_files, api_type="hotel", control_mechanisms=None):
    """
    Group experiments by capacity, API, and control, and calculate the average and standard deviation
    of recovery time for each group.
    """
    if control_mechanisms is None:
        control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'topdown', 'rajomon']

    # Define the interfaces for different API types
    if api_type == "social":
        interfaces = ['compose', 'user-timeline', 'home-timeline']
    elif api_type == "alibaba":
        interfaces = ['S_102000854', 'S_149998854', 'S_161142529']
    else:
        interfaces = ['search-hotel', 'reserve-hotel']

    # Filter and group by 'capacity', 'Request' (API), and 'overload_control'
    results = []

    for interface in interfaces:
        for control in control_mechanisms:
            subset = df[(df['Request'] == interface) & (df['overload_control'] == control)]

            if not subset.empty:
                # Group by capacity and calculate mean and std for recovery time
                grouped = subset.groupby('capacity')['recovery_time'].agg(['mean', 'std']).reset_index()
                grouped['Request'] = interface
                grouped['overload_control'] = control

                # Append results
                results.append(grouped)

    # Concatenate all results into a single dataframe
    result_df = pd.concat(results, ignore_index=True)
    
    return result_df


def convert_recovery_time_to_milliseconds(df, recovery_time_col):
    """
    Convert recovery time from timedelta to seconds.
    """
    df[recovery_time_col] = pd.to_timedelta(df[recovery_time_col]).dt.total_seconds() * 1000
    return df


# def plot_recovery_time(df, apis, x_col, y_col, title):
#     # Create a line plot for each API
#     fig, axs = plt.subplots(1, len(apis), figsize=(15, 5), sharey=True)
    
#     for i, api in enumerate(apis):
#         ax = axs[i]
#         api_data = df[df['Request'] == api]
        
#         # Group by capacity instead of time
#         # first, convert time to milliseconds
#         api_data = convert_recovery_time_to_milliseconds(api_data, y_col)

#         api_avg_recovery_time = api_data.groupby(x_col)[y_col].mean().reset_index()

#         ax.plot(api_avg_recovery_time[x_col], api_avg_recovery_time[y_col], marker='o', label=f'{api} Avg Recovery Time')
#         ax.set_xlabel('Capacity (RPS)')
#         ax.set_title(api)
#         ax.grid(True)
    
#     axs[0].set_ylabel('Average Recovery Time (seconds)')
#     fig.suptitle(title)
#     plt.savefig('recovery-time-plot.pdf')
#     print("Plot saved to recovery-time-plot.pdf")
#     plt.show()


def plot_recovery_vs_load(hotel_df, control_mechanisms):
    """
    Plot Recovery Time vs. Load for each control mechanism.
    """
    # Set up the plot
    fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharey=True)

    # Define color and line style mappings for control mechanisms
    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'topdown': '#F44336',
        'rajomon': '#FF9800'
    }

    labelDict = {
        'plain': 'No Control',
        'breakwater': 'Breakwater',
        'breakwaterd': 'Breakwaterd',
        'topdown': 'TopFull',
        'dagor': 'Dagor',
        'rajomon': 'Rajomon',
    }
    markers = {
        'plain': 'o',
        'breakwater': 's',
        'breakwaterd': '^',
        'dagor': 'v',
        'topdown': '.',
        'rajomon': 'x',
    }

    # Filter to include only non-concurrent runs for compose and search-hotel
    filtered_df = hotel_df[
        (hotel_df['Request'].isin(['compose', 'search-hotel'])) &
        (hotel_df['concurrency'] == False)
    ]

    # Create limits for each subplot
    load_limits = {
        'search-hotel': (6000, 18000),
        'compose': (3000, 11000)
    }

    # Iterate over interfaces and control mechanisms
    for i, interface in enumerate(['search-hotel', 'compose']):
        ax = axs[i]
        for control in control_mechanisms:
            subset = filtered_df[
                (filtered_df['Request'] == interface) &
                (filtered_df['overload_control'] == control) &
                (filtered_df['Load'].between(*load_limits[interface]))
            ]
            # for compose only take odd loads (3000, 5000, 7000, 9000) for search-hotel take even loads (6000, 8000, 10000, 12000, 14000, 16000, 18000)
            if interface == 'compose':
                subset = subset[subset['Load'] % 2000 == 1000]
            else:
                subset = subset[subset['Load'] % 2000 == 0]
            if not subset.empty:
                # Plot the data for each interface and control mechanism
                ax.plot(subset['Load'], subset['recovery_ms_mean'],
                        label=labelDict.get(control, control),
                        color=colors.get(control, '#333333'),
                        marker=markers.get(control, 'o'),
                        linewidth=2)
                        # Set labels, title, and limits
        ax.set_xlabel('Load (kRPS)')
        ax.set_title(interface.replace('-', ' ').title())
        ax.set_xlim(load_limits[interface])
        ax.grid(True)

    axs[0].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axs[1].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axs[1].yaxis.set_major_formatter(FuncFormatter(format_ticks))

    # change the x-axis gridlines to be 5
    axs[0].xaxis.set_major_locator(MaxNLocator(nbins=7))
    axs[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
    # set x ticks to be odd numbers of 000
    axs[1].set_xticks(np.arange(3000, 12000, 2000))

    axs[0].set_xlim(5000, 19000)
    axs[1].set_xlim(2000, 12000)

    axs[0].set_ylabel('Recovery Time (s)')

    handles, labels = axs[0].get_legend_handles_labels()

    # Create a single legend
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=3)
    # Customize the legend and plot
    # axs[1].legend(loc='best', frameon=False) let's put the legend outside the plot on top
    plt.tight_layout()

    plt.savefig('recovery-time-vs-load.pdf', bbox_inches='tight')
    print("Plot saved to recovery-time-vs-load.pdf")
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_recovery_vs_load_concurrent(hotel_df, control_mechanisms):
    """
    Plot Recovery Time vs. Load for each control mechanism.
    """
    # Set up the plot
    fig, axs = plt.subplots(1, 5, figsize=(6, 2), sharey=True)

    # Define color and line style mappings for control mechanisms
    colors = {
        'plain': '#F44336',
        'breakwater': '#2196F3',
        'breakwaterd': '#0D47A1',
        'dagor': '#4CAF50',
        'topdown': '#F44336',
        'rajomon': '#FF9800'
    }

    labelDict = {
        'plain': 'No Control',
        'breakwater': 'Breakwater',
        'breakwaterd': 'Breakwaterd',
        'topdown': 'TopFull',
        'dagor': 'Dagor',
        'rajomon': 'Rajomon',
    }
    markers = {
        'plain': 'o',
        'breakwater': 's',
        'breakwaterd': '^',
        'dagor': 'v',
        'topdown': '.',
        'rajomon': 'x',
    }

    APIs = ['search-hotel', 'reserve-hotel', 'compose', 'user-timeline', 'home-timeline']

    # Filter to include only non-concurrent runs for compose and search-hotel
    filtered_df = hotel_df[
        (hotel_df['Request'].isin(APIs)) &
        (hotel_df['concurrency'] == True)
    ]

    # Create limits for each subplot
    load_limits = {
        'search-hotel': (6000, 18000),
        'compose': (3000, 11000)
    }

    # Iterate over interfaces and control mechanisms
    for i, interface in enumerate(APIs):
        ax = axs[i]
        for control in control_mechanisms:
            subset = filtered_df[
                (filtered_df['Request'] == interface) &
                (filtered_df['overload_control'] == control) &
                (filtered_df['Load'].between(*load_limits[interface]))
            ]
            # for compose only take odd loads (3000, 5000, 7000, 9000) for search-hotel take even loads (6000, 8000, 10000, 12000, 14000, 16000, 18000)
            if interface == 'compose':
                subset = subset[subset['Load'] % 2000 == 1000]
            else:
                subset = subset[subset['Load'] % 2000 == 0]
            if not subset.empty:
                # Plot the data for each interface and control mechanism
                ax.plot(subset['Load'], subset['recovery_ms_mean'],
                        label=labelDict.get(control, control),
                        color=colors.get(control, '#333333'),
                        marker=markers.get(control, 'o'),
                        linewidth=2)
                        # Set labels, title, and limits
        ax.set_xlabel('Load (kRPS)')
        ax.set_title(interface.replace('-', ' ').title())
        ax.set_xlim(load_limits[interface])
        ax.grid(True)

    axs[0].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axs[1].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axs[1].yaxis.set_major_formatter(FuncFormatter(format_ticks))

    axs[0].set_ylabel('Recovery Time (s)')

    handles, labels = axs[0].get_legend_handles_labels()

    # Create a single legend
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=3)
    # Customize the legend and plot
    # axs[1].legend(loc='best', frameon=False) let's put the legend outside the plot on top
    plt.tight_layout()

    plt.savefig('recovery-time-vs-load.pdf', bbox_inches='tight')
    print("Plot saved to recovery-time-vs-load.pdf")
    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    generate_csv = False
    # Save the result to a new CSV file
    output_file = 'grouped_recover_time.csv'

    if generate_csv:
        input_file = 'all-experiments-recover.csv'  # Replace with the actual CSV file path

        # Load and process the data
        grouped_data = load_and_group_data(input_file)

        grouped_data.to_csv(output_file, index=False)
        print(f"Grouped experiment results saved to {output_file}")

    # 
    method = os.getenv('METHOD', 'both-hotel')
 
    control_mechanisms = ['dagor', 'breakwater', 'breakwaterd', 'topdown', 'rajomon']
    # Load data
    hotel_df = pd.DataFrame()

    # social = True
    alibaba = False

    if alibaba:
        interfaces = ['S_102000854', 'S_149998854', 'S_161142529']
    else:
        interfaces = ['search-hotel', 'reserve-hotel']
        interfaces += ['compose', 'user-timeline', 'home-timeline']


    for interface in interfaces:
        df = load_data_from_csv(output_file, method=interface, given_parameter=parameter_files)
        assert df is not None, f"Dataframe for {interface} is None"
        df['Request'] = interface
        # report the file_count column for each interface and capacity
        # df['file_count'] = df['file_count'].astype(int)
        # for control in control_mechanisms:
            # subset = df[(df['Request'] == interface) & (df['overload_control'] == control)]
            # print(f"{interface} {control} file_count: {subset['file_count'].values}")
        hotel_df = pd.concat([hotel_df, df])

    df = hotel_df # within 10000 capacity
    # df = hotel_df[hotel_df['Load'] <= 12000]
    
    # print the df for overload_control = rajomon and load = 5000
    print(df[(df['interface'] == 'search-hotel') & (df['Load'] == 6000) & (df['concurrency'] == False) ])

    plot_recovery_vs_load(df, control_mechanisms)

if __name__ == "__main__":
    main()

