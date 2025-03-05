import re
import pandas as pd
import numpy as np
from datetime import timedelta
from utils import calculate_goodput_dynamic, read_data, calculate_throughput_dynamic, calculate_goodput_dynamic, calculate_loadshedded, calculate_ratelimited, calculate_tail_latency_dynamic
from visualize import convert_to_dataframe
from slo import get_slo

def load_data(filename):
    """
    Calculate the recovery goodput from the given file.
    """
    # the method is the word between `social-` and `-control` in the filename
    method = re.findall(r"social-(.*?)-control", filename)[0]
    timestamp = re.findall(r"-\d+_\d+", filename)[0]
    SLO = get_slo(method, tight=False, all_methods=False)

    data = read_data(filename)
    df = convert_to_dataframe(data)
    df = calculate_throughput_dynamic(df)
    df = calculate_goodput_dynamic(df, SLO)
    df = calculate_loadshedded(df)
    df = calculate_ratelimited(df)
    df = calculate_tail_latency_dynamic(df)
    if df is None or df.empty:
        print(f"Data is empty for file: {filename}")
        return None
    return df   
    
def calculate_recovery_goodput(df):
    """
    Calculate the recovery goodput from the given file.
    """
    if df is None or df.empty:
        return None
    warmup_period = df[df.index <= df.index.min() + pd.Timedelta(seconds=5)]
    max_warmup_goodput = warmup_period['goodput'].max() if not warmup_period.empty else None

    # Calculate goodput for the recovery period (last 10 seconds)
    recovery_period = df[df.index >= df.index.max() - pd.Timedelta(seconds=10)]
    max_recovery_goodput = recovery_period['goodput'].max() if not recovery_period.empty else None
    
    # Check if either of the goodput values are None
    if max_warmup_goodput is None:
        return None
    if max_recovery_goodput is None:
        return None
    
    # Return the lower of the two goodput values
    return min(max_warmup_goodput, max_recovery_goodput)

def calculate_recovery_time(df, recovery_goodput):
    """
    Calculate the time taken to recover to the given goodput.
    """
    # Find the first time after the warmup where goodput exceeds the stable goodput threshold
    post_warmup_period = df[df.index > df.index.min() + pd.Timedelta(seconds=5)]
    recovery_point = post_warmup_period[post_warmup_period['goodput'] >= recovery_goodput]

    if not recovery_point.empty:
        recovery_time = recovery_point.index.min() - post_warmup_period.index.min()
    else:
        recovery_time = None  # Could not find a stable goodput recovery point

    # Return the recovery goodput and recovery time
    return recovery_time


def example():
    # List of filenames and control mechanisms
    filenames = [
        "social-search-hotel-control-rajomon-parallel-capacity-10000-0824_2252.json",
        "social-search-hotel-control-breakwater-parallel-capacity-10000-0826_2030.json",
        "social-search-hotel-control-breakwaterd-parallel-capacity-10000-0822_1857.json",
        "social-search-hotel-control-dagor-parallel-capacity-10000-0822_1712.json",
        "social-search-hotel-control-topdown-parallel-capacity-10000-0903_2224.json"
    ]

    control_mechanisms = ["Rajomon", "Breakwater", "Breakwaterd", "Dagor", "TopFull"]

    # Dictionary to store the recovery goodput for each control mechanism
    recovery_goodputs = {}
    recovery_time = {}

    # Iterate over each file and calculate recovery goodput
    for filename, control_mechanism in zip(filenames, control_mechanisms):
        df = load_data(filename)
        # Calculate goodput for the warmup period (first 5 seconds)
        recovery_goodput_value = calculate_recovery_goodput(df)
        if recovery_goodput_value is None:
            print(f"Recovery goodput is None for file: {filename}")
            continue
        recover_time = calculate_recovery_time(df, recovery_goodput_value)
        if recover_time is None:
            print(f"Recovery time is None for file: {filename}")
            continue
        recovery_goodputs[control_mechanism] = recovery_goodput_value

        recover_time = recover_time.total_seconds()
        recovery_time[control_mechanism] = recover_time

        print(f"Recovery goodput for {control_mechanism}: {recovery_goodput_value}")
        print(f"Recovery time for {control_mechanism}: {recover_time}")


    # Output the recovery goodputs for all control mechanisms
    print("\nRecovery Goodput Summary:")
    # for control, goodput in recovery_goodputs.items():

def process_and_attach_recovery_metrics(input_file, output_file, chunk_size=1000):
    """
    Process the input CSV in chunks, calculate recovery metrics, and write them to the output CSV file.
    """
    # Open output file for writing
    with pd.read_csv(input_file, chunksize=chunk_size) as reader:
        for chunk_index, chunk in enumerate(reader):
            recovery_goodput_values = []
            recovery_time_values = []

            # Process each row in the chunk
            for index, row in chunk.iterrows():
                filename = row['filename']

                try:
                    data_df = load_data(filename)

                    # Calculate recovery goodput
                    recovery_goodput = calculate_recovery_goodput(data_df)

                    if recovery_goodput is None:
                        recovery_goodput_values.append(None)
                        recovery_time_values.append(None)
                        continue

                    # Calculate recovery time
                    recovery_time = calculate_recovery_time(data_df, recovery_goodput)

                    recovery_goodput_values.append(recovery_goodput)
                    recovery_time_values.append(recovery_time)

                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")
                    recovery_goodput_values.append(None)
                    recovery_time_values.append(None)

            # Add new columns to the chunk
            chunk['recovery_goodput'] = recovery_goodput_values
            chunk['recovery_time'] = recovery_time_values

            # Write the chunk to the output CSV
            chunk_mode = 'a' if chunk_index > 0 else 'w'
            header = chunk_index == 0
            chunk.to_csv(output_file, mode=chunk_mode, header=header, index=False)
            print(f"Processed chunk {chunk_index + 1}, written to {output_file}")
   
def main():
    input_file = 'all-experiments.csv'
    output_file = 'all-experiments-recover.csv'

    # Process the file and attach recovery metrics
    process_and_attach_recovery_metrics(input_file, output_file)


if __name__ == "__main__":
    main()