import pandas as pd
import json

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import os
import random

# Define the directory containing the files
folder = os.path.expanduser("/z/rajomon-nsdi/ghz-results")
# folder = os.path.expanduser("~/Sync/Git/protobuf/ghz-results")

# Define the patterns for interface and control
interfaces = ["compose", "user-timeline", "home-timeline", "search-hotel", "reserve-hotel", "S_102000854", "S_149998854", "S_161142529"]
controls = ["breakwater", "breakwaterd", "dagor", "topdown", "charon", "plain"]

# Initialize an empty DataFrame
columns = ['Application', 'Control', 'Ave Latency']
latency_df = pd.DataFrame(columns=columns)

# Regex for extracting latency value
avelatency_regex = re.compile(r"Average:\s+(\d+\.\d+)\s+ms")

# 50 % in 4.10 ms is the median latency
median_regex = re.compile(r"90 % in (\d+\.\d+) ms")
median = True

# Correct regex pattern to match the filenames
filename_pattern = r"social-(.*?)-control-(.*?)-parallel-capacity-2000-(0917|0918)(.*?)output" 

# Function to process files and extract latency
def process_files(files):
    data = []
    for filename in files:
        matches = re.search(filename_pattern, filename)
        if not matches:
            print("No match for ", filename)
            continue
        application = matches.group(1)
        control = matches.group(2)

        with open(filename, 'r') as file:
            for line in file:
                if median:
                    medmatch = median_regex.search(line)
                    if medmatch:
                        median_latency = float(medmatch.group(1))
                        data.append({'Application': application, 'Control': control, 'Ave Latency': median_latency})
                        break
                else:
                    avematch = avelatency_regex.search(line)
                    if avematch:
                        ave_latency = float(avematch.group(1))
                        data.append({'Application': application, 'Control': control, 'Ave Latency': ave_latency})
                        break
    return data

# Collect and process matching files for each interface and control
for interface in interfaces:
    for control in controls:
        pattern = f"social-{interface}-control-{control}-parallel-capacity-2000-0917*.json.output"
        matching_files = glob.glob(os.path.join(folder, pattern))
        matching_files += glob.glob(os.path.join(folder, pattern.replace("0917", "0918")))
        if matching_files:
            random.seed(12)
            sampled_files = random.sample(matching_files, min(len(matching_files), 3))
            latency_df = pd.concat([latency_df, pd.DataFrame(process_files(sampled_files))], ignore_index=True)
            # add one column for the length of the sampled files
            print(f"Interface: {interface}, Control: {control}, Files: {len(sampled_files)}")

# Calculate the mean of average latency
mean_latencies = latency_df.groupby(['Application', 'Control'])['Ave Latency'].mean().reset_index()

# Use the mean of 'plain' as the baseline and deduct the latency from all other control mechanisms
baseline_latencies = mean_latencies[mean_latencies['Control'] == 'plain'].set_index('Application')['Ave Latency']
# use the min of 'plain' as the baseline
# baseline_latencies = mean_latencies[mean_latencies['Control'] == 'plain'].groupby('Application')['Ave Latency'].max()

mean_latencies['Baseline'] = mean_latencies['Application'].map(baseline_latencies)
mean_latencies['Deducted Latency'] = mean_latencies['Ave Latency'] - mean_latencies['Baseline']

# Report the deducted mean average latency for all apps / interfaces
report_df = mean_latencies[mean_latencies['Control'] != 'plain'][['Application', 'Control', 'Deducted Latency']]
# replace rajomon with Rajomon
control_name = {
    "dagor": "Dagor",
    "charon": "Rajomon",
    "breakwater": "Breakwater",
    "breakwaterd": "Breakwaterd",
    "topdown": "TopFull",
}
report_df['Control'] = report_df['Control'].replace(control_name)

# put Rajomon before other controls
report_df['Control'] = pd.Categorical(report_df['Control'], ['Rajomon', 'Breakwater', 'Breakwaterd', 'Dagor', 'TopFull'])

print(report_df)
# cap the latency at 0 ms
report_df['Deducted Latency'] = report_df['Deducted Latency'].clip(lower=0.0)

# Plotting
plt.figure(figsize=(2.5, 2.6))
# use max and min for the whiskers, and not show the outliers
# use gray scale for the plot with hatches
# make box thinner, transparent box 
sns.boxplot(x='Control', y='Deducted Latency', data=report_df, palette='rocket', showfliers=False, whis=[0, 100], ax=plt.gca(), width=0.5, boxprops=dict(alpha=.5))
# plot in log scale for y axis
plt.yscale('log')
plt.ylim(0.01, 20)
plt.xticks(rotation=45)
plt.grid(axis='y', which='major')
plt.title('')
plt.xlabel('')
plt.ylabel('Latency Overhead (ms)')
plt.tight_layout()
plt.savefig('overhead_boxplot_3.pdf', bbox_inches='tight')
print(f"Figure saved to {os.getcwd()}/overhead_boxplot_3.pdf")
plt.show()
