from cProfile import label
import json, sys, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Get the filename from the command-line arguments
filename = sys.argv[1]

# Read in the JSON file
with open(filename, 'r') as f:
    data = json.load(f)

# convert JSON to pandas DataFrame
df = pd.DataFrame(data["details"])


# convert the timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# set the timestamp column as the dataframe index
df.set_index('timestamp', inplace=True)

# Convert latency to milliseconds
df['latency'] = df['latency'] / 1000000

# plot histogram of the 'latency' column
# Get latency column data
latency = df['latency']

# Compute PDF
pdf, bins = np.histogram(latency, bins=50, density=True)
pdf_x = (bins[:-1] + bins[1:]) / 2

# Compute CDF
cdf_x = np.sort(latency)
cdf_y = np.arange(1, len(cdf_x)+1) / len(cdf_x)

# Plot PDF and CDF
fig, ax1 = plt.subplots()

ax1.plot(pdf_x, pdf, label='PDF')
ax1.set_xlabel('Latency (millisecond)')
ax1.set_ylabel('PDF')

ax2 = ax1.twinx()
ax2.plot(cdf_x, cdf_y, color='orange', label='CDF')
ax2.set_ylabel('CDF')

# ax1.legend()
# ax2.legend()

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Position the legends
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
ax2.legend().remove()

plt.savefig(filename+'.latency.png')
plt.show()

# Calculate throughput (number of requests per second)
requests_per_second = df['status'].resample('1S').count()
df['throughput'] = requests_per_second.reindex(df.index, method='ffill')

# Calculate moving average of latency
df['latency_ma'] = df['latency'].rolling(window=50).mean()


# Calculate the tail latency over time (e.g., 99th percentile)
tail_latency = df['latency'].rolling(window=50).quantile(0.99)

# Create a new column 'tail_latency' in the DataFrame
df['tail_latency'] = tail_latency

# Plot data
fig, ax1 = plt.subplots(figsize=(12, 4))

# Latency plot
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Latencies (ms)', color=color)
# ax1.plot(df.index, df['latency'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(df.index, df['latency_ma'], color='orange', linestyle='--', label='Latency')
ax1.plot(df.index, df['tail_latency'], color='green', linestyle='-.', label='Tail Latency')
# ax1.legend()

# Throughput plot
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Throughput (req/s)', color=color)
ax2.plot(df.index, df['throughput'], color=color, label='Throughput')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(1000, ax2.get_ylim()[1])  # Set the y-axis minimum limit to 1000
ax2.grid(False)
# ax2.legend(['Throughput'])


# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Position the legends
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
ax2.legend().remove()

concurrent_clients = re.findall(r"\d+", filename)[0]

# Set the title
plt.title(f"Number of Concurrent Clients: {concurrent_clients}")

plt.savefig(filename+'.timeseries.png')

plt.show()

