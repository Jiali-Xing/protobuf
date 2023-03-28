from cProfile import label
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read in the JSON file
with open('data.json', 'r') as f:
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

ax1.legend()
ax2.legend()
plt.savefig('cdf.png')
plt.show()

# Calculate throughput (number of requests per second)
requests_per_second = df['status'].resample('1S').count()
df['throughput'] = requests_per_second.reindex(df.index, method='ffill')

# Calculate moving average of latency
df['latency_ma'] = df['latency'].rolling(window=50).mean()

# Plot data
fig, ax1 = plt.subplots(figsize=(12, 4))

# Latency plot
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Latency (ms)', color=color)
# ax1.plot(df.index, df['latency'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(df.index, df['latency_ma'], color='orange', linestyle='--', label='Latency, Moving Average')
ax1.legend(loc='upper left')

# Throughput plot
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Throughput (req/s)', color=color)
ax2.plot(df.index, df['throughput'], color=color, label='Throughput')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(False)
ax2.legend(['Throughput'])
ax1.legend(loc='upper right')

plt.savefig('timeseries.png')

plt.show()

