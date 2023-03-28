from cProfile import label
import json
import pandas as pd
import matplotlib.pyplot as plt

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
ax1.plot(df.index, df['latency_ma'], color='orange', linestyle='--', label='Latency (MA)')
ax1.legend(loc='upper left')

# Throughput plot
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Throughput (req/s)', color=color)
ax2.plot(df.index, df['throughput'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(False)
ax2.legend(['Throughput'])


plt.savefig('myplot.png')

plt.show()

