import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import os

# Regular expression pattern to extract relevant information

# # # Parse the text and extract the needed information
# # matches = re.findall(pattern, text_data)

# # Create a DataFrame from the extracted data
# df = pd.DataFrame(matches, columns=['Application', 'Control', 'Latency'])
# df['Latency'] = df['Latency'].astype(float)  # Convert latency to a float

# # for each app and control, get the average latency and surpress the rows.
# # then add one column for how many times the control was run for that app.
# df['Count'] = df.groupby(['Application', 'Control'])['Latency'].transform('count')
# # make it an int
# df['Count'] = df['Count'].astype(int)

folder = os.path.expanduser("~/Sync/Git/charon-experiments-results")
# Define the patterns to search for
patterns = [
    "social-S*-control-*-parallel-capacity-4000-01*put",
    "social-S_10*-control-*-parallel-capacity-6000-01*put",
    "social-compose-control-*-parallel-capacity-4000-01*put",
    "social-*-http-control-*-parallel-capacity-2000-01*put",
]

# Initialize an empty DataFrame
columns = ['Application', 'Control', 'Ave Latency', 'Median Latency', 'Count']
latency_df = pd.DataFrame(columns=columns)

# Regex for extracting latency value
avelatency_regex = re.compile(r"Average:\s+(\d+\.\d+)\s+ms")
minlatency_regex = re.compile(r"Fastest:\s+(\d+\.\d+)\s+ms")
# match `25 % in xxx ms`
quantile_regex = re.compile(r"25 % in (\d+\.\d+) ms")

# median latency
median_regex = re.compile(r"50 % in (\d+\.\d+) ms")

count_regex = re.compile(r"Count:\s+(\d+)")

# the above part read latency data from output files and put them into a dataframe.
# the following part is to calculate the latency from json files.
            
def read_data(filename):
    # first, make sure that the file exist
    if not os.path.exists(filename):
        print("File ", filename, " does not exist")
        return None
    # and that the file is not empty
    if os.path.getsize(filename) == 0:
        print("File ", filename, " is empty")
        return None
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
            return data["details"]
        except:
            print("File ", filename, " is not valid")
            return None


def convert_to_dataframe(data):
    df_from_data = pd.DataFrame(data)
    # if df is empty, return empty df and print the message
    if df_from_data.empty:
        print("DataFrame is empty")
        return df_from_data
    df_from_data['timestamp'] = pd.to_datetime(df_from_data['timestamp'])

    df_from_data.set_index('timestamp', inplace=True)
    df_from_data['latency'] = df_from_data['latency'] / 1000000
    # drop the rows if the `status` is Unavailable
    df_from_data = df_from_data[df_from_data['status'] != 'Unavailable']
    df_from_data = df_from_data[df_from_data['status'] != 'Canceled']
    # if df is empty, return empty df and print the message
    if len(df_from_data) == 0:
        return df_from_data
    
    offset = 1
    # remove the data within first couple seconds of df
    df_from_data = df_from_data[df_from_data.index > df_from_data.index[0] + pd.Timedelta(seconds=offset)]

    min_timestamp = df_from_data.index.min()   
    df_from_data.index = df_from_data.index - min_timestamp + pd.Timestamp('2000-01-01')
    return df_from_data


def sample_average_latency(filename, size=1000):
    data = read_data(filename)
    df = convert_to_dataframe(data)
    
    if df.empty:
        print("DataFrame is empty for ", filename)
        return None
    
    # sample the data
    df = df.sample(n=size, random_state=1, replace=True)
    # calculate the average latency
    average_latency = df['latency'].mean()
    
    return average_latency


# create a map to map application name to the short name
app_map = {
    "S_149998854": "S_2",
    "S_102000854": "S_1",
    "S_161142529": "S_3",
    "compose": "POST",
    "home-timeline": "HOMET",
    "user-timeline": "USERT",
    # "hotels-http": "HOTEL",
    # "recommendations-http": "RECOMMEND",
    "reservation-http": "RESERVE",
    # "user-http": "USER",
}

# Process each pattern
for pattern in patterns:
    for filename in glob.glob(os.path.join(folder, pattern)):
        # Extract Application and Control from filename
        # parts = filename.split('-')
        # application = parts[1]
        # control = parts[3]
        pattern = r"social-(.*?)-control-([a-zA-Z]+)-parallel-capacity-\d+-\d+_\d+.json.output"
        # # replace 4000 with 5000 for S_102000854
        # if "S_102000854" in filename:
        #     pattern = r"social-(S_[0-9]+|compose)-control-([a-zA-Z]+)-parallel-capacity-6000-\d+_\d+.json.output"
        matches = re.search(pattern, filename)
        if not matches:
            print("No match for ", filename)
            continue
        application = matches.group(1)
        control = matches.group(2)

        # skip application that is not in the map
        if application not in app_map:
            continue

        # Search for latency values inside the file
        with open(filename, 'r') as file:
            for line in file:
                avematch = avelatency_regex.search(line)
                minmatch = minlatency_regex.search(line)
                quantilematch = quantile_regex.search(line)
                medmatch = median_regex.search(line)
                cntmatch = count_regex.search(line)
                if cntmatch:
                    cnt = int(cntmatch.group(1))
                if avematch:
                    ave = float(avematch.group(1))
                if medmatch:
                    med = float(medmatch.group(1))
                if minmatch:
                    minlat = float(minmatch.group(1))
                if quantilematch:
                    quanlat = float(quantilematch.group(1))

            if cnt < 1000:
                continue
            ave = sample_average_latency(filename.split('.')[0] + '.json')
            # Append to DataFrame
            # latency_df = latency_df.append({'Application': application, 
            #                                 'Control': control, 
            #                                 'Ave Latency': ave, 
            #                                 'Median Latency': med,
            #                                 'Count': int(cnt),
            #                                 'Min Latency': min
            #                                 },
            #                                 ignore_index=True)
            latency_df = pd.concat([latency_df, pd.DataFrame({'Application': app_map[application],
                                                                'Control': control,
                                                                'Ave Latency': ave,
                                                                'Median Latency': med,
                                                                'Count': int(cnt),
                                                                'Min Latency': minlat,
                                                                '25th Percentile': quanlat
                                                                }, index=[0])],
                                     ignore_index=True)


df = latency_df.groupby(['Application', 'Control']).agg({
    'Ave Latency': 'mean',
    'Median Latency': 'mean',
    'Min Latency': 'mean',
    'Count': 'mean',
    '25th Percentile': 'mean'
}).reset_index()
# Convert DataFrame to CSV
csv_data = df.to_csv(index=False)

print(csv_data)

# # Assuming df is your existing DataFrame
latency = 'Min Latency'
baseline_latency = 'Min Latency'

# Compute the baseline latency for each application (i.e., 'plain')
baseline_latencies = df[df['Control'] == 'plain'].set_index('Application')[baseline_latency]

# Calculate the overhead for each control scheme compared to plain
df['Overhead'] = df.apply(lambda row: row[latency] - baseline_latencies[row['Application']], axis=1)
# df overhead percentage
df['Overhead Percentage'] = df.apply(lambda row: row['Overhead'] / baseline_latencies[row['Application']] * 100, axis=1)

# Filter out the 'plain' control as we are only interested in overhead of other controls
overhead_df = df[df['Control'] != 'plain']

# rename the column `charon` to `our model`
overhead_df['Control'] = overhead_df['Control'].replace('charon', 'our model')

# Plotting
plt.figure(figsize=(4.5, 2), dpi=300)
# plot should be visiable when printed black and white
sns.set_palette("colorblind")
sns.set_style("whitegrid"
              , {'grid.linestyle': '--'}
             )
sns.barplot(x='Control', y='Overhead Percentage', hue='Application', data=overhead_df) 

# make y scale log
# plt.yscale('log')
# add grid
# plt.grid(True, which="both", ls="--")
plt.xlabel('')
plt.ylabel('Latency\nOverhead (%)')
# plt.title('Latency Overhead by Control Scheme and Application Compared to No Control')
plt.xticks(rotation=10)
# put legend outside the plot on the right
plt.legend(title='Interfaces', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('overhead2024.pdf', format='pdf', bbox_inches='tight')
plt.show()
print(f"Figure saved to {os.getcwd()}/overhead2024.pdf")
