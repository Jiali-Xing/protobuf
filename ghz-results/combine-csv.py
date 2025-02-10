import pandas as pd

# Read both CSV files into DataFrames
df1 = pd.read_csv('grouped_hotel-2.csv')
df2 = pd.read_csv('grouped_hotel-1.csv')


# Merge the two DataFrames on the first three columns (interface, control_scheme, and capacity)
merged_df = pd.merge(df1, df2, on=['interface', 'control_scheme', 'capacity'], how='outer', suffixes=('', '_df2'))

# For each column in df2 (except the first three), replace the values in df1 with those from df2 when available
for col in df2.columns:
    if col not in ['interface', 'control_scheme', 'capacity']:
        # Replace df1 values with df2 values, keeping df2 values when present
        merged_df[col] = merged_df[col + '_df2'].combine_first(merged_df[col])

# Drop the duplicate columns that came from df2
drop_columns = [col for col in merged_df.columns if '_df2' in col]
final_df = merged_df.drop(columns=drop_columns)

# Fill missing values if any (optional step based on your needs)
final_df.fillna(method='ffill', inplace=True)  # Use forward fill or adjust this to fill missing values as needed

# Save the final merged result to a new CSV file
final_df.to_csv('grouped_hotel.csv', index=False)

print("Merge completed and saved to merged_grouped_hotel.csv")