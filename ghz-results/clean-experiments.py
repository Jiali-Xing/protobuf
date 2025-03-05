import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('all-experiments.csv')

# Function to check if a value is an integer
def is_integer(val):
    try:
        # Check if the value is an integer or a float that is an integer (like 139.0)
        if isinstance(val, int):
            return True
        if isinstance(val, float) and val.is_integer():
            return True
        return False
    except ValueError:
        return False

# Filter the DataFrame to keep only rows where PRICE_STEP is an integer
cleaned_df = df[df['PRICE_STEP'].apply(is_integer)]

# Print the number of rows removed
print(f"Number of rows removed: {len(df) - len(cleaned_df)}")

# Save the cleaned DataFrame back to a new CSV file
cleaned_df.to_csv('cleaned_all_experiments.csv', index=False)

print("Cleaned data saved to 'cleaned_all_experiments.csv'.")
