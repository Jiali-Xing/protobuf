import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate the coefficient of variation (CV)
def calculate_cv(values):
    mean = np.mean(values)
    stddev = np.std(values)
    if mean != 0:
        cv = stddev / mean
    else:
        cv = 0
    return cv

# Directory containing the parameter files
# PARAM_DIR = os.path.expanduser('/z/rajomon-nov/')
PARAM_DIR = os.path.expanduser('~/Sync/Git/protobuf/baysian-opt')

# Specific files to process
FILES_PATTERN = [
    "bopt_False_{}_compose_gpt1-best.json",
    "bopt_False_{}_all-alibaba_gpt1-best.json",
    "bopt_False_{}_search-hotel_gpt1-best.json"
]

# Array of control mechanisms
CONTROLS = ["dagor", "rajomon", "breakwater", "breakwaterd"]

# Parameter categories
control_target_params = ["DAGOR_QUEUING_THRESHOLD", "LATENCY_THRESHOLD", "BREAKWATER_SLO", "BREAKWATERD_SLO"]
update_interval_params = ["DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL", "PRICE_UPDATE_RATE", "BREAKWATER_RTT", "BREAKWATERD_RTT"]
update_step_width_params = ["DAGOR_ALPHA", "DAGOR_BETA", "PRICE_STEP", "BREAKWATER_A", "BREAKWATER_B", "BREAKWATERD_A", "BREAKWATERD_B"]
client_params = []  # Other parameters

# DataFrame to store results
cv_data = {
    "control": [],
    "category": [],
    "cv": [],
    "max_cv": []
}

# Specific files to process
FILES = {
    'bopt_False_dagor_search-hotel_gpt0-10000_08-23.json',
    'bopt_False_breakwater_search-hotel_gpt1-10000_08-22.json',
    'bopt_False_breakwaterd_search-hotel_gpt1-10000_08-22.json',
    'bopt_False_rajomon_search-hotel_gpt1-10000_08-24.json',
    'bopt_False_dagor_compose_gpt1-5000_08-15.json',
    'bopt_False_breakwater_compose_gpt0-5000_08-27.json',
    'bopt_False_breakwaterd_compose_gpt0-5000_08-25.json',
    'bopt_False_rajomon_compose_gpt1-5000_08-25.json',
    'bopt_False_dagor_all-alibaba_gpt1-10000_09-13.json',
    'bopt_False_breakwater_all-alibaba_gpt1-10000_09-12.json',
    'bopt_False_breakwaterd_all-alibaba_gpt1-10000_09-12.json',
    'bopt_False_rajomon_all-alibaba_gpt1-10000_09-12.json',
}

# Loop over each control mechanism
for control in CONTROLS:
    print(f"Calculating CV for control: {control}")
    
    params = {}

    # # Read parameter values from each file
    # for pattern in FILES_PATTERN:
    #     filename = pattern.format(control)
    for filename in FILES:
        if control not in filename:
            continue
        filepath = os.path.join(PARAM_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                data = json.load(file)
                for key, value in data['parameters'].items():
                    if key not in params:
                        params[key] = []
                    if isinstance(value, str) and value.endswith('us'):
                        value = float(value[:-2])  # Remove 'us' and convert to float
                    elif isinstance(value, str) and value.endswith('ms'):
                        value = float(value[:-2]) * 1000  # Convert ms to us
                    elif isinstance(value, str) and not value.endswith('s'):
                        continue
                    params[key].append(float(value))
        else:
            print(f"File not found: {filename}")

    # Calculate and store CV for each parameter
    category_cv = {
        "Control\nTarget": [],
        "Update\nInterval": [],
        "Update\nStep Width": [],
        "Client\nParams": []
    }
    
    control_name = {
        "dagor": "Dagor",
        "rajomon": "Rajomon",
        "breakwater": "Breakwater",
        "breakwaterd": "Breakwaterd"
    }

    for key, values in params.items():
        cv = calculate_cv(values)
        if not np.isnan(cv):
            controlName = control_name[control]
            
            if key in control_target_params:
                category_cv["Control\nTarget"].append(cv)
            elif key in update_interval_params:
                category_cv["Update\nInterval"].append(cv)
            elif key in update_step_width_params:
                category_cv["Update\nStep Width"].append(cv)
            else:
                category_cv["Client\nParams"].append(cv)
            print(f"Parameter: {key}, CV: {cv}")

    for category, cvs in category_cv.items():
        if cvs:  # Only add if there are CVs in the category
            avg_cv = np.mean(cvs)
            max_cv = np.max(cvs)
            cv_data["control"].append(controlName)
            cv_data["category"].append(category)
            cv_data["cv"].append(avg_cv)
            cv_data["max_cv"].append(max_cv)
            print(f"Category: {category}, Avg CV: {avg_cv}")

    print()

# Create DataFrame
df = pd.DataFrame(cv_data)

sns.set_palette("rocket")

# Create bar plot of mean CV values for each control mechanism and category
plt.figure(figsize=(2.3, 1.5))
mean_cv = df.pivot(index='control', columns='category', values='cv')
# put Rajomon first, then Breakwater, then Breakwaterd, then Dagor
mean_cv = mean_cv.reindex(index=['Rajomon', 'Breakwater', 'Breakwaterd', 'Dagor'])
ax = mean_cv.plot(kind='bar', ax=plt.gca(), alpha=0.6, edgecolor='black')


# # Apply hatch patterns for grayscale differentiation
# hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
# for i, bar in enumerate(ax.patches):
#     bar.set_hatch(hatches[i % len(hatches)])

# plt.title('Coefficient of Variation of Parameters by Category')
# plt.xlabel('Control Mechanism')
plt.grid(axis='y')
plt.xlabel('')
plt.ylabel('Average CV\nby Category')
plt.xticks(rotation=25)
# plt.legend(title='Parameter\nCategory', bbox_to_anchor=(1.05, 1), loc='upper left')
# plot legend outside of the plot on top
plt.legend(title='Parameter Category', bbox_to_anchor=(0.3, 1.9), loc='upper center', ncol=2)


# Save the figure
plt.savefig('cv_barplot_by_category.pdf', bbox_inches='tight')
print(f'Bar plot saved to cv_barplot_by_category.pdf')

# Show the plot
plt.show()
