import os
import shutil

# Define the parameter files
parameter_files = {
    'search-hotel': {
        'dagor': 'bopt_False_dagor_search-hotel_gpt0-10000_08-23.json',
        'breakwater': 'bopt_False_breakwater_search-hotel_gpt1-10000_08-22.json',
        'breakwaterd': 'bopt_False_breakwaterd_search-hotel_gpt1-10000_08-22.json',
        'rajomon': 'bopt_False_rajomon_search-hotel_gpt1-10000_08-24.json',
    },
    'compose': {
        'dagor': 'bopt_False_dagor_compose_gpt1-5000_08-15.json',
        'breakwater': 'bopt_False_breakwater_compose_gpt0-5000_08-27.json',
        'breakwaterd': 'bopt_False_breakwaterd_compose_gpt0-5000_08-25.json',
        'rajomon': 'bopt_False_rajomon_compose_gpt1-5000_08-25.json',
    },
    'alibaba': {
        'dagor': 'bopt_False_dagor_all-alibaba_gpt1-10000_09-13.json',
        'breakwater': 'bopt_False_breakwater_all-alibaba_gpt1-10000_09-12.json',
        'breakwaterd': 'bopt_False_breakwaterd_all-alibaba_gpt1-10000_09-12.json',
        'rajomon': 'bopt_False_rajomon_all-alibaba_gpt1-10000_09-12.json',
    },
}


# Specific files to process for each app
FILES_PATTERN = {
    'search-hotel': [
        "bopt_False_{}_search-hotel_gpt1-best.json"
    ],
    'compose': [
        "bopt_False_{}_compose_gpt1-best.json"
    ],
    'alibaba': [
        "bopt_False_{}_S_102000854_gpt1-best.json",
        "bopt_False_{}_S_161142529_gpt1-best.json",
        "bopt_False_{}_S_149998854_gpt1-best.json"
    ]
}

PARAM_DIR = os.path.expanduser('~/Sync/Git/protobuf/baysian-opt')
# Define source and destination directories
source_dir = PARAM_DIR
destination_dir = source_dir  # Replace with the actual destination directory


# Function to copy and rename files correctly based on the app
def copy_and_rename_files():
    for app, files in parameter_files.items():
        for method, file_name in files.items():
            if app in FILES_PATTERN:
                for pattern in FILES_PATTERN[app]:
                    new_file_name = pattern.format(method)
                    
                    # Define full paths
                    source_file = os.path.join(source_dir, file_name)
                    dest_file = os.path.join(destination_dir, new_file_name)
                    
                    # Only copy if the source file exists, skip missing files
                    if os.path.exists(source_file):
                        shutil.copy(source_file, dest_file)
                        print(f"File copied: {source_file} to {dest_file}")
                    else:
                        print(f"File missing: {source_file}")

# Run the function
copy_and_rename_files()
