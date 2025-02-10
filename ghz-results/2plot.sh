#!/bin/bash

# Check if the input argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input-file>"
  exit 1
fi

# Extract the base name of the input file
input_file="$1"
base_name=$(basename "$input_file" .json)

# Generate the other filenames
# reserve_file="${base_name/social-search/social-reserve}.json"
search_file="${base_name/social-reserve/social-search}.json"

# Create the Python commands
command1="python visualize.py $input_file"
command2="python visualize.py $search_file"

# Execute the commands
$command1
$command2

