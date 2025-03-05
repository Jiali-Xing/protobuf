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
user_timeline_file="${base_name/social-compose/social-user-timeline}.json"
home_timeline_file="${base_name/social-compose/social-home-timeline}.json"

# Create the Python commands
command1="python visualize.py $input_file"
command2="python visualize.py $user_timeline_file"
command3="python visualize.py $home_timeline_file"

# Execute the commands
$command1
$command2
$command3

