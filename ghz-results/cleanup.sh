#!/bin/bash

# Loop through each json.output file
for output_file in social-*-control-*.json.output; do
    # Check if the file contains '[OK]'
    if ! grep -q '\[OK\]' "$output_file"; then
        # Remove the output file
	mv "$output_file" ../archived_results/
        
        # Remove the corresponding .json file if it exists
        json_file="${output_file%.output}"
        if [[ -f "$json_file" ]]; then
            mv "$json_file" ../archived_results/
        fi
    fi
done
