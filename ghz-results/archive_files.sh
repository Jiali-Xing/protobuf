#!/bin/bash

# Directory where the files are located
directory="$HOME/Sync/Git/protobuf/ghz-results/"

# # Find files and extract timestamps
# for file in $(find "$directory" -name "social-user-timeline-control-*-parallel-capacity-*-*.json"); do
#     timestamp=$(echo "$file" | sed -n 's/.*parallel-capacity-\([0-9]\+\)-\(.*\)\.json/\2/p')
#     capacity=$(echo "$file" | sed -n 's/.*parallel-capacity-\([0-9]\+\)-\(.*\)\.json/\1/p')
 
#     # # Find all files with this timestamp and move them
#     # find "$directory" -name "*$timestamp.json" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
#     # find "$directory" -name "*$timestamp.json.output" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;

#     # Skip capacity 6000
#     # if [ "$capacity" != "6000" ]; then
#         # Find all files with this timestamp and move them
#     find "$directory" -name "*$timestamp.json" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
#     find "$directory" -name "*$timestamp.json.output" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
#     # fi
# done
# # Find files and extract timestamps
# for file in $(find "$directory" -name "social-user-http-control-*-parallel-capacity-*-*.json"); do
#     timestamp=$(echo "$file" | sed -n 's/.*parallel-capacity-\([0-9]\+\)-\(.*\)\.json/\2/p')
#     capacity=$(echo "$file" | sed -n 's/.*parallel-capacity-\([0-9]\+\)-\(.*\)\.json/\1/p')
 
#     # # Find all files with this timestamp and move them
#     # find "$directory" -name "*$timestamp.json" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
#     # find "$directory" -name "*$timestamp.json.output" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;

#     # Skip capacity 6000
#     # if [ "$capacity" != "6000" ]; then
#         # Find all files with this timestamp and move them
#     find "$directory" -name "*$timestamp.json" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
#     find "$directory" -name "*$timestamp.json.output" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
#     # fi
# done

# # notice that before 0124_1715 is the old version of the experiment, with in total 1000 clients
# # after 0124_1715 is the new version of the experiment, with in total 1000x #interfaces clients


# Function to process each file
process_file() {
    file=$1
    timestamp=$(echo "$file" | sed -n 's/.*parallel-capacity-\([0-9]\+\)-\(.*\)\.json/\2/p')
    capacity=$(echo "$file" | sed -n 's/.*parallel-capacity-\([0-9]\+\)-\(.*\)\.json/\1/p')

    # Skip capacity 6000 and timestamps newer than 0124_1715
    if [ "$timestamp" \< "0128_1715" ]; then
    # if [ "$timestamp" \< "0124_1715" ]; then
        # Find all files with this timestamp and move them
        find "$directory" -name "*$timestamp.json" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
        find "$directory" -name "*$timestamp.json.output" -exec mv {} ~/Sync/Git/charon-experiments/old_mixed/ \;
    fi
}

# # Iterate over the specific files and process them
# for file in $(find "$directory" -name "social-user-timeline-control-*-parallel-capacity-*-*.json"); do
#     process_file "$file"
# done

for file in $(find "$directory" -name "social-user-http-control-*-parallel-capacity-*-*.json"); do
    process_file "$file"
done