#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-c charon] [-b breakwater] [-d dagor] [-a breakwaterd]"
    exit 1
}

# Parse command-line options
CHARON=false
BREAKWATER=false
DAGOR=false
BREAKWATERD=false

while getopts "acbd" opt; do
    case ${opt} in
        a )
            BREAKWATERD=true
            ;;
        c )
            CHARON=true
            ;;
        b )
            BREAKWATER=true
            ;;
        d )
            DAGOR=true
            ;;
        \? )
            usage
            ;;
    esac
done

# Export the METHOD environment variable
export METHOD=both-motivate-monotonic

# Define variables
CAPACITY_STEP=2000
CAPACITY_START=10000
CAPACITY_END=30000
# CONTROLS=("charon" "breakwater")

# Check if at least one control option is provided
if [ "$CHARON" = false ] && [ "$BREAKWATER" = false ] && [ "$DAGOR" = false ] && [ "$BREAKWATERD" = false ]; then
    usage
fi

# Function to run experiments
run_experiments() {
    local CONTROL=$1
    for ((LOAD = CAPACITY_START; LOAD <= CAPACITY_END; LOAD += CAPACITY_STEP)); do
        # Construct the parameter file pattern
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_motivate-set_*_06-10.json | sort -V | tail -n 1)
        
        # Run the compound experiments script
        if [ -n "$PARAM_FILE" ]; then
            echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
            bash ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"
        else
            echo "No parameter file found for control: $CONTROL"
        fi
    done
}

# Loop through each control
if [ "$CHARON" = true ]; then
    run_experiments "charon"
fi

if [ "$BREAKWATER" = true ]; then
    run_experiments "breakwater"
fi

if [ "$DAGOR" = true ]; then
    run_experiments "dagor"
fi

if [ "$BREAKWATERD" = true ]; then
    run_experiments "breakwaterd"
fi
