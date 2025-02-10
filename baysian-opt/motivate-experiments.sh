#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-c rajomon] [-b breakwater] [-d dagor] [-a breakwaterd] [-t topdown] [-g <config_name>]"
    exit 1
}

# Parse command-line options
RAJOMON=false
BREAKWATER=false
DAGOR=false
TOPDOWN=false
BREAKWATERD=false
CONFIG_NAME="gpt0-30000_09-06"

while getopts "acbdtg:" opt; do
    case ${opt} in
        a )
            BREAKWATERD=true
            ;;
        c )
            RAJOMON=true
            ;;
        b )
            BREAKWATER=true
            ;;
        d )
            DAGOR=true
            ;;
        t )
            TOPDOWN=true
            ;;
        g )
            CONFIG_NAME="$OPTARG"
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND-1))

# Export the METHOD environment variable
export METHOD=both-motivate
export WARMUP_LOAD=10

# Define variables
CAPACITY_STEP=2000
CAPACITY_START=6000
CAPACITY_END=30000

# Check if at least one control option is provided
if [ "$RAJOMON" = false ] && [ "$BREAKWATER" = false ] && [ "$DAGOR" = false ] && [ "$BREAKWATERD" = false ] && [ "$TOPDOWN" = false ]; then
    usage
fi

# Function to run experiments
run_experiments() {
    local CONTROL=$1

    local first_run="true"
    for ((LOAD = CAPACITY_START; LOAD <= CAPACITY_END; LOAD += CAPACITY_STEP)); do
        # Construct the parameter file pattern
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_motivate-set_${CONFIG_NAME}.json | sort -V | tail -n 1)

        # Run the compound experiments script
        if [ -n "$PARAM_FILE" ]; then
            if [ "$first_run" = "true" ] || [ "$CONTROL" = "breakwaterd" ] || [ "$CONTROL" = "breakwater" ]; then
                echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
                bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"
                first_run="false"
            else
                echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
                echo "Skipping the k8s setup for the subsequent runs"
                bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE" --skip
            fi
        else
            echo "No parameter file found for control: $CONTROL"
        fi
    done
}

# Function to run experiments with --topdown
run_experiments_topdown() {
    for ((LOAD = CAPACITY_START; LOAD <= CAPACITY_END; LOAD += CAPACITY_STEP)); do
        echo "Running experiment with topdown, load: $LOAD, method: $METHOD"
        bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" --topdown
    done
}

# Loop through each control
if [ "$RAJOMON" = true ]; then
    run_experiments "rajomon"
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

if [ "$TOPDOWN" = true ]; then
    run_experiments_topdown
fi
