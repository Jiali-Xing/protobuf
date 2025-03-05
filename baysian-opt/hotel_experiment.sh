#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-c rajomon] [-b breakwater] [-d dagor] [-a breakwaterd] [-t topdown] [-g <config_name>] [--start <start_value>] [--end <end_value>]"
    exit 1
}

# Parse command-line options
RAJOMON=false
BREAKWATER=false
DAGOR=false
TOPDOWN=false
BREAKWATERD=false
CONFIG_NAME="gpt0-7000_08-14"


# Define variables
CAPACITY_STEP=2000
CAPACITY_START=2000
CAPACITY_END=16000

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -a)
            BREAKWATERD=true
            ;;
        -c)
            RAJOMON=true
            ;;
        -b)
            BREAKWATER=true
            ;;
        -d)
            DAGOR=true
            ;;
        -t)
            TOPDOWN=true
            ;;
        -g)
            CONFIG_NAME="$2"
            shift
            ;;
        --start)
            CAPACITY_START="$2"
            shift
            ;;
        --end)
            CAPACITY_END="$2"
            shift
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Check if at least one control option is provided
if [ "$RAJOMON" = false ] && [ "$BREAKWATER" = false ] && [ "$DAGOR" = false ] && [ "$BREAKWATERD" = false ] && [ "$TOPDOWN" = false ]; then
    usage
fi

# Export the METHOD environment variable
export METHOD=both-hotel
export WARMUP_LOAD=2001

# Function to run experiments
run_experiments() {
    local CONTROL=$1

    local first_run="true" # Variable to track the first run
    
    for ((LOAD = CAPACITY_END; LOAD >= CAPACITY_START; LOAD -= CAPACITY_STEP)); do
        # Construct the parameter file pattern
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_search-hotel_${CONFIG_NAME}.json | sort -V | tail -n 1)
        
        # Run the compound experiments script
        if [ -n "$PARAM_FILE" ]; then
            echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
            
            # source the env
            source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh
            # if [ "$first_run" = "true" ] or interceptor is breakwaterd or breakwater
            if [ "$first_run" = "true" ]; then
                echo "Setup k8s"
                ssh -p 22 -i ${private_key} ${node_username}@${node_address} "cd hotelApp/ && ./setup-k8s-initial.sh hotel"
                sleep 10
                # For the first experiment, don't use the --skip flag
                bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"
                first_run="false" # Set first_run to false after the first run
            else
                echo "Setup k8s"
                ssh -p 22 -i ${private_key} ${node_username}@${node_address} "cd hotelApp/ && ./setup-k8s-initial.sh hotel"
                sleep 10
                # For subsequent experiments, use the --skip flag
                bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE" 
            fi
        else
            echo "No parameter file found for control: $CONTROL"
        fi

        # give the machine some time to cool down
        sleep 20
    done
}

# Function to run experiments with --topdown
run_experiments_topdown() {
    for ((LOAD = CAPACITY_START; LOAD <= CAPACITY_END; LOAD += CAPACITY_STEP)); do
        echo "Running experiment with topdown, load: $LOAD, method: $METHOD"
        bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" --topdown --skip
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
