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
BREAKWATERD=false
TOPDOWN=false
CONFIG_NAME="gpt0-7000_08-14"

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
export METHOD=search-hotel
export WARMUP_LOAD=4800

# Define variables
CAPACITY_STEP=2000
CAPACITY_START=6000
CAPACITY_END=18000

# Debugging output to check the options parsed
echo "RAJOMON: $RAJOMON, BREAKWATER: $BREAKWATER, DAGOR: $DAGOR, BREAKWATERD: $BREAKWATERD, TOPDOWN: $TOPDOWN"

# Check if at least one control option is provided
if [ "$RAJOMON" = false ] && [ "$BREAKWATER" = false ] && [ "$DAGOR" = false ] && [ "$BREAKWATERD" = false ] && [ "$TOPDOWN" = false ]; then
    usage
fi

# Function to run experiments with parameter file
run_experiments_with_param() {
    local CONTROL=$1
    local first_run="true"
    for ((LOAD = CAPACITY_END; LOAD >= CAPACITY_START; LOAD -= CAPACITY_STEP)); do
        # Construct the parameter file pattern
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_search-hotel_${CONFIG_NAME}.json | sort -V | tail -n 1)
        
        # source the env
        source ~/Sync/Git/service-app/cloudlab/scripts/envs.sh
        # Run the compound experiments script with the parameter file
        if [ -n "$PARAM_FILE" ]; then
            if [ "$first_run" = "true" ]; then
                echo "Setup k8s"
                ssh -p 22 -i ${private_key} ${node_username}@${node_address} "cd hotelApp/ && ./setup-k8s-initial.sh hotel"

                echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
                bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"
                first_run="false"
            else
                # echo "Setup k8s"
                # ssh -p 22 -i ${private_key} ${node_username}@${node_address} "cd hotelApp/ && ./setup-k8s-initial.sh hotel"

                echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
                bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"
                # echo "[SKIP] k8s re-deployment is skipped"
                # bash ~/Sync/Git/service-app/cloudlab/scripts/overload-experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"  --skip
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

# Run experiments based on the chosen control option
if [ "$TOPDOWN" = true ]; then
    run_experiments_topdown
elif [ "$RAJOMON" = true ]; then
    run_experiments_with_param "rajomon"
elif [ "$BREAKWATER" = true ]; then
    run_experiments_with_param "breakwater"
elif [ "$DAGOR" = true ]; then
    run_experiments_with_param "dagor"
elif [ "$BREAKWATERD" = true ]; then
    run_experiments_with_param "breakwaterd"
fi
