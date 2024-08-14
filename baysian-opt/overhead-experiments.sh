#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-c charon] [-b breakwater] [-d dagor] [-a breakwaterd] [-p plain]"
    exit 1
}

# Parse command-line options
CHARON=false
BREAKWATER=false
DAGOR=false
BREAKWATERD=false
PLAIN=false

while getopts "pacbd" opt; do
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
        p )
            PLAIN=true
            ;;
        \? )
            usage
            ;;
    esac
done

# Export the environment variables
export WARMUP_LOAD=2000
export LOAD=2001

# Define variables
CONTROLS=("charon" "breakwater" "dagor" "breakwaterd" "plain")

# Check if at least one control option is provided
if [ "$CHARON" = false ] && [ "$BREAKWATER" = false ] && [ "$DAGOR" = false ] && [ "$BREAKWATERD" = false ] && [ "$PLAIN" = false ]; then
    usage
fi

# Function to run experiments
run_experiments() {
    local CONTROL=$1
    local METHOD=$2

    # if PLAIN is true, then only run the plain control, no parameter file is needed
    if [ "$PLAIN" = true ]; then
        echo "Running experiment with control: $CONTROL, load: $LOAD, method: $METHOD"
        bash ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c "$LOAD" -s parallel
        return
    fi

    # Determine the parameter file
    if [[ "$METHOD" == "compose" || "$METHOD" == "home-timeline" || "$METHOD" == "user-timeline" ]]; then
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_compose_gpt1-best.json | sort -V | tail -n 1)
    elif [[ "$METHOD" == "search-hotel" || "$METHOD" == "reserve-hotel" ]]; then
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_search-hotel_gpt1-best.json | sort -V | tail -n 1)
    else
        PARAM_FILE=$(ls ~/Sync/Git/protobuf/baysian-opt/bopt_False_${CONTROL}_${METHOD}_gpt1-best.json | sort -V | tail -n 1)
    fi
    
    # Run the compound experiments script
    if [ -n "$PARAM_FILE" ]; then
        echo "Running experiment with control: $CONTROL, load: $LOAD, param file: $PARAM_FILE, method: $METHOD"
        bash ~/Sync/Git/service-app/cloudlab/scripts/compound_experiments.sh -c "$LOAD" -s parallel --"$CONTROL" --param "$PARAM_FILE"
    else
        echo "No parameter file found for control: $CONTROL"
    fi
}

# Loop through each control and method based on the METHOD environment variable
if [ "$METHOD" = "all-social" ]; then
    METHODS=("compose" "home-timeline" "user-timeline")
elif [ "$METHOD" = "all-alibaba" ]; then
    METHODS=("S_102000854" "S_149998854" "S_161142529")
elif [ "$METHOD" = "both-hotel" ]; then
    METHODS=("search-hotel" "reserve-hotel")
else
    echo "Invalid METHOD environment variable"
    exit 1
fi

for CONTROL in "${CONTROLS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        case $CONTROL in
            "charon")
                if [ "$CHARON" = true ]; then
                    run_experiments "$CONTROL" "$METHOD"
                fi
                ;;
            "breakwater")
                if [ "$BREAKWATER" = true ]; then
                    run_experiments "$CONTROL" "$METHOD"
                fi
                ;;
            "dagor")
                if [ "$DAGOR" = true ]; then
                    run_experiments "$CONTROL" "$METHOD"
                fi
                ;;
            "breakwaterd")
                if [ "$BREAKWATERD" = true ]; then
                    run_experiments "$CONTROL" "$METHOD"
                fi
                ;;
            "plain")
                run_experiments "$CONTROL" "$METHOD"
                ;;
        esac
    done
done
