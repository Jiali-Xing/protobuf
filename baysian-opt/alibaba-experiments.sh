#!/bin/bash

# Define the commands to run
commands=(
    "python bayesian_optimization_all.py S_102000854"
    "python bayesian_optimization_all.py S_149998854"
    "python bayesian_optimization_all.py S_161142529"
)

# Loop to run each command 5 times
for cmd in "${commands[@]}"; do
    for i in {1..5}; do
        echo "Running: $cmd (Iteration $i)"
        $cmd
    done
done

