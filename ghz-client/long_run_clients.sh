#!/bin/bash
# Run this file from the kube master node in order to run the ghz experiment

# Get the names of all deployments
deployments=$(kubectl get deployments -o custom-columns=":metadata.name" --no-headers)
echo "Deployments: $deployments"

# Loop through each deployment and wait for it to complete
for deployment in $deployments; do
  kubectl rollout status deployment/$deployment
  # echo "Deployment $deployment is ready."
done

# entrypoint is 
echo "ENTRY_POINT: $ENTRY_POINT"

# Get the Cluster IP of grpc-service-1
SERVICE_A_IP=$(kubectl get service $ENTRY_POINT -o=jsonpath='{.spec.clusterIP}')

# Get the NodePort (if available) of grpc-service-1
SERVICE_A_NODEPORT=$(kubectl get service $ENTRY_POINT -o=jsonpath='{.spec.ports[0].nodePort}')

SERVICE_A_URL="$SERVICE_A_IP:50051"

# Export the SERVICE_A_URL as an environment variable
export SERVICE_A_URL

# Display the URL
echo "SERVICE_A_URL: $SERVICE_A_URL"

# if RL_TIERS env is set, then echo it
if [ -n "$RL_TIERS" ]; then
  echo "RL_TIERS: $RL_TIERS"
fi
# if AQM_ENABLED env is set, then echo it
if [ -n "$AQM_ENABLED" ]; then
  echo "AQM_ENABLED: $AQM_ENABLED"
fi

# Export LOAD_INCREASE environment variable
export LOAD_INCREASE=true

# Function to run clientcall with randomized capacity
run_clientcall() {
    # Generate a random capacity between 100 and 2000
    local CAPACITY=$((RANDOM % 1901 + 100))
    
    # Print the capacity value
    echo "Capacity: $CAPACITY"
    
    # Export the CAPACITY environment variable
    export CAPACITY
    
    # Run the clientcall command
    ~/protobuf/ghz-client/clientcall &
}

# Initial call to run clientcall
run_clientcall

# Rolling process: Launch a new randomized clientcall every 4 seconds
while true; do
    sleep 4
    run_clientcall
    # clean disk space, empty the ../ghz-results/ folder
    rm -rf ../ghz-results/*
done
