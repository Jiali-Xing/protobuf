#!/bin/bash
# Run this file from the kube master node in order to run the ghz experiment

# Function to get node IP from pod name
get_node_ip() {
    local NODE_NAME=$1
    grep "$NODE_NAME" /etc/hosts | awk '{print $1}'
}

# Function to get the service port number
get_service_port() {
    local APP=$1
    local ENTRY_POINT=$2
    kubectl get svc -o wide | grep "$ENTRY_POINT" | awk -F '[[:space:]]+|,' '{for (i=1; i<=NF; i++) if ($i ~ /[0-9]+:[0-9]+\/TCP/) {split($i, ports, ":"); print ports[2]; break}}' | sed 's/\/TCP//'
}

# Prompt user for app and service IP/port
echo "Enter the app name (social or hotel): "
read APP

# Determine METHOD and entry point based on app name
if [ "$APP" == "social" ]; then
    METHOD="compose"
    ENTRY_POINT="nginx"
elif [ "$APP" == "hotel" ]; then
    METHOD="search-hotel"
    ENTRY_POINT="frontend"
else
    echo "Unknown app. Exiting."
    exit 1
fi

# Display app configuration
echo "App: $APP"
echo "Method: $METHOD"
echo "Entry Point: $ENTRY_POINT"

# Run kubectl to find the pod and node
kubectl get pods -o wide | grep "$ENTRY_POINT" > pod_info.txt
NODE_NAME=$(awk '{print $7}' pod_info.txt)
echo "Node: $NODE_NAME"

# Get node IP address
NODE_IP=$(get_node_ip "$NODE_NAME")
if [ -z "$NODE_IP" ]; then
    echo "Failed to find IP for node $NODE_NAME. Exiting."
    exit 1
fi
echo "Node IP: $NODE_IP"

# Get service port number
SERVICE_PORT=$(get_service_port "$APP" "$ENTRY_POINT")
if [ -z "$SERVICE_PORT" ]; then
    echo "Failed to find port for service $ENTRY_POINT. Exiting."
    exit 1
fi
echo "Service Port: $SERVICE_PORT"

# Combine IP and port to create SERVICE_A_URL
SERVICE_A_URL="$NODE_IP:$SERVICE_PORT"
export SERVICE_A_URL
# also export method
export METHOD

echo "SERVICE_A_URL: $SERVICE_A_URL"


# Export LOAD_INCREASE environment variable
export LOAD_INCREASE=true

# Function to run clientcall with randomized capacity
run_clientcall() {
    # Generate a random capacity between 100 and 2000
    # if `hotel` is in METHOD, then capacity is between 100 and 4000
    # local CAPACITY=$((RANDOM % 1901 + 100))
    if [[ $METHOD == *"hotel"* ]]; then
      CAPACITY=$((RANDOM % 3901 + 100))
    else
      CAPACITY=$((RANDOM % 1901 + 100))
    fi
    
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
    rm -rf ../ghz-results/*output
    rm -rf ../ghz-results/*json
done
