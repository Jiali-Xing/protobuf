# query the service for the IP address from user input
# from stdin
echo "Enter the IP address and port of the service you want to query: "
read SERVICE_A_URL

# Export the SERVICE_A_URL as an environment variable
export SERVICE_A_URL

# Display the URL
echo "SERVICE_A_URL: $SERVICE_A_URL"

# if AQM_ENABLED env is set, then echo it
if [ -n "$AQM_ENABLED" ]; then
  echo "AQM_ENABLED: $AQM_ENABLED"
fi

# Export LOAD_INCREASE environment variable
export LOAD_INCREASE=true

# Define an array of methods for the Alibaba services
methods=("S_102000854" "S_149998854" "S_161142529")

# Function to run clientcall with randomized capacity
run_clientcall() {
    # Generate a random capacity between 100 and 2000
    # first, randomize the method to be either `motivate-get` or `motivate-set`

    # Randomly select one of the Alibaba methods
    METHOD=${methods[$RANDOM % ${#methods[@]}]}
    export METHOD

    # then randomize the capacity to be around 4000
    CAPACITY=$((RANDOM % 7001 + 500))

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
    rm -rf ../ghz-results/*json
    rm -rf ../ghz-results/*output
done
