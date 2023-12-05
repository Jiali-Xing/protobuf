#!/bin/bash
# Run this file from the kube master node in order to run the ghz experiment

# # # First, delete and re-deploy the services
# kubectl delete deployment --all
# kubectl delete service --all
# kubectl apply -f /users/jiali/service-app/cloudlab/deploy/hardcode.yaml

# # # wait for the deployment to be ready
# kubectl wait --for=condition=Running -f /users/jiali/service-app/cloudlab/deploy/hardcode.yaml --timeout=20s
# Get the names of all deployments
deployments=$(kubectl get deployments -o custom-columns=":metadata.name" --no-headers)
echo "Deployments: $deployments"

# Loop through each deployment and wait for it to complete
for deployment in $deployments; do
  kubectl rollout status deployment/$deployment
  echo "Deployment $deployment is ready."
done

# sleep 20

# entrypoint is 
echo "ENTRY_POINT: $ENTRY_POINT"
    # export METHOD=compose 
    # export CONSTANT_LOAD=true
    # # export the INTERCEPT env var as true or false
    # export INTERCEPT=$3
    # # export SUBCALL=parallel
    # export SUBCALL=$2
    # export PROFILING=true

    # # export the environment variables to set the options for charon (priceUpdateRate, latencyThreshold, priceStep, priceStrategy)
    # export PRICE_UPDATE_RATE=10ms
    # export LATENCY_THRESHOLD=500000us
    # export PRICE_STEP=10
    # export PRICE_STRATEGY=proportional
    # export LAZY_UPDATE=true
    # # export ENTRY_POINT as the corresponding one for the method from file msgraph.yaml
    # export ENTRY_POINT=nginx-web-server
    # export RATE_LIMITING=true
# echo all the env vars above
for var in METHOD CONSTANT_LOAD INTERCEPT SUBCALL PROFILING PRICE_UPDATE_RATE LATENCY_THRESHOLD PRICE_STEP PRICE_STRATEGY LAZY_UPDATE ENTRY_POINT RATE_LIMITING DEBUG_INFO; do
    echo "$var: ${!var}"
done

# Get the Cluster IP of grpc-service-1
SERVICE_A_IP=$(kubectl get service $ENTRY_POINT -o=jsonpath='{.spec.clusterIP}')

# Get the NodePort (if available) of grpc-service-1
SERVICE_A_NODEPORT=$(kubectl get service $ENTRY_POINT -o=jsonpath='{.spec.ports[0].nodePort}')

# Check if NodePort is available, if not, use the Cluster IP
# if [ -z "$SERVICE_A_NODEPORT" ]; then
#   SERVICE_A_URL="$SERVICE_A_IP:50051"
# else
#   # Use NodePort if available
#   SERVICE_A_URL="$(kubectl get nodes -o=jsonpath='{.items[0].status.addresses[0].address}'):${SERVICE_A_NODEPORT}"
# fi

SERVICE_A_URL="$SERVICE_A_IP:50051"

# Export the SERVICE_A_URL as an environment variable
export SERVICE_A_URL

# Display the URL
echo "SERVICE_A_URL: $SERVICE_A_URL"

# # Get the namespace_name and pod_name based on the prefix "grpc-service-1"
# namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 'grpc-service-1' | awk '{print $2}')
# pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 'grpc-service-1')

# # Check if a pod with the prefix "grpc-service-1" exists
# if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
#   echo "No pod with the prefix 'grpc-service-1' found."
#   exit 1
# fi

# # Print the namespace_name and pod_name
# echo "Namespace Name: $namespace_name"
# echo "Pod Name: $pod_name"

# # Remove the server output file on server pod 1, if it exists
# # kubectl exec -it -n $namespace_name $pod_name -- /bin/sh -c "touch /root/server.output"
# kubectl exec -it -n $namespace_name $pod_name -- /bin/sh -c "echo '' > /root/grpc-service-1.output"

copy_msgraph_yaml() {
  if [ -z "$1" ]; then
    echo "Usage: copy_msgraph_yaml <x>"
    exit 1
  fi

  prefix="grpc-service-$1"

  # Get the namespace_name and pod_name based on the prefix
  namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 "$prefix" | awk '{print $2}')
  pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 "$prefix")

  # Check if a pod with the given prefix exists
  if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
    echo "No pod with the prefix '$prefix' found."
    exit 1
  fi

  # Print the namespace_name and pod_name
  echo "Namespace Name: $namespace_name"
  echo "Pod Name: $pod_name"

  # Copy the local msgraph.yaml file to the server pod
  kubectl cp ~/service-app/services/protobuf-grpc/msgraph.yaml "$namespace_name"/"$pod_name":/go/service-app/services/protobuf-grpc/msgraph.yaml
}

# use function to clear server output
clear_output() {
  if [ -z "$1" ]; then
    echo "Usage: clear_output <x>"
    exit 1
  fi

  prefix="grpc-service-$1"

  # Get the namespace_name and pod_name based on the prefix
  namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 "$prefix" | awk '{print $2}')
  pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 "$prefix")

  # Check if a pod with the given prefix exists
  if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
    echo "No pod with the prefix '$prefix' found."
    exit 1
  fi

  # Print the namespace_name and pod_name
  echo "Namespace Name: $namespace_name"
  echo "Pod Name: $pod_name"

  # Remove the server output file on the server pod, if it exists
  kubectl exec -it -n "$namespace_name" "$pod_name" -- /bin/sh -c "echo '' > /root/grpc-service-$1.output"
}

# Use function to clear server output
start_monitering() {
  if [ -z "$1" ]; then
    echo "Usage: start moniter <x>"
    exit 1
  fi

  prefix="grpc-service-$1"

  # Get the namespace_name and pod_name based on the prefix
  namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 "$prefix" | awk '{print $2}')
  pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 "$prefix")

  # Check if a pod with the given prefix exists
  if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
    echo "No pod with the prefix '$prefix' found."
    exit 1
  fi

  # Print the namespace_name and pod_name
  echo "Namespace Name: $namespace_name"
  echo "Pod Name: $pod_name"

  # Create a new file to record the output
  output_file="/root/service-$1.output"
#   touch "$output_file" in the pod
  kubectl exec -i -n "$namespace_name" "$pod_name" -- /bin/sh -c "touch $output_file"

  # Continuously monitor the server output file and record new lines to the new file
  kubectl exec -n "$namespace_name" "$pod_name" -- /bin/sh -c "tail -f /root/grpc-service-$1.output > $output_file " &
  tail_pid=$!

  # Wait for the user to stop monitoring
  echo "Monitoring server output..."
#   echo "Monitoring server output... Press CTRL+C to stop."
#   trap 'kill $tail_pid; echo "Monitoring stopped."; exit' INT
#   wait
}

# Function to stop monitoring server output
stop_monitoring() {
  if [ -z "$1" ]; then
    echo "Usage: stop_monitoring <x>"
    exit 1
  fi

  prefix="grpc-service-$1"

  # Get the namespace_name and pod_name based on the prefix
  namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 "$prefix" | awk '{print $2}')
  pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 "$prefix")

  # Check if a pod with the given prefix exists
  if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
    echo "No pod with the prefix '$prefix' found."
    exit 1
  fi

  # Print the namespace_name and pod_name
  echo "Namespace Name: $namespace_name"
  echo "Pod Name: $pod_name"

  # Get the process ID of the tail command inside the pod
#   tail_pid=$(kubectl exec -n "$namespace_name" "$pod_name" -- ps -ef | grep "tail -f /root/grpc-service-$1.output" | grep -v grep | awk '{print $2}')
# just do kill all tail on the pod
    kubectl exec -n "$namespace_name" "$pod_name" -- killall tail

#   if [ -z "$tail_pid" ]; then
#     echo "No monitoring process found for grpc-service-$1."
#     exit 1
#   fi

#   # Stop the tail command inside the pod
#   kubectl exec -n "$namespace_name" "$pod_name" -- kill "$tail_pid"

  echo "Monitoring for grpc-service-$1 has been stopped."

  # Find the process ID of the tail command
#   tail_pid=$(ps -ef | grep "tail -f /root/grpc-service-$1.output" | grep -v grep | awk '{print $2}')

#   if [ -z "$tail_pid" ]; then
#     echo "No monitoring process found for grpc-service-$1."
#     exit 1
#   fi

#   # Stop the tail command
#   kill "$tail_pid"

    pkill -f "tail -f /root/grpc-service"
  echo "Monitoring for grpc-service-$1 has been stopped."
}

# Loop over grpc-service-x from 1 to 7
# for ((x=1; x<=7; x++)); do
#     start_monitering "$x"
# done

# Print the `CAPACITY` env var of the current pod
echo "Capacity: $CAPACITY"

# Run the client
~/protobuf/ghz-client/clientcall 

# kubectl cp "$namespace_name"/"$pod_name":/root/grpc-service-1.output ~/server.output

copy_output() {
  if [ -z "$1" ]; then
    echo "Usage: copy_output <x>"
    exit 1
  fi

  prefix="grpc-service-$1"

  # Get the namespace_name and pod_name based on the prefix
  namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 "$prefix" | awk '{print $2}')
  pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 "$prefix")

  # Check if a pod with the given prefix exists
  if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
    echo "No pod with the prefix '$prefix' found."
    exit 1
  fi

  # Print the namespace_name and pod_name
  echo "Namespace Name: $namespace_name"
  echo "Pod Name: $pod_name"

  # Copy the server output file from the server pod to the local machine
  kubectl cp "$namespace_name"/"$pod_name":/root/service-$1.output ~/grpc-service-$1.output
}

# # Loop over grpc-service-x from 1 to 7
# for ((x=1; x<=7; x++)); do
#     stop_monitoring "$x"
#     copy_output "$x"
# done

copy_deathstar_output() {
  target_file="deathstar_*.output"

  # Loop over all pods
  kubectl get pods -o=jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.namespace}{"\n"}{end}' | while read -r pod; do
    pod_name=$(echo "$pod" | cut -d' ' -f1)
    namespace_name=$(echo "$pod" | cut -d' ' -f2)

    echo "Debug: Checking Namespace=$namespace_name, Pod=$pod_name for Target File=$target_file"

    # List matching files in the pod
    matching_files=$(kubectl exec -n "$namespace_name" "$pod_name" -- sh -c "ls /root/ | grep 'deathstar_'")

    if [ -n "$matching_files" ]; then
      echo "Namespace: $namespace_name, Pod: $pod_name has the target files."
      
      # Loop over matching files and copy them individually
      for file in $matching_files; do
        # local_file="${namespace_name}-${pod_name}-$(date +%s)-$file"
        local_file="$file"

        echo "Copying $file to $local_file"
        kubectl cp "$namespace_name/$pod_name:/root/$file" ~/"$local_file"
      done
    else
      echo "Target files not found in Pod $pod_name in Namespace $namespace_name."
    fi
  done
}

copy_deathstar_output

# copy pprof files from server pods their location is /go/service-app/services/protobuf-grpc/{service-name}.pprof
copy_deathstar_pprof() {
  # target_file="*.pprof" but not starting with "grpc-service"
  target_file="*.pprof"

  # Loop over all pods
  kubectl get pods -o=jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.namespace}{"\n"}{end}' | while read -r pod; do
    pod_name=$(echo "$pod" | cut -d' ' -f1)
    namespace_name=$(echo "$pod" | cut -d' ' -f2)

    echo "Debug: Checking Namespace=$namespace_name, Pod=$pod_name for Target File=$target_file"

    # List matching files in the pod
    matching_files=$(kubectl exec -n "$namespace_name" "$pod_name" -- sh -c "ls /go/service-app/services/protobuf-grpc/ | grep '\.pprof$'")
    # matching_files=$(kubectl exec -n "$namespace_name" "$pod_name" -- sh -c "ls /go/service-app/services/protobuf-grpc/ | grep 'pprof'")

    if [ -n "$matching_files" ]; then
      echo "Namespace: $namespace_name, Pod: $pod_name has the target files."
      
      # Loop over matching files and copy them individually
      for file in $matching_files; do
        # # local_file="${namespace_name}-${pod_name}-$(date +%s)-$file"
        # local_file="$file"

        # kubectl cp "$namespace_name/$pod_name:/go/service-app/services/protobuf-grpc/$file" ~/"$local_file"
        if [[ ! $file =~ ^grpc-service ]]; then
          # local_file="${namespace_name}-${pod_name}-$(date +%s)-$file"
          local_file="$file"

          echo "Profiling $file"
          # # if INTERCEPT is `plain` then use `go tool pprof -list` without focus on the intercept function
          if [ "$INTERCEPT" = "plain" ]; then
            kubectl exec -n "$namespace_name" "$pod_name" -- /bin/sh -c "go tool pprof -list greeting $file > ~/$file.txt"
          # otherwise, use `go tool pprof -list` with corresponding interceptor as focus
          else
            kubectl exec -n "$namespace_name" "$pod_name" -- /bin/sh -c "go tool pprof -list $INTERCEPT $file > ~/$file.txt"
          fi

          echo "Copying $file to $local_file"
          kubectl cp "$namespace_name/$pod_name:/go/service-app/services/protobuf-grpc/$file" ~/"$local_file"
          kubectl cp "$namespace_name/$pod_name:/root/$file.txt" ~/"$local_file.txt"
        fi
      done
    else
      echo "Target files not found in Pod $pod_name in Namespace $namespace_name."
    fi
  done
  # client profiling file is ~/protobuf/ghz-client/Client.pprof. cp it to ~/
  # cd to the dir, then go tool pprof -list $INTERCEPT Client.pprof > ~/Client.pprof.txt
  cp ~/protobuf/ghz-client/Client.pprof ~/Client.pprof
  /usr/local/go/bin/go tool pprof -list $INTERCEPT ~/Client.pprof > ~/Client.pprof.txt 
}

# if env var PROFILING is true, copy pprof files from server pods
if [ "$PROFILING" = true ]; then
  copy_deathstar_pprof
fi