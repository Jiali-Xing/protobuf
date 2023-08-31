#!/bin/bash
# Run this file from the kube master node in order to run the ghz experiment

# # First, delete and re-deploy the services
kubectl delete deployment --all
kubectl delete service --all
kubectl apply -f /users/jiali/service-app/cloudlab/deploy/hardcode.yaml

# # wait for the deployment to be ready
kubectl wait --for=condition=Ready -f /users/jiali/service-app/cloudlab/deploy/hardcode.yaml --timeout=20s

# entrypoint is name of service_a 
ENTRY_POINT="social-graph-mongodb"

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

  # Loop over all pods and copy the target file
  for pod in $(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.namespace}{"\n"}{end}'); do
    pod_name=$(echo "$pod" | cut -d' ' -f1)
    namespace_name=$(echo "$pod" | cut -d' ' -f2)

    # Check if the pod contains the target file
    if kubectl exec -n "$namespace_name" "$pod_name" -- ls "/root/$target_file" > /dev/null 2>&1; then
      # Print the namespace_name and pod_name
      echo "Namespace Name: $namespace_name"
      echo "Pod Name: $pod_name"

      # Copy the target file from the pod to the local machine
      kubectl cp "$namespace_name"/"$pod_name":/root/"$target_file" ~/"$target_file"
    fi
  done
}

copy_deathstar_output