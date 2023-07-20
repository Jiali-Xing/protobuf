#!/bin/bash
# Run this file from the kube master node in order to run the ghz experiment

# Get the Cluster IP of grpc-service-1
SERVICE_A_IP=$(kubectl get service grpc-service-1 -o=jsonpath='{.spec.clusterIP}')

# Get the NodePort (if available) of grpc-service-1
SERVICE_A_NODEPORT=$(kubectl get service grpc-service-1 -o=jsonpath='{.spec.ports[0].nodePort}')

# Check if NodePort is available, if not, use the Cluster IP
if [ -z "$SERVICE_A_NODEPORT" ]; then
  SERVICE_A_URL="$SERVICE_A_IP:50051"
else
  # Use NodePort if available
  SERVICE_A_URL="$(kubectl get nodes -o=jsonpath='{.items[0].status.addresses[0].address}'):${SERVICE_A_NODEPORT}"
fi

# Export the SERVICE_A_URL as an environment variable
export SERVICE_A_URL

# Display the URL
echo "SERVICE_A_URL: $SERVICE_A_URL"

export CAPACITY=700

# Get the namespace_name and pod_name based on the prefix "grpc-service-1"
namespace_name=$(kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.namespace}{"\n"}{end}' | grep -m 1 'grpc-service-1' | awk '{print $2}')
pod_name=$(kubectl get pods -n "$namespace_name" -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m 1 'grpc-service-1')

# Check if a pod with the prefix "grpc-service-1" exists
if [ -z "$namespace_name" ] || [ -z "$pod_name" ]; then
  echo "No pod with the prefix 'grpc-service-1' found."
  exit 1
fi

# Print the namespace_name and pod_name
echo "Namespace Name: $namespace_name"
echo "Pod Name: $pod_name"

# Remove the server output file on server pod 1, if it exists
# kubectl exec -it -n $namespace_name $pod_name -- /bin/sh -c "touch /root/server.output"
# kubectl exec -it -n $namespace_name $pod_name -- /bin/sh -c "rm /root/server.output"

# Run the client
~/protobuf/ghz-client/clientcall 

kubectl cp "$namespace_name"/"$pod_name":/root/grpc-service-1.output ~/server.output
