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

# Print the `CAPACITY` env var of the current pod
echo "Capacity: $CAPACITY"

# Run the client
~/protobuf/ghz-client/clientcall 


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