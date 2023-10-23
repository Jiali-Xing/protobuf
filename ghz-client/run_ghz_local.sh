#!/bin/bash

# for x in 10 100 1000 5000; do
#     cd /home/ying/Sync/Git/k8s-istio-observe-backend/services/protobuf-grpc
#     bash kill_services.sh
#     bash start_services.sh
#     cd /home/ying/Sync/Git/protobuf/ghz-client
#     go run ./main.go $x
# done

# define var capacity
export CAPACITY=1000
# define the method to run
# export METHOD=compose
export METHOD=echo
# export the INTERCEPT env var as true or false
export INTERCEPT=breakwater
export PROFILING=true
export YAML_FILE=local.yaml

# export CLOUDLAB=false

cd /home/ying/Sync/Git/service-app/services/protobuf-grpc
# activate charon and run msgraph.py
# conda activate charon
# python msgraph.py

bash kill_services.sh
bash start_services.sh
cd /home/ying/Sync/Git/protobuf/ghz-client
./clientcall > ghz.output
cd /home/ying/Sync/Git/service-app/services/protobuf-grpc
bash kill_services.sh
