#!/bin/bash

for x in 10 100 1000 5000; do
    cd /home/ying/Sync/Git/k8s-istio-observe-backend/services/protobuf-grpc
    bash kill_services.sh
    bash start_services.sh
    cd /home/ying/Sync/Git/protobuf/ghz-client
    go run ./main.go $x
done

