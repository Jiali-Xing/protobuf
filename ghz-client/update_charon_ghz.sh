#!/bin/bash
 

#!/bin/bash

message="$1"

cd ~/Sync/Git/akis/charon
git add . ; git commit -m "$message"
git push

cd ~/Sync/Git/ghz
go get github.com/tgiannoukos/charon
git add . ; git commit -m "$message"
git push

cd /home/ying/Sync/Git/k8s-istio-observe-backend/services/protobuf-grpc
bash goget_charon.sh

cd /home/ying/Sync/Git/protobuf/ghz-client
go get github.com/Jiali-Xing/ghz
go get github.com/tgiannoukos/charon

for x in 2000; do
    cd /home/ying/Sync/Git/k8s-istio-observe-backend/services/protobuf-grpc
    bash start_services.sh
    cd /home/ying/Sync/Git/protobuf/ghz-client
    go run ./main.go $x
    cd /home/ying/Sync/Git/k8s-istio-observe-backend/services/protobuf-grpc
    bash kill_services.sh
done

cd /home/ying/Sync/Git/protobuf/ghz-results
python visualize.py ./charon_stepup_nclients_2000.json

