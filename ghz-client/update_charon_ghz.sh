#!/bin/bash
 

#!/bin/bash

message="$1"

cd ~/Sync/Git/akis/charon
git add . ; git commit -m "$message"
git push

cd ~/Sync/Git/ghz
go get github.com/tgiannoukos/charon@beta
git add . ; git commit -m "$message"
git push

cd /home/ying/Sync/Git/service-app/services/protobuf-grpc
bash goget_charon.sh

cd /home/ying/Sync/Git/protobuf/ghz-client
go get github.com/Jiali-Xing/ghz
go get github.com/tgiannoukos/charon@beta


