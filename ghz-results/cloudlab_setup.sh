#!/bin/bash
# The folders are specific to Jiali laptop. Change them if you want to use it.
# cloudlab is in my ssh/config 

rsync -azP ~/Sync/Git/protobuf cloudlab:/users/jiali/
rsync -azP ~/Sync/Git/k8s-istio-observe-backend cloudlab:/users/jiali/
rsync -azP ~/.ssh/cloudlab cloudlab:/users/jiali/.ssh/

ssh cloudlab <<-'ENDSSH'
    #commands to run on remote host
    wget https://go.dev/dl/go1.20.2.linux-amd64.tar.gz
    sudo su
    rm -rf /usr/local/go && tar -C /usr/local -xzf go1.20.2.linux-amd64.tar.gz
    exit
 
    export PATH=$PATH:/usr/local/go/bin
    git config --global url.git@github.com:.insteadOf https://github.com/
    go env -w GOPRIVATE=github.com/tgiannoukos/charon
    mv ~/protobuf/config ~/.ssh/config

    cd k8s-istio-observe-backend/services/protobuf-grpc/
    bash goget_charon.sh 
    bash start_services.sh

    cd ~/protobuf
    wget https://github.com/bojand/ghz/releases/download/v0.114.0/ghz-linux-x86_64.tar.gz
    tar -xvf ghz-linux-x86_64.tar.gz

    ./ghz --insecure  -n 100000  -m '{"tokens":"100"}'  --proto ./greeting.proto  --call greeting.v3.GreetingService/Greeting  0.0.0.0:50051

ENDSSH