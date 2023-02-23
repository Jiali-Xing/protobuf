#!/bin/sh
#Jiali on linux

# to run the test with ghz
ghz --insecure \
  -n 10000 \
  -m '{"tokens":"100"}' \
  --proto ./greeting.proto \
  --call greeting.v3.GreetingService/Greeting \
  0.0.0.0:50051

