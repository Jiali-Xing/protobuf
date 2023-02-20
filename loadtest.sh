#!/bin/sh
#Jiali on linux

# to run the test with ghz
ghz --insecure \
  -n 1000 \
  -m '{"tokens":"10"}' \
  --proto ./greeting.proto \
  --call greeting.v3.GreetingService/Greeting \
  0.0.0.0:50051

