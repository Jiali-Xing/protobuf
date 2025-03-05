#!/bin/sh
#Jiali on linux

# to run the test with ghz

ghz --insecure -O html -o test.html \
  -n 50000 \
  -m '{"tokens":"100"}' \
  --proto ./greeting.proto \
  --load-schedule="step" \
  --load-start=2000 --load-step=2000 --load-end=6000 --load-step-duration=2s \
  --call greeting.v3.GreetingService/Greeting \
  0.0.0.0:50051

