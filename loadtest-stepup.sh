ghz --insecure -O pretty -o step.json \
  -m '{"tokens":"100", "request-id":"{{.RequestNumber}}", "timestamp":"{{.TimestampUnix}}"}' \
  -n 100000 \
  --proto ./greeting.proto \
  --load-schedule="step" \
  --load-start=500 --load-step=100 --load-end=4000 --load-step-duration=1s \
  --call greeting.v3.GreetingService/Greeting \
  0.0.0.0:50051

