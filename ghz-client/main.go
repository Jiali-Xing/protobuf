package main

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/Jiali-Xing/ghz/printer"
	"github.com/Jiali-Xing/ghz/runner"
	pb "github.com/Jiali-Xing/protobuf"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	// printer_wo_charon "github.com/bojand/ghz/printer"
	// runner_wo_charon "github.com/bojand/ghz/runner"
)

// ghz --insecure -O html -o test.html \
//   -n 50000 \
//   -m '{"tokens":"100"}' \
//   --proto ./greeting.proto \
//   --load-schedule="step" \
//   --load-start=2000 --load-step=2000 --load-end=6000 --load-step-duration=2s \
//   --call \
//   0.0.0.0:50051

var (
	logLevel     = getEnv("LOG_LEVEL", "info")
	serviceName  = getEnv("SERVICE_NAME", "Client")
	message      = getEnv("GREETING", "Hello, from Client!")
	URLServiceA  = getEnv("SERVICE_A_URL", "localhost:50051")
	log          = logrus.New()
	enableCharon = true
	runDuration  = time.Second*5 + 2
	loadSchedule = "step"
	loadStart    = uint(20000)
	loadEnd      = uint(50000)
	loadStep     = 5000
)

func getHostname() string {
	hostname, err := os.Hostname()
	if err != nil {
		log.Error(err)
	}
	return hostname
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func main() {
	md := make(map[string]string)
	md["request-id"] = "{{.RequestNumber}}"
	md["timestamp"] = "{{.TimestampUnix}}"

	// data := make(map[string]string)
	// data["order_id"] = "{{newUUID}}"
	// data["item_id"] = "{{newUUID}}"
	// data["sku"] = "{{randomString 8 }}"
	// data["product_name"] = "{{randomString 0}}"

	requestGreeting := pb.Greeting{
		Id:       uuid.New().String(),
		Service:  serviceName,
		Message:  message,
		Created:  time.Now().Local().String(),
		Hostname: getHostname(),
	}

	// Get the concurrency number from the second input argument
	concurrencyStr := os.Args[1]
	concurrency, err := strconv.Atoi(concurrencyStr)
	if err != nil {
		fmt.Println("Invalid concurrency value:", concurrencyStr)
		os.Exit(1)
	}

	charonOptions := map[string]interface{}{
		"rateLimiting":       true,
		"loadShedding":       true,
		"pinpointQueuing":    false,
		"pinpointLatency":    false,
		"pinpointThroughput": true,
		"debug":              false,
		"debugFreq":          int64(2000),
		"tokensLeft":         int64(0),
		// "latencyThreshold":   time.Millisecond * 7,
	}

	report, err := runner.Run(
		"greeting.v3.GreetingService/Greeting",
		URLServiceA,
		runner.WithProtoFile("../greeting.proto", []string{}),
		runner.WithData(&pb.GreetingRequest{Greeting: &requestGreeting}),
		runner.WithMetadata(md),
		runner.WithConcurrency(uint(concurrency)),
		runner.WithConnections(uint(concurrency)),
		runner.WithInsecure(true),
		// runner.WithTotalRequests(3),
		// runner.WithRPS(2000),
		// runner.WithAsync(true),
		runner.WithRunDuration(runDuration),
		runner.WithLoadSchedule(loadSchedule),
		runner.WithLoadStart(loadStart),
		runner.WithLoadEnd(loadEnd),
		runner.WithLoadStep(loadStep),
		runner.WithLoadStepDuration(time.Second*1),
		runner.WithCharon(enableCharon),
		runner.WithCharonEntry("50051"),
		runner.WithCharonOptions(charonOptions),
	)

	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	toStd := printer.ReportPrinter{
		Out:    os.Stdout,
		Report: report,
	}

	toStd.Print("summary")

	var filename string
	if enableCharon {
		filename = fmt.Sprintf("../ghz-results/charon_stepup_nclients_%d.json", concurrency)
	} else {
		filename = fmt.Sprintf("../ghz-results/baseline_stepup_nclients_%d.json", concurrency)
	}

	file, err := os.Create(filename)

	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	toFile := printer.ReportPrinter{
		Out:    file,
		Report: report,
	}

	toFile.Print("pretty")
}
