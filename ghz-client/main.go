package main

import (
	"fmt"
	"os"
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
	logLevel    = getEnv("LOG_LEVEL", "info")
	serviceName = getEnv("SERVICE_NAME", "Client")
	message     = getEnv("GREETING", "Hello, from Client!")
	URLServiceA = getEnv("SERVICE_A_URL", "localhost:50051")
	log         = logrus.New()
	intercept   = false
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
	// md := make(map[string]string)
	// md["tokens"] = "100"

	requestGreeting := pb.Greeting{
		Id:       uuid.New().String(),
		Service:  serviceName,
		Message:  message,
		Created:  time.Now().Local().String(),
		Hostname: getHostname(),
	}

	report, err := runner.Run(
		"greeting.v3.GreetingService/Greeting",
		URLServiceA,
		runner.WithProtoFile("../greeting.proto", []string{}),
		runner.WithData(&pb.GreetingRequest{Greeting: &requestGreeting}),
		// runner.WithMetadata(md),
		runner.WithInsecure(true),
		runner.WithTotalRequests(1),
		runner.WithCharon(false),
	)

	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	printer := printer.ReportPrinter{
		Out:    os.Stdout,
		Report: report,
	}

	printer.Print("summary")
}
