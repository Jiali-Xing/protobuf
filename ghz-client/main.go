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
	logLevel     = getEnv("LOG_LEVEL", "info")
	serviceName  = getEnv("SERVICE_NAME", "Client")
	message      = getEnv("GREETING", "Hello, from Client!")
	URLServiceA  = getEnv("SERVICE_A_URL", "localhost:50051")
	log          = logrus.New()
	enableCharon = false
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
		runner.WithConcurrency(100),
		runner.WithInsecure(true),
		runner.WithTotalRequests(5000),
		// runner.WithRPS(2000),
		runner.WithLoadStart(1000),
		runner.WithLoadEnd(3000),
		runner.WithLoadStep(200),
		runner.WithLoadStepDuration(1),
		runner.WithLoadStart(1000),
		runner.WithCharon(enableCharon),
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

	file, err := os.Create("baseline_stepup.json")
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
