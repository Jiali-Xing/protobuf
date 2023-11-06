package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"strconv"
	"time"

	bw "github.com/Jiali-Xing/breakwater-grpc/breakwater"
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
	// logLevel     = getEnv("LOG_LEVEL", "info")
	serviceName = getEnv("SERVICE_NAME", "Client")
	message     = getEnv("GREETING", "Hello, from Client!")
	URLServiceA = getEnv("SERVICE_A_URL", "localhost:50051")
	log         = logrus.New()
	// read from the environment variable, and convert it to bool
	// interceptor, _ = strconv.ParseBool(getEnv("INTERCEPT", "true"))
	// interceptor if the INTERCEPT environment variable is set to charon
	interceptor = getEnv("INTERCEPT", "charon")

	loadSchedule = "step"
	// runDuration  = time.Second * 10
	// loadStart    = uint(10000)
	// loadEnd      = uint(30000)
	// loadStep     = 1000
	constantLoadStr = getEnv("CONSTANT_LOAD", "false")
	// make it a boolean
	constantLoad, _ = strconv.ParseBool(constantLoadStr)
	runDuration     = time.Second * 10
	capacity        = func() int {
		capacityStr := getEnv("CAPACITY", "4000")
		parsedCapacity, err := strconv.Atoi(capacityStr)
		if err != nil {
			log.Warnf("Invalid capacity value, using default: 4000")
			return 4000
		}
		return parsedCapacity
	}()
	loadStart        = uint(capacity / 2)
	loadEnd          = uint(capacity * 3 / 2)
	loadStep         = capacity / 2
	loadStepDuration = time.Second * 2

	// read the inferface/method from the environment variable
	method  = getEnv("METHOD", "echo")
	subcall = getEnv("SUBCALL", "sequential")

	// rateLimiting = getEnv("RATE_LIMITING", "true") convert to bool
	rateLimiting, _ = strconv.ParseBool(getEnv("RATE_LIMITING", "true"))
	entry_point     = getEnv("ENTRY_POINT", "nginx-web-server")
	profiling, _    = strconv.ParseBool(getEnv("PROFILING", "true"))

	debug, _ = strconv.ParseBool(getEnv("DEBUG_INFO", "false"))
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
	md["method"] = method
	// md["reqid"]
	// print the metadata used
	for k, v := range md {
		fmt.Printf("Metadata: %s=%s\n", k, v)
	}

	requestGreeting := pb.Greeting{
		Id:       uuid.New().String(),
		Service:  serviceName,
		Message:  message,
		Created:  time.Now().Local().String(),
		Hostname: getHostname(),
	}

	// Get the concurrency number from the second input argument
	// concurrencyStr := os.Args[1]
	// concurrency, err := strconv.Atoi(concurrencyStr)
	// if err != nil {
	// 	fmt.Println("Invalid concurrency value:", concurrencyStr)
	// 	os.Exit(1)
	// }
	concurrency := 1000

	// arg, err := strconv.ParseInt(os.Args[1], 10, 64)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// read the environment

	charonOptions := map[string]interface{}{
		// "rateLimitWaiting": true,
		"rateLimiting": rateLimiting,
		"debug":        debug,
		"debugFreq":    int64(1000),
		"tokensLeft":   int64(0),
		"initprice":    int64(10),
		// "clientTimeOut":   time.Millisecond * time.Duration(arg),
		// "clientTimeOut":   time.Duration(0),
		"tokenUpdateStep": int64(10),
		"tokenUpdateRate": time.Millisecond * 10,
		// "randomRateLimit": int64(35),
		// "invokeAfterRL":   true,
		// "clientBackoff": time.Millisecond * 0,
		"tokenRefillDist": "poisson",
		"tokenStrategy":   "uniform",
		// "latencyThreshold":   time.Millisecond * 7,
	}

	// print all the options
	fmt.Printf("Constant load: %s\n", constantLoadStr)
	fmt.Printf("Capacity: %d\n", capacity)
	fmt.Printf("Run duration: %s\n", runDuration)
	fmt.Printf("Interceptor is: %s\n", interceptor)
	// fmt.Printf("Method: %s\n", method)
	fmt.Printf("Charon options: %v\n", charonOptions)

	fmt.Printf("Profiling: %v\n", profiling)
	fmt.Printf("Debug Enabled: %v\n", debug)

	var err error
	var report *runner.Report

	if profiling {
		// ... Inside your main or init function
		f, err := os.Create(serviceName + ".pprof")
		if err != nil {
			log.Fatalf("Could not create pprof file: %v", err)
			return
		}

		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("Could not start CPU profile: %v", err)
			return
		} else {
			log.Println("Started CPU profiling at", time.Now().Local().String())
		}

		// stop the CPU profile and write the profiling data to the file after 9 seconds
		go func() {
			time.Sleep(9 * time.Second)
			pprof.StopCPUProfile()
			f.Close()
			log.Println("Stopped CPU profiling at", time.Now().Local().String())
		}()
	}

	breakwaterOptions := bw.BWParametersDefault
	breakwaterOptions.Verbose = debug

	if constantLoad {
		report, err = runner.Run(
			"greeting.v3.GreetingService/Greeting",
			URLServiceA,
			runner.WithProtoFile("../greeting.proto", []string{}),
			runner.WithData(&pb.GreetingRequest{Greeting: &requestGreeting}),
			runner.WithMetadata(md),
			runner.WithConcurrency(uint(concurrency)),
			runner.WithConnections(uint(concurrency)),
			runner.WithInsecure(true),
			runner.WithRPS(uint(capacity)),
			runner.WithRunDuration(runDuration),
			runner.WithLoadSchedule("const"),
			runner.WithMethod(method),
			runner.WithInterceptor(interceptor),
			runner.WithInterceptorEntry(entry_point),
			runner.WithCharonOptions(charonOptions),
			runner.WithBreakwaterOptions(breakwaterOptions),
			runner.WithEnableCompression(false),
			runner.WithTimeout(100*time.Millisecond),
		)
	} else {
		report, err = runner.Run(
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
			runner.WithLoadStepDuration(loadStepDuration),
			runner.WithMethod(method),
			runner.WithInterceptor(interceptor),
			runner.WithInterceptorEntry(entry_point),
			runner.WithCharonOptions(charonOptions),
			runner.WithBreakwaterOptions(breakwaterOptions),
			runner.WithEnableCompression(false),
			runner.WithTimeout(100*time.Millisecond),
		)
	}

	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	toStd := printer.ReportPrinter{
		Out:    os.Stdout,
		Report: report,
	}

	toStd.Print("summary")

	// filename := fmt.Sprintf("../ghz-results/xxx json. where xxx tells us if interceptor is enabled or not")
	// and method, constant load or not, capacity, etc.
	// filename := fmt.Sprintf("../ghz-results/.json", enableCharon, method, constantLoadStr, capacity)
	// enableCharonStr := strconv.FormatBool(enableCharon)
	// filename := fmt.Sprintf("../ghz-results/charon-%s-method-%s-constantload-%s-capacity-%d.json", enableCharonStr, method, constantLoadStr, capacity)
	// if enableCharon, name it charon-xxx, otherwise, plain-xxx
	filename := ""
	// if interceptor == "charon" || interceptor == "breakwater" {
	// else {
	filename = fmt.Sprintf("../ghz-results/social-%s-%s-%s-capacity-%d.json", method, interceptor, subcall, capacity)
	// writeToFile(filename, report)
	// }

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
