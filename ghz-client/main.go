package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"strconv"
	"time"

	bw "github.com/Jiali-Xing/breakwater-grpc/breakwater"
	dagor "github.com/Jiali-Xing/dagor-grpc/dagor"

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

	// loadSchedule = "step"
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

	// loadReduction is true or false
	loadReduction, _ = strconv.ParseBool(getEnv("LOAD_REDUCTION", "false"))

	loadStart        = uint(capacity / 2)
	loadEnd          = uint(capacity)
	loadStep         = capacity / 2
	loadStepDuration = time.Second * 3

	// read the inferface/method from the environment variable
	method  = getEnv("METHOD", "echo")
	subcall = getEnv("SUBCALL", "sequential")

	// rateLimiting = getEnv("RATE_LIMITING", "true") convert to bool
	rateLimiting, _ = strconv.ParseBool(getEnv("RATE_LIMITING", "true"))
	entry_point     = getEnv("ENTRY_POINT", "nginx-web-server")
	profiling, _    = strconv.ParseBool(getEnv("PROFILING", "true"))
	// latencyThreshold = getEnv("LATENCY_THRESHOLD", "10ms") convert to time.Duration
	// latencyThreshold, _ = time.ParseDuration(getEnv("LATENCY_THRESHOLD", "10ms"))

	priceUpdateRate  time.Duration
	latencyThreshold time.Duration
	priceStep        int64
	priceStrategy    string
	lazyUpdate       bool

	breakwaterSLO           time.Duration
	breakwaterClientTimeout time.Duration
	breakwaterInitialCredit int64
	serverSideInterceptOnly = false

	dagorQueuingThresh                time.Duration
	dagorAlpha                        float64
	dagorBeta                         float64
	dagorAdmissionLevelUpdateInterval time.Duration
	dagorUmax                         int

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

	concurrency := 1000

	interceptorConfigs := GetCharonConfigs()
	fmt.Println("Charon Configurations:")
	fmt.Println(interceptorConfigs[serviceName])

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
			log.Println("[ghz client] Started CPU profiling at", time.Now().Local().String())
		}

		// stop the CPU profile and write the profiling data to the file after 9 seconds
		go func() {
			time.Sleep(9 * time.Second)
			pprof.StopCPUProfile()
			f.Close()
			log.Println("[ghz client] Stopped CPU profiling at", time.Now().Local().String())
		}()
	}

	// Iterate over the charonConfig slice and assign values based on the Name field
	for _, config := range interceptorConfigs[serviceName] {
		switch config.Name {
		case "INTERCEPT":
			interceptor = config.Value
		case "PRICE_UPDATE_RATE":
			priceUpdateRate, _ = time.ParseDuration(config.Value)
		case "LATENCY_THRESHOLD":
			latencyThreshold, _ = time.ParseDuration(config.Value)
		case "PRICE_STEP":
			priceStep, _ = strconv.ParseInt(config.Value, 10, 64)
		case "PRICE_STRATEGY":
			priceStrategy = config.Value
		case "LAZY_UPDATE":
			lazyUpdate, _ = strconv.ParseBool(config.Value)
		// and one optional field: 'SIDE': 'server_only'
		case "BREAKWATER_SLO":
			breakwaterSLO, _ = time.ParseDuration(config.Value)
		case "BREAKWATER_CLIENT_EXPIRATION":
			breakwaterClientTimeout, _ = time.ParseDuration(config.Value)
		case "BREAKWATER_INITIAL_CREDIT":
			breakwaterInitialCredit, _ = strconv.ParseInt(config.Value, 10, 64)
		case "SIDE":
			// if the side is server_only, then set the serverSideInterceptOnly to true
			if config.Value == "server_only" {
				serverSideInterceptOnly = true
			}
		case "DAGOR_QUEUING_THRESHOLD":
			dagorQueuingThresh, _ = time.ParseDuration(config.Value)
		case "DAGOR_ALPHA":
			dagorAlpha, _ = strconv.ParseFloat(config.Value, 64)
		case "DAGOR_BETA":
			dagorBeta, _ = strconv.ParseFloat(config.Value, 64)
		case "DAGOR_ADMISSION_LEVEL_UPDATE_INTERVAL":
			dagorAdmissionLevelUpdateInterval, _ = time.ParseDuration(config.Value)
		case "DAGOR_UMAX":
			dagorUmax, _ = strconv.Atoi(config.Value)
		}
	}

	charonOptions := map[string]interface{}{
		"rateLimiting":       rateLimiting,
		"loadShedding":       true,
		"pinpointQueuing":    true,
		"pinpointThroughput": false,
		"pinpointLatency":    false,
		"debug":              debug,
		"lazyResponse":       lazyUpdate,
		"priceUpdateRate":    priceUpdateRate,
		"guidePrice":         int64(-1),
		"priceStrategy":      priceStrategy,
		"latencyThreshold":   latencyThreshold,
		"priceStep":          priceStep,
		"priceAggregation":   "maximal",
		"tokensLeft":         int64(0),
		"initprice":          int64(10),
		"tokenUpdateStep":    int64(10),
		"tokenUpdateRate":    time.Millisecond * 10,
		"tokenRefillDist":    "poisson",
		"tokenStrategy":      "uniform",
	}

	// 	// if interceptor is false, then the priceTable is nil
	breakwaterOptions := bw.BWParametersDefault

	breakwaterOptions.Verbose = debug
	breakwaterOptions.SLO = breakwaterSLO.Microseconds()
	breakwaterOptions.ClientExpiration = breakwaterClientTimeout.Microseconds()
	breakwaterOptions.InitialCredits = breakwaterInitialCredit
	breakwaterOptions.LoadShedding = false
	breakwaterOptions.ServerSide = false

	// print the breakwater config for debugging
	// log.Printf("Breakwater Config: %v", breakwaterOptions)

	// Define Dagor parameters (assuming you have these values defined)
	dagorParams := dagor.DagorParam{
		// Set the parameters accordingly
		NodeName: serviceName,
		BusinessMap: map[string]int{
			"compose":       1,
			"home-timeline": 2,
			"user-timeline": 2,
			"S_149998854":   1,
			"S_161142529":   2,
			"S_102000854":   2,
		},
		EntryService:                 false,
		IsEnduser:                    true,
		QueuingThresh:                dagorQueuingThresh,
		AdmissionLevelUpdateInterval: dagorAdmissionLevelUpdateInterval,
		Alpha:                        dagorAlpha,
		Beta:                         dagorBeta,
		Umax:                         dagorUmax,
		Bmax:                         9,
		Debug:                        debug,
	}

	if loadReduction {
		// capacity = capacity / 4
		// constantLoad = true
		// runDuration = time.Second * 3
		loadStart = uint(capacity * 3 / 4)
		loadEnd = uint(capacity / 2)
		loadStep = -capacity / 2
		// loadStepDuration = time.Second * 3
	}

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
			runner.WithDagorOptions(dagorParams),
			runner.WithAsync(true),
			runner.WithEnableCompression(false),
			runner.WithTimeout(time.Second),
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
			runner.WithRunDuration(runDuration),
			runner.WithLoadSchedule("step"),
			runner.WithLoadStart(loadStart),
			runner.WithLoadEnd(loadEnd),
			runner.WithLoadStep(loadStep),
			runner.WithLoadStepDuration(loadStepDuration),
			runner.WithMethod(method),
			runner.WithInterceptor(interceptor),
			runner.WithInterceptorEntry(entry_point),
			runner.WithCharonOptions(charonOptions),
			runner.WithBreakwaterOptions(breakwaterOptions),
			runner.WithDagorOptions(dagorParams),
			runner.WithAsync(true),
			runner.WithEnableCompression(false),
			runner.WithTimeout(time.Second),
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
