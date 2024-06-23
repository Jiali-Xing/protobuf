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
	hotelpb "github.com/Jiali-Xing/hotelproto"
	pb "github.com/Jiali-Xing/protobuf"
	socialpb "github.com/Jiali-Xing/socialproto"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
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

	interceptor = getEnv("INTERCEPT", "charon")

	constantLoadStr = getEnv("CONSTANT_LOAD", "false")
	// make it a boolean
	constantLoad, _ = strconv.ParseBool(constantLoadStr)
	// runDuration     = time.Second * 10 unless specified in the environment variable
	runDuration, _ = time.ParseDuration(getEnv("RUN_DURATION", "20s"))
	capacity       = func() int {
		capacityStr := getEnv("CAPACITY", "4000")
		parsedCapacity, err := strconv.Atoi(capacityStr)
		if err != nil {
			log.Warnf("Invalid capacity value, using default: 4000")
			return 4000
		}
		return parsedCapacity
	}()

	warmup_load, _ = strconv.Atoi(getEnv("WARMUP_LOAD", "1000"))

	concurrency, _ = strconv.Atoi(getEnv("CONCURRENCY", "1000"))

	// loadReduction is true or false
	loadReduction, _ = strconv.ParseBool(getEnv("LOAD_REDUCTION", "false"))
	loadIncrease, _  = strconv.ParseBool(getEnv("LOAD_INCREASE", "false"))

	// loadStart        is from environment variable
	loadStart           = uint(warmup_load)
	loadEnd             = uint(capacity)
	loadStep            = int(loadEnd - loadStart)
	loadStepDuration, _ = time.ParseDuration(getEnv("LOAD_STEP_DURATION", "10s"))

	// read the inferface/method from the environment variable
	method  = getEnv("METHOD", "echo")
	subcall = getEnv("SUBCALL", "sequential")

	// rateLimiting = getEnv("RATE_LIMITING", "true") convert to bool
	rateLimiting, _ = strconv.ParseBool(getEnv("RATE_LIMITING", "true"))
	entry_point     = getEnv("ENTRY_POINT", "nginx-web-server")
	profiling, _    = strconv.ParseBool(getEnv("PROFILING", "true"))

	priceUpdateRate  time.Duration
	tokenUpdateRate  time.Duration
	latencyThreshold time.Duration
	priceStep        int64
	priceStrategy    string
	lazyUpdate       bool

	breakwaterSLO           time.Duration
	breakwaterClientTimeout time.Duration
	breakwaterInitialCredit int64
	// serverSideInterceptOnly = false

	dagorQueuingThresh                time.Duration
	dagorAlpha                        float64
	dagorBeta                         float64
	dagorAdmissionLevelUpdateInterval time.Duration
	dagorUmax                         int

	debug, _ = strconv.ParseBool(getEnv("DEBUG_INFO", "false"))

	// locations = []string{
	// 	"new-york-city-ny-0", "los-angeles-ca-0", "chicago-il-0", "houston-tx-0", "phoenix-az-0", "philadelphia-pa-0", "san-antonio-tx-0", "san-diego-ca-0", "dallas-tx-0", "san-jose-ca-0", "austin-tx-0",
	// }
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

	var req interface{}

	username := "user1"
	password := "password1"
	// we randomize the user and password and hotel location for each request in the frontend of the hotel and social services

	// Declare proto file and method variables
	var protoFile string
	var protoCall string

	switch method {
	case "search-hotel":
		// location := locations[rand.Intn(len(locations))]
		req = &hotelpb.SearchHotelsRequest{
			InDate:   "2023-04-17",
			OutDate:  "2023-04-19",
			Location: "new-york-city-ny-0",
		}

		// Declare proto file and method variables
		protoFile = "../frontend.proto"
		protoCall = "hotelproto.FrontendService/SearchHotels"

	case "reserve-hotel":
		req = &hotelpb.FrontendReservationRequest{
			HotelId:  "1",
			InDate:   "2023-04-17",
			OutDate:  "2023-04-19",
			Rooms:    1,
			Username: username,
			Password: password,
		}
		protoFile = "../frontend.proto"
		protoCall = "hotelproto.FrontendService/FrontendReservation"

	case "home-timeline":
		req = &socialpb.ReadHomeTimelineRequest{
			UserId: username,
		}
		protoFile = "../nginx.proto"
		protoCall = "socialproto.NginxService/ReadHomeTimeline"

	case "user-timeline":
		req = &socialpb.ReadUserTimelineRequest{
			UserId: username,
		}
		protoFile = "../nginx.proto"
		protoCall = "socialproto.NginxService/ReadUserTimeline"

	case "compose":
		req = &socialpb.ComposePostRequest{
			CreatorId: username,
			Text:      "This is a sample post",
		}
		protoFile = "../nginx.proto"
		protoCall = "socialproto.NginxService/ComposePost"

	default:
		requestGreetings := &pb.Greeting{
			Id:       uuid.New().String(),
			Service:  serviceName,
			Message:  message,
			Created:  time.Now().Local().String(),
			Hostname: getHostname(),
		}
		req = &pb.GreetingRequest{Greeting: requestGreetings}
		protoFile = "../greeting.proto"
		protoCall = "greeting.v3.GreetingService/Greeting"
	}

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
		case "TOKEN_UPDATE_RATE":
			tokenUpdateRate, _ = time.ParseDuration(config.Value)
		case "LATENCY_THRESHOLD":
			latencyThreshold, _ = time.ParseDuration(config.Value)
		case "PRICE_STEP":
			priceStep, _ = strconv.ParseInt(config.Value, 10, 64)
		case "PRICE_STRATEGY":
			priceStrategy = config.Value
		case "LAZY_UPDATE":
			lazyUpdate, _ = strconv.ParseBool(config.Value)
		case "RATE_LIMITING":
			rateLimiting, _ = strconv.ParseBool(config.Value)
		// and one optional field: 'SIDE': 'server_only'
		case "BREAKWATER_SLO":
			breakwaterSLO, _ = time.ParseDuration(config.Value)
		case "BREAKWATER_CLIENT_EXPIRATION":
			breakwaterClientTimeout, _ = time.ParseDuration(config.Value)
		case "BREAKWATER_INITIAL_CREDIT":
			breakwaterInitialCredit, _ = strconv.ParseInt(config.Value, 10, 64)
		// case "SIDE":
		// 	// if the side is server_only, then set the serverSideInterceptOnly to true
		// 	if config.Value == "server_only" {
		// 		serverSideInterceptOnly = true
		// 	}
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
		"loadShedding":       false,
		"pinpointQueuing":    false,
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
		"tokenUpdateRate":    tokenUpdateRate,
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
			"compose":              1,
			"home-timeline":        2,
			"user-timeline":        3,
			"S_149998854":          2,
			"S_161142529":          3,
			"S_102000854":          1,
			"hotels-http":          3,
			"reservation-http":     1,
			"user-http":            2,
			"recommendations-http": 4,
			"motivate-set":         1,
			"motivate-get":         2,
			"search-hotel":         1,
			"store-hotel":          2,
			"reserve-hotel":        3,
		},
		EntryService:                 false,
		IsEnduser:                    true,
		QueuingThresh:                dagorQueuingThresh,
		AdmissionLevelUpdateInterval: dagorAdmissionLevelUpdateInterval,
		Alpha:                        dagorAlpha,
		Beta:                         dagorBeta,
		Umax:                         dagorUmax,
		Bmax:                         4,
		Debug:                        debug,
	}

	var defactoInterceptor string
	// the de facto interceptor is dagor when interceptor is set to "dagorf"
	if interceptor == "dagorf" {
		defactoInterceptor = "dagor"
	} else {
		defactoInterceptor = interceptor
	}
	log.Printf("De facto interceptor: %s", defactoInterceptor)

	if loadReduction {
		loadStart = uint(capacity * 90 / 100)
		loadEnd = uint(capacity * 60 / 100)
	}
	if loadIncrease {
		loadStart = uint(capacity * 10 / 100)
		loadEnd = uint(capacity * 40 / 100)
	}
	loadStep = int(loadEnd) - int(loadStart)
	log.Printf("De facto load: Start=%d, End=%d, Step=%d", loadStart, loadEnd, loadStep)

	report, err = runner.Run(
		protoCall,
		URLServiceA,
		runner.WithProtoFile(protoFile, []string{}),
		runner.WithData(req),
		runner.WithMetadata(md),
		runner.WithConcurrency(uint(concurrency)),
		runner.WithConnections(uint(concurrency)),
		runner.WithInsecure(true),
		runner.WithRunDuration(runDuration),
		runner.WithLoadSchedule("step"),
		runner.WithLoadStart(loadStart),
		runner.WithLoadEnd(loadEnd),
		runner.WithLoadStep(loadStep),
		runner.WithLoadStepDuration(loadStepDuration),
		runner.WithMethod(method),
		runner.WithInterceptor(defactoInterceptor),
		runner.WithInterceptorEntry(entry_point),
		runner.WithCharonOptions(charonOptions),
		runner.WithBreakwaterOptions(breakwaterOptions),
		runner.WithDagorOptions(dagorParams),
		runner.WithAsync(true),
		runner.WithEnableCompression(false),
		runner.WithTimeout(time.Second),
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

	// filename := fmt.Sprintf("../ghz-results/xxx json. where xxx tells us if interceptor is enabled or not")
	// and method, constant load or not, capacity, etc.
	// filename := fmt.Sprintf("../ghz-results/.json", enableCharon, method, constantLoadStr, capacity)
	// enableCharonStr := strconv.FormatBool(enableCharon)
	// filename := fmt.Sprintf("../ghz-results/charon-%s-method-%s-constantload-%s-capacity-%d.json", enableCharonStr, method, constantLoadStr, capacity)
	// if enableCharon, name it charon-xxx, otherwise, plain-xxx
	filename := ""

	filename = fmt.Sprintf("../ghz-results/social-%s-%s-%s-capacity-%d.json", method, interceptor, subcall, capacity)

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
