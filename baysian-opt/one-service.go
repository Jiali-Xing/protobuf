// author: Gary A. Stafford
// site: https://programmaticponderings.com
// license: MIT License
// purpose: Service - gRPC/Protobuf

package main

import (
	"context"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/tgiannoukos/charon"

	lrf "github.com/banzaicloud/logrus-runtime-formatter"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	pb "github.com/Jiali-Xing/protobuf"
)

var (
	logLevel    = getEnv("LOG_LEVEL", "info")
	port        = "0.0.0.0:" + getEnv("APP_PORT", "undefined")
	serviceName = "Service " + getEnv("SERVICE_NAME", "undefined")
	message     = getEnv("GREETING", "Hello, from "+serviceName+"!")
	greetings   []*pb.Greeting
	log         = logrus.New()
	intercept   = true
	RT          = 10
)

type greetingServiceServer struct {
	pb.UnimplementedGreetingServiceServer
	pt *charon.PriceTable
}

func busyLoop(c chan<- int, quit chan bool) {
	for {
		if <-quit {
			return
		}
	}
}

func computation(duration int) {
	// Jiali: the following block implements the fake computation
	quit := make(chan bool)
	busyChan := make(chan int)
	go busyLoop(busyChan, quit)
	select {
	case busyResult := <-busyChan:
		log.Println(busyResult)
	case <-time.After(time.Duration(duration) * time.Millisecond):
		// log.Println("timed out")
	}
	quit <- true
	return
}

func (s *greetingServiceServer) Greeting(ctx context.Context, req *pb.GreetingRequest) (*pb.GreetingResponse, error) {
	log.Debugf(serviceName+": GreetingRequest: %v", req.GetGreeting())
	var greetings []*pb.Greeting
	// var URLDownstream string

	requestGreeting := pb.Greeting{
		Id:       uuid.New().String(),
		Service:  serviceName,
		Message:  message,
		Created:  time.Now().Local().String(),
		Hostname: getHostname(),
	}

	greetings = append(greetings, &requestGreeting)

	computation(RT)

	// // Read the URLs for the downstream services from the environment variables
	// for i := 1; URLDownstream != "undefined"; i++ {
	// 	URLDownstream = getEnv("DOWNSTREAM_"+fmt.Sprint(i)+"_URL", "undefined")

	// 	if URLDownstream != "undefined" {
	// 		if i == 1 {
	// 			ctx = context.WithValue(ctx, "pricetable", s.pt)
	// 		}
	// 		fmt.Printf("DOWNSTREAM_"+fmt.Sprint(i)+"_URL %s\n", URLDownstream)
	// 		callGrpcService(ctx, &requestGreeting, URLDownstream)
	// 	}
	// }

	// // Read the URLs for the downstream services when the environment variables are not set
	// for i := 3; i <= len(os.Args[1:]); i++ {
	// 	if i == 3 {
	// 		ctx = context.WithValue(ctx, "pricetable", s.pt)
	// 	}
	// 	callGrpcService(ctx, &requestGreeting, "localhost:"+os.Args[i])
	// 	break
	// }

	return &pb.GreetingResponse{
		Greeting: greetings,
	}, nil
}

func callGrpcService(ctx context.Context, requestGreeting *pb.Greeting, address string) {
	conn, err := createGRPCConn(ctx, address)
	if err != nil {
		log.Fatal(err)
	}
	defer func(conn *grpc.ClientConn) {
		err := conn.Close()
		if err != nil {
			log.Error(err)
		}
	}(conn)

	headersIn, _ := metadata.FromIncomingContext(ctx)
	log.Debugf("headersIn: %s", headersIn)

	client := pb.NewGreetingServiceClient(conn)
	_, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	// append the request-id and timestamp of headersIn to the outgoing context
	ctx = metadata.AppendToOutgoingContext(ctx, "request-id", headersIn["request-id"][0], "timestamp", headersIn["timestamp"][0])

	headersOut, _ := metadata.FromOutgoingContext(ctx)
	log.Debugf("headersOut: %s", headersOut)

	defer cancel()

	responseGreetings, err := client.Greeting(ctx, &pb.GreetingRequest{Greeting: requestGreeting})
	if err != nil {
		log.Debug(err)
	}
	// log.Info(responseGreetings.GetGreeting())

	for _, responseGreeting := range responseGreetings.GetGreeting() {
		greetings = append(greetings, responseGreeting)
	}
}

func createGRPCConn(ctx context.Context, addr string) (*grpc.ClientConn, error) {
	var opts []grpc.DialOption

	pt := ctx.Value("pricetable").(*charon.PriceTable)
	if intercept {
		opts = append(opts,
			grpc.WithUnaryInterceptor(pt.UnaryInterceptorClient),
			grpc.WithInsecure(),
			grpc.WithBlock())
	} else {
		opts = append(opts,
			grpc.WithInsecure(),
			grpc.WithBlock())
	}
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	return conn, nil
}

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

func run() error {
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatal(err)
	}

	// define the initial price for each service, int64
	initialPrice := int64(0)
	// callGraph is a map with key as a string and value as slices of strings
	callGraph := make(map[string][]string)

	// var URLDownstream string
	// // Read the URLs for the downstream services from the environment variables
	// for i := 1; URLDownstream != "undefined"; i++ {
	// 	URLDownstream = getEnv("DOWNSTREAM_"+fmt.Sprint(i)+"_URL", "undefined")

	// 	if URLDownstream != "undefined" {
	// 		fmt.Printf("DOWNSTREAM_"+fmt.Sprint(i)+"_URL %s\n", URLDownstream)
	// 		// append the URL of downstream service as a piece of slices to the value of callGraph
	// 		callGraph["echo"] = append(callGraph["echo"], URLDownstream)
	// 	}
	// }

	// // Read the URLs for the downstream services when the environment variables are not set
	// for i := 3; i <= len(os.Args[1:]); i++ {
	// 	// append the URL of downstream service to the callGraph's slices
	// 	callGraph["echo"] = append(callGraph["echo"], os.Args[i])
	// }

	// make a slice of int64 to store the os args
	var args []int64
	// convert all the os args to int64
	for i := 3; i <= len(os.Args[1:]); i++ {
		arg, err := strconv.ParseInt(os.Args[i], 10, 64)
		if err != nil {
			log.Fatal(err)
		}
		args = append(args, arg)
	}

	charonOptions := map[string]interface{}{
		"initprice":       initialPrice,
		"rateLimiting":    true,
		"loadShedding":    true,
		"pinpointQueuing": true,
		// "pinpointThroughput": true,
		"pinpointLatency": false,
		"debug":           true,
		// "priceStrategy":   "exponential",
		// "priceStrategy":   "proportional",
		"debugFreq":       int64(20000),
		"priceUpdateRate": time.Millisecond * time.Duration(args[0]),
		// "clientTimeOut":    time.Millisecond * time.Duration(args[2]),
		"guidePrice": int64(args[3]),
		// "guidePrice":       int64(20),
		"latencyThreshold": time.Microsecond * time.Duration(args[1]),
		// "clientTimeOut":    time.Millisecond * 35,
		// "throughputThreshold": int64(args[1] * args[0]),
		"priceStep": int64(args[2]),
		// "priceUpdateRate":    time.Second,
	}

	priceTable := charon.NewCharon(
		os.Args[2],
		callGraph,
		charonOptions,
	)

	var grpcServer *grpc.Server
	if intercept {
		grpcServer = grpc.NewServer(grpc.UnaryInterceptor(priceTable.UnaryInterceptor))
	} else {
		grpcServer = grpc.NewServer()
	}
	pb.RegisterGreetingServiceServer(grpcServer, &greetingServiceServer{pt: priceTable})
	return grpcServer.Serve(lis)
}

func init() {
	childFormatter := logrus.JSONFormatter{}
	runtimeFormatter := &lrf.Formatter{ChildFormatter: &childFormatter}
	runtimeFormatter.Line = true
	log.Formatter = runtimeFormatter
	log.Out = os.Stdout
	level, err := logrus.ParseLevel(logLevel)
	if err != nil {
		log.Error(err)
	}
	log.Level = level
}

func main() {

	// fmt.Printf("port %s\n", port)
	// fmt.Printf("serviceName %s\n", serviceName)

	// Set the port and the service name from the command line arguments
	if port == "0.0.0.0:undefined" {
		port = "0.0.0.0:" + os.Args[2]
	}

	if serviceName == "Service undefined" {
		serviceName = "Service " + os.Args[1]
		message = getEnv("GREETING", "Hello, from "+serviceName+"!")
	}

	if err := run(); err != nil {
		log.Fatal(err)
		os.Exit(1)
	}
}
