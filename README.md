# protobuf
Here I put the protobuf file for gRPC microservices. Copied from garystafford / protobuf

To generate compile the protobuf file:
`protoc --go_out=. --go_opt=paths=source_relative \                            
    --go-grpc_out=. --go-grpc_opt=paths=source_relative greeting.proto`

To install all the depandencies, see the [tutorial](https://grpc.io/docs/languages/go/quickstart/)
