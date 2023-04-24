// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v3.21.12
// source: greeting.proto

package protobuf

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// GreetingServiceAClient is the client API for GreetingServiceA service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceAClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceAClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceAClient(cc grpc.ClientConnInterface) GreetingServiceAClient {
	return &greetingServiceAClient{cc}
}

func (c *greetingServiceAClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceA/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceAServer is the server API for GreetingServiceA service.
// All implementations must embed UnimplementedGreetingServiceAServer
// for forward compatibility
type GreetingServiceAServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceAServer()
}

// UnimplementedGreetingServiceAServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceAServer struct {
}

func (UnimplementedGreetingServiceAServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceAServer) mustEmbedUnimplementedGreetingServiceAServer() {}

// UnsafeGreetingServiceAServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceAServer will
// result in compilation errors.
type UnsafeGreetingServiceAServer interface {
	mustEmbedUnimplementedGreetingServiceAServer()
}

func RegisterGreetingServiceAServer(s grpc.ServiceRegistrar, srv GreetingServiceAServer) {
	s.RegisterService(&GreetingServiceA_ServiceDesc, srv)
}

func _GreetingServiceA_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceAServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceA/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceAServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceA_ServiceDesc is the grpc.ServiceDesc for GreetingServiceA service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceA_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceA",
	HandlerType: (*GreetingServiceAServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceA_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceBClient is the client API for GreetingServiceB service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceBClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceBClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceBClient(cc grpc.ClientConnInterface) GreetingServiceBClient {
	return &greetingServiceBClient{cc}
}

func (c *greetingServiceBClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceB/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceBServer is the server API for GreetingServiceB service.
// All implementations must embed UnimplementedGreetingServiceBServer
// for forward compatibility
type GreetingServiceBServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceBServer()
}

// UnimplementedGreetingServiceBServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceBServer struct {
}

func (UnimplementedGreetingServiceBServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceBServer) mustEmbedUnimplementedGreetingServiceBServer() {}

// UnsafeGreetingServiceBServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceBServer will
// result in compilation errors.
type UnsafeGreetingServiceBServer interface {
	mustEmbedUnimplementedGreetingServiceBServer()
}

func RegisterGreetingServiceBServer(s grpc.ServiceRegistrar, srv GreetingServiceBServer) {
	s.RegisterService(&GreetingServiceB_ServiceDesc, srv)
}

func _GreetingServiceB_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceBServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceB/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceBServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceB_ServiceDesc is the grpc.ServiceDesc for GreetingServiceB service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceB_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceB",
	HandlerType: (*GreetingServiceBServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceB_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceCClient is the client API for GreetingServiceC service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceCClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceCClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceCClient(cc grpc.ClientConnInterface) GreetingServiceCClient {
	return &greetingServiceCClient{cc}
}

func (c *greetingServiceCClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceC/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceCServer is the server API for GreetingServiceC service.
// All implementations must embed UnimplementedGreetingServiceCServer
// for forward compatibility
type GreetingServiceCServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceCServer()
}

// UnimplementedGreetingServiceCServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceCServer struct {
}

func (UnimplementedGreetingServiceCServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceCServer) mustEmbedUnimplementedGreetingServiceCServer() {}

// UnsafeGreetingServiceCServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceCServer will
// result in compilation errors.
type UnsafeGreetingServiceCServer interface {
	mustEmbedUnimplementedGreetingServiceCServer()
}

func RegisterGreetingServiceCServer(s grpc.ServiceRegistrar, srv GreetingServiceCServer) {
	s.RegisterService(&GreetingServiceC_ServiceDesc, srv)
}

func _GreetingServiceC_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceCServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceC/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceCServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceC_ServiceDesc is the grpc.ServiceDesc for GreetingServiceC service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceC_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceC",
	HandlerType: (*GreetingServiceCServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceC_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceDClient is the client API for GreetingServiceD service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceDClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceDClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceDClient(cc grpc.ClientConnInterface) GreetingServiceDClient {
	return &greetingServiceDClient{cc}
}

func (c *greetingServiceDClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceD/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceDServer is the server API for GreetingServiceD service.
// All implementations must embed UnimplementedGreetingServiceDServer
// for forward compatibility
type GreetingServiceDServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceDServer()
}

// UnimplementedGreetingServiceDServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceDServer struct {
}

func (UnimplementedGreetingServiceDServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceDServer) mustEmbedUnimplementedGreetingServiceDServer() {}

// UnsafeGreetingServiceDServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceDServer will
// result in compilation errors.
type UnsafeGreetingServiceDServer interface {
	mustEmbedUnimplementedGreetingServiceDServer()
}

func RegisterGreetingServiceDServer(s grpc.ServiceRegistrar, srv GreetingServiceDServer) {
	s.RegisterService(&GreetingServiceD_ServiceDesc, srv)
}

func _GreetingServiceD_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceDServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceD/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceDServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceD_ServiceDesc is the grpc.ServiceDesc for GreetingServiceD service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceD_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceD",
	HandlerType: (*GreetingServiceDServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceD_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceEClient is the client API for GreetingServiceE service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceEClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceEClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceEClient(cc grpc.ClientConnInterface) GreetingServiceEClient {
	return &greetingServiceEClient{cc}
}

func (c *greetingServiceEClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceE/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceEServer is the server API for GreetingServiceE service.
// All implementations must embed UnimplementedGreetingServiceEServer
// for forward compatibility
type GreetingServiceEServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceEServer()
}

// UnimplementedGreetingServiceEServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceEServer struct {
}

func (UnimplementedGreetingServiceEServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceEServer) mustEmbedUnimplementedGreetingServiceEServer() {}

// UnsafeGreetingServiceEServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceEServer will
// result in compilation errors.
type UnsafeGreetingServiceEServer interface {
	mustEmbedUnimplementedGreetingServiceEServer()
}

func RegisterGreetingServiceEServer(s grpc.ServiceRegistrar, srv GreetingServiceEServer) {
	s.RegisterService(&GreetingServiceE_ServiceDesc, srv)
}

func _GreetingServiceE_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceEServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceE/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceEServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceE_ServiceDesc is the grpc.ServiceDesc for GreetingServiceE service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceE_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceE",
	HandlerType: (*GreetingServiceEServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceE_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceFClient is the client API for GreetingServiceF service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceFClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceFClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceFClient(cc grpc.ClientConnInterface) GreetingServiceFClient {
	return &greetingServiceFClient{cc}
}

func (c *greetingServiceFClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceF/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceFServer is the server API for GreetingServiceF service.
// All implementations must embed UnimplementedGreetingServiceFServer
// for forward compatibility
type GreetingServiceFServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceFServer()
}

// UnimplementedGreetingServiceFServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceFServer struct {
}

func (UnimplementedGreetingServiceFServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceFServer) mustEmbedUnimplementedGreetingServiceFServer() {}

// UnsafeGreetingServiceFServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceFServer will
// result in compilation errors.
type UnsafeGreetingServiceFServer interface {
	mustEmbedUnimplementedGreetingServiceFServer()
}

func RegisterGreetingServiceFServer(s grpc.ServiceRegistrar, srv GreetingServiceFServer) {
	s.RegisterService(&GreetingServiceF_ServiceDesc, srv)
}

func _GreetingServiceF_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceFServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceF/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceFServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceF_ServiceDesc is the grpc.ServiceDesc for GreetingServiceF service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceF_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceF",
	HandlerType: (*GreetingServiceFServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceF_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceGClient is the client API for GreetingServiceG service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceGClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceGClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceGClient(cc grpc.ClientConnInterface) GreetingServiceGClient {
	return &greetingServiceGClient{cc}
}

func (c *greetingServiceGClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceG/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceGServer is the server API for GreetingServiceG service.
// All implementations must embed UnimplementedGreetingServiceGServer
// for forward compatibility
type GreetingServiceGServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceGServer()
}

// UnimplementedGreetingServiceGServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceGServer struct {
}

func (UnimplementedGreetingServiceGServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceGServer) mustEmbedUnimplementedGreetingServiceGServer() {}

// UnsafeGreetingServiceGServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceGServer will
// result in compilation errors.
type UnsafeGreetingServiceGServer interface {
	mustEmbedUnimplementedGreetingServiceGServer()
}

func RegisterGreetingServiceGServer(s grpc.ServiceRegistrar, srv GreetingServiceGServer) {
	s.RegisterService(&GreetingServiceG_ServiceDesc, srv)
}

func _GreetingServiceG_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceGServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceG/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceGServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceG_ServiceDesc is the grpc.ServiceDesc for GreetingServiceG service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceG_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceG",
	HandlerType: (*GreetingServiceGServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceG_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}

// GreetingServiceHClient is the client API for GreetingServiceH service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type GreetingServiceHClient interface {
	Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error)
}

type greetingServiceHClient struct {
	cc grpc.ClientConnInterface
}

func NewGreetingServiceHClient(cc grpc.ClientConnInterface) GreetingServiceHClient {
	return &greetingServiceHClient{cc}
}

func (c *greetingServiceHClient) Greeting(ctx context.Context, in *GreetingRequest, opts ...grpc.CallOption) (*GreetingResponse, error) {
	out := new(GreetingResponse)
	err := c.cc.Invoke(ctx, "/greeting.v3.GreetingServiceH/Greeting", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// GreetingServiceHServer is the server API for GreetingServiceH service.
// All implementations must embed UnimplementedGreetingServiceHServer
// for forward compatibility
type GreetingServiceHServer interface {
	Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error)
	mustEmbedUnimplementedGreetingServiceHServer()
}

// UnimplementedGreetingServiceHServer must be embedded to have forward compatible implementations.
type UnimplementedGreetingServiceHServer struct {
}

func (UnimplementedGreetingServiceHServer) Greeting(context.Context, *GreetingRequest) (*GreetingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Greeting not implemented")
}
func (UnimplementedGreetingServiceHServer) mustEmbedUnimplementedGreetingServiceHServer() {}

// UnsafeGreetingServiceHServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to GreetingServiceHServer will
// result in compilation errors.
type UnsafeGreetingServiceHServer interface {
	mustEmbedUnimplementedGreetingServiceHServer()
}

func RegisterGreetingServiceHServer(s grpc.ServiceRegistrar, srv GreetingServiceHServer) {
	s.RegisterService(&GreetingServiceH_ServiceDesc, srv)
}

func _GreetingServiceH_Greeting_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GreetingRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(GreetingServiceHServer).Greeting(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/greeting.v3.GreetingServiceH/Greeting",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(GreetingServiceHServer).Greeting(ctx, req.(*GreetingRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// GreetingServiceH_ServiceDesc is the grpc.ServiceDesc for GreetingServiceH service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var GreetingServiceH_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "greeting.v3.GreetingServiceH",
	HandlerType: (*GreetingServiceHServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Greeting",
			Handler:    _GreetingServiceH_Greeting_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "greeting.proto",
}
