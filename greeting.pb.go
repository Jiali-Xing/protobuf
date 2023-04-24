// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.21.12
// source: greeting.proto

package protobuf

import (
	_ "google.golang.org/genproto/googleapis/api/annotations"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type Greeting struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id       string `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`
	Service  string `protobuf:"bytes,2,opt,name=service,proto3" json:"service,omitempty"`
	Message  string `protobuf:"bytes,3,opt,name=message,proto3" json:"message,omitempty"`
	Created  string `protobuf:"bytes,4,opt,name=created,proto3" json:"created,omitempty"`
	Hostname string `protobuf:"bytes,5,opt,name=hostname,proto3" json:"hostname,omitempty"`
}

func (x *Greeting) Reset() {
	*x = Greeting{}
	if protoimpl.UnsafeEnabled {
		mi := &file_greeting_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Greeting) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Greeting) ProtoMessage() {}

func (x *Greeting) ProtoReflect() protoreflect.Message {
	mi := &file_greeting_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Greeting.ProtoReflect.Descriptor instead.
func (*Greeting) Descriptor() ([]byte, []int) {
	return file_greeting_proto_rawDescGZIP(), []int{0}
}

func (x *Greeting) GetId() string {
	if x != nil {
		return x.Id
	}
	return ""
}

func (x *Greeting) GetService() string {
	if x != nil {
		return x.Service
	}
	return ""
}

func (x *Greeting) GetMessage() string {
	if x != nil {
		return x.Message
	}
	return ""
}

func (x *Greeting) GetCreated() string {
	if x != nil {
		return x.Created
	}
	return ""
}

func (x *Greeting) GetHostname() string {
	if x != nil {
		return x.Hostname
	}
	return ""
}

type GreetingRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Greeting *Greeting `protobuf:"bytes,1,opt,name=greeting,proto3" json:"greeting,omitempty"`
}

func (x *GreetingRequest) Reset() {
	*x = GreetingRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_greeting_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GreetingRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GreetingRequest) ProtoMessage() {}

func (x *GreetingRequest) ProtoReflect() protoreflect.Message {
	mi := &file_greeting_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GreetingRequest.ProtoReflect.Descriptor instead.
func (*GreetingRequest) Descriptor() ([]byte, []int) {
	return file_greeting_proto_rawDescGZIP(), []int{1}
}

func (x *GreetingRequest) GetGreeting() *Greeting {
	if x != nil {
		return x.Greeting
	}
	return nil
}

type GreetingResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Greeting []*Greeting `protobuf:"bytes,1,rep,name=greeting,proto3" json:"greeting,omitempty"`
}

func (x *GreetingResponse) Reset() {
	*x = GreetingResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_greeting_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GreetingResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GreetingResponse) ProtoMessage() {}

func (x *GreetingResponse) ProtoReflect() protoreflect.Message {
	mi := &file_greeting_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GreetingResponse.ProtoReflect.Descriptor instead.
func (*GreetingResponse) Descriptor() ([]byte, []int) {
	return file_greeting_proto_rawDescGZIP(), []int{2}
}

func (x *GreetingResponse) GetGreeting() []*Greeting {
	if x != nil {
		return x.Greeting
	}
	return nil
}

var File_greeting_proto protoreflect.FileDescriptor

var file_greeting_proto_rawDesc = []byte{
	0x0a, 0x0e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x12, 0x0b, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x1a, 0x1c, 0x67,
	0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x61, 0x6e, 0x6e, 0x6f, 0x74, 0x61,
	0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x84, 0x01, 0x0a, 0x08,
	0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x09, 0x52, 0x02, 0x69, 0x64, 0x12, 0x18, 0x0a, 0x07, 0x73, 0x65, 0x72, 0x76,
	0x69, 0x63, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x73, 0x65, 0x72, 0x76, 0x69,
	0x63, 0x65, 0x12, 0x18, 0x0a, 0x07, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x03, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x07, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x12, 0x18, 0x0a, 0x07,
	0x63, 0x72, 0x65, 0x61, 0x74, 0x65, 0x64, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x63,
	0x72, 0x65, 0x61, 0x74, 0x65, 0x64, 0x12, 0x1a, 0x0a, 0x08, 0x68, 0x6f, 0x73, 0x74, 0x6e, 0x61,
	0x6d, 0x65, 0x18, 0x05, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x68, 0x6f, 0x73, 0x74, 0x6e, 0x61,
	0x6d, 0x65, 0x22, 0x44, 0x0a, 0x0f, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x31, 0x0a, 0x08, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e,
	0x67, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x15, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69,
	0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x08,
	0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x22, 0x45, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65,
	0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x31, 0x0a, 0x08,
	0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x15,
	0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65,
	0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x08, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x32,
	0x72, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69,
	0x63, 0x65, 0x41, 0x12, 0x5e, 0x0a, 0x08, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12,
	0x1c, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72,
	0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e,
	0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65,
	0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3,
	0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x67, 0x72, 0x65, 0x65, 0x74,
	0x69, 0x6e, 0x67, 0x32, 0x72, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x53,
	0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x42, 0x12, 0x5e, 0x0a, 0x08, 0x47, 0x72, 0x65, 0x65, 0x74,
	0x69, 0x6e, 0x67, 0x12, 0x1c, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76,
	0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x1a, 0x1d, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e,
	0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x67,
	0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x32, 0x72, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65, 0x74,
	0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x43, 0x12, 0x5e, 0x0a, 0x08, 0x47,
	0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12, 0x1c, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69,
	0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67,
	0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d, 0x2f, 0x61,
	0x70, 0x69, 0x2f, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x32, 0x72, 0x0a, 0x10, 0x47,
	0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x44, 0x12,
	0x5e, 0x0a, 0x08, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12, 0x1c, 0x2e, 0x67, 0x72,
	0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69,
	0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e, 0x67, 0x72, 0x65, 0x65,
	0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f,
	0x12, 0x0d, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x32,
	0x72, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69,
	0x63, 0x65, 0x45, 0x12, 0x5e, 0x0a, 0x08, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12,
	0x1c, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72,
	0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e,
	0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65,
	0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3,
	0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x67, 0x72, 0x65, 0x65, 0x74,
	0x69, 0x6e, 0x67, 0x32, 0x72, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x53,
	0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x46, 0x12, 0x5e, 0x0a, 0x08, 0x47, 0x72, 0x65, 0x65, 0x74,
	0x69, 0x6e, 0x67, 0x12, 0x1c, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76,
	0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x1a, 0x1d, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e,
	0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x67,
	0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x32, 0x72, 0x0a, 0x10, 0x47, 0x72, 0x65, 0x65, 0x74,
	0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x47, 0x12, 0x5e, 0x0a, 0x08, 0x47,
	0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12, 0x1c, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69,
	0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67,
	0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d, 0x2f, 0x61,
	0x70, 0x69, 0x2f, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x32, 0x72, 0x0a, 0x10, 0x47,
	0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x48, 0x12,
	0x5e, 0x0a, 0x08, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x12, 0x1c, 0x2e, 0x67, 0x72,
	0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69,
	0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e, 0x67, 0x72, 0x65, 0x65,
	0x74, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f,
	0x12, 0x0d, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x67, 0x72, 0x65, 0x65, 0x74, 0x69, 0x6e, 0x67, 0x42,
	0x20, 0x5a, 0x1e, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x4a, 0x69,
	0x61, 0x6c, 0x69, 0x2d, 0x58, 0x69, 0x6e, 0x67, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75,
	0x66, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_greeting_proto_rawDescOnce sync.Once
	file_greeting_proto_rawDescData = file_greeting_proto_rawDesc
)

func file_greeting_proto_rawDescGZIP() []byte {
	file_greeting_proto_rawDescOnce.Do(func() {
		file_greeting_proto_rawDescData = protoimpl.X.CompressGZIP(file_greeting_proto_rawDescData)
	})
	return file_greeting_proto_rawDescData
}

var file_greeting_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_greeting_proto_goTypes = []interface{}{
	(*Greeting)(nil),         // 0: greeting.v3.Greeting
	(*GreetingRequest)(nil),  // 1: greeting.v3.GreetingRequest
	(*GreetingResponse)(nil), // 2: greeting.v3.GreetingResponse
}
var file_greeting_proto_depIdxs = []int32{
	0,  // 0: greeting.v3.GreetingRequest.greeting:type_name -> greeting.v3.Greeting
	0,  // 1: greeting.v3.GreetingResponse.greeting:type_name -> greeting.v3.Greeting
	1,  // 2: greeting.v3.GreetingServiceA.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 3: greeting.v3.GreetingServiceB.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 4: greeting.v3.GreetingServiceC.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 5: greeting.v3.GreetingServiceD.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 6: greeting.v3.GreetingServiceE.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 7: greeting.v3.GreetingServiceF.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 8: greeting.v3.GreetingServiceG.Greeting:input_type -> greeting.v3.GreetingRequest
	1,  // 9: greeting.v3.GreetingServiceH.Greeting:input_type -> greeting.v3.GreetingRequest
	2,  // 10: greeting.v3.GreetingServiceA.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 11: greeting.v3.GreetingServiceB.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 12: greeting.v3.GreetingServiceC.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 13: greeting.v3.GreetingServiceD.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 14: greeting.v3.GreetingServiceE.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 15: greeting.v3.GreetingServiceF.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 16: greeting.v3.GreetingServiceG.Greeting:output_type -> greeting.v3.GreetingResponse
	2,  // 17: greeting.v3.GreetingServiceH.Greeting:output_type -> greeting.v3.GreetingResponse
	10, // [10:18] is the sub-list for method output_type
	2,  // [2:10] is the sub-list for method input_type
	2,  // [2:2] is the sub-list for extension type_name
	2,  // [2:2] is the sub-list for extension extendee
	0,  // [0:2] is the sub-list for field type_name
}

func init() { file_greeting_proto_init() }
func file_greeting_proto_init() {
	if File_greeting_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_greeting_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Greeting); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_greeting_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GreetingRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_greeting_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GreetingResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_greeting_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   8,
		},
		GoTypes:           file_greeting_proto_goTypes,
		DependencyIndexes: file_greeting_proto_depIdxs,
		MessageInfos:      file_greeting_proto_msgTypes,
	}.Build()
	File_greeting_proto = out.File
	file_greeting_proto_rawDesc = nil
	file_greeting_proto_goTypes = nil
	file_greeting_proto_depIdxs = nil
}
