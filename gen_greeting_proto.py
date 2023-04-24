# Define the base text for the file
base_text = """syntax = "proto3";
package greeting.v3;

import "google/api/annotations.proto";

option go_package = "github.com/Jiali-Xing/protobuf";

message Greeting {{
  string id = 1;
  string service = 2;
  string message = 3;
  string created = 4;
  string hostname = 5;
}}

message GreetingRequest {{
  Greeting greeting = 1;
}}

message GreetingResponse {{
  repeated Greeting greeting = 1;
}}

{service_text}
"""

# Define the service text template
service_text_template = """
service {service_name} {{
  rpc Greeting (GreetingRequest) returns (GreetingResponse) {{
    option (google.api.http) = {{
      get: "/api/{service_path}"
    }};
  }}
}}
"""

# Define the service names and paths
services = [
    {
        "name": "GreetingServiceA",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceB",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceC",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceD",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceE",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceF",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceG",
        "path": "greeting"
    },
    {
        "name": "GreetingServiceH",
        "path": "greeting"
    }
]

# Generate the service text for each service
service_texts = []
for service in services:
    service_name = service["name"]
    service_path = service["path"]
    service_text = service_text_template.format(
        service_name=service_name,
        service_path=service_path
    )
    service_texts.append(service_text)

# Combine the base text with the service texts
full_text = base_text.format(service_text="".join(service_texts))

# Write the text to a file
with open("greeting.proto", "w") as f:
    f.write(full_text)
