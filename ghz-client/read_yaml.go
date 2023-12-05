package main

import (
	"fmt"
	"os"

	// "github.com/labstack/gommon/log"
	"gopkg.in/yaml.v2"
)

var (
	// yamlFile = "msgraph.yaml" if env var MSGRAPH_YAML is not set
	yamlFile = "msgraph.yaml"
)

// Node represents a node in the call graph
type Node struct {
	Name       string   `yaml:"name"`
	Value      string   `yaml:"value"`
	URL        string   `yaml:"URL"`
	Charon     []Config `yaml:"charon"`
	Downstream []string `yaml:"downstream"`
	Server     []Config `yaml:"server"`
	ID         string   `yaml:"id"`
}

// Config represents a configuration entry
type Config struct {
	Name  string `yaml:"name"`
	Value string `yaml:"value"`
}

// Application represents the call graph structure
type Application struct {
	CallGraph map[string][]string `yaml:"callgraph"`
	Interface string              `yaml:"interface"`
}

// func main() {
// 	callGraph := GetCallGraph()
// 	fmt.Println("Call Graph:")
// 	fmt.Println(callGraph)

// 	downstreams := GetDownstreamURLs()
// 	fmt.Println("Downstreams:")
// 	fmt.Println(downstreams)

// 	serverConfigs := GetServerConfigs()
// 	fmt.Println("Server Configurations:")
// 	fmt.Println(serverConfigs)

// 	charonConfigs := GetCharonConfigs()
// 	fmt.Println("Charon Configurations:")
// 	fmt.Println(charonConfigs)
// }

// func SwapKeys(callGraph map[string]map[string][]string) map[string]map[string][]string {
func SwapKeys(applications []Application) map[string]map[string][]string {
	// swappedGraph := make(map[string]map[string][]string)

	// for method, downstreams := range callGraph {
	// 	for node, services := range downstreams {
	// 		if swappedGraph[node] == nil {
	// 			swappedGraph[node] = make(map[string][]string)
	// 		}
	// 		swappedGraph[node][method] = services
	// 	}
	// }

	// return swappedGraph
	downstreamMappings := make(map[string]map[string][]string)

	for _, app := range applications {
		callGraph := app.CallGraph
		interfaceName := app.Interface

		// now, copy the value (slices of string) of callGraph into downstreamMappings[serviceName][interfaceName] (slice of string)
		for upstream, downstreams := range callGraph {
			// assign callGraph[upstream] to downstreamMappings[upstream][interfaceName]
			for _, downstream := range downstreams {
				if downstreamMappings[upstream] == nil {
					downstreamMappings[upstream] = make(map[string][]string)
				}
				if downstreamMappings[upstream][interfaceName] == nil {
					downstreamMappings[upstream][interfaceName] = make([]string, 0)
				}
				downstreamMappings[upstream][interfaceName] = append(downstreamMappings[upstream][interfaceName], downstream)
			}
		}
	}
	return downstreamMappings
}

func GetCallGraph() map[string]map[string][]string {
	callGraph := make(map[string]map[string][]string)

	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return getCallGraphFromEnv()
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Applications []Application `yaml:"applications"`
		Nodes        []Node        `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	callGraph = SwapKeys(cfg.Applications)

	return callGraph
}

func FindURLFromName(nodes []Node, targetName string) (string, bool) {
	for _, node := range nodes {
		if node.Name == targetName {
			return node.URL, true
		}
	}
	return "", false
}

func ReplaceDownstreamNamesWithUrls(callGraph map[string]map[string][]string, nodes []Node) map[string]map[string][]string {
	newCallGraph := make(map[string]map[string][]string)

	for node, downstreams := range callGraph {
		newDownstreams := make(map[string][]string)
		for method, names := range downstreams {
			var urls []string
			for _, name := range names {
				if url, found := FindURLFromName(nodes, name); found {
					urls = append(urls, url)
				}
			}
			newDownstreams[method] = urls
		}
		newCallGraph[node] = newDownstreams
	}

	return newCallGraph
}

func MergeDownstreamURLs(callGraph map[string]map[string][]string) map[string][]string {
	mergedGraph := make(map[string][]string)

	for node, applications := range callGraph {
		for _, urls := range applications {
			if mergedGraph[node] == nil {
				// mergedGraph[node] is a slice of string
				mergedGraph[node] = make([]string, 0)
			}
			// append the url to mergedGraph[node]
			for _, url := range urls {
				if !contains(mergedGraph[node], url) {
					mergedGraph[node] = append(mergedGraph[node], url)
				}
			}
		}
	}

	return mergedGraph
}

// func GetDownstreamNames() map[string][]string {
func GetDownstreamNames() map[string][]string {
	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return getDownstreamsFromEnv()
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Applications []Application `yaml:"applications"`
		Nodes        []Node        `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	// This function returns the downstreams names for a given node, in the same format as previous functions
	callGraph := SwapKeys(cfg.Applications)

	// Merge downstreams across interfaces
	mergedDownstreams := MergeDownstreamURLs(callGraph)

	return mergedDownstreams
}

func GetDownstreamURLs() map[string][]string {
	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return getDownstreamsFromEnv()
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Applications []Application `yaml:"applications"`
		Nodes        []Node        `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	// This function returns the downstreams urls for a given node, in the same format as previous functions
	callGraph := SwapKeys(cfg.Applications)
	// replace the donwstream node name with the node url in the callgraph value (inner map)
	nodes := cfg.Nodes
	downstreamsURLs := ReplaceDownstreamNamesWithUrls(callGraph, nodes)

	// Merge downstreams across interfaces
	mergedDownstreams := MergeDownstreamURLs(downstreamsURLs)

	return mergedDownstreams
}

// func GetNodes() []Node returns the list of nodes
func GetNodes() []Node {
	nodes := make([]Node, 0)

	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return nodes
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Nodes []Node `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	nodes = cfg.Nodes

	return nodes
}

func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

func GetServerConfigs() map[string][]Config {
	serverConfigs := make(map[string][]Config)

	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return serverConfigs
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Applications []Application `yaml:"applications"`
		Nodes        []Node        `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	for _, node := range cfg.Nodes {
		serverConfigs[node.Name] = node.Server
	}

	return serverConfigs
}

func GetCharonConfigs() map[string][]Config {
	charonConfigs := make(map[string][]Config)

	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return charonConfigs
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Applications []Application `yaml:"applications"`
		Nodes        []Node        `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	for _, node := range cfg.Nodes {
		charonConfigs[node.Name] = node.Charon
	}

	return charonConfigs
}

func getCallGraphFromEnv() map[string]map[string][]string {
	callGraph := make(map[string]map[string][]string)

	MY_URL := os.Getenv("MY_URL")
	if MY_URL != "" {
		downstreams := []string{}
		for i := 1; ; i++ {
			URL := os.Getenv("DOWNSTREAM_" + fmt.Sprint(i) + "_URL")
			if URL == "" {
				break
			}
			downstreams = append(downstreams, URL)
		}
		callGraph[MY_URL] = map[string][]string{"echo": downstreams}
	}

	return callGraph
}

func getDownstreamsFromEnv() map[string][]string {
	downstreams := make(map[string][]string)

	MY_URL := os.Getenv("MY_URL")
	if MY_URL != "" {
		downstreamURLs := []string{}
		for i := 1; ; i++ {
			URL := os.Getenv("DOWNSTREAM_" + fmt.Sprint(i) + "_URL")
			if URL == "" {
				break
			}
			downstreamURLs = append(downstreamURLs, URL)
		}
		downstreams[MY_URL] = downstreamURLs
	}

	return downstreams
}

// func getNodeList () gives the list of nodes by their name
func getNodeList() []string {
	nodeList := make([]string, 0)

	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		log.Fatal(err)
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Nodes []Node `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	for _, node := range cfg.Nodes {
		nodeList = append(nodeList, node.Name)
	}

	return nodeList
}

// func getURL(name) gives the url of a node given its name
func getURL(name string) string {
	if _, err := os.Stat(yamlFile); os.IsNotExist(err) {
		return ""
	}

	data, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatal(err)
	}

	var cfg struct {
		Nodes []Node `yaml:"nodes"`
	}

	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		log.Fatal(err)
	}

	for _, node := range cfg.Nodes {
		if node.Name == name {
			return node.URL
		}
	}

	return ""
}
