package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func main() {
	// Set the seed for random number generation
	rand.Seed(time.Now().UnixNano())

	lambda := 1.0 / 20.0 // Lambda value of 1/20

	for i := 0; i < 10; i++ {
		// Generate a random number from the exponential distribution
		x := -math.Log(1-rand.Float64()) / lambda

		fmt.Printf("Random number %d: %f\n", i+1, x)
	}
}
