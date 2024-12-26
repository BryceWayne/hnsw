package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/BryceWayne/hnsw"
)

var (
	dimension      = flag.Int("d", 128, "Vector dimension")
	numVectors     = flag.Int("n", 1000, "Number of vectors to insert")
	connections    = flag.Int("m", 16, "Max connections per layer")
	maxConnections = flag.Int("mmax", 32, "Max connections at layer 0")
	efConstruction = flag.Int("ef", 100, "EF construction parameter")
	searchK        = flag.Int("k", 10, "Number of nearest neighbors to find")
	useEuclidean   = flag.Bool("euclidean", true, "Use Euclidean distance (false for Cosine)")
	outputFile     = flag.String("o", "hnsw_index.gob", "Output file for saved index")
	batchSize      = flag.Int("batch", 25, "Batch size for parallel insertions")
)

type BenchmarkResult struct {
	Dimension      int           `json:"dimension"`
	NumVectors     int           `json:"num_vectors"`
	Connections    int           `json:"connections"`
	MaxConnections int           `json:"max_connections"`
	EfConstruction int           `json:"ef_construction"`
	DistanceMetric string        `json:"distance_metric"`
	BuildTime      time.Duration `json:"build_time"`
	SearchTime     time.Duration `json:"search_time"`
	AverageResults []int         `json:"average_results"`
	MemoryUsage    uint64        `json:"memory_usage"`
}

func generateRandomVector(dim int) hnsw.Vector {
	vec := make(hnsw.Vector, dim)
	for i := range vec {
		vec[i] = rand.Float64()
	}
	return vec
}

func getMemoryUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}

func main() {
	flag.Parse()
	rand.Seed(time.Now().UnixNano())

	distanceFunc := hnsw.Euclidean
	if !*useEuclidean {
		distanceFunc = hnsw.Cosine
	}

	index := hnsw.New(
		*dimension,
		*connections,
		*maxConnections,
		*efConstruction,
		distanceFunc,
	)

	startBuild := time.Now()
	numBatches := (*numVectors + *batchSize - 1) / *batchSize
	vectors := make([]hnsw.Vector, *numVectors)

	fmt.Println("Generating vectors...")
	for i := range vectors {
		vectors[i] = generateRandomVector(*dimension)
	}

	fmt.Println("Building index...")
	for b := 0; b < numBatches; b++ {
		start := b * *batchSize
		end := min(start+*batchSize, *numVectors)

		var wg sync.WaitGroup
		for i := start; i < end; i++ {
			wg.Add(1)
			go func(id int, vec hnsw.Vector) {
				defer wg.Done()
				index.Insert(id, vec)
			}(i, vectors[i])
		}
		wg.Wait()

		progress := float64(end) / float64(*numVectors) * 100
		fmt.Printf("\rProgress: %.1f%%", progress)
	}
	fmt.Println()

	buildTime := time.Since(startBuild)

	queryVector := generateRandomVector(*dimension)
	startSearch := time.Now()
	results := index.Search(queryVector, *searchK)
	searchTime := time.Since(startSearch)

	result := BenchmarkResult{
		Dimension:      *dimension,
		NumVectors:     *numVectors,
		Connections:    *connections,
		MaxConnections: *maxConnections,
		EfConstruction: *efConstruction,
		DistanceMetric: "euclidean",
		BuildTime:      buildTime,
		SearchTime:     searchTime,
		AverageResults: results,
		MemoryUsage:    getMemoryUsage(),
	}
	if !*useEuclidean {
		result.DistanceMetric = "cosine"
	}

	jsonResult, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(jsonResult))

	if err := index.Save(*outputFile); err != nil {
		log.Fatalf("Failed to save index: %v", err)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
