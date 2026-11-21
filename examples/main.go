// Command examples demonstrates building, querying, filtering, compacting,
// and persisting an HNSW index with github.com/BryceWayne/hnsw.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"

	"github.com/BryceWayne/hnsw"
)

var (
	dimension      = flag.Int("d", 128, "Vector dimension")
	numVectors     = flag.Int("n", 1000, "Number of vectors to insert")
	connections    = flag.Int("m", 16, "Max connections per upper layer")
	maxConnections = flag.Int("mmax", 32, "Max connections at layer 0")
	efConstruction = flag.Int("ef", 100, "EF construction parameter")
	efSearch       = flag.Int("ef-search", 50, "EF search parameter")
	searchK        = flag.Int("k", 10, "Number of nearest neighbors to find")
	useEuclidean   = flag.Bool("euclidean", true, "Use Euclidean distance (false for Cosine)")
	outputFile     = flag.String("o", "hnsw_index.bin", "Output file for saved index")
	workerCount    = flag.Int("workers", runtime.GOMAXPROCS(0), "Number of worker threads")
)

type benchmarkResult struct {
	Dimension      int                      `json:"dimension"`
	NumVectors     int                      `json:"num_vectors"`
	Connections    int                      `json:"connections"`
	MaxConnections int                      `json:"max_connections"`
	EfConstruction int                      `json:"ef_construction"`
	DistanceMetric string                   `json:"distance_metric"`
	BuildTime      time.Duration            `json:"build_time"`
	SearchTime     time.Duration            `json:"search_time"`
	Results        []hnsw.SearchResult[int] `json:"results"`
	FilteredCount  int                      `json:"filtered_count"`
	MemoryBytes    int64                    `json:"memory_bytes"`
}

func generateRandomVector(rng *rand.Rand, dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()
	}
	return vec
}

func main() {
	flag.Parse()

	metric := hnsw.MetricEuclidean
	if !*useEuclidean {
		metric = hnsw.MetricCosine
	}

	index, err := hnsw.New[int](
		*dimension,
		hnsw.WithM(*connections),
		hnsw.WithMmax(*maxConnections),
		hnsw.WithEfConstruction(*efConstruction),
		hnsw.WithEfSearch(*efSearch),
		hnsw.WithMetric(metric),
		hnsw.WithSeed(42),
	)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	fmt.Println("Generating vectors...")
	items := make([]hnsw.Item[int], *numVectors)
	for i := range items {
		items[i] = hnsw.Item[int]{
			ID:     i,
			Vector: generateRandomVector(rng, *dimension),
		}
	}

	fmt.Println("Building index...")
	startBuild := time.Now()
	if err := index.InsertBatch(items, hnsw.WithWorkers(*workerCount)); err != nil {
		log.Fatalf("Failed to insert batch: %v", err)
	}
	buildTime := time.Since(startBuild)

	// Demonstrate single Insert (upsert) on key 0 if present.
	if *numVectors > 0 {
		if err := index.Insert(0, generateRandomVector(rng, *dimension)); err != nil {
			log.Fatalf("Failed to upsert vector: %v", err)
		}
	}

	queryVector := generateRandomVector(rng, *dimension)
	startSearch := time.Now()
	results, err := index.Search(queryVector, *searchK, hnsw.WithSearchEf(*efSearch))
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}
	searchTime := time.Since(startSearch)

	filtered, err := index.SearchFilter(queryVector, *searchK, func(id int) bool {
		return id%2 == 0
	})
	if err != nil {
		log.Fatalf("Failed filtered search: %v", err)
	}

	// Demonstrate Delete + Compact lifecycle on a temporary key.
	tempID := *numVectors + 1
	if err := index.Insert(tempID, generateRandomVector(rng, *dimension)); err != nil {
		log.Fatalf("Failed to insert temporary key: %v", err)
	}
	if !index.Delete(tempID) {
		log.Fatalf("Expected Delete(%d) to return true", tempID)
	}
	if err := index.Compact(); err != nil {
		log.Fatalf("Failed to compact index: %v", err)
	}

	stats := index.Stats()
	summary := benchmarkResult{
		Dimension:      stats.Dim,
		NumVectors:     stats.Len,
		Connections:    stats.M,
		MaxConnections: stats.Mmax,
		EfConstruction: stats.EfConstruction,
		DistanceMetric: stats.Metric.String(),
		BuildTime:      buildTime,
		SearchTime:     searchTime,
		Results:        results,
		FilteredCount:  len(filtered),
		MemoryBytes:    stats.MemoryBytes,
	}

	jsonResult, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal result: %v", err)
	}
	fmt.Println(string(jsonResult))

	// Stream round-trip via WriteTo / ReadFrom.
	var buf bytes.Buffer
	if _, err := index.WriteTo(&buf); err != nil {
		log.Fatalf("Failed to write index to buffer: %v", err)
	}
	streamed, err := hnsw.ReadFrom[int](&buf)
	if err != nil {
		log.Fatalf("Failed to read index from buffer: %v", err)
	}
	if streamed.Len() != index.Len() {
		log.Fatalf("Streamed index length %d != original %d", streamed.Len(), index.Len())
	}

	// File persistence round-trip via SaveFile / LoadFile.
	if err := index.SaveFile(*outputFile); err != nil {
		log.Fatalf("Failed to save index: %v", err)
	}
	reloaded, err := hnsw.LoadFile[int](*outputFile)
	if err != nil {
		log.Fatalf("Failed to reload index: %v", err)
	}
	if reloaded.Len() != index.Len() {
		log.Fatalf("Reloaded index length %d != original %d", reloaded.Len(), index.Len())
	}
}
