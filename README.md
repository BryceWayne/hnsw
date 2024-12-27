# HNSW (Hierarchical Navigable Small World) Implementation in Go

High-performance HNSW implementation with AVX2 optimizations and parallel search.

## Installation

```bash
go get github.com/BryceWayne/hnsw
go get golang.org/x/sys/cpu  # Required for AVX2 detection
```

Build with AVX2 support:
```bash
GOAMD64=v3 go build
```

## Quick Start

```go
import (
    "github.com/BryceWayne/hnsw"
    "golang.org/x/sys/cpu"
)

func main() {
    if cpu.X86.HasAVX2 {
        println("Using AVX2 optimizations")
    }

    // Create index
    index := hnsw.New(
        128,    // dimension
        16,     // M (max connections per layer)
        32,     // Mmax (max connections at layer 0)
        100,    // efConstruction
        hnsw.Euclidean,  // AVX2-optimized distance function
    )

    // Batch insert with parallel processing
    vectors := make(map[int]hnsw.Vector, 1000)
    for i := 0; i < 1000; i++ {
        vectors[i] = generateRandomVector(128)
    }
    index.BatchInsert(vectors, 100) // batch size 100

    // Parallel search
    results := index.Search(queryVector, 10) // ~9µs per search
}
```

## Full Example

```go
package main

import (
    "flag"
    "fmt"
    "math/rand"
    "sync"
    "time"
    "github.com/BryceWayne/hnsw"
    // Uncomment for AVX2 optimizations
    // "golang.org/x/sys/cpu"
)

var (
    dimension      = flag.Int("d", 128, "Vector dimension")
    numVectors     = flag.Int("n", 1000, "Number of vectors")
    connections    = flag.Int("m", 16, "Max connections per layer")
    maxConnections = flag.Int("mmax", 32, "Max connections at layer 0")
    efConstruction = flag.Int("ef", 100, "EF construction parameter")
    searchK        = flag.Int("k", 10, "Number of nearest neighbors")
    batchSize      = flag.Int("batch", 25, "Batch size for parallel insertions")
    parallel       = flag.Bool("parallel", true, "Enable parallel search")
    workerCount    = flag.Int("workers", runtime.GOMAXPROCS(0), "Number of worker threads") // Default is runtime.NumCPU()
)

func main() {
    flag.Parse()
    rand.Seed(time.Now().UnixNano())

    // Initialize index
    index := hnsw.New(
        *dimension,
        *connections,
        *maxConnections,
        *efConstruction,
        hnsw.Euclidean,
    )

    // Generate and insert vectors in batches
    vectors := make([]hnsw.Vector, *numVectors)
    for i := range vectors {
        vectors[i] = generateRandomVector(*dimension)
    }

    numBatches := (*numVectors + *batchSize - 1) / *batchSize
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
    }

    // Search
    queryVector := generateRandomVector(*dimension)

    // Default parallel search
    results := index.Search(query, 10)

    // Custom config for sequential search
    config := hnsw.SearchConfig{
        UseParallel: *useParallel,
        WorkerCount: *workerCount,
    }
    results := index.SearchWithConfig(query, 10, config)

    fmt.Printf("Found neighbors: %v\n", results)
}
```

Run with:
```bash
go run main.go -d 256 -n 10000 -m 32 -ef 100 -k 20 -batch 25
```

Run with custom parameters:
```bash
go run main.go \
  -d 256 \           # dimension
  -n 10000 \         # number of vectors
  -m 32 \            # max connections
  -ef 100 \          # ef construction
  -k 20 \            # k nearest neighbors
  -batch 25 \        # batch size
  -parallel=true \   # use parallel search
  -workers 16        # worker threads
```

Flag descriptions:
```
  -d int
        Vector dimension (default 128)
  -n int
        Number of vectors (default 1000)
  -m int
        Max connections per layer (default 16)
  -mmax int
        Max connections at layer 0 (default 32)
  -ef int
        EF construction parameter (default 100)
  -k int
        Number of nearest neighbors (default 10)
  -batch int
        Batch size for parallel insertions (default 25)
  -parallel
        Use parallel search (default true)
  -workers int
        Number of worker threads (default: CPU cores)
```

Performance impact:
- Parallel search (-parallel=true): ~20x speedup
- Worker count (-workers): Scale with available CPU cores
- Batch size (-batch): Memory vs speed tradeoff

## API Reference

### Types

```go
type Vector []float64
type DistanceFunc func(Vector, Vector) float64
```

### Functions

#### New
```go
func New(dim, m, mmax, efConstruction int, distanceFunc DistanceFunc) *HNSW
```
Creates new HNSW index:
- `dim`: Vector dimension
- `m`: Max connections per layer
- `mmax`: Max connections at layer 0
- `efConstruction`: Search quality during construction (recommend 100-200)
- `distanceFunc`: Distance metric function (Euclidean or Cosine provided)

#### Insert
```go
func (h *HNSW) Insert(id int, vec Vector)
```
Inserts vector with given ID. Thread-safe.

#### Search 
```go
func (h *HNSW) Search(vec Vector, k int) []int
```
Returns IDs of k nearest neighbors. Thread-safe.

#### Delete
```go
func (h *HNSW) Delete(id int)
```
Removes vector from index. Thread-safe.

#### Save/Load
```go 
func (h *HNSW) Save(filename string) error
func Load(filename string, distanceFunc DistanceFunc) (*HNSW, error)
```
Persistence functions.

### Distance Functions

- `Euclidean`: Standard Euclidean distance
- `Cosine`: Cosine similarity as distance

## Performance

Sample benchmark (256d vectors, 10k points):
```json
{
  "dimension": 256,
  "num_vectors": 10000,
  "connections": 32,
  "max_connections": 32,
  "ef_construction": 100,
  "distance_metric": "euclidean",
  "build_time": 556174141967,
  "search_time": 376427279,
  "memory_usage": 33609840
}
```

## Project Structure

```
hnsw/
├── examples/      # Example usage
├── distance.go    # Distance metrics
├── hnsw.go       # Main HNSW implementation  
├── node.go       # Node implementation
├── serialize.go  # Serialization logic
└── types.go      # Core data types
```

## Algorithm Overview

HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search that creates a layered graph structure. Each layer is a "small world" graph, with the number of connections between nodes decreasing as you go up the layers.

### Search Process
![HNSW Search Process](https://raw.githubusercontent.com/BryceWayne/hnsw/refs/heads/root/docs/images/search-process.svg)

1. Begin at entry point in highest layer
2. Explore current layer to find closest node
3. Descend to next layer and repeat
4. Final search in bottom layer (Layer 0)

### Insertion Process
![HNSW Insertion](https://raw.githubusercontent.com/BryceWayne/hnsw/refs/heads/root/docs/images/insertion-process.svg)

1. Randomly select maximum level for new node
2. Find nearest neighbors at each level
3. Create bidirectional connections
4. Maintain connection limits through pruning

### Network Properties
![HNSW Growth](https://raw.githubusercontent.com/BryceWayne/hnsw/refs/heads/root/docs/images/growth-process.svg)

The network maintains efficiency through:
- Balance of short/long-range connections
- Limited connections per node (M/Mmax)
- Hierarchical navigation structure

### EF Parameter Impact
![EF Impact](https://raw.githubusercontent.com/BryceWayne/hnsw/refs/heads/root/docs/images/ef-impact.svg)

EF (Exploration Factor) controls:
- Lower EF: Faster search, less accurate
- Higher EF: Slower search, more accurate

### Parameter Tuning

1. **M/Mmax** (Connection Limits):
   - `M`: 12-16 for high-dimensional data
   - `Mmax`: Usually 2*M for ground layer

2. **EF** (Search Quality):
   - Lower values (64): Faster, less accurate
   - Higher values (128+): Slower, more accurate
   - Construction: 100-200 recommended

3. **Distance Metrics**:
   - Cosine: Best for text/embedding vectors
   - Euclidean: Better for coordinate-based data

## Testing

Run the test suite:
```bash
go test -v ./...
```

Run benchmarks:
```bash
go test -bench=. ./...
```

## Performance

Current test results:

```
go test -v ./...
?       github.com/BryceWayne/hnsw/examples     [no test files]
=== RUN   TestBatchOperations
--- PASS: TestBatchOperations (0.01s)
=== RUN   TestEuclidean
--- PASS: TestEuclidean (0.00s)
=== RUN   TestCosine
--- PASS: TestCosine (0.00s)
=== RUN   TestNew
--- PASS: TestNew (0.00s)
=== RUN   TestInsertAndSearch
--- PASS: TestInsertAndSearch (0.00s)
=== RUN   TestSearch
=== RUN   TestSearch/Find_nearest_to_origin
=== RUN   TestSearch/Find_two_nearest
--- PASS: TestSearch (0.00s)
    --- PASS: TestSearch/Find_nearest_to_origin (0.00s)
    --- PASS: TestSearch/Find_two_nearest (0.00s)
=== RUN   TestParallelSearch
--- PASS: TestParallelSearch (0.00s)
=== RUN   TestConcurrentSearches
--- PASS: TestConcurrentSearches (0.01s)
=== RUN   TestDelete
--- PASS: TestDelete (0.00s)
=== RUN   TestSaveLoad
--- PASS: TestSaveLoad (0.00s)
=== RUN   TestConcurrentInserts
--- PASS: TestConcurrentInserts (0.00s)
PASS
ok      github.com/BryceWayne/hnsw      0.022s
```


Current benchmark results (Intel i9-12900K):

```
Operation                            Time/op
Sequential Insert                    ~263µs
Sequential Search                    ~154µs
Parallel Search                      ~29µs
Batch Insert (100 vectors)          ~250µs
Batch Search                        ~174ns

Dimension Scaling:
Size      Dim    Time/op
1K        32     ~42µs
1K        128    ~36µs
1K        512    ~23µs
10K       32     ~21µs
10K       128    ~20µs
10K       512    ~15µs
100K      32     ~20µs
100K      128    ~18µs
100K      512    ~13µs
```

Key Features:
- AVX2 vectorized distance calculations
- 5.3x speedup with parallel search
- Efficient scaling with dimensions
- Optimal batch operations

### Hardware Requirements
- AVX2 support for SIMD optimizations
- Multiple CPU cores for parallel search
- Recommended: 16GB+ RAM for 100K+ vectors

### Tuning Guidelines
1. Batch Size: 
   - Small datasets (<10K): 25-50
   - Large datasets: 100-200

2. Worker Count:
   - Default: Number of CPU cores
   - High load: 2x CPU cores

3. Memory vs Speed:
   - Lower M (8-12): Less memory, slower search
   - Higher M (16-32): Faster search, more memory

### Performance Tips

1. **Index Construction**:
   - Use batch insertions (25-50 vectors per batch)
   - Pre-generate vectors before insertion
   - Lower `efConstruction` (100) for faster builds

2. **Search Optimization**:
   - Adjust EF based on accuracy needs
   - Use appropriate distance metric

3. **Memory Management**:
   - Monitor node count vs. available RAM
   - Consider saving/loading for large datasets

## License

MIT License

Copyright (c) 2024 Bryce Wayne