# HNSW (Hierarchical Navigable Small World) Implementation in Go

High-performance implementation of HNSW algorithm for approximate nearest neighbor search.

## Installation

```bash
go get github.com/BryceWayne/hnsw
```

## Quick Start

```go
import "github.com/BryceWayne/hnsw"

// Create index
index := hnsw.New(
    128,    // dimension
    16,     // M (max connections per layer)
    32,     // Mmax (max connections at layer 0)
    100,    // efConstruction (construction-time ef)
    hnsw.Euclidean,  // distance function
)

// Insert vectors
index.Insert(1, vector1)
index.Insert(2, vector2)

// Search 
results := index.Search(queryVector, 10)

// Save/Load
index.Save("index.hnsw")
loadedIndex, _ := hnsw.Load("index.hnsw", hnsw.Euclidean)
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
)

var (
    dimension      = flag.Int("d", 128, "Vector dimension")
    numVectors     = flag.Int("n", 1000, "Number of vectors")
    connections    = flag.Int("m", 16, "Max connections per layer")
    maxConnections = flag.Int("mmax", 32, "Max connections at layer 0")
    efConstruction = flag.Int("ef", 100, "EF construction parameter")
    searchK        = flag.Int("k", 10, "Number of nearest neighbors")
    batchSize      = flag.Int("batch", 25, "Batch size for parallel insertions")
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
    results := index.Search(queryVector, *searchK)
    fmt.Printf("Found neighbors: %v\n", results)
}
```

Run with:
```bash
go run main.go -d 256 -n 10000 -m 32 -ef 100 -k 20 -batch 25
```

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