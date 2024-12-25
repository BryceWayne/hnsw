# HNSW (Hierarchical Navigable Small World) Implementation in Go

A high-performance implementation of the HNSW algorithm for approximate nearest neighbor search, optimized for high-dimensional vector spaces.

## Algorithm Overview

HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search that creates a layered graph structure. Each layer is a "small world" graph, with the number of connections between nodes decreasing as you go up the layers.

### Search Process
![HNSW Search Process](https://raw.githubusercontent.com/BryceWayne/hnsw/main/docs/images/search-process.svg)

The search process starts at the top layer and descends through the hierarchy:
1. Begin at the entry point in the highest layer
2. Explore the current layer to find the closest node to the query
3. Descend to the next layer and repeat
4. Final search is performed in the bottom layer (Layer 0)

### Insertion Process
![HNSW Insertion](https://raw.githubusercontent.com/BryceWayne/hnsw/main/docs/images/insertion-process.svg)

When inserting a new node:
1. Randomly select maximum level for the new node
2. Find nearest neighbors at each level
3. Create bidirectional connections
4. Maintain connection limits (M/Mmax) through pruning

### Small World Properties
![HNSW Growth](https://raw.githubusercontent.com/BryceWayne/hnsw/main/docs/images/growth-process.svg)

The network maintains its efficiency through:
- Balance between short and long-range connections
- Limited connections per node (M/Mmax parameters)
- Hierarchical structure for efficient navigation

### EF Parameter Impact
![EF Impact](https://raw.githubusercontent.com/BryceWayne/hnsw/main/docs/mages/ef-impact.svg)

The EF (Exploration Factor) parameter controls the trade-off between:
- Search speed (lower EF)
- Search accuracy (higher EF)

## Code Structure

### Core Types

```go
type Vector []float64

type Node struct {
    ID       int
    Vector   Vector
    Levels   []*Level
    MaxLevel int
    sync.RWMutex
}

type HNSW struct {
    Nodes           map[int]*Node
    EntryPoint      *Node
    MaxLevel        int
    M               int    // max connections per layer
    Mmax            int    // max connections at layer 0
    EfConstruction  int
    Dim             int
    DistanceFunc    DistanceFunc
}
```

### Key Functions

#### NewHNSW
```go
func NewHNSW(dim, m, mmax, efConstruction int, distanceFunc DistanceFunc) *HNSW
```
Creates a new HNSW index with specified parameters:
- `dim`: Vector dimension
- `m`: Maximum connections per node (except ground layer)
- `mmax`: Maximum connections in ground layer
- `efConstruction`: Search quality during construction
- `distanceFunc`: Distance metric function

#### Insert
```go
func (h *HNSW) Insert(id int, vec Vector)
```
Inserts a new vector into the index:
1. Randomly assigns a maximum level
2. Finds insertion points through layer traversal
3. Creates connections at each level
4. Maintains connection limits through pruning

#### Search
```go
func (h *HNSW) Search(vec Vector, k int) []int
```
Finds k nearest neighbors for a query vector:
1. Traverses from top layer to bottom
2. Uses EF parameter to control search breadth
3. Returns IDs of k closest vectors

#### Delete
```go
func (h *HNSW) Delete(id int)
```
Marks a vector as deleted and updates entry point if needed.

#### Save/Load
```go
func (h *HNSW) Save(filename string) error
func LoadHNSW(filename string) (*HNSW, error)
```
Persistence functions for saving/loading the index.

## Usage

```go
// Create a new HNSW index
index := NewHNSW(
    128,    // dimension
    16,     // M (max connections per layer)
    32,     // Mmax (max connections at layer 0)
    200,    // efConstruction
    CosineDistance,  // distance function
)

// Insert vectors
index.Insert(1, vector1)
index.Insert(2, vector2)

// Search for nearest neighbors
results := index.Search(queryVector, 10)

// Save index to file
index.Save("index.hnsw")

// Load index from file
loadedIndex, err := LoadHNSW("index.hnsw")
```

## Pruning Strategy

![HNSW Pruning](https://raw.githubusercontent.com/BryceWayne/hnsw/main/docs/images/pruning-process.svg)

Pruning is crucial for maintaining the HNSW graph's efficiency and "small world" properties. The implementation uses several strategies:

### Connection Pruning

1. **Maximum Connections**:
   - Each node has a maximum number of connections (M)
   - Ground layer (L0) has higher limit (Mmax)
   - When exceeded, prune to keep best connections

2. **Distance-Based Selection**:
   ```go
   maxConnections := h.M
   if level == 0 {
       maxConnections = h.Mmax
   }
   ```

3. **Pruning Process**:
   - Sort connections by distance
   - Keep closest neighbors up to M/Mmax
   - Maintain bidirectional links
   ```go
   // Sort by distance
   sort.Slice(conns, func(i, j int) bool {
       return conns[i].distance < conns[j].distance
   })
   ```

### Optimization Strategies

1. **Connection Deduplication**:
   - Prevent duplicate connections
   - Check before adding new links
   ```go
   for _, conn := range node.Levels[level].Connections {
       if conn.ID == newNode.ID {
           return // Already connected
       }
   }
   ```

2. **Long-range Connections**:
   - Occasionally keep some longer-range connections
   - Improves graph navigability
   - Prevents local clustering
   ```go
   if level > 0 && rand.Float64() < 0.2 {
       // Keep one long-range connection
   }
   ```

3. **Layer-Specific Treatment**:
   - Ground layer (L0) keeps more connections
   - Higher layers maintain sparser connections
   - Balances search speed and accuracy

### Impact on Performance

1. **Search Efficiency**:
   - Fewer connections to traverse
   - Better quality neighbors
   - Faster convergence

2. **Memory Usage**:
   - Controlled growth
   - Efficient storage
   - Predictable scaling

3. **Build Time**:
   - One-time cost during insertion
   - Maintains index quality
   - Prevents degradation over time

## Parameter Tuning

### Essential Parameters

1. **M/Mmax** (Connection Limits):
   - `M`: 12-16 for high-dimensional data
   - `Mmax`: Usually 2*M for ground layer

2. **EF** (Search Quality):
   - Lower values (64): Faster, less accurate
   - Higher values (128+): Slower, more accurate
   - Construction: 200+ recommended

3. **Distance Metrics**:
   - Cosine: Best for text/embedding vectors
   - Euclidean: Better for coordinate-based data

### Performance Tips

1. **Index Construction**:
   - Higher `efConstruction` for better quality
   - Batch insertions for better throughput

2. **Search Optimization**:
   - Adjust EF based on accuracy needs
   - Use appropriate distance metric

3. **Memory Management**:
   - Monitor node count vs. available RAM
   - Consider saving/loading for large datasets

## License

[Add your license information here]