package main

import (
	"encoding/gob"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"
)

func init() {
	// Register types for gob serialization
	gob.Register(&serialNode{})
	gob.Register(&Level{})
	gob.Register(Vector{})
	gob.Register(&SerializableHNSW{})
}

type SerializableHNSW struct {
	Nodes          map[int]*serialNode
	EntryPointID   int // Store ID instead of pointer
	MaxLevel       int
	M              int
	Mmax           int
	EfConstruction int
	Dim            int
	DeletedNodes   map[int]bool
}

type Vector []float64

type DistanceFunc func(Vector, Vector) float64

type Node struct {
	ID       int
	Vector   Vector
	Levels   []*Level
	MaxLevel int
	sync.RWMutex
}

type Level struct {
	Connections []*Node
}

// Serializable version of Level
type serialLevel struct {
	ConnectionIDs []int // Store IDs instead of pointers
}

type serialNode struct {
	ID       int
	Vector   Vector
	Levels   []*serialLevel // Changed from Level to serialLevel
	MaxLevel int
}

type HNSW struct {
	Nodes          map[int]*Node
	EntryPoint     *Node
	MaxLevel       int
	M              int // max number of connections
	Mmax           int // max number of connections for level 0
	EfConstruction int
	Dim            int
	DistanceFunc   DistanceFunc
	mutex          sync.RWMutex
	deletedNodes   map[int]bool
}

func NewHNSW(dim, m, mmax, efConstruction int, distanceFunc DistanceFunc) *HNSW {
	return &HNSW{
		Nodes:          make(map[int]*Node),
		MaxLevel:       0,
		M:              m,
		Mmax:           mmax,
		EfConstruction: efConstruction,
		Dim:            dim,
		DistanceFunc:   distanceFunc,
		deletedNodes:   make(map[int]bool),
	}
}

func (h *HNSW) Insert(id int, vec Vector) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	// Create new node with initial level
	newNode := &Node{
		ID:     id,
		Vector: vec,
		Levels: make([]*Level, 1),
	}
	newNode.Levels[0] = &Level{Connections: make([]*Node, 0)}

	// First node handling
	if len(h.Nodes) == 0 {
		h.EntryPoint = newNode
		h.Nodes[id] = newNode
		return
	}

	// Find entry point
	currentNode := h.EntryPoint
	currentDist := h.DistanceFunc(currentNode.Vector, vec)

	// Search through existing layers
	maxLevel := min(h.MaxLevel, len(currentNode.Levels)-1)
	for level := maxLevel; level >= 0; level-- {
		changed := true
		for changed {
			changed = false
			currentNode.RLock()
			if level < len(currentNode.Levels) && currentNode.Levels[level] != nil {
				for _, neighbor := range currentNode.Levels[level].Connections {
					if neighbor == nil {
						continue
					}
					neighborDist := h.DistanceFunc(neighbor.Vector, vec)
					if neighborDist < currentDist {
						currentNode.RUnlock()
						currentNode = neighbor
						currentDist = neighborDist
						changed = true
						break
					}
				}
			}
			if !changed {
				currentNode.RUnlock()
			}
		}
	}

	// Generate random level
	newLevel := h.randomLevel()
	if newLevel > h.MaxLevel {
		h.MaxLevel = newLevel
	}

	// Initialize all levels
	for i := len(newNode.Levels); i <= newLevel; i++ {
		newNode.Levels = append(newNode.Levels, &Level{
			Connections: make([]*Node, 0),
		})
	}

	// Build connections for each level
	for level := 0; level <= newLevel; level++ {
		neighbors := h.searchLayer(currentNode, vec, h.M, level)
		if level < len(newNode.Levels) {
			newNode.Levels[level].Connections = neighbors
			for _, neighbor := range neighbors {
				h.addConnection(neighbor, newNode, level)
			}
		}
	}

	h.Nodes[id] = newNode
}

// Add this helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (h *HNSW) Search(vec Vector, k int) []int {
	h.mutex.RLock()
	defer h.mutex.RUnlock()

	if len(h.Nodes) == 0 {
		return nil
	}

	currentNode := h.EntryPoint
	for level := min(h.MaxLevel, len(currentNode.Levels)-1); level > 0; level-- {
		changed := true
		for changed {
			changed = false
			currentNode.RLock()
			if level < len(currentNode.Levels) && currentNode.Levels[level] != nil {
				for _, neighbor := range currentNode.Levels[level].Connections {
					if h.DistanceFunc(neighbor.Vector, vec) < h.DistanceFunc(currentNode.Vector, vec) {
						currentNode.RUnlock()
						currentNode = neighbor
						changed = true
						break
					}
				}
			}
			if !changed {
				currentNode.RUnlock()
			}
		}
	}

	candidates := h.searchLayer(currentNode, vec, h.EfConstruction, 0)
	sort.Slice(candidates, func(i, j int) bool {
		return h.DistanceFunc(candidates[i].Vector, vec) < h.DistanceFunc(candidates[j].Vector, vec)
	})

	if len(candidates) > k {
		candidates = candidates[:k]
	}

	result := make([]int, 0, len(candidates))
	for _, node := range candidates {
		if !h.deletedNodes[node.ID] {
			result = append(result, node.ID)
		}
	}

	return result
}

func (h *HNSW) Delete(id int) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if _, exists := h.Nodes[id]; !exists {
		return
	}

	h.deletedNodes[id] = true
	delete(h.Nodes, id)

	if h.EntryPoint.ID == id {
		for _, node := range h.Nodes {
			h.EntryPoint = node
			break
		}
	}
}

func (h *HNSW) Save(filename string) error {
	h.mutex.RLock()
	defer h.mutex.RUnlock()

	serializable := &SerializableHNSW{
		Nodes:          make(map[int]*serialNode),
		MaxLevel:       h.MaxLevel,
		M:              h.M,
		Mmax:           h.Mmax,
		EfConstruction: h.EfConstruction,
		Dim:            h.Dim,
		DeletedNodes:   h.deletedNodes,
	}

	if h.EntryPoint != nil {
		serializable.EntryPointID = h.EntryPoint.ID
	}

	// Convert nodes with serializable levels
	for id, node := range h.Nodes {
		sNode := &serialNode{
			ID:       node.ID,
			Vector:   node.Vector,
			MaxLevel: node.MaxLevel,
			Levels:   make([]*serialLevel, len(node.Levels)),
		}

		// Convert each level's connections to IDs
		for i, level := range node.Levels {
			sLevel := &serialLevel{
				ConnectionIDs: make([]int, len(level.Connections)),
			}
			for j, conn := range level.Connections {
				sLevel.ConnectionIDs[j] = conn.ID
			}
			sNode.Levels[i] = sLevel
		}

		serializable.Nodes[id] = sNode
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(serializable)
}

func LoadHNSW(filename string, distanceFunc DistanceFunc) (*HNSW, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var serialized SerializableHNSW
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&serialized); err != nil {
		return nil, err
	}

	h := &HNSW{
		Nodes:          make(map[int]*Node),
		MaxLevel:       serialized.MaxLevel,
		M:              serialized.M,
		Mmax:           serialized.Mmax,
		EfConstruction: serialized.EfConstruction,
		Dim:            serialized.Dim,
		DistanceFunc:   distanceFunc,
		deletedNodes:   serialized.DeletedNodes,
		mutex:          sync.RWMutex{},
	}

	// First pass: create all nodes
	for id, sNode := range serialized.Nodes {
		node := &Node{
			ID:       sNode.ID,
			Vector:   sNode.Vector,
			MaxLevel: sNode.MaxLevel,
			Levels:   make([]*Level, len(sNode.Levels)),
			RWMutex:  sync.RWMutex{},
		}
		h.Nodes[id] = node
	}

	// Second pass: restore connections using IDs
	for id, sNode := range serialized.Nodes {
		node := h.Nodes[id]
		for i, sLevel := range sNode.Levels {
			level := &Level{
				Connections: make([]*Node, len(sLevel.ConnectionIDs)),
			}
			for j, connID := range sLevel.ConnectionIDs {
				level.Connections[j] = h.Nodes[connID]
			}
			node.Levels[i] = level
		}

		if id == serialized.EntryPointID {
			h.EntryPoint = node
		}
	}

	return h, nil
}

func (h *HNSW) randomLevel() int {
	level := 0
	for rand.Float64() < 0.5 && level < 32 {
		level++
	}
	return level
}

func (h *HNSW) searchLayer(entryPoint *Node, vec Vector, ef int, level int) []*Node {
	if level >= len(entryPoint.Levels) {
		return []*Node{entryPoint}
	}

	visited := sync.Map{}
	visitedResults := sync.Map{} // Track visited nodes that are potential results

	// Initialize candidates with entry point
	candidates := []*Node{entryPoint}
	visited.Store(entryPoint.ID, true)

	// Initialize result set
	results := []*Node{entryPoint}
	visitedResults.Store(entryPoint.ID, true)

	// Calculate distance to entry point
	entryDist := h.DistanceFunc(entryPoint.Vector, vec)
	furthestDist := entryDist

	var resultsMutex sync.Mutex

	for len(candidates) > 0 {
		// Get current candidate
		currentNode := candidates[0]
		currentDist := h.DistanceFunc(currentNode.Vector, vec)
		candidates = candidates[1:]

		// If we've found something further than our worst candidate
		if currentDist > furthestDist {
			continue
		}

		// Check neighbors
		currentNode.RLock()
		if level < len(currentNode.Levels) && currentNode.Levels[level] != nil {
			for _, neighbor := range currentNode.Levels[level].Connections {
				if neighbor == nil {
					continue
				}

				// Skip if already visited
				if _, seen := visited.LoadOrStore(neighbor.ID, true); seen {
					continue
				}

				neighborDist := h.DistanceFunc(neighbor.Vector, vec)

				// Update results if this is a better candidate
				resultsMutex.Lock()
				if len(results) < ef || neighborDist < furthestDist {
					if _, seen := visitedResults.LoadOrStore(neighbor.ID, true); !seen {
						results = append(results, neighbor)
					}

					// Sort results by distance
					sort.Slice(results, func(i, j int) bool {
						return h.DistanceFunc(results[i].Vector, vec) < h.DistanceFunc(results[j].Vector, vec)
					})

					// Keep only ef closest results
					if len(results) > ef {
						results = results[:ef]
					}

					// Update furthest distance
					furthestDist = h.DistanceFunc(results[len(results)-1].Vector, vec)
				}
				resultsMutex.Unlock()

				// Add to candidates if it could lead to better results
				candidates = append(candidates, neighbor)
			}
		}
		currentNode.RUnlock()

		// Sort candidates by distance
		sort.Slice(candidates, func(i, j int) bool {
			return h.DistanceFunc(candidates[i].Vector, vec) < h.DistanceFunc(candidates[j].Vector, vec)
		})
	}

	return results
}

func (h *HNSW) addConnection(node, newNode *Node, level int) {
	node.Lock()
	defer node.Unlock()

	// Ensure level exists
	for len(node.Levels) <= level {
		node.Levels = append(node.Levels, &Level{})
	}

	// Get max connections for this level
	maxConnections := h.M
	if level == 0 {
		maxConnections = h.Mmax
	}

	// Check if connection already exists
	for _, conn := range node.Levels[level].Connections {
		if conn.ID == newNode.ID {
			return // Already connected
		}
	}

	// Calculate distances for all connections including the new one
	type connDist struct {
		node     *Node
		distance float64
	}

	conns := make([]connDist, 0, len(node.Levels[level].Connections)+1)

	// Add existing connections
	for _, conn := range node.Levels[level].Connections {
		dist := h.DistanceFunc(node.Vector, conn.Vector)
		conns = append(conns, connDist{conn, dist})
	}

	// Add new connection
	newDist := h.DistanceFunc(node.Vector, newNode.Vector)
	conns = append(conns, connDist{newNode, newDist})

	// Sort by distance
	sort.Slice(conns, func(i, j int) bool {
		return conns[i].distance < conns[j].distance
	})

	// Keep only the closest connections up to maxConnections
	node.Levels[level].Connections = make([]*Node, 0, maxConnections)
	for i := 0; i < len(conns) && i < maxConnections; i++ {
		node.Levels[level].Connections = append(node.Levels[level].Connections, conns[i].node)
	}

	// Optional: Keep some long-range connections for better graph navigability
	if level > 0 && len(conns) > maxConnections {
		// Replace one close connection with a longer-range one
		longRangeIdx := len(conns) - 1
		if rand.Float64() < 0.2 { // 20% chance to keep a long-range connection
			node.Levels[level].Connections[maxConnections-1] = conns[longRangeIdx].node
		}
	}
}

// Distance metrics
func EuclideanDistance(v1, v2 Vector) float64 {
	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func CosineDistance(v1, v2 Vector) float64 {
	dot := 0.0
	norm1 := 0.0
	norm2 := 0.0
	for i := range v1 {
		dot += v1[i] * v2[i]
		norm1 += v1[i] * v1[i]
		norm2 += v2[i] * v2[i]
	}
	return 1 - (dot / (math.Sqrt(norm1) * math.Sqrt(norm2)))
}

// generateRandomVector creates a vector of given dimension with random values
func generateRandomVector(dim int) Vector {
	vec := make(Vector, dim)
	for i := range vec {
		vec[i] = rand.Float64() // generates values between 0 and 1
	}
	return vec
}

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create a new HNSW index with Euclidean distance
	dim := 128
	index := NewHNSW(
		dim, // dimension
		16,  // M (max connections per layer)
		32,  // Mmax (max connections at layer 0)
		200, // efConstruction
		EuclideanDistance,
	)

	// Generate some random vectors and insert them concurrently
	numVectors := 1000
	var wg sync.WaitGroup

	for i := 0; i < numVectors; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vec := generateRandomVector(dim)
			index.Insert(id, vec)
		}(i)
	}
	wg.Wait()

	// Create a query vector and search for nearest neighbors
	queryVector := generateRandomVector(dim)
	k := 10 // number of nearest neighbors to find
	results := index.Search(queryVector, k)

	log.Printf("Found %d nearest neighbors: %v", len(results), results)

	// Delete a vector
	index.Delete(5)

	// Save the index to a file
	if err := index.Save("hnsw_index.gob"); err != nil {
		log.Fatalf("Failed to save index: %v", err)
	}

	// Load the index from a file
	loadedIndex, err := LoadHNSW("hnsw_index.gob", EuclideanDistance)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	// Verify loaded index works
	results = loadedIndex.Search(queryVector, k)
	log.Printf("Search results from loaded index: %v", results)

	// Example with cosine distance
	cosineIndex := NewHNSW(
		dim,
		16,
		32,
		200,
		CosineDistance,
	)

	// Insert a few vectors in the cosine index
	for i := 0; i < 10; i++ {
		vec := generateRandomVector(dim)
		cosineIndex.Insert(i, vec)
	}

	// Search with cosine distance
	cosineResults := cosineIndex.Search(queryVector, k)
	log.Printf("Cosine distance search results: %v", cosineResults)
}
