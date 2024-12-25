package hnsw

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
)

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

type HNSW struct {
	Nodes           map[int]*Node
	EntryPoint      *Node
	MaxLevel        int
	M               int // max number of connections
	Mmax            int // max number of connections for level 0
	EfConstruction  int
	Dim             int
	DistanceFunc    DistanceFunc
	mutex           sync.RWMutex
	deletedNodes    map[int]bool
}

func NewHNSW(dim, m, mmax, efConstruction int, distanceFunc DistanceFunc) *HNSW {
	return &HNSW{
		Nodes:           make(map[int]*Node),
		MaxLevel:        0,
		M:               m,
		Mmax:            mmax,
		EfConstruction:  efConstruction,
		Dim:             dim,
		DistanceFunc:    distanceFunc,
		deletedNodes:    make(map[int]bool),
	}
}

func (h *HNSW) Insert(id int, vec Vector) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	newNode := &Node{
		ID:     id,
		Vector: vec,
	}

	if len(h.Nodes) == 0 {
		h.EntryPoint = newNode
		h.Nodes[id] = newNode
		return
	}

	currentNode := h.EntryPoint
	currentDist := h.DistanceFunc(currentNode.Vector, vec)

	for level := h.MaxLevel; level >= 0; level-- {
		changed := true
		for changed {
			changed = false
			currentNode.RLock()
			for _, neighbor := range currentNode.Levels[level].Connections {
				neighborDist := h.DistanceFunc(neighbor.Vector, vec)
				if neighborDist < currentDist {
					currentNode.RUnlock()
					currentNode = neighbor
					currentDist = neighborDist
					changed = true
					break
				}
			}
			if !changed {
				currentNode.RUnlock()
			}
		}
	}

	newLevel := h.randomLevel()
	if newLevel > h.MaxLevel {
		h.MaxLevel = newLevel
	}

	for level := 0; level <= newLevel; level++ {
		neighbors := h.searchLayer(currentNode, vec, h.M, level)
		newNode.Levels = append(newNode.Levels, &Level{Connections: neighbors})
		for _, neighbor := range neighbors {
			h.addConnection(neighbor, newNode, level)
		}
	}

	h.Nodes[id] = newNode
}

func (h *HNSW) Search(vec Vector, k int) []int {
	h.mutex.RLock()
	defer h.mutex.RUnlock()

	if len(h.Nodes) == 0 {
		return nil
	}

	currentNode := h.EntryPoint
	for level := h.MaxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			currentNode.RLock()
			for _, neighbor := range currentNode.Levels[level].Connections {
				if h.DistanceFunc(neighbor.Vector, vec) < h.DistanceFunc(currentNode.Vector, vec) {
					currentNode.RUnlock()
					currentNode = neighbor
					changed = true
					break
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

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(h)
}

func LoadHNSW(filename string) (*HNSW, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var h HNSW
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&h)
	if err != nil {
		return nil, err
	}

	return &h, nil
}

func (h *HNSW) randomLevel() int {
	level := 0
	for rand.Float64() < 0.5 && level < 32 {
		level++
	}
	return level
}

func (h *HNSW) searchLayer(entryPoint *Node, vec Vector, ef int, level int) []*Node {
	visited := sync.Map{}
	candidates := []*Node{entryPoint}
	visited.Store(entryPoint.ID, true)

	var candidatesMutex sync.Mutex

	for len(candidates) > 0 {
		candidatesMutex.Lock()
		currentNode := candidates[0]
		candidates = candidates[1:]
		candidatesMutex.Unlock()

		currentNode.RLock()
		for _, neighbor := range currentNode.Levels[level].Connections {
			if _, isVisited := visited.LoadOrStore(neighbor.ID, true); !isVisited {
				candidatesMutex.Lock()
				candidates = append(candidates, neighbor)
				candidatesMutex.Unlock()
			}
		}
		currentNode.RUnlock()

		candidatesMutex.Lock()
		sort.Slice(candidates, func(i, j int) bool {
			return h.DistanceFunc(candidates[i].Vector, vec) < h.DistanceFunc(candidates[j].Vector, vec)
		})

		if len(candidates) > ef {
			candidates = candidates[:ef]
		}
		candidatesMutex.Unlock()
	}

	return candidates
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


func main() {
  // Create a new HNSW index with Euclidean distance
  index := NewHNSW(128, 16, 32, 200, EuclideanDistance)
  
  // Insert vectors concurrently
  var wg sync.WaitGroup
  for i := 0; i < 1000; i++ {
      wg.Add(1)
      go func(id int) {
          defer wg.Done()
          vec := generateRandomVector(128)
          index.Insert(id, vec)
      }(i)
  }
  wg.Wait()
  
  // Search for nearest neighbors
  results := index.Search(Vector{...}, 10)
  
  // Delete a vector
  index.Delete(5)
  
  // Save the index to a file
  err := index.Save("hnsw_index.gob")
  if err != nil {
      log.Fatal(err)
  }
  
  // Load the index from a file
  loadedIndex, err := LoadHNSW("hnsw_index.gob")
  if err != nil {
      log.Fatal(err)
  }
  
  // Use a different distance metric
  cosineIndex := NewHNSW(128, 16, 32, 200, CosineDistance)
}
