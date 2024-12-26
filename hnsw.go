package hnsw

import (
    "encoding/gob"
    "math/rand"
    "os"
    "sort"
    "sync"
)

func init() {
    gob.Register(&serialNode{})
    gob.Register(&Level{})
    gob.Register(Vector{})
    gob.Register(&SerializableHNSW{})
}

// SerializableHNSW represents the serializable form of HNSW
type SerializableHNSW struct {
    Nodes          map[int]*serialNode
    EntryPointID   int
    MaxLevel       int
    M              int
    Mmax           int
    EfConstruction int
    Dim            int
    DeletedNodes   map[int]bool
}

// HNSW represents the hierarchical navigable small world graph
type HNSW struct {
    Nodes          map[int]*Node
    EntryPoint     *Node
    MaxLevel       int
    M              int
    Mmax           int
    EfConstruction int
    Dim            int
    DistanceFunc   DistanceFunc
    mutex          sync.RWMutex
    deletedNodes   map[int]bool
}

// New creates a new HNSW index
func New(dim, m, mmax, efConstruction int, distanceFunc DistanceFunc) *HNSW {
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

// Insert adds a new vector to the index
func (h *HNSW) Insert(id int, vec Vector) {
    h.mutex.Lock()
    defer h.mutex.Unlock()

    newNode := &Node{
        ID:     id,
        Vector: vec,
        Levels: make([]*Level, 1),
    }
    newNode.Levels[0] = &Level{Connections: make([]*Node, 0)}

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

// Search finds k nearest neighbors for the given vector
func (h *HNSW) Search(vec Vector, k int) []int {
    h.mutex.RLock()
    defer h.mutex.RUnlock()

    if len(h.Nodes) == 0 {
        return nil
    }

    currentNode := h.EntryPoint
    if currentNode == nil {
        return []int{}
    }

    // Search through layers
    for level := h.MaxLevel; level > 0; level-- {
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

    // Get candidates from bottom layer
    candidates := h.searchLayer(currentNode, vec, h.EfConstruction, 0)

    // Filter deleted nodes and sort by distance
    validCandidates := make([]*Node, 0, len(candidates))
    for _, node := range candidates {
        if !h.deletedNodes[node.ID] {
            validCandidates = append(validCandidates, node)
        }
    }

    if len(validCandidates) == 0 {
        return []int{}
    }

    // Sort remaining candidates
    sort.Slice(validCandidates, func(i, j int) bool {
        return h.DistanceFunc(validCandidates[i].Vector, vec) < h.DistanceFunc(validCandidates[j].Vector, vec)
    })

    // Return k closest
    count := min(k, len(validCandidates))
    result := make([]int, count)
    for i := 0; i < count; i++ {
        result[i] = validCandidates[i].ID
    }

    return result
}

// Delete removes a vector from the index
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

// Save persists the index to a file
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

// Load reads the index from a file
func Load(filename string, distanceFunc DistanceFunc) (*HNSW, error) {
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

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
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
