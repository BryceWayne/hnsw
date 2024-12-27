// node.go
package hnsw

import "sync"

// Level represents a layer in the HNSW graph
type Level struct {
    Connections []*Node
}

// Node represents a point in the HNSW graph
type Node struct {
    ID       int
    Vector   Vector
    Levels   []*Level
    MaxLevel int
    sync.RWMutex
}

// serialLevel is used for serialization
type serialLevel struct {
    ConnectionIDs []int
}

// serialNode is used for serialization
type serialNode struct {
    ID       int
    Vector   Vector
    Levels   []*serialLevel
    MaxLevel int
}
