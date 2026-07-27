package hnsw

import (
	"sync"
)

// Global pools for reusable objects to reduce GC pressure

var visitedMapPool = sync.Pool{
	New: func() interface{} {
		// Start with a reasonable size, but it will grow
		return make(map[int]bool, 1024)
	},
}

var nodeDistHeapPool = sync.Pool{
	New: func() interface{} {
		// Initial capacity
		h := make(nodeDistHeap, 0, 1024)
		return &h
	},
}

func getVisitedMap() map[int]bool {
	m := visitedMapPool.Get().(map[int]bool)
	clear(m)
	return m
}

func putVisitedMap(m map[int]bool) {
	visitedMapPool.Put(m)
}

func getNodeDistHeap() *nodeDistHeap {
	h := nodeDistHeapPool.Get().(*nodeDistHeap)
	return h
}

// resetNodeDistHeap clears pointers to avoid memory leaks
func resetNodeDistHeap(h *nodeDistHeap) {
	old := *h
	for i := range old {
		old[i] = nil
	}
	*h = old[:0]
}

func putNodeDistHeap(h *nodeDistHeap) {
	resetNodeDistHeap(h)
	nodeDistHeapPool.Put(h)
}

var floatSlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]float64, 0, 4096)
		return &s
	},
}

func getFloatSlice() *[]float64 {
	return floatSlicePool.Get().(*[]float64)
}

func putFloatSlice(s *[]float64) {
	old := *s
	*s = old[:0]
	floatSlicePool.Put(s)
}

var nodeSlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]*Node, 0, 1024)
		return &s
	},
}

func getNodeSlice() *[]*Node {
	return nodeSlicePool.Get().(*[]*Node)
}

func putNodeSlice(s *[]*Node) {
	old := *s
	for i := range old {
		old[i] = nil
	}
	*s = old[:0]
	nodeSlicePool.Put(s)
}
