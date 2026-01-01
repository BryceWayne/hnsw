package hnsw

import (
	"math/rand"
	"testing"
)

func TestOptimizationLogic(t *testing.T) {
	// Verify that the pooled objects are working correctly and not leaking state
	h := New(16, 16, 32, 100, Euclidean)

	// Create vectors
	for i := 0; i < 1000; i++ {
		vec := make(Vector, 16)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		h.Insert(i, vec)
	}

	// Perform multiple searches to trigger reuse
	for i := 0; i < 100; i++ {
		query := make(Vector, 16)
		for j := range query {
			query[j] = rand.Float64()
		}

		results := h.Search(query, 10)
		if len(results) == 0 {
			t.Errorf("Search returned no results on iteration %d", i)
		}
	}
}

func TestVisitedMapPooling(t *testing.T) {
	// Get a map
	m1 := getVisitedMap()
	m1[1] = true
	m1[2] = true

	// Put it back
	putVisitedMap(m1)

	// Get it again
	m2 := getVisitedMap()

	// Verify it's empty
	if len(m2) != 0 {
		t.Errorf("Pooled map was not cleared! Len: %d", len(m2))
	}

	putVisitedMap(m2)
}

func TestNodeDistHeapPooling(t *testing.T) {
	// Get a heap
	h1 := getNodeDistHeap()
	*h1 = append(*h1, &nodeDist{node: &Node{}, dist: 1.0})

	// Put it back
	putNodeDistHeap(h1)

	// Get it again
	h2 := getNodeDistHeap()

	// Verify it's empty
	if len(*h2) != 0 {
		t.Errorf("Pooled heap was not cleared! Len: %d", len(*h2))
	}

	putNodeDistHeap(h2)
}
