// Description: Tests for batch operations.
// batch_test.go
package hnsw

import (
	"math/rand"
	"testing"
)

func TestBatchOperations(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Test BatchInsert
	vectors := make(map[int]Vector)
	for i := 0; i < 100; i++ {
		vectors[i] = Vector{float64(i / 10), float64(i % 10)}
	}
	h.BatchInsert(vectors)

	if len(h.Nodes) != 100 {
		t.Errorf("BatchInsert: got %d nodes, want 100", len(h.Nodes))
	}

	// Test BatchSearch
	queries := make([]Vector, 5)
	for i := range queries {
		queries[i] = Vector{float64(i), float64(i)}
	}

	config := DefaultSearchConfig()
	results := h.BatchSearch(queries, 5, config)

	if len(results) != len(queries) {
		t.Errorf("BatchSearch: got %d results, want %d", len(results), len(queries))
	}

	// Test BatchDelete
	ids := make([]int, 10)
	for i := range ids {
		ids[i] = i
	}
	h.BatchDelete(ids)

	for _, id := range ids {
		if _, exists := h.Nodes[id]; exists {
			t.Errorf("BatchDelete: node %d still exists", id)
		}
	}
}

func BenchmarkBatchOperations(b *testing.B) {
	dim := 128

	b.Run("BatchInsert", func(b *testing.B) {
		h := New(dim, 16, 32, 100, Euclidean)
		vectors := make(map[int]Vector)

		for i := 0; i < b.N; i++ {
			vectors[i] = make(Vector, dim)
			for j := range vectors[i] {
				vectors[i][j] = rand.Float64()
			}
		}

		b.ResetTimer()
		h.BatchInsert(vectors)
	})

	b.Run("BatchSearch", func(b *testing.B) {
		h := New(dim, 16, 32, 100, Euclidean)
		queries := make([]Vector, b.N)

		for i := range queries {
			queries[i] = make(Vector, dim)
			for j := range queries[i] {
				queries[i][j] = rand.Float64()
			}
		}

		config := DefaultSearchConfig()
		b.ResetTimer()
		h.BatchSearch(queries, 10, config)
	})
}
