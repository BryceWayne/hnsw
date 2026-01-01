package hnsw

import (
	"testing"
)

func TestDimensionMismatch(t *testing.T) {
	h := New(4, 16, 32, 100, Euclidean)

	// Test Insert with wrong dimension
	defer func() {
		if r := recover(); r == nil {
			t.Error("Insert should panic or return error on dimension mismatch (expecting panic for now)")
		}
	}()
	h.Insert(1, Vector{1, 2, 3}) // Dimension 3, expected 4
}

func TestEmptyGraph(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Search on empty graph
	results := h.Search(Vector{1, 1}, 5)
	if len(results) != 0 {
		t.Errorf("Search on empty graph should return empty results, got %v", results)
	}
}

func TestKGreaterThanN(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)
	h.Insert(1, Vector{1, 1})
	h.Insert(2, Vector{2, 2})

	// Search with K > N
	results := h.Search(Vector{1.5, 1.5}, 10)
	if len(results) != 2 {
		t.Errorf("Search with K > N should return all %d nodes, got %d", 2, len(results))
	}
}

func TestDuplicateInsert(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)
	h.Insert(1, Vector{1, 1})

	// Re-insert same ID with different vector
	h.Insert(1, Vector{2, 2})

	if len(h.Nodes) != 1 {
		t.Errorf("After duplicate insert, node count should be 1, got %d", len(h.Nodes))
	}

	// Verify the vector was updated
	node := h.Nodes[1]
	if node.Vector[0] != 2 || node.Vector[1] != 2 {
		t.Errorf("Duplicate insert did not update vector. Got %v", node.Vector)
	}
}

func TestDeleteNonExistent(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)
	h.Insert(1, Vector{1, 1})

	h.Delete(999) // Should not panic

	if len(h.Nodes) != 1 {
		t.Errorf("Delete non-existent should not change node count")
	}
}

func TestNewInvalidParams(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("New() with invalid params should panic")
		}
	}()
	New(2, -1, 32, 100, Euclidean)
}
