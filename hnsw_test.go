package hnsw

import (
	"os"
	"testing"
)

func TestNew(t *testing.T) {
	h := New(128, 16, 32, 100, Euclidean)
	if h.Dim != 128 || h.M != 16 || h.Mmax != 32 || h.EfConstruction != 100 {
		t.Errorf("New() created HNSW with incorrect parameters")
	}
}

func TestInsertAndSearch(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Insert some vectors
	vectors := []Vector{
		{0, 0},
		{1, 1},
		{2, 2},
		{3, 3},
	}

	for i, v := range vectors {
		h.Insert(i, v)
	}

	// Search for nearest neighbors
	query := Vector{1.1, 1.1}
	results := h.Search(query, 2)

	if len(results) != 2 {
		t.Errorf("Search() returned %d results; want 2", len(results))
	}

	if results[0] != 1 { // Closest should be {1,1}
		t.Errorf("Search() first result = %d; want 1", results[0])
	}
}

func TestDelete(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Insert test vectors
	h.Insert(1, Vector{1, 1})
	h.Insert(2, Vector{2, 2})
	h.Insert(3, Vector{3, 3})

	// Initial search
	initialResults := h.Search(Vector{1.1, 1.1}, 1)
	if len(initialResults) != 1 || initialResults[0] != 1 {
		t.Errorf("Initial search failed, got %v, want [1]", initialResults)
	}

	// Delete vector
	h.Delete(1)

	// Search after deletion
	results := h.Search(Vector{1.1, 1.1}, 1)
	if len(results) == 0 {
		t.Error("Search() should return results after delete")
	}
	if results[0] == 1 {
		t.Error("Search() returned deleted vector")
	}

	// Verify deleted ID is not searchable
	for _, id := range results {
		if id == 1 {
			t.Error("Deleted ID found in search results")
		}
	}
}

func TestSaveLoad(t *testing.T) {
	filename := "test_index.hnsw"
	defer os.Remove(filename)

	// Create and save index
	h1 := New(2, 16, 32, 100, Euclidean)
	h1.Insert(1, Vector{1, 1})
	h1.Insert(2, Vector{2, 2})

	err := h1.Save(filename)
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	// Load index
	h2, err := Load(filename, Euclidean)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	// Compare search results
	query := Vector{1.1, 1.1}
	results1 := h1.Search(query, 1)
	results2 := h2.Search(query, 1)

	if len(results1) != len(results2) || results1[0] != results2[0] {
		t.Errorf("Search results differ after load. Got %v; want %v", results2, results1)
	}
}

func TestConcurrentInserts(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)
	done := make(chan bool)

	go func() {
		h.Insert(1, Vector{1, 1})
		done <- true
	}()

	go func() {
		h.Insert(2, Vector{2, 2})
		done <- true
	}()

	<-done
	<-done

	results := h.Search(Vector{1.5, 1.5}, 2)
	if len(results) != 2 {
		t.Error("Concurrent Insert() failed")
	}
}

func BenchmarkInsert(b *testing.B) {
	h := New(128, 16, 32, 100, Euclidean)
	vec := make(Vector, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Insert(i, vec)
	}
}

func BenchmarkSearch(b *testing.B) {
	h := New(128, 16, 32, 100, Euclidean)
	vec := make(Vector, 128)

	// Insert some vectors first
	for i := 0; i < 1000; i++ {
		h.Insert(i, vec)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(vec, 10)
	}
}
