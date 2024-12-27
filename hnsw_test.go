// hnsw_test.go
package hnsw

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sync"
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

	// Insert test vectors in a grid pattern
	vectors := []Vector{
		{0, 0},
		{1, 1},
		{2, 2},
		{3, 3},
	}

	for i, v := range vectors {
		h.Insert(i, v)
	}

	// Test search
	query := Vector{1.1, 1.1}
	config := SearchConfig{
		UseParallel: false, // Use sequential search for stability in tests
	}
	results := h.SearchWithConfig(query, 2, config)

	if len(results) != 2 {
		t.Fatalf("Search() returned %d results; want 2", len(results))
	}

	// Verify closest point is (1,1)
	if results[0] != 1 {
		t.Errorf("Search() first result = %d; want 1", results[0])
	}
}

func TestSearch(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Create test data
	testPoints := []struct {
		id     int
		vector Vector
	}{
		{0, Vector{0, 0}},
		{1, Vector{0, 1}},
		{2, Vector{1, 0}},
		{3, Vector{1, 1}},
	}

	for _, point := range testPoints {
		h.Insert(point.id, point.vector)
	}

	// Test cases
	tests := []struct {
		name    string
		query   Vector
		k       int
		wantIDs []int
	}{
		{
			name:    "Find nearest to origin",
			query:   Vector{0.1, 0.1},
			k:       1,
			wantIDs: []int{0},
		},
		{
			name:    "Find two nearest",
			query:   Vector{0.5, 0.5},
			k:       2,
			wantIDs: []int{0, 3}, // or {0, 2} or {1, 3} depending on implementation
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := SearchConfig{UseParallel: false}
			got := h.SearchWithConfig(tt.query, tt.k, config)

			if len(got) != len(tt.wantIDs) {
				t.Errorf("got %v results, want %v", len(got), len(tt.wantIDs))
			}

			// Verify distances are monotonically increasing
			var lastDist float64
			for i, id := range got {
				dist := h.DistanceFunc(h.Nodes[id].Vector, tt.query)
				if i > 0 && dist < lastDist {
					t.Errorf("results not sorted by distance: %v at position %d", dist, i)
				}
				lastDist = dist
			}
		})
	}
}

// TestParallelSearch verifies parallel search functionality
func TestParallelSearch(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Create grid of points
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			h.Insert(i*10+j, Vector{float64(i), float64(j)})
		}
	}

	query := Vector{1.5, 1.5}
	config := SearchConfig{
		UseParallel: true,
		WorkerCount: runtime.NumCPU(),
	}
	results := h.SearchWithConfig(query, 4, config)

	if len(results) == 0 {
		t.Error("Parallel search returned no results")
		return
	}

	// Verify distances are correct
	firstDist := h.DistanceFunc(h.Nodes[results[0]].Vector, query)
	for i := 1; i < len(results); i++ {
		dist := h.DistanceFunc(h.Nodes[results[i]].Vector, query)
		if dist < firstDist {
			t.Errorf("Result %d (dist=%f) closer than first result (dist=%f)", i, dist, firstDist)
		}
	}
}

// TestConcurrentSearches verifies concurrent search safety
func TestConcurrentSearches(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Insert test data
	for i := 0; i < 100; i++ {
		x := float64(i / 10)
		y := float64(i % 10)
		h.Insert(i, Vector{x, y})
	}

	var wg sync.WaitGroup
	queries := []Vector{
		{0, 0},
		{5, 5},
		{9, 9},
		{3, 7},
	}

	results := make([][]int, len(queries))
	config := DefaultSearchConfig()

	for i := range queries {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = h.SearchWithConfig(queries[idx], 5, config)
		}(i)
	}
	wg.Wait()

	for i, r := range results {
		if len(r) != 5 {
			t.Errorf("Query %d: got %d results, want 5", i, len(r))
			t.Logf("Query point: %v", queries[i])
			for j, id := range r {
				t.Logf("Result %d: point %v", j, h.Nodes[id].Vector)
			}
		}
	}
}

// TestDelete verifies deletion functionality
func TestDelete(t *testing.T) {
	h := New(2, 16, 32, 100, Euclidean)

	// Insert test vectors with known positions
	h.Insert(1, Vector{1, 1})
	h.Insert(2, Vector{2, 2})
	h.Insert(3, Vector{3, 3})

	config := SearchConfig{UseParallel: false} // Use sequential search for stability

	// Initial search
	query := Vector{1.1, 1.1}
	initialResults := h.SearchWithConfig(query, 1, config)
	if len(initialResults) == 0 {
		t.Fatal("Initial search returned no results")
	}
	if initialResults[0] != 1 {
		t.Errorf("Initial search failed, got %v, want [1]", initialResults)
	}

	// Delete vector
	h.Delete(1)

	// Search after deletion
	results := h.SearchWithConfig(query, 1, config)
	if len(results) == 0 {
		t.Error("Search() should return results after delete")
	} else if results[0] == 1 {
		t.Error("Search() returned deleted vector")
	}

	// Verify proper ordering
	deletedDist := h.DistanceFunc(Vector{1, 1}, query)
	resultDist := h.DistanceFunc(h.Nodes[results[0]].Vector, query)
	if resultDist < deletedDist {
		t.Error("Search returned closer point than deleted point")
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

func BenchmarkParallelSearch(b *testing.B) {
	h := New(128, 16, 32, 100, Euclidean)

	// Insert test data
	for i := 0; i < 1000; i++ {
		vec := make(Vector, 128)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		h.Insert(i, vec)
	}

	query := make(Vector, 128)
	for i := range query {
		query[i] = rand.Float64()
	}

	config := DefaultSearchConfig()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			h.SearchWithConfig(query, 10, config)
		}
	})
}

func BenchmarkSearchComparison(b *testing.B) {
	sizes := []int{1000, 10000, 100000}
	dimensions := []int{32, 128, 512}

	for _, size := range sizes {
		for _, dim := range dimensions {
			b.Run(fmt.Sprintf("Size_%d_Dim_%d", size, dim), func(b *testing.B) {
				h := New(dim, 16, 32, 100, Euclidean)

				// Insert vectors
				for i := 0; i < size; i++ {
					vec := make(Vector, dim)
					for j := range vec {
						vec[j] = rand.Float64()
					}
					h.Insert(i, vec)
				}

				query := make(Vector, dim)
				for i := range query {
					query[i] = rand.Float64()
				}

				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						h.Search(query, 10)
					}
				})
			})
		}
	}
}
