package hnsw

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestBatchOperations(t *testing.T) {
	idx, err := New[int](2, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Empty batch calls should be safe no-ops
	if err := idx.InsertBatch(nil); err != nil {
		t.Fatalf("InsertBatch(nil) failed: %v", err)
	}
	if n := idx.DeleteBatch(nil); n != 0 {
		t.Errorf("DeleteBatch(nil) = %d, want 0", n)
	}
	emptyRes, err := idx.SearchBatch(nil, 5)
	if err != nil || len(emptyRes) != 0 {
		t.Errorf("SearchBatch(nil) = (%v, %v), want ([], nil)", emptyRes, err)
	}

	// Invalid worker option
	if err := idx.InsertBatch(nil, WithWorkers(0)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithWorkers(0) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := idx.SearchBatch(nil, 5, WithWorkers(-1)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("SearchBatch WithWorkers(-1) err = %v, want ErrInvalidConfig", err)
	}

	// InsertBatch with duplicate IDs (last-write-wins)
	items := make([]Item[int], 0, 101)
	for i := 0; i < 100; i++ {
		items = append(items, Item[int]{
			ID:     i,
			Vector: []float32{float32(i / 10), float32(i % 10)},
		})
	}
	// Overwrite ID 0 at the end with (99, 99)
	items = append(items, Item[int]{ID: 0, Vector: []float32{99, 99}})

	if err := idx.InsertBatch(items, WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}
	if idx.Len() != 100 {
		t.Fatalf("Len = %d, want 100", idx.Len())
	}
	if v0, ok := idx.Get(0); !ok || v0[0] != 99 || v0[1] != 99 {
		t.Fatalf("duplicate ID 0 = (%v, %v), want ([99 99], true)", v0, ok)
	}

	// Pre-validation atomicity: invalid item in batch must not mutate index
	badItems := []Item[int]{
		{ID: 500, Vector: []float32{1, 1}},
		{ID: 501, Vector: []float32{1, 2, 3}}, // mismatch
	}
	if err := idx.InsertBatch(badItems, WithWorkers(2)); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("InsertBatch dim mismatch err = %v, want ErrDimensionMismatch", err)
	}
	if idx.Contains(500) || idx.Len() != 100 {
		t.Errorf("InsertBatch mutated index on validation failure: Contains(500)=%v Len=%d", idx.Contains(500), idx.Len())
	}

	// SearchBatch with deterministic ordering
	queries := make([][]float32, 5)
	for i := range queries {
		queries[i] = []float32{float32(i), float32(i)}
	}
	results, err := idx.SearchBatch(queries, 5, WithWorkers(2), WithSearchEf(32))
	if err != nil {
		t.Fatalf("SearchBatch failed: %v", err)
	}
	if len(results) != len(queries) {
		t.Fatalf("SearchBatch len = %d, want %d", len(results), len(queries))
	}
	for qi, r := range results {
		if len(r) != 5 {
			t.Errorf("query %d len = %d, want 5", qi, len(r))
		}
		for j, sr := range r {
			if !idx.Contains(sr.ID) {
				t.Errorf("query %d returned unknown id %d", qi, sr.ID)
			}
			if j > 0 && sr.Distance < r[j-1].Distance {
				t.Errorf("query %d not sorted at %d: %v < %v", qi, j, sr.Distance, r[j-1].Distance)
			}
		}
	}

	// DeleteBatch (including duplicate and non-existent IDs)
	delIDs := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9999}
	removed := idx.DeleteBatch(delIDs)
	if removed != 10 {
		t.Errorf("DeleteBatch removed = %d, want 10", removed)
	}
	if idx.Len() != 90 {
		t.Errorf("Len after DeleteBatch = %d, want 90", idx.Len())
	}

	deletedSet := make(map[int]bool)
	for _, id := range delIDs {
		deletedSet[id] = true
		if idx.Contains(id) {
			t.Errorf("id %d still present after DeleteBatch", id)
		}
	}

	afterDel, err := idx.SearchBatch(queries, 5, WithWorkers(1))
	if err != nil {
		t.Fatalf("SearchBatch after delete failed: %v", err)
	}
	for qi, r := range afterDel {
		if len(r) != 5 {
			t.Errorf("afterDel query %d len = %d, want 5", qi, len(r))
		}
		for _, sr := range r {
			if deletedSet[sr.ID] {
				t.Errorf("afterDel query %d returned deleted id %d", qi, sr.ID)
			}
		}
	}
}

func TestBatchContextCancellation(t *testing.T) {
	idx, err := New[int](8, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := idx.InsertBatch([]Item[int]{{ID: 1, Vector: make([]float32, 8)}}, WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Errorf("pre-cancelled InsertBatch err = %v, want context.Canceled", err)
	}
	if idx.Len() != 0 {
		t.Errorf("pre-cancelled InsertBatch mutated index: Len=%d", idx.Len())
	}

	if _, err := idx.SearchBatch([][]float32{make([]float32, 8)}, 1, WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Errorf("pre-cancelled SearchBatch err = %v, want context.Canceled", err)
	}
}

func TestBatchEdgeCasesAndCoverage(t *testing.T) {
	idx, err := New[int](3, WithM(4), WithMmax(8), WithEfConstruction(32), WithSeed(11))
	if err != nil {
		t.Fatal(err)
	}

	// SearchBatch on empty index returns empty slices
	emptyBatch, err := idx.SearchBatch([][]float32{{1, 2, 3}, {4, 5, 6}}, 3, WithWorkers(1))
	if err != nil || len(emptyBatch) != 2 || len(emptyBatch[0]) != 0 || len(emptyBatch[1]) != 0 {
		t.Fatalf("empty SearchBatch = (%+v, %v), want [[], []]", emptyBatch, err)
	}

	// Sequential InsertBatch with WithWorkers(1)
	items := make([]Item[int], 20)
	for i := range items {
		items[i] = Item[int]{ID: i, Vector: []float32{float32(i), 0, 0}}
	}
	if err := idx.InsertBatch(items, WithWorkers(1)); err != nil {
		t.Fatalf("InsertBatch(WithWorkers(1)) failed: %v", err)
	}
	if idx.Len() != 20 {
		t.Fatalf("Len = %d, want 20", idx.Len())
	}

	// NaN pre-validation atomicity on InsertBatch
	nanItems := []Item[int]{
		{ID: 500, Vector: []float32{1, 1, 1}},
		{ID: 501, Vector: []float32{1, float32(math.NaN()), 1}},
	}
	if err := idx.InsertBatch(nanItems); !errors.Is(err, ErrNaNOrInf) {
		t.Errorf("InsertBatch NaN err = %v, want ErrNaNOrInf", err)
	}
	if idx.Contains(500) || idx.Len() != 20 {
		t.Errorf("InsertBatch NaN mutated index: Contains(500)=%v Len=%d", idx.Contains(500), idx.Len())
	}

	// SearchBatch validation errors
	if _, err := idx.SearchBatch([][]float32{{0, 0, 0}}, 0); !errors.Is(err, ErrInvalidK) {
		t.Errorf("SearchBatch k=0 err = %v, want ErrInvalidK", err)
	}
	if _, err := idx.SearchBatch([][]float32{{0, 0}}, 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("SearchBatch dim mismatch err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := idx.SearchBatch([][]float32{{0, float32(math.Inf(1)), 0}}, 1); !errors.Is(err, ErrNaNOrInf) {
		t.Errorf("SearchBatch Inf err = %v, want ErrNaNOrInf", err)
	}
	if _, err := idx.SearchBatch([][]float32{{0, 0, 0}}, 1, WithSearchEf(0)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("SearchBatch WithSearchEf(0) err = %v, want ErrInvalidConfig", err)
	}

	// Sequential SearchBatch cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := idx.SearchBatch([][]float32{{0, 0, 0}, {1, 0, 0}}, 2, WithWorkers(1), WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Errorf("SearchBatch(WithWorkers(1), canceled) err = %v, want context.Canceled", err)
	}

	// DeleteBatch with removed <= 8 (2-hop repair branch)
	if removed := idx.DeleteBatch([]int{0, 1, 2, 3, 999}); removed != 4 {
		t.Errorf("DeleteBatch(<=8) removed = %d, want 4", removed)
	}
	if idx.Len() != 16 {
		t.Errorf("Len after DeleteBatch(<=8) = %d, want 16", idx.Len())
	}
	res, err := idx.Search([]float32{2, 0, 0}, 3)
	if err != nil || len(res) != 3 {
		t.Fatalf("Search after DeleteBatch(<=8) = (%+v, %v)", res, err)
	}

	// DeleteBatch removing all remaining nodes -> empty index
	remaining := make([]int, 0, 16)
	for i := 4; i < 20; i++ {
		remaining = append(remaining, i)
	}
	if removed := idx.DeleteBatch(remaining); removed != 16 {
		t.Errorf("DeleteBatch(all) removed = %d, want 16", removed)
	}
	if idx.Len() != 0 || idx.Stats().EntryPointSet {
		t.Fatalf("expected empty index after DeleteBatch(all): %+v", idx.Stats())
	}
}

func BenchmarkBatchSizes(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	batchSizes := []int{64, 128, 256, 512}
	dim := 256
	size := 10000

	idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(64), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}
	items := make([]Item[int], size)
	for i := 0; i < size; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: vec}
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	allQueries := make([][]float32, 512)
	for i := range allQueries {
		allQueries[i] = make([]float32, dim)
		for j := range allQueries[i] {
			allQueries[i][j] = rng.Float32()
		}
	}

	for _, bSize := range batchSizes {
		b.Run(fmt.Sprintf("batch=%d", bSize), func(b *testing.B) {
			batchQueries := allQueries[:bSize]
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := idx.SearchBatch(batchQueries, 10); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
				b.ReportMetric(float64(b.N*len(batchQueries))/elapsed, "queries/s")
			}
		})
	}
}

func BenchmarkBatchOperations(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128

	b.Run("InsertBatch", func(b *testing.B) {
		const batchCount = 1000
		items := make([]Item[int], batchCount)
		for i := 0; i < batchCount; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()
			}
			items[i] = Item[int]{ID: i, Vector: v}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
			if err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			if err := idx.InsertBatch(items); err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		elapsed := b.Elapsed()
		if elapsed > 0 {
			totalVecs := float64(b.N * batchCount)
			b.ReportMetric(totalVecs/elapsed.Seconds(), "vecs/s")
			b.ReportMetric(float64(elapsed.Nanoseconds())/totalVecs, "ns/vec")
		}
	})

	b.Run("SearchBatch", func(b *testing.B) {
		idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
		if err != nil {
			b.Fatal(err)
		}
		items := make([]Item[int], 1000)
		for i := range items {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()
			}
			items[i] = Item[int]{ID: i, Vector: v}
		}
		if err := idx.InsertBatch(items); err != nil {
			b.Fatal(err)
		}

		queries := make([][]float32, 100)
		for i := range queries {
			queries[i] = make([]float32, dim)
			for j := range queries[i] {
				queries[i][j] = rng.Float32()
			}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := idx.SearchBatch(queries, 10); err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
			b.ReportMetric(float64(b.N*len(queries))/elapsed, "queries/s")
		}
	})
}
