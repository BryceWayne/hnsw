package hnsw

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestNewValidationAndStats(t *testing.T) {
	if _, err := New[int](0); !errors.Is(err, ErrInvalidDimension) {
		t.Errorf("New(0) err = %v, want ErrInvalidDimension", err)
	}
	if _, err := New[int](-5); !errors.Is(err, ErrInvalidDimension) {
		t.Errorf("New(-5) err = %v, want ErrInvalidDimension", err)
	}
	if _, err := New[int](8, WithM(1)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithM(1) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithM(65536)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithM(65536) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithM(40000)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithM(40000) default Mmax>65535 err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithMmax(65536)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithMmax(65536) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithM(16), WithMmax(8)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("Mmax<M err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithEfConstruction(0)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("EfConstruction(0) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithEfSearch(0)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("EfSearch(0) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithCustomDistance(nil)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithCustomDistance(nil) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithMetric(Metric(99))); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("invalid metric err = %v, want ErrInvalidConfig", err)
	}

	idx, err := New[int](128,
		nil, // nil option ignored safely
		WithM(12),
		WithEfConstruction(150),
		WithEfSearch(40),
		WithMetric(MetricSquaredEuclidean),
		WithHeuristic(false),
		WithSeed(77),
	)
	if err != nil {
		t.Fatalf("New valid failed: %v", err)
	}
	if idx.Dim() != 128 || idx.Len() != 0 {
		t.Fatalf("Dim=%d Len=%d, want 128, 0", idx.Dim(), idx.Len())
	}
	st := idx.Stats()
	if st.Dim != 128 || st.M != 12 || st.Mmax != 24 || st.EfConstruction != 150 ||
		st.EfSearch != 40 || st.Metric != MetricSquaredEuclidean || st.Heuristic || st.EntryPointSet {
		t.Fatalf("unexpected Stats: %+v", st)
	}
}

func TestInsertSearchValidationAndMetrics(t *testing.T) {
	idx, err := New[int](3, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}

	// Empty index search returns empty slice, nil error
	res, err := idx.Search([]float32{1, 2, 3}, 5)
	if err != nil || len(res) != 0 {
		t.Fatalf("empty search = (%v, %v), want ([], nil)", res, err)
	}

	// Validation errors
	if err := idx.Insert(1, []float32{1, 2}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Insert dim mismatch = %v, want ErrDimensionMismatch", err)
	}
	if err := idx.Insert(1, []float32{1, float32(math.NaN()), 3}); !errors.Is(err, ErrNaNOrInf) {
		t.Errorf("Insert NaN = %v, want ErrNaNOrInf", err)
	}
	if err := idx.Insert(1, []float32{1, float32(math.Inf(1)), 3}); !errors.Is(err, ErrNaNOrInf) {
		t.Errorf("Insert Inf = %v, want ErrNaNOrInf", err)
	}

	if err := idx.Insert(1, []float32{0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Insert(2, []float32{3, 4, 0}); err != nil {
		t.Fatal(err)
	}

	if _, err := idx.Search([]float32{0, 0, 0}, 0); !errors.Is(err, ErrInvalidK) {
		t.Errorf("k=0 err = %v, want ErrInvalidK", err)
	}
	if _, err := idx.Search([]float32{0, 0, 0}, -1); !errors.Is(err, ErrInvalidK) {
		t.Errorf("k=-1 err = %v, want ErrInvalidK", err)
	}
	if _, err := idx.Search([]float32{0, 0}, 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Search dim mismatch = %v, want ErrDimensionMismatch", err)
	}
	if _, err := idx.Search([]float32{0, float32(math.NaN()), 0}, 1); !errors.Is(err, ErrNaNOrInf) {
		t.Errorf("Search NaN = %v, want ErrNaNOrInf", err)
	}
	if _, err := idx.Search([]float32{0, 0, 0}, 1, WithSearchEf(0)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("WithSearchEf(0) = %v, want ErrInvalidConfig", err)
	}

	// k = math.MaxInt must not panic or OOM
	res, err = idx.Search([]float32{0, 0, 0}, math.MaxInt)
	if err != nil || len(res) != 2 {
		t.Fatalf("k=MaxInt = (%v, %v), want len 2", res, err)
	}
	if res[0].ID != 1 || res[0].Distance != 0 || res[1].ID != 2 || math.Abs(float64(res[1].Distance-5)) > 1e-5 {
		t.Fatalf("Euclidean results = %+v, want IDs [1(0), 2(5)]", res)
	}

	// MetricSquaredEuclidean
	sqIdx, err := New[int](3, WithMetric(MetricSquaredEuclidean), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	_ = sqIdx.Insert(1, []float32{0, 0, 0})
	_ = sqIdx.Insert(2, []float32{3, 4, 0})
	sqRes, err := sqIdx.Search([]float32{0, 0, 0}, 2)
	if err != nil || len(sqRes) != 2 || math.Abs(float64(sqRes[1].Distance-25)) > 1e-5 {
		t.Fatalf("SquaredEuclidean results = %+v, err=%v; want Distance=25", sqRes, err)
	}

	// MetricDotProduct: highest dot product first, Distance == -dot
	dpIdx, err := New[int](3, WithMetric(MetricDotProduct), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	_ = dpIdx.Insert(1, []float32{1, 0, 0})
	_ = dpIdx.Insert(2, []float32{5, 0, 0})
	_ = dpIdx.Insert(3, []float32{-2, 0, 0})
	dpRes, err := dpIdx.Search([]float32{2, 0, 0}, 3)
	if err != nil || len(dpRes) != 3 {
		t.Fatalf("DotProduct search failed: %v", err)
	}
	if dpRes[0].ID != 2 || dpRes[0].Distance != -10 || dpRes[1].ID != 1 || dpRes[1].Distance != -2 || dpRes[2].ID != 3 || dpRes[2].Distance != 4 {
		t.Fatalf("DotProduct ordering/distances = %+v", dpRes)
	}
}

func TestCustomDistanceAndSliceCapSafety(t *testing.T) {
	// L1 (Manhattan) distance that attempts append(a, 999) to verify 3-index slice cap safety
	l1WithAppend := func(a, b []float32) float32 {
		_ = append(a, 999)
		_ = append(b, 888)
		var sum float32
		for i := range a {
			d := a[i] - b[i]
			if d < 0 {
				d = -d
			}
			sum += d
		}
		return sum
	}

	idx, err := New[int](4, WithCustomDistance(l1WithAppend), WithSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		if err := idx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	// Verify stored vectors were NOT corrupted by append inside custom distance
	for i := 0; i < 10; i++ {
		v, ok := idx.Get(i)
		if !ok || v[0] != float32(i) || v[1] != 0 || v[2] != 0 || v[3] != 0 {
			t.Fatalf("vector %d corrupted: %v (ok=%v)", i, v, ok)
		}
	}
	res, err := idx.Search([]float32{3.1, 0, 0, 0}, 2)
	if err != nil || len(res) != 2 || res[0].ID != 3 {
		t.Fatalf("custom search = (%+v, %v), want ID 3 first", res, err)
	}

	// Verify heuristic=false works cleanly too
	noHeur, err := New[int](4, WithHeuristic(false), WithM(4), WithSeed(12))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		_ = noHeur.Insert(i, []float32{float32(i), float32(i % 5), 0, 0})
	}
	res2, err := noHeur.Search([]float32{25, 0, 0, 0}, 3)
	if err != nil || len(res2) != 3 || res2[0].ID != 25 {
		t.Fatalf("WithHeuristic(false) search = (%+v, %v)", res2, err)
	}
}

func TestUpsertDeleteCompactClearAndEntryReelection(t *testing.T) {
	idx, err := New[int](4, WithM(8), WithMmax(16), WithEfConstruction(50), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}

	v := []float32{1, 0, 0, 0}
	if err := idx.Insert(0, v); err != nil {
		t.Fatal(err)
	}
	// Caller slice mutation must not affect stored vector
	v[0] = 999
	if stored, ok := idx.Get(0); !ok || stored[0] != 1 {
		t.Fatalf("caller mutation leaked: %v", stored)
	}
	// Mutating Get() return must not affect stored vector
	stored, _ := idx.Get(0)
	stored[0] = 777
	if stored2, _ := idx.Get(0); stored2[0] != 1 {
		t.Fatalf("Get() slice mutation leaked: %v", stored2)
	}

	for i := 1; i < 60; i++ {
		if err := idx.Insert(i, []float32{float32(i), float32(i), 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	// Upsert node 0 far away and verify reachability + zero orphans
	far := []float32{1000, 1000, 0, 0}
	if err := idx.Insert(0, far); err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 60 {
		t.Fatalf("Len after upsert = %d, want 60", idx.Len())
	}
	res1, err := idx.Search(far, 1)
	if err != nil || len(res1) != 1 || res1[0].ID != 0 {
		t.Fatalf("Search(far, 1) = (%+v, %v), want ID 0", res1, err)
	}
	resAll, err := idx.Search(far, 60)
	if err != nil || len(resAll) != 60 {
		t.Fatalf("Search(far, 60) len = %d, err = %v", len(resAll), err)
	}

	// Delete the current entry point specifically and verify re-election
	epID := idx.extKeyAt(uint32(idx.entryState.Load()))
	if !idx.Delete(epID) {
		t.Fatalf("Delete(entryPoint=%d) returned false", epID)
	}
	if idx.Delete(epID) {
		t.Fatalf("second Delete(entryPoint=%d) returned true", epID)
	}
	if idx.Contains(epID) {
		t.Fatalf("Contains(%d) true after Delete", epID)
	}

	st := idx.Stats()
	if st.Len != 59 || st.Deleted < 2 || !st.EntryPointSet {
		t.Fatalf("unexpected Stats after upsert+delete: %+v", st)
	}

	// Compact reclaims tombstones
	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact failed: %v", err)
	}
	stAfter := idx.Stats()
	if stAfter.Len != 59 || stAfter.Deleted != 0 || !stAfter.EntryPointSet {
		t.Fatalf("unexpected Stats after Compact: %+v", stAfter)
	}
	if res, err := idx.Search(far, 1); err != nil || len(res) != 1 || res[0].ID != 0 {
		t.Fatalf("Search after Compact = (%+v, %v)", res, err)
	}

	// Delete all remaining nodes -> empty index
	for i := 0; i < 60; i++ {
		idx.Delete(i)
	}
	if idx.Len() != 0 || idx.Stats().EntryPointSet {
		t.Fatalf("expected empty index after deleting all nodes: %+v", idx.Stats())
	}
	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact on all-deleted failed: %v", err)
	}
	if idx.Stats().Capacity != 0 {
		t.Fatalf("Capacity after Compact of empty = %d, want 0", idx.Stats().Capacity)
	}

	// Clear resets non-empty index
	_ = idx.Insert(1, []float32{1, 2, 3, 4})
	idx.Clear()
	if idx.Len() != 0 || idx.Contains(1) || idx.Stats().EntryPointSet {
		t.Fatalf("Clear did not reset index: %+v", idx.Stats())
	}
}

func TestSearchFilterAndCancellation(t *testing.T) {
	idx, err := New[int](4, WithM(8), WithMmax(16), WithEfConstruction(64), WithEfSearch(16), WithSeed(9))
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 200; i++ {
		if err := idx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	// Delete some matching and non-matching nodes to exercise tombstones + filter together
	idx.Delete(100)
	idx.Delete(101)

	// Selective filter (only IDs divisible by 10, which is 10% of live nodes)
	query := []float32{102, 0, 0, 0}
	res, err := idx.SearchFilter(query, 5, func(id int) bool {
		return id%10 == 0
	})
	if err != nil {
		t.Fatalf("SearchFilter failed: %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("SearchFilter len = %d, want 5", len(res))
	}
	for _, sr := range res {
		if sr.ID%10 != 0 || sr.ID == 100 {
			t.Errorf("unexpected filtered result: %+v", sr)
		}
	}
	if res[0].ID != 110 && res[0].ID != 90 {
		t.Errorf("closest multiple of 10 to 102 (excluding 100) = %d, want 110 or 90", res[0].ID)
	}

	// All-false filter returns empty slice without error
	none, err := idx.SearchFilter(query, 5, func(int) bool { return false })
	if err != nil || len(none) != 0 {
		t.Fatalf("all-false filter = (%v, %v), want ([], nil)", none, err)
	}

	// Cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := idx.Search(query, 5, WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled Search err = %v, want context.Canceled", err)
	}
	if _, err := idx.SearchFilter(query, 5, nil, WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled SearchFilter err = %v, want context.Canceled", err)
	}
}

func TestSearchScratchEpochWraparound(t *testing.T) {
	s := newSearchScratch(0, 32)
	s.epoch = math.MaxUint32 - 1

	s.begin(100)
	if s.epoch != math.MaxUint32 {
		t.Fatalf("epoch = %d, want MaxUint32", s.epoch)
	}
	if !s.visit(42) || s.visit(42) {
		t.Fatal("visit(42) failed before wraparound")
	}

	// Next begin wraps epoch to 0 -> must clear visited and set epoch = 1
	s.begin(100)
	if s.epoch != 1 {
		t.Fatalf("wrapped epoch = %d, want 1", s.epoch)
	}
	if !s.visit(42) {
		t.Fatal("visit(42) returned false after epoch wraparound")
	}

	// Dynamic growth mid-search preserves existing visited marks
	if !s.visit(250) {
		t.Fatal("visit(250) failed on dynamic growth")
	}
	if s.visit(42) {
		t.Fatal("visit(42) lost after dynamic growth")
	}
}

func TestSearchZeroAllocs(t *testing.T) {
	if raceDetectorEnabled {
		t.Skip("sync.Pool intentionally discards pooled items under -race")
	}
	idx, err := New[int](128, WithM(16), WithMmax(32), WithEfConstruction(100), WithEfSearch(50), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	items := make([]Item[int], 500)
	for i := range items {
		v := make([]float32, 128)
		for j := range v {
			v[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items); err != nil {
		t.Fatal(err)
	}

	query := make([]float32, 128)
	for j := range query {
		query[j] = rng.Float32()
	}

	// Warm up sync.Pool before measuring.
	for i := 0; i < 10; i++ {
		if _, err := idx.Search(query, 10); err != nil {
			t.Fatal(err)
		}
	}

	allocs := testing.AllocsPerRun(100, func() {
		res, err := idx.Search(query, 10)
		if err != nil || len(res) != 10 {
			t.Fatalf("unexpected search result: len=%d err=%v", len(res), err)
		}
	})
	t.Logf("Search(dim=128, k=10) allocs/op = %.1f", allocs)
	if allocs > 2 {
		t.Fatalf("Search allocs/op = %.1f, want <= 2", allocs)
	}
}

func TestRecall10K(t *testing.T) {
	const (
		numVecs    = 10000
		numQueries = 100
		dim        = 32
	)
	rng := rand.New(rand.NewSource(2026))

	idx, err := New[int](dim,
		WithM(16),
		WithMmax(32),
		WithEfConstruction(200),
		WithEfSearch(100),
		WithMetric(MetricEuclidean),
		WithSeed(2026),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Clustered + Gaussian distribution (16 cluster centers + Gaussian jitter)
	const numClusters = 16
	centers := make([][]float32, numClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for j := range centers[c] {
			centers[c][j] = float32(rng.NormFloat64() * 3.0)
		}
	}

	vecs := make([][]float32, numVecs)
	items := make([]Item[int], numVecs)
	for i := 0; i < numVecs; i++ {
		v := make([]float32, dim)
		c := centers[i%numClusters]
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64()*0.5)
		}
		vecs[i] = v
		items[i] = Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items); err != nil {
		t.Fatal(err)
	}

	type idDist struct {
		id   int
		dist float32
	}
	dists := make([]idDist, numVecs)
	top1Hits := 0
	top10Overlap := 0

	for q := 0; q < numQueries; q++ {
		query := make([]float32, dim)
		c := centers[q%numClusters]
		for j := range query {
			query[j] = c[j] + float32(rng.NormFloat64()*0.5)
		}

		for i := 0; i < numVecs; i++ {
			d, err := Euclidean(vecs[i], query)
			if err != nil {
				t.Fatal(err)
			}
			dists[i] = idDist{id: i, dist: d}
		}
		sort.Slice(dists, func(i, j int) bool {
			if dists[i].dist != dists[j].dist {
				return dists[i].dist < dists[j].dist
			}
			return dists[i].id < dists[j].id
		})

		got10, err := idx.Search(query, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(got10) != 10 {
			t.Fatalf("got %d results, want 10", len(got10))
		}
		if got10[0].ID == dists[0].id {
			top1Hits++
		}
		oracle := make(map[int]struct{}, 10)
		for i := 0; i < 10; i++ {
			oracle[dists[i].id] = struct{}{}
		}
		for _, r := range got10 {
			if _, ok := oracle[r.ID]; ok {
				top10Overlap++
			}
		}
	}

	recall1 := float64(top1Hits) / float64(numQueries)
	recall10 := float64(top10Overlap) / float64(numQueries*10)
	t.Logf("10K Clustered Recall@1=%.4f Recall@10=%.4f", recall1, recall10)
	if recall1 < 0.95 {
		t.Errorf("Recall@1 = %.4f, want >= 0.95", recall1)
	}
	if recall10 < 0.95 {
		t.Errorf("Recall@10 = %.4f, want >= 0.95", recall10)
	}
}

type customStructKey struct {
	Tenant string
	Seq    int64
}

func TestPersistenceRoundTripAndCorruption(t *testing.T) {
	// 1. int keys + tombstones + SaveFile / LoadFile
	t.Run("int_save_load_with_tombstones", func(t *testing.T) {
		rng := rand.New(rand.NewSource(123))
		idx, err := New[int](16, WithM(8), WithMmax(16), WithEfConstruction(64), WithSeed(123))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 60; i++ {
			v := make([]float32, 16)
			for j := range v {
				v[j] = rng.Float32()
			}
			if err := idx.Insert(i, v); err != nil {
				t.Fatal(err)
			}
		}
		for i := 0; i < 15; i++ {
			idx.Delete(i)
		}

		path := filepath.Join(t.TempDir(), "idx.bin")
		if err := idx.SaveFile(path); err != nil {
			t.Fatalf("SaveFile failed: %v", err)
		}

		loaded, err := LoadFile[int](path, WithEfSearch(80))
		if err != nil {
			t.Fatalf("LoadFile failed: %v", err)
		}
		if loaded.Len() != 45 || loaded.Stats().Deleted != 0 || loaded.Stats().EfSearch != 80 {
			t.Fatalf("loaded stats unexpected: %+v", loaded.Stats())
		}

		query := make([]float32, 16)
		for j := range query {
			query[j] = 0.5
		}
		r1, err := idx.Search(query, 10)
		if err != nil {
			t.Fatal(err)
		}
		r2, err := loaded.Search(query, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(r1) != len(r2) || len(r2) != 10 {
			t.Fatalf("result lens differ: %d vs %d", len(r1), len(r2))
		}
		for _, sr := range r2 {
			if sr.ID < 15 {
				t.Errorf("loaded search returned deleted key %d", sr.ID)
			}
		}

		// Type mismatch: loading int index as string must return ErrCorruptedIndex
		if _, err := LoadFile[string](path); !errors.Is(err, ErrCorruptedIndex) {
			t.Errorf("key type mismatch err = %v, want ErrCorruptedIndex", err)
		}
		// Conflicting metric option must return ErrInvalidConfig
		if _, err := LoadFile[int](path, WithMetric(MetricCosine)); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("conflicting metric err = %v, want ErrInvalidConfig", err)
		}
		if _, err := LoadFile[int](path, WithCustomDistance(func(_, _ []float32) float32 { return 0 })); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("custom dist on builtin index err = %v, want ErrInvalidConfig", err)
		}
	})

	// 2. string keys + custom struct keys (gob fallback)
	t.Run("string_and_struct_keys", func(t *testing.T) {
		sIdx, err := New[string](4, WithMetric(MetricDotProduct), WithSeed(5))
		if err != nil {
			t.Fatal(err)
		}
		_ = sIdx.Insert("alpha", []float32{1, 0, 0, 0})
		_ = sIdx.Insert("beta", []float32{2, 0, 0, 0})
		var buf bytes.Buffer
		if _, err := sIdx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		sLoaded, err := ReadFrom[string](&buf)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sLoaded.Search([]float32{1, 0, 0, 0}, 2)
		if err != nil || len(res) != 2 || res[0].ID != "beta" {
			t.Fatalf("string loaded search = (%+v, %v)", res, err)
		}

		cIdx, err := New[customStructKey](4, WithSeed(6))
		if err != nil {
			t.Fatal(err)
		}
		_ = cIdx.Insert(customStructKey{"t1", 1}, []float32{1, 2, 3, 4})
		_ = cIdx.Insert(customStructKey{"t2", 2}, []float32{5, 6, 7, 8})
		buf.Reset()
		if _, err := cIdx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		cLoaded, err := ReadFrom[customStructKey](&buf)
		if err != nil {
			t.Fatal(err)
		}
		if !cLoaded.Contains(customStructKey{"t1", 1}) || cLoaded.Len() != 2 {
			t.Fatalf("struct key reload failed: Len=%d", cLoaded.Len())
		}
	})

	// 3. WithCustomDistance serialization contract
	t.Run("custom_distance_roundtrip", func(t *testing.T) {
		l1 := func(a, b []float32) float32 {
			var s float32
			for i := range a {
				d := a[i] - b[i]
				if d < 0 {
					d = -d
				}
				s += d
			}
			return s
		}
		idx, err := New[int](3, WithCustomDistance(l1), WithSeed(1))
		if err != nil {
			t.Fatal(err)
		}
		_ = idx.Insert(10, []float32{1, 1, 1})
		_ = idx.Insert(20, []float32{5, 5, 5})
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		raw := buf.Bytes()

		if _, err := ReadFrom[int](bytes.NewReader(raw)); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("custom index without WithCustomDistance err = %v, want ErrInvalidConfig", err)
		}
		loaded, err := ReadFrom[int](bytes.NewReader(raw), WithCustomDistance(l1))
		if err != nil {
			t.Fatalf("ReadFrom custom failed: %v", err)
		}
		res, err := loaded.Search([]float32{1, 1, 1}, 1)
		if err != nil || len(res) != 1 || res[0].ID != 10 {
			t.Fatalf("custom loaded search = (%+v, %v)", res, err)
		}
	})

	// 4. Corruption, truncation, CRC32 flip, and OOM-attack header
	t.Run("corruption_and_oom_guards", func(t *testing.T) {
		idx, _ := New[int](4, WithSeed(1))
		for i := 0; i < 10; i++ {
			_ = idx.Insert(i, []float32{float32(i), 0, 0, 0})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		valid := buf.Bytes()

		// Truncation at multiple offsets
		for _, cut := range []int{0, 3, 20, 48, len(valid) / 2, len(valid) - 1} {
			if _, err := ReadFrom[int](bytes.NewReader(valid[:cut])); !errors.Is(err, ErrCorruptedIndex) {
				t.Errorf("truncation at %d err = %v, want ErrCorruptedIndex", cut, err)
			}
		}

		// Bit flips at header, payload, and CRC trailer
		for _, pos := range []int{0, 10, len(valid) / 2, len(valid) - 2} {
			corrupted := append([]byte(nil), valid...)
			corrupted[pos] ^= 0xFF
			if _, err := ReadFrom[int](bytes.NewReader(corrupted)); !errors.Is(err, ErrCorruptedIndex) {
				t.Errorf("bitflip at %d err = %v, want ErrCorruptedIndex", pos, err)
			}
		}

		// OOM-attack: claim LiveCount = 0x0FFFFFFF in a small buffer
		oom := append([]byte(nil), valid[:64]...)
		binary.LittleEndian.PutUint32(oom[40:44], 0x0FFFFFFF)
		if _, err := ReadFrom[int](bytes.NewReader(oom)); !errors.Is(err, ErrCorruptedIndex) {
			t.Errorf("OOM attack err = %v, want ErrCorruptedIndex", err)
		}

		// Corrupted Mmax > 65535 and M > 65535 must return ErrCorruptedIndex
		badMmax := append([]byte(nil), valid...)
		binary.LittleEndian.PutUint32(badMmax[20:24], 65536)
		if _, err := ReadFrom[int](bytes.NewReader(badMmax)); !errors.Is(err, ErrCorruptedIndex) {
			t.Errorf("mmax=65536 err = %v, want ErrCorruptedIndex", err)
		}
		badM := append([]byte(nil), valid...)
		binary.LittleEndian.PutUint32(badM[16:20], 65536)
		binary.LittleEndian.PutUint32(badM[20:24], 65536)
		if _, err := ReadFrom[int](bytes.NewReader(badM)); !errors.Is(err, ErrCorruptedIndex) {
			t.Errorf("m=65536 err = %v, want ErrCorruptedIndex", err)
		}

		// Missing file returns OS error
		if _, err := LoadFile[int](filepath.Join(t.TempDir(), "missing.bin")); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("missing file err = %v, want ErrNotExist", err)
		}
	})
}

func TestConcurrentStressRace(t *testing.T) {
	idx, err := New[int](16, WithM(12), WithMmax(24), WithEfConstruction(64), WithMetric(MetricCosine), WithSeed(99))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < 64; i++ {
		v := make([]float32, 16)
		for j := range v {
			v[j] = rng.Float32() + 0.1
		}
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup
	// Mutating / search workers
	for g := 0; g < 6; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			lrng := rand.New(rand.NewSource(int64(gid + 1)))
			for step := 0; step < 40; step++ {
				v := make([]float32, 16)
				for j := range v {
					v[j] = lrng.Float32() + 0.1
				}
				id := 100 + gid*40 + step
				if err := idx.Insert(id, v); err != nil {
					t.Errorf("Insert err: %v", err)
					return
				}
				if _, err := idx.Search(v, 5); err != nil {
					t.Errorf("Search err: %v", err)
					return
				}
				if _, err := idx.SearchFilter(v, 5, func(k int) bool { return k%2 == 0 }); err != nil {
					t.Errorf("SearchFilter err: %v", err)
					return
				}
				_, _ = idx.Get(id)
				_ = idx.Contains(id)
				_ = idx.Stats()
				if step%4 == 0 {
					idx.Delete(id)
				}
				if step%10 == 0 {
					// Concurrent upsert on existing key
					_ = idx.Insert(gid, v)
				}
			}
		}(g)
	}

	// Concurrent Compact + WriteTo goroutines
	wg.Add(1)
	go func() {
		defer wg.Done()
		var buf bytes.Buffer
		for i := 0; i < 5; i++ {
			buf.Reset()
			if _, err := idx.WriteTo(&buf); err != nil {
				t.Errorf("WriteTo err: %v", err)
				return
			}
			if err := idx.Compact(); err != nil {
				t.Errorf("Compact err: %v", err)
				return
			}
		}
	}()

	wg.Wait()
}

func TestPanicRecoveryDoesNotDeadlock(t *testing.T) {
	// 1. Panicking SearchFilter predicate
	idx, err := New[int](4, WithM(4), WithMmax(8), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if err := idx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic from SearchFilter predicate")
			}
		}()
		_, _ = idx.SearchFilter([]float32{5, 0, 0, 0}, 5, func(_ int) bool {
			panic("boom filter")
		})
	}()

	done := make(chan struct{})
	go func() {
		defer close(done)
		if !idx.Delete(0) {
			t.Errorf("Delete(0) returned false")
		}
		if err := idx.Compact(); err != nil {
			t.Errorf("Compact failed: %v", err)
		}
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("deadlocked on Delete/Compact after SearchFilter panic")
	}

	// 2. Panicking WithCustomDistance callback during Search and Insert
	shouldPanic := false
	cIdx, err := New[int](4, WithM(4), WithMmax(8), WithSeed(2), WithCustomDistance(func(a, b []float32) float32 {
		if shouldPanic {
			panic("boom customDist")
		}
		var sum float32
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return sum
	}))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if err := cIdx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	shouldPanic = true
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic from custom DistanceFunc in Search")
			}
		}()
		_, _ = cIdx.Search([]float32{5, 0, 0, 0}, 5)
	}()
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic from custom DistanceFunc in Insert")
			}
		}()
		_ = cIdx.Insert(100, []float32{100, 0, 0, 0})
	}()
	shouldPanic = false

	done2 := make(chan struct{})
	go func() {
		defer close(done2)
		if !cIdx.Delete(1) {
			t.Errorf("Delete(1) returned false")
		}
		if err := cIdx.Compact(); err != nil {
			t.Errorf("Compact failed: %v", err)
		}
		res, err := cIdx.Search([]float32{5, 0, 0, 0}, 3)
		if err != nil || len(res) != 3 {
			t.Errorf("Search after panic recovery = (%+v, %v)", res, err)
		}
	}()
	select {
	case <-done2:
	case <-time.After(3 * time.Second):
		t.Fatal("deadlocked on Delete/Compact after custom DistanceFunc panic")
	}
}

func TestConcurrentMultilevelInsertAndSearch(t *testing.T) {
	// Small M (M=2, Mmax=4) yields frequent upper-layer nodes (mL = 1/ln(2) ~ 1.44).
	idx, err := New[int](8, WithM(2), WithMmax(4), WithEfConstruction(32), WithEfSearch(32), WithSeed(77))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(77))
	const initialNodes = 40
	for i := 0; i < initialNodes; i++ {
		v := make([]float32, 8)
		for j := range v {
			v[j] = rng.Float32()
		}
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			lrng := rand.New(rand.NewSource(int64(1000 + gid)))
			for step := 0; step < 100; step++ {
				v := make([]float32, 8)
				for j := range v {
					v[j] = lrng.Float32()
				}
				if err := idx.Insert(initialNodes+gid*100+step, v); err != nil {
					t.Errorf("Insert error: %v", err)
					return
				}
			}
		}(g)
	}
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			lrng := rand.New(rand.NewSource(int64(2000 + gid)))
			q := make([]float32, 8)
			for step := 0; step < 150; step++ {
				for j := range q {
					q[j] = lrng.Float32()
				}
				res, err := idx.Search(q, 10)
				if err != nil {
					t.Errorf("Search error: %v", err)
					return
				}
				if len(res) != 10 {
					t.Errorf("Search(k=10) returned %d results with >=%d live nodes (upper-layer exposure?)", len(res), initialNodes)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	if idx.Stats().MaxLevel < 2 {
		t.Fatalf("expected multi-level graph with M=2, got MaxLevel=%d", idx.Stats().MaxLevel)
	}
}

func BenchmarkInsert(b *testing.B) {
	const numVecs = 500
	const dim = 128
	rng := rand.New(rand.NewSource(42))
	vecs := make([][]float32, numVecs)
	for i := range vecs {
		vecs[i] = make([]float32, dim)
		for j := range vecs[i] {
			vecs[i][j] = rng.Float32()
		}
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
		for j := 0; j < numVecs; j++ {
			if err := idx.Insert(j, vecs[j]); err != nil {
				b.Fatal(err)
			}
		}
	}
	b.StopTimer()
	elapsed := b.Elapsed()
	if elapsed > 0 {
		totalVecs := float64(b.N * numVecs)
		b.ReportMetric(totalVecs/elapsed.Seconds(), "vecs/s")
		b.ReportMetric(float64(elapsed.Nanoseconds())/totalVecs, "ns/vec")
	}
}

func testNumericKeyRoundTrip[K comparable](t *testing.T, name string, keys []K) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		idx, err := New[K](4, WithM(4), WithMmax(8), WithEfConstruction(32), WithSeed(11))
		if err != nil {
			t.Fatal(err)
		}
		for i, k := range keys {
			if err := idx.Insert(k, []float32{float32(i + 1), 0, 0, 0}); err != nil {
				t.Fatalf("Insert(%v) failed: %v", k, err)
			}
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		loaded, err := ReadFrom[K](&buf, WithMetric(MetricEuclidean))
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		if loaded.Len() != len(keys) {
			t.Fatalf("loaded Len = %d, want %d", loaded.Len(), len(keys))
		}
		for _, k := range keys {
			if !loaded.Contains(k) {
				t.Fatalf("missing key %v after roundtrip", k)
			}
		}
		res, err := loaded.Search([]float32{1, 0, 0, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != keys[0] {
			t.Fatalf("Search = (%+v, %v), want ID=%v", res, err, keys[0])
		}
	})
}

func TestSerializationAllKeyCodecsAndErrors(t *testing.T) {
	testNumericKeyRoundTrip(t, "int64", []int64{-10, 0, 42})
	testNumericKeyRoundTrip(t, "int32", []int32{-5, 1, 99})
	testNumericKeyRoundTrip(t, "uint64", []uint64{1, 2, 1 << 40})
	testNumericKeyRoundTrip(t, "uint32", []uint32{1, 2, 300})
	testNumericKeyRoundTrip(t, "uint", []uint{7, 8, 9})
	testNumericKeyRoundTrip(t, "int16", []int16{-12, 3, 45})
	testNumericKeyRoundTrip(t, "uint16", []uint16{10, 20, 30})
	testNumericKeyRoundTrip(t, "int8", []int8{-3, 0, 12})
	testNumericKeyRoundTrip(t, "uint8", []uint8{1, 2, 3})
	testNumericKeyRoundTrip(t, "bytes16", [][16]byte{{1}, {2}, {3}})

	// Empty index round-trip
	t.Run("empty_index_roundtrip", func(t *testing.T) {
		idx, err := New[string](8, WithMetric(MetricCosine), WithEfSearch(33), WithSeed(5))
		if err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		loaded, err := ReadFrom[string](&buf)
		if err != nil {
			t.Fatalf("empty ReadFrom failed: %v", err)
		}
		st := loaded.Stats()
		if st.Len != 0 || st.Dim != 8 || st.Metric != MetricCosine || st.EfSearch != 33 || st.EntryPointSet {
			t.Fatalf("unexpected empty loaded stats: %+v", st)
		}
	})

	// Empty keyKindGob struct-key round-trips (built-in metric, SaveFile/LoadFile, WithCustomDistance, delete-all) + populated round-trip
	t.Run("empty_gob_builtin_metric", func(t *testing.T) {
		idx, err := New[customStructKey](8, WithMetric(MetricEuclidean), WithEfSearch(24), WithSeed(7))
		if err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("empty gob WriteTo failed: %v", err)
		}
		loaded, err := ReadFrom[customStructKey](&buf)
		if err != nil {
			t.Fatalf("empty gob ReadFrom failed: %v", err)
		}
		st := loaded.Stats()
		if st.Len != 0 || st.Dim != 8 || st.Metric != MetricEuclidean || st.EfSearch != 24 || st.EntryPointSet {
			t.Fatalf("unexpected empty gob loaded stats: %+v", st)
		}
		res, err := loaded.Search(make([]float32, 8), 3)
		if err != nil || len(res) != 0 {
			t.Fatalf("empty gob Search = (%+v, %v), want empty nil", res, err)
		}

		// SaveFile / LoadFile round-trip on empty keyKindGob index
		path := filepath.Join(t.TempDir(), "empty_gob.bin")
		if err := idx.SaveFile(path); err != nil {
			t.Fatalf("empty gob SaveFile failed: %v", err)
		}
		fromFile, err := LoadFile[customStructKey](path)
		if err != nil {
			t.Fatalf("empty gob LoadFile failed: %v", err)
		}
		if fromFile.Len() != 0 || fromFile.Stats().EntryPointSet {
			t.Fatalf("unexpected empty gob LoadFile stats: %+v", fromFile.Stats())
		}
	})

	t.Run("empty_gob_custom_distance", func(t *testing.T) {
		l1 := func(a, b []float32) float32 {
			var s float32
			for i := range a {
				d := a[i] - b[i]
				if d < 0 {
					d = -d
				}
				s += d
			}
			return s
		}
		idx, err := New[customStructKey](4, WithCustomDistance(l1), WithSeed(9))
		if err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("empty custom gob WriteTo failed: %v", err)
		}
		raw := buf.Bytes()
		if _, err := ReadFrom[customStructKey](bytes.NewReader(raw)); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("empty custom gob ReadFrom without WithCustomDistance = %v, want ErrInvalidConfig", err)
		}
		loaded, err := ReadFrom[customStructKey](bytes.NewReader(raw), WithCustomDistance(l1))
		if err != nil {
			t.Fatalf("empty custom gob ReadFrom failed: %v", err)
		}
		st := loaded.Stats()
		if st.Len != 0 || st.Dim != 4 || st.EntryPointSet {
			t.Fatalf("unexpected empty custom gob stats: %+v", st)
		}
		k := customStructKey{Tenant: "alpha", Seq: 1}
		if err := loaded.Insert(k, []float32{1, 2, 3, 4}); err != nil {
			t.Fatalf("Insert into reloaded empty custom gob failed: %v", err)
		}
		res, err := loaded.Search([]float32{1, 2, 3, 4}, 1)
		if err != nil || len(res) != 1 || res[0].ID != k {
			t.Fatalf("Search on reloaded empty custom gob = (%+v, %v), want %+v", res, err, k)
		}
	})

	t.Run("delete_all_gob_roundtrip", func(t *testing.T) {
		idx, err := New[customStructKey](4, WithMetric(MetricCosine), WithSeed(13))
		if err != nil {
			t.Fatal(err)
		}
		keys := []customStructKey{
			{Tenant: "t1", Seq: 1},
			{Tenant: "t2", Seq: 2},
			{Tenant: "t3", Seq: 3},
		}
		for i, k := range keys {
			if err := idx.Insert(k, []float32{float32(i + 1), 1, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		for _, k := range keys {
			if !idx.Delete(k) {
				t.Fatalf("Delete(%+v) returned false", k)
			}
		}
		if idx.Len() != 0 {
			t.Fatalf("Len after delete-all = %d, want 0", idx.Len())
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("delete-all gob WriteTo failed: %v", err)
		}
		loaded, err := ReadFrom[customStructKey](&buf)
		if err != nil {
			t.Fatalf("delete-all gob ReadFrom failed: %v", err)
		}
		st := loaded.Stats()
		if st.Len != 0 || st.Deleted != 0 || st.EntryPointSet || st.Metric != MetricCosine {
			t.Fatalf("unexpected delete-all gob stats: %+v", st)
		}
		for _, k := range keys {
			if loaded.Contains(k) {
				t.Fatalf("loaded delete-all gob still contains %+v", k)
			}
		}
	})

	testNumericKeyRoundTrip(t, "gob_struct_populated", []customStructKey{
		{Tenant: "ns1", Seq: 10},
		{Tenant: "ns2", Seq: 20},
		{Tenant: "ns3", Seq: 30},
	})

	// Corrupt header & graph variants
	t.Run("corrupt_header_and_graph_variants", func(t *testing.T) {
		idx, err := New[uint32](4, WithM(4), WithMmax(8), WithSeed(1))
		if err != nil {
			t.Fatal(err)
		}
		for i := uint32(0); i < 6; i++ {
			if err := idx.Insert(i+1, []float32{float32(i), 0, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		valid := buf.Bytes()

		mutateAndExpectCorrupt := func(name string, mut func(b []byte)) {
			t.Helper()
			b := append([]byte(nil), valid...)
			mut(b)
			if _, err := ReadFrom[uint32](bytes.NewReader(b)); !errors.Is(err, ErrCorruptedIndex) {
				t.Errorf("%s: err = %v, want ErrCorruptedIndex", name, err)
			}
		}

		mutateAndExpectCorrupt("bad_magic", func(b []byte) { b[0] = 'X' })
		mutateAndExpectCorrupt("bad_version", func(b []byte) { binary.LittleEndian.PutUint32(b[4:8], 99) })
		mutateAndExpectCorrupt("zero_dim", func(b []byte) { binary.LittleEndian.PutUint32(b[8:12], 0) })
		mutateAndExpectCorrupt("bad_metric", func(b []byte) { b[12] = 99 })
		mutateAndExpectCorrupt("bad_heuristic", func(b []byte) { b[13] = 2 })
		mutateAndExpectCorrupt("nonzero_reserved", func(b []byte) { binary.LittleEndian.PutUint16(b[14:16], 1) })
		mutateAndExpectCorrupt("zero_efConstruction", func(b []byte) { binary.LittleEndian.PutUint32(b[24:28], 0) })
		mutateAndExpectCorrupt("zero_efSearch", func(b []byte) { binary.LittleEndian.PutUint32(b[28:32], 0) })
		mutateAndExpectCorrupt("entryPt_oob", func(b []byte) { binary.LittleEndian.PutUint32(b[48:52], 999) })
		mutateAndExpectCorrupt("maxLvl_oob", func(b []byte) { binary.LittleEndian.PutUint32(b[44:48], 100) })

		// Duplicate key in key section (starts right after 48 + 7 + len("uint32") = 61)
		keyOffset := 48 + 7 + len("uint32")
		mutateAndExpectCorrupt("duplicate_key", func(b []byte) {
			copy(b[keyOffset+4:keyOffset+8], b[keyOffset:keyOffset+4])
		})

		// Non-finite vector float (vectors follow 6 uint32 keys = 24 bytes)
		vecOffset := keyOffset + 6*4
		mutateAndExpectCorrupt("nan_vector", func(b []byte) {
			binary.LittleEndian.PutUint32(b[vecOffset:vecOffset+4], math.Float32bits(float32(math.NaN())))
		})

		// Graph adjacency invalid node level count (follows 6 * 4 * 4 = 96 vector bytes)
		graphOffset := vecOffset + 6*4*4
		mutateAndExpectCorrupt("zero_numLevels", func(b []byte) {
			b[graphOffset] = 0
		})
		mutateAndExpectCorrupt("degree_exceeds_liveCount", func(b []byte) {
			binary.LittleEndian.PutUint32(b[graphOffset+1:graphOffset+5], 6)
		})

		// Empty index with non-invalid entry point
		emptyIdx, err := New[uint32](4, WithSeed(1))
		if err != nil {
			t.Fatal(err)
		}
		var ebuf bytes.Buffer
		if _, err := emptyIdx.WriteTo(&ebuf); err != nil {
			t.Fatal(err)
		}
		eb := ebuf.Bytes()
		binary.LittleEndian.PutUint32(eb[48:52], 0)
		if _, err := ReadFrom[uint32](bytes.NewReader(eb)); !errors.Is(err, ErrCorruptedIndex) {
			t.Errorf("empty_bad_entryPt: err = %v, want ErrCorruptedIndex", err)
		}
	})
}

func TestMultiMetricRecallAndLifecycle(t *testing.T) {
	metrics := []struct {
		name   string
		metric Metric
		distFn func(a, b []float32) (float32, error)
	}{
		{"SquaredEuclidean", MetricSquaredEuclidean, SquaredEuclidean},
		{"Cosine", MetricCosine, Cosine},
		{"DotProduct", MetricDotProduct, func(a, b []float32) (float32, error) {
			d, err := DotProduct(a, b)
			return -d, err
		}},
	}

	const (
		numVecs    = 600
		numQueries = 30
		dim        = 24
		k          = 10
	)

	for _, tc := range metrics {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(707))
			idx, err := New[int](dim,
				WithM(16),
				WithMmax(32),
				WithEfConstruction(128),
				WithEfSearch(64),
				WithMetric(tc.metric),
				WithSeed(707),
			)
			if err != nil {
				t.Fatal(err)
			}

			vecs := make([][]float32, numVecs)
			items := make([]Item[int], numVecs)
			for i := 0; i < numVecs; i++ {
				v := make([]float32, dim)
				var norm float64
				for j := range v {
					v[j] = float32(rng.NormFloat64())
					norm += float64(v[j]) * float64(v[j])
				}
				inv := float32(1.0 / math.Sqrt(norm))
				for j := range v {
					v[j] *= inv
				}
				vecs[i] = v
				items[i] = Item[int]{ID: i, Vector: v}
			}
			if err := idx.InsertBatch(items); err != nil {
				t.Fatal(err)
			}

			queries := make([][]float32, numQueries)
			for q := 0; q < numQueries; q++ {
				v := make([]float32, dim)
				var norm float64
				for j := range v {
					v[j] = float32(rng.NormFloat64())
					norm += float64(v[j]) * float64(v[j])
				}
				inv := float32(1.0 / math.Sqrt(norm))
				for j := range v {
					v[j] *= inv
				}
				queries[q] = v
			}

			measureRecall := func(target *Index[int], deletedLimit int, ef int) float64 {
				type pair struct {
					id   int
					dist float32
				}
				hits := 0
				for _, qv := range queries {
					var dists []pair
					for i := deletedLimit; i < numVecs; i++ {
						d, err := tc.distFn(vecs[i], qv)
						if err != nil {
							t.Fatal(err)
						}
						dists = append(dists, pair{id: i, dist: d})
					}
					sort.Slice(dists, func(i, j int) bool {
						if dists[i].dist != dists[j].dist {
							return dists[i].dist < dists[j].dist
						}
						return dists[i].id < dists[j].id
					})
					oracle := make(map[int]struct{}, k)
					for i := 0; i < k; i++ {
						oracle[dists[i].id] = struct{}{}
					}
					got, err := target.Search(qv, k, WithSearchEf(ef))
					if err != nil {
						t.Fatal(err)
					}
					for _, r := range got {
						if _, ok := oracle[r.ID]; ok {
							hits++
						}
					}
				}
				return float64(hits) / float64(numQueries*k)
			}

			rLow := measureRecall(idx, 0, 10)
			rHigh := measureRecall(idx, 0, 100)
			if rHigh < 0.90 {
				t.Fatalf("%s Recall@10 (ef=100) = %.4f, want >= 0.90", tc.name, rHigh)
			}
			if rHigh < rLow {
				t.Fatalf("%s Recall@10 did not improve/hold with higher ef: ef=10 %.4f > ef=100 %.4f", tc.name, rLow, rHigh)
			}

			// Delete 10% of vectors, Compact, WriteTo/ReadFrom, and verify recall preservation
			for i := 0; i < 60; i++ {
				if !idx.Delete(i) {
					t.Fatalf("Delete(%d) returned false", i)
				}
			}
			if err := idx.Compact(); err != nil {
				t.Fatal(err)
			}
			var buf bytes.Buffer
			if _, err := idx.WriteTo(&buf); err != nil {
				t.Fatal(err)
			}
			reloaded, err := ReadFrom[int](&buf)
			if err != nil {
				t.Fatal(err)
			}
			rAfter := measureRecall(reloaded, 60, 100)
			if rAfter < 0.90 {
				t.Fatalf("%s Recall@10 after Delete+Compact+Reload = %.4f, want >= 0.90", tc.name, rAfter)
			}
		})
	}

	// dim=1 full lifecycle + nil filter + k > Len with tombstones
	t.Run("dim1_lifecycle_and_tombstone_k_bounds", func(t *testing.T) {
		idx, err := New[int](1, WithM(4), WithMmax(8), WithSeed(3))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 5; i++ {
			if err := idx.Insert(i, []float32{float32(i * 10)}); err != nil {
				t.Fatal(err)
			}
		}
		idx.Delete(1)
		idx.Delete(3)
		// SearchFilter with filter == nil behaves like Search, and k > liveCount returns liveCount results
		res, err := idx.SearchFilter([]float32{20}, 10, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 3 || res[0].ID != 2 || res[0].Distance != 0 {
			t.Fatalf("SearchFilter(nil, k=10) = %+v, want 3 live results with ID=2 first", res)
		}
		if err := idx.Compact(); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		loaded, err := ReadFrom[int](&buf)
		if err != nil {
			t.Fatal(err)
		}
		if loaded.Dim() != 1 || loaded.Len() != 3 {
			t.Fatalf("loaded dim=%d len=%d, want 1, 3", loaded.Dim(), loaded.Len())
		}
	})
}
