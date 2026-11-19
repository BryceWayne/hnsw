package hnsw_test

import (
	"bytes"
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/BryceWayne/hnsw"
)

func e2eManhattan(a, b []float32) float32 {
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

// Tier 1 — Feature F1: Index Construction & Options (5 cases)
func TestE2E_Tier1_F1_Construction(t *testing.T) {
	t.Run("DefaultOptions", func(t *testing.T) {
		idx := e2eMustNew[int](t, 8)
		if idx.Dim() != 8 {
			t.Fatalf("Dim() = %d, want 8", idx.Dim())
		}
		if idx.Len() != 0 {
			t.Fatalf("Len() = %d, want 0", idx.Len())
		}
		st := idx.Stats()
		if st.Dim != 8 || st.Len != 0 || st.Deleted != 0 || st.M != 16 || st.Mmax != 32 ||
			st.EfConstruction != 200 || st.EfSearch != 50 || st.Metric != hnsw.MetricEuclidean ||
			!st.Heuristic || st.EntryPointSet {
			t.Fatalf("unexpected default stats: %+v", st)
		}
	})

	t.Run("ExplicitTuningOptions", func(t *testing.T) {
		idx := e2eMustNew[int](t, 16,
			hnsw.WithM(8),
			hnsw.WithMmax(20),
			hnsw.WithEfConstruction(80),
			hnsw.WithEfSearch(25),
			hnsw.WithHeuristic(false),
			hnsw.WithSeed(7),
		)
		st := idx.Stats()
		if st.M != 8 || st.Mmax != 20 || st.EfConstruction != 80 || st.EfSearch != 25 || st.Heuristic {
			t.Fatalf("unexpected tuned stats: %+v", st)
		}
	})

	t.Run("MetricSelectionAndOverride", func(t *testing.T) {
		idxCos := e2eMustNew[int](t, 4,
			hnsw.WithCustomDistance(e2eManhattan),
			hnsw.WithMetric(hnsw.MetricCosine),
		)
		if idxCos.Stats().Metric != hnsw.MetricCosine {
			t.Fatalf("Metric = %v, want cosine", idxCos.Stats().Metric)
		}
		e2eMustInsert(t, idxCos, 1, []float32{2, 0, 0, 0})
		e2eMustInsert(t, idxCos, 2, []float32{0, 2, 0, 0})
		resCos, err := idxCos.Search([]float32{4, 0, 0, 0}, 1)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(resCos) != 1 || resCos[0].ID != 1 || !e2eApproxEqual(resCos[0].Distance, 0, 1e-5, 1e-5) {
			t.Fatalf("expected cosine match ID=1 dist~0, got %+v", resCos)
		}

		idxCust := e2eMustNew[int](t, 3,
			hnsw.WithMetric(hnsw.MetricCosine),
			hnsw.WithCustomDistance(e2eManhattan),
		)
		if idxCust.Stats().Metric.String() != "custom" {
			t.Fatalf("Metric.String() = %q, want %q", idxCust.Stats().Metric.String(), "custom")
		}
		e2eMustInsert(t, idxCust, 1, []float32{1, 2, 3})
		resCust, err := idxCust.Search([]float32{0, 0, 0}, 1)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(resCust) != 1 || !e2eApproxEqual(resCust[0].Distance, 6.0, 1e-5, 1e-5) {
			t.Fatalf("expected L1 dist=6, got %+v", resCust)
		}
	})

	t.Run("DeterministicSeedReproducibility", func(t *testing.T) {
		vecs := e2eMakeUniformVecs(101, 20, 6)
		query := []float32{0.2, -0.4, 0.1, 0.5, -0.2, 0.3}

		build := func() *hnsw.Index[int] {
			idx := e2eMustNew[int](t, 6, hnsw.WithM(4), hnsw.WithSeed(42))
			for i, v := range vecs {
				e2eMustInsert(t, idx, i, v)
			}
			return idx
		}

		idxA := build()
		idxB := build()
		if idxA.Stats().MaxLevel != idxB.Stats().MaxLevel {
			t.Fatalf("MaxLevel mismatch: %d vs %d", idxA.Stats().MaxLevel, idxB.Stats().MaxLevel)
		}
		resA, err := idxA.Search(query, 5)
		if err != nil {
			t.Fatalf("Search A failed: %v", err)
		}
		resB, err := idxB.Search(query, 5)
		if err != nil {
			t.Fatalf("Search B failed: %v", err)
		}
		if len(resA) != len(resB) || len(resA) != 5 {
			t.Fatalf("result lengths mismatch: %d vs %d", len(resA), len(resB))
		}
		for i := range resA {
			if resA[i].ID != resB[i].ID || math.Float32bits(resA[i].Distance) != math.Float32bits(resB[i].Distance) {
				t.Fatalf("result[%d] differs: %+v vs %+v", i, resA[i], resB[i])
			}
		}
	})

	t.Run("DimMethodMatchesConfig", func(t *testing.T) {
		for _, dim := range []int{1, 4, 32, 128} {
			idx := e2eMustNew[int](t, dim)
			if idx.Dim() != dim || idx.Stats().Dim != dim {
				t.Fatalf("dim mismatch: Dim()=%d, Stats().Dim=%d, want %d", idx.Dim(), idx.Stats().Dim, dim)
			}
		}
	})
}

// Tier 1 — Feature F2: Insert / Upsert & Point CRUD (5 cases)
func TestE2E_Tier1_F2_CRUD(t *testing.T) {
	t.Run("InsertGetContainsLen", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		vecs := e2eMakeUniformVecs(11, 5, 3)
		for i, v := range vecs {
			e2eMustInsert(t, idx, i+10, v)
		}
		if idx.Len() != 5 {
			t.Fatalf("Len() = %d, want 5", idx.Len())
		}
		for i, v := range vecs {
			id := i + 10
			if !idx.Contains(id) {
				t.Fatalf("Contains(%d) = false, want true", id)
			}
			got, ok := idx.Get(id)
			if !ok {
				t.Fatalf("Get(%d) ok = false, want true", id)
			}
			e2eAssertVecEqual(t, got, v)
		}
		if idx.Contains(999) {
			t.Fatalf("Contains(999) = true, want false")
		}
	})

	t.Run("UpsertReplacesVectorWithoutDup", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		e2eMustInsert(t, idx, 1, []float32{0, 0, 0})
		for i := 2; i <= 11; i++ {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		newVec := []float32{100, 0, 0}
		e2eMustInsert(t, idx, 1, newVec)

		if idx.Len() != 11 {
			t.Fatalf("Len() after upsert = %d, want 11", idx.Len())
		}
		got, ok := idx.Get(1)
		if !ok {
			t.Fatalf("Get(1) failed")
		}
		e2eAssertVecEqual(t, got, newVec)

		res, err := idx.Search([]float32{100, 0, 0}, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) == 0 || res[0].ID != 1 || res[0].Distance != 0 {
			t.Fatalf("expected ID=1 at dist=0, got %+v", res)
		}
	})

	t.Run("InsertDefensiveCopy", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		v := []float32{1, 2, 3}
		e2eMustInsert(t, idx, 7, v)
		v[0] = 999
		got, ok := idx.Get(7)
		if !ok {
			t.Fatalf("Get(7) failed")
		}
		e2eAssertVecEqual(t, got, []float32{1, 2, 3})
	})

	t.Run("GetReturnsDefensiveCopy", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		e2eMustInsert(t, idx, 7, []float32{1, 2, 3})
		got, ok := idx.Get(7)
		if !ok {
			t.Fatalf("Get(7) failed")
		}
		got[0] = 888
		got2, ok2 := idx.Get(7)
		if !ok2 {
			t.Fatalf("second Get(7) failed")
		}
		e2eAssertVecEqual(t, got2, []float32{1, 2, 3})
	})

	t.Run("MultipleSequentialUpsertsSameKey", func(t *testing.T) {
		idx := e2eMustNew[int](t, 2)
		coords := [][]float32{{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 6}}
		for _, v := range coords {
			e2eMustInsert(t, idx, 42, v)
		}
		if idx.Len() != 1 {
			t.Fatalf("Len() = %d, want 1", idx.Len())
		}
		got, ok := idx.Get(42)
		if !ok {
			t.Fatalf("Get(42) failed")
		}
		e2eAssertVecEqual(t, got, []float32{5, 6})
		res, err := idx.Search([]float32{5, 6}, 1)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 1 || res[0].ID != 42 || res[0].Distance != 0 {
			t.Fatalf("unexpected search result: %+v", res)
		}
	})
}

// Tier 1 — Feature F3: Built-in & Custom Distance Metrics (5 cases)
func TestE2E_Tier1_F3_Metrics(t *testing.T) {
	t.Run("EuclideanExactDistance", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithMetric(hnsw.MetricEuclidean))
		e2eMustInsert(t, idx, 1, []float32{0, 0, 0})
		e2eMustInsert(t, idx, 2, []float32{3, 4, 0})
		res, err := idx.Search([]float32{0, 0, 0}, 2)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 2 || res[0].ID != 1 || res[0].Distance != 0 ||
			res[1].ID != 2 || !e2eApproxEqual(res[1].Distance, 5.0, 1e-6, 1e-6) {
			t.Fatalf("unexpected Euclidean results: %+v", res)
		}
	})

	t.Run("SquaredEuclideanExactDistance", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithMetric(hnsw.MetricSquaredEuclidean))
		e2eMustInsert(t, idx, 1, []float32{0, 0, 0})
		e2eMustInsert(t, idx, 2, []float32{3, 4, 0})
		res, err := idx.Search([]float32{0, 0, 0}, 2)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 2 || res[0].ID != 1 || res[0].Distance != 0 ||
			res[1].ID != 2 || !e2eApproxEqual(res[1].Distance, 25.0, 1e-6, 1e-6) {
			t.Fatalf("unexpected SquaredEuclidean results: %+v", res)
		}
	})

	t.Run("CosineScaleInvarianceAndLosslessGet", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithMetric(hnsw.MetricCosine))
		orig := []float32{30, 40, 0}
		e2eMustInsert(t, idx, 1, orig)
		e2eMustInsert(t, idx, 2, []float32{0, 0, 5})
		res, err := idx.Search([]float32{3, 4, 0}, 2)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 2 || res[0].ID != 1 || !e2eApproxEqual(res[0].Distance, 0, 1e-5, 1e-5) ||
			res[1].ID != 2 || !e2eApproxEqual(res[1].Distance, 1.0, 1e-5, 1e-5) {
			t.Fatalf("unexpected Cosine results: %+v", res)
		}
		got, ok := idx.Get(1)
		if !ok {
			t.Fatalf("Get(1) failed")
		}
		e2eAssertVecEqual(t, got, orig)
	})

	t.Run("DotProductRankingNegativeDistance", func(t *testing.T) {
		idx := e2eMustNew[int](t, 2, hnsw.WithMetric(hnsw.MetricDotProduct))
		e2eMustInsert(t, idx, 1, []float32{1, 0})
		e2eMustInsert(t, idx, 2, []float32{5, 0})
		e2eMustInsert(t, idx, 3, []float32{-2, 0})
		res, err := idx.Search([]float32{2, 0}, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 3 ||
			res[0].ID != 2 || !e2eApproxEqual(res[0].Distance, -10, 1e-5, 1e-5) ||
			res[1].ID != 1 || !e2eApproxEqual(res[1].Distance, -2, 1e-5, 1e-5) ||
			res[2].ID != 3 || !e2eApproxEqual(res[2].Distance, 4, 1e-5, 1e-5) {
			t.Fatalf("unexpected DotProduct results: %+v", res)
		}
	})

	t.Run("CustomManhattanDistance", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithCustomDistance(e2eManhattan))
		e2eMustInsert(t, idx, 1, []float32{3, 2, 0})  // L1 to {3,1,0} = 1
		e2eMustInsert(t, idx, 2, []float32{0, 1, 0})  // L1 = 3
		e2eMustInsert(t, idx, 3, []float32{10, 1, 0}) // L1 = 7
		e2eMustInsert(t, idx, 4, []float32{3, 1, 5})  // L1 = 5
		e2eMustInsert(t, idx, 5, []float32{1, 1, 1})  // L1 = 3
		res, err := idx.Search([]float32{3, 1, 0}, 2)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 2 || res[0].ID != 1 || !e2eApproxEqual(res[0].Distance, 1.0, 1e-5, 1e-5) {
			t.Fatalf("unexpected custom L1 results: %+v", res)
		}
	})
}

// Tier 1 — Feature F4: Search, EfSearch, Filtering & Context (5 cases)
func TestE2E_Tier1_F4_Search(t *testing.T) {
	t.Run("AscendingDistanceOrdering", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4, hnsw.WithSeed(123))
		vecs := e2eMakeUniformVecs(123, 30, 4)
		items := make([]hnsw.Item[int], len(vecs))
		for i, v := range vecs {
			items[i] = hnsw.Item[int]{ID: i, Vector: v}
			e2eMustInsert(t, idx, i, v)
		}
		query := []float32{0.1, -0.2, 0.3, 0.4}
		res, err := idx.Search(query, 10)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 10 {
			t.Fatalf("got %d results, want 10", len(res))
		}
		e2eAssertSortedDistances(t, res)
		oracle := e2eExactKNN(items, query, 1, e2eOracleDistFunc(t, hnsw.MetricEuclidean), nil)
		if res[0].ID != oracle[0].ID {
			t.Fatalf("top-1 ID = %d, oracle = %d", res[0].ID, oracle[0].ID)
		}
	})

	t.Run("WithSearchEfOverride", func(t *testing.T) {
		const count = 400
		const queryCount = 30
		const dim = 32

		vecs, queries := e2eMakeClusteredDataset(77, count, queryCount, dim, 16, false)
		idx := e2eMustNew[int](t, dim,
			hnsw.WithM(4),
			hnsw.WithMmax(8),
			hnsw.WithEfConstruction(40),
			hnsw.WithEfSearch(2),
			hnsw.WithSeed(77),
		)
		items := make([]hnsw.Item[int], count)
		for i, v := range vecs {
			items[i] = hnsw.Item[int]{ID: i, Vector: v}
			e2eMustInsert(t, idx, i, v)
		}

		distFn := e2eOracleDistFunc(t, hnsw.MetricEuclidean)
		hitsDefault := 0
		hitsWithOverride := 0
		for _, q := range queries {
			oracle := e2eExactKNN(items, q, 1, distFn, nil)
			resDef, err := idx.Search(q, 1)
			if err != nil {
				t.Fatalf("default Search failed: %v", err)
			}
			if len(resDef) == 1 && resDef[0].ID == oracle[0].ID {
				hitsDefault++
			}
			resOver, err := idx.Search(q, 1, hnsw.WithSearchEf(64))
			if err != nil {
				t.Fatalf("override Search failed: %v", err)
			}
			if len(resOver) == 1 && resOver[0].ID == oracle[0].ID {
				hitsWithOverride++
			}
		}
		if idx.Stats().EfSearch != 2 {
			t.Fatalf("Stats().EfSearch mutated to %d, want 2", idx.Stats().EfSearch)
		}
		if hitsWithOverride <= hitsDefault {
			t.Fatalf("expected WithSearchEf(64) to recover more top-1 hits than EfSearch=2: default=%d, override=%d", hitsDefault, hitsWithOverride)
		}
	})

	t.Run("SearchFilterSelectivePredicate", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4, hnsw.WithSeed(55))
		vecs := e2eMakeUniformVecs(55, 30, 4)
		for i, v := range vecs {
			e2eMustInsert(t, idx, i, v)
		}
		res, err := idx.SearchFilter([]float32{0, 0, 0, 0}, 5, func(id int) bool {
			return id%3 == 0
		})
		if err != nil {
			t.Fatalf("SearchFilter failed: %v", err)
		}
		if len(res) != 5 {
			t.Fatalf("got %d results, want 5", len(res))
		}
		e2eAssertSortedDistances(t, res)
		for _, r := range res {
			if r.ID%3 != 0 {
				t.Fatalf("filter violated for ID=%d", r.ID)
			}
		}
	})

	t.Run("SearchFilterTraversalThroughNonMatching", func(t *testing.T) {
		idx := e2eMustNew[int](t, 2, hnsw.WithM(4), hnsw.WithEfSearch(32), hnsw.WithSeed(9))
		for i := range 10 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0})
		}
		e2eMustInsert(t, idx, 10, []float32{10, 0})

		res, err := idx.SearchFilter([]float32{0, 0}, 1, func(id int) bool {
			return id >= 10
		})
		if err != nil {
			t.Fatalf("SearchFilter failed: %v", err)
		}
		if len(res) != 1 || res[0].ID != 10 {
			t.Fatalf("expected ID=10 via non-matching traversal, got %+v", res)
		}
	})

	t.Run("ValidBackgroundContext", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		for i := range 5 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 1, 0})
		}
		res, err := idx.Search([]float32{0, 1, 0}, 3, hnsw.WithContext(context.Background()))
		if err != nil {
			t.Fatalf("Search with background context failed: %v", err)
		}
		if len(res) != 3 {
			t.Fatalf("got %d results, want 3", len(res))
		}
	})
}

// Tier 1 — Feature F5: Batch Operations & Worker Pools (5 cases)
func TestE2E_Tier1_F5_Batch(t *testing.T) {
	t.Run("InsertBatchHappyPath", func(t *testing.T) {
		idx := e2eMustNew[int](t, 6, hnsw.WithSeed(31))
		vecs := e2eMakeUniformVecs(31, 50, 6)
		items := make([]hnsw.Item[int], len(vecs))
		for i, v := range vecs {
			items[i] = hnsw.Item[int]{ID: i, Vector: v}
		}
		if err := idx.InsertBatch(items, hnsw.WithWorkers(4)); err != nil {
			t.Fatalf("InsertBatch failed: %v", err)
		}
		if idx.Len() != 50 {
			t.Fatalf("Len() = %d, want 50", idx.Len())
		}
		for i, v := range vecs {
			if !idx.Contains(i) {
				t.Fatalf("missing ID %d", i)
			}
			got, ok := idx.Get(i)
			if !ok {
				t.Fatalf("Get(%d) false", i)
			}
			e2eAssertVecEqual(t, got, v)
		}
	})

	t.Run("InsertBatchDuplicateLastWriteWins", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		items := []hnsw.Item[int]{
			{ID: 7, Vector: []float32{1, 1, 1}},
			{ID: 8, Vector: []float32{2, 2, 2}},
			{ID: 7, Vector: []float32{9, 9, 9}},
		}
		if err := idx.InsertBatch(items); err != nil {
			t.Fatalf("InsertBatch failed: %v", err)
		}
		if idx.Len() != 2 {
			t.Fatalf("Len() = %d, want 2", idx.Len())
		}
		got, ok := idx.Get(7)
		if !ok {
			t.Fatalf("Get(7) false")
		}
		e2eAssertVecEqual(t, got, []float32{9, 9, 9})
	})

	t.Run("SearchBatchQueryOrderPreserved", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		targets := [][]float32{
			{100, 0, 0},
			{0, 100, 0},
			{0, 0, 100},
			{-100, 0, 0},
			{0, -100, 0},
			{0, 0, -100},
		}
		for i, v := range targets {
			e2eMustInsert(t, idx, i, v)
		}
		res, err := idx.SearchBatch(targets, 1, hnsw.WithWorkers(3))
		if err != nil {
			t.Fatalf("SearchBatch failed: %v", err)
		}
		if len(res) != len(targets) {
			t.Fatalf("len(res) = %d, want %d", len(res), len(targets))
		}
		for i := range targets {
			if len(res[i]) != 1 || res[i][0].ID != i {
				t.Fatalf("query[%d] got %+v, want ID=%d", i, res[i], i)
			}
		}
	})

	t.Run("DeleteBatchRemovesSubset", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		for i := range 20 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0, 0})
		}
		removed := idx.DeleteBatch([]int{1, 3, 5, 7})
		if removed != 4 {
			t.Fatalf("DeleteBatch removed = %d, want 4", removed)
		}
		if idx.Len() != 16 {
			t.Fatalf("Len() = %d, want 16", idx.Len())
		}
		for _, id := range []int{1, 3, 5, 7} {
			if idx.Contains(id) {
				t.Fatalf("deleted ID %d still present", id)
			}
		}
		res, err := idx.Search([]float32{5, 0, 0, 0}, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		for _, r := range res {
			if r.ID == 1 || r.ID == 3 || r.ID == 5 || r.ID == 7 {
				t.Fatalf("deleted ID %d returned by Search", r.ID)
			}
		}
	})

	t.Run("DeleteBatchEmptyAndNoOp", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		e2eMustInsert(t, idx, 1, []float32{1, 2, 3})
		if removed := idx.DeleteBatch(nil); removed != 0 {
			t.Fatalf("DeleteBatch(nil) = %d, want 0", removed)
		}
		if removed := idx.DeleteBatch([]int{}); removed != 0 {
			t.Fatalf("DeleteBatch(empty) = %d, want 0", removed)
		}
		if idx.Len() != 1 {
			t.Fatalf("Len() = %d, want 1", idx.Len())
		}
	})
}

// Tier 1 — Feature F6: Delete, Compact, Clear & Stats (5 cases)
func TestE2E_Tier1_F6_Lifecycle(t *testing.T) {
	t.Run("DeleteUpdatesLenContainsAndSearch", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		for i := range 10 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		if !idx.Delete(5) {
			t.Fatalf("Delete(5) = false, want true")
		}
		if idx.Len() != 9 || idx.Contains(5) {
			t.Fatalf("post-delete state invalid: Len=%d Contains(5)=%v", idx.Len(), idx.Contains(5))
		}
		res, err := idx.Search([]float32{5, 0, 0}, 9)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		for _, r := range res {
			if r.ID == 5 {
				t.Fatalf("deleted ID 5 returned in Search")
			}
		}
	})

	t.Run("CompactReclaimsTombstones", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		for i := range 20 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 1, 0, 0})
		}
		e2eMustInsert(t, idx, 0, []float32{0.5, 1, 0, 0})
		e2eMustInsert(t, idx, 1, []float32{1.5, 1, 0, 0})
		for i := 2; i < 6; i++ {
			if !idx.Delete(i) {
				t.Fatalf("Delete(%d) failed", i)
			}
		}
		before := idx.Stats()
		if before.Deleted < 4 || before.Len != 16 {
			t.Fatalf("unexpected pre-compact stats: %+v", before)
		}
		if err := idx.Compact(); err != nil {
			t.Fatalf("Compact failed: %v", err)
		}
		after := idx.Stats()
		if after.Deleted != 0 || after.Len != 16 {
			t.Fatalf("unexpected post-compact stats: %+v", after)
		}
		res, err := idx.Search([]float32{0.5, 1, 0, 0}, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 5 || res[0].ID != 0 {
			t.Fatalf("expected ID=0 top-1 after Compact, got %+v", res)
		}
	})

	t.Run("CompactNoOpWhenNoTombstones", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		for i := range 10 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		if err := idx.Compact(); err != nil {
			t.Fatalf("Compact failed: %v", err)
		}
		st := idx.Stats()
		if st.Len != 10 || st.Deleted != 0 {
			t.Fatalf("unexpected stats after no-op Compact: %+v", st)
		}
	})

	t.Run("ClearResetsAndRetainsConfig", func(t *testing.T) {
		idx := e2eMustNew[int](t, 6, hnsw.WithMetric(hnsw.MetricCosine), hnsw.WithM(8))
		vecs := e2eMakeUniformVecs(8, 15, 6)
		for i, v := range vecs {
			e2eMustInsert(t, idx, i, v)
		}
		idx.Clear()
		st := idx.Stats()
		if st.Len != 0 || st.Deleted != 0 || st.Capacity != 0 || st.EntryPointSet ||
			st.Dim != 6 || st.Metric != hnsw.MetricCosine || st.M != 8 {
			t.Fatalf("unexpected stats after Clear: %+v", st)
		}
		e2eMustInsert(t, idx, 100, []float32{1, 0, 0, 0, 0, 0})
		if idx.Len() != 1 || !idx.Contains(100) {
			t.Fatalf("re-insert after Clear failed")
		}
	})

	t.Run("StatsAccuracyAcrossLifecycle", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		s0 := idx.Stats()
		if s0.Len != 0 || s0.EntryPointSet {
			t.Fatalf("initial stats invalid: %+v", s0)
		}
		for i := range 12 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0, 0})
		}
		s1 := idx.Stats()
		if s1.Len != 12 || s1.Deleted != 0 || !s1.EntryPointSet || s1.MemoryBytes <= 0 || s1.Capacity < 12 {
			t.Fatalf("populated stats invalid: %+v", s1)
		}
		for i := range 3 {
			idx.Delete(i)
		}
		s2 := idx.Stats()
		if s2.Len != 9 || s2.Deleted != 3 {
			t.Fatalf("post-delete stats invalid: %+v", s2)
		}
		if err := idx.Compact(); err != nil {
			t.Fatalf("Compact failed: %v", err)
		}
		s3 := idx.Stats()
		if s3.Len != 9 || s3.Deleted != 0 {
			t.Fatalf("post-compact stats invalid: %+v", s3)
		}
		idx.Clear()
		s4 := idx.Stats()
		if s4.Len != 0 || s4.Deleted != 0 || s4.Capacity != 0 || s4.EntryPointSet {
			t.Fatalf("post-clear stats invalid: %+v", s4)
		}
	})
}

// Tier 1 — Feature F7: Streaming & File Persistence (5 cases)
func TestE2E_Tier1_F7_Persistence(t *testing.T) {
	t.Run("WriteToReadFromRoundTrip", func(t *testing.T) {
		idx := e2eMustNew[int](t, 8, hnsw.WithMetric(hnsw.MetricCosine), hnsw.WithM(8), hnsw.WithSeed(19))
		vecs := e2eMakeUniformVecs(19, 25, 8)
		for i, v := range vecs {
			e2eMustInsert(t, idx, i, v)
		}
		var buf bytes.Buffer
		written, err := idx.WriteTo(&buf)
		if err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		if written <= 0 || int(written) != buf.Len() {
			t.Fatalf("written=%d, buf.Len()=%d", written, buf.Len())
		}
		reloaded, err := hnsw.ReadFrom[int](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		if reloaded.Stats().Metric != hnsw.MetricCosine || reloaded.Stats().M != 8 || reloaded.Len() != 25 {
			t.Fatalf("reloaded stats mismatch: %+v", reloaded.Stats())
		}
		for i, v := range vecs {
			got, ok := reloaded.Get(i)
			if !ok {
				t.Fatalf("reloaded missing key %d", i)
			}
			e2eAssertVecEqual(t, got, v)
		}
		q := vecs[7]
		wantRes, err := idx.Search(q, 5)
		if err != nil {
			t.Fatalf("orig Search failed: %v", err)
		}
		gotRes, err := reloaded.Search(q, 5)
		if err != nil {
			t.Fatalf("reloaded Search failed: %v", err)
		}
		if len(gotRes) != len(wantRes) || gotRes[0].ID != wantRes[0].ID {
			t.Fatalf("search mismatch after reload: got %+v, want %+v", gotRes, wantRes)
		}
	})

	t.Run("SaveFileLoadFileRoundTrip", func(t *testing.T) {
		idx := e2eMustNew[string](t, 4)
		for i := range 10 {
			key := string(rune('a' + i))
			e2eMustInsert(t, idx, key, []float32{float32(i), 1, 2, 3})
		}
		path := filepath.Join(t.TempDir(), "index.hnsw")
		if err := idx.SaveFile(path); err != nil {
			t.Fatalf("SaveFile failed: %v", err)
		}
		reloaded, err := hnsw.LoadFile[string](path)
		if err != nil {
			t.Fatalf("LoadFile failed: %v", err)
		}
		if reloaded.Len() != 10 {
			t.Fatalf("reloaded.Len() = %d, want 10", reloaded.Len())
		}
		res, err := reloaded.Search([]float32{4, 1, 2, 3}, 1)
		if err != nil {
			t.Fatalf("reloaded Search failed: %v", err)
		}
		if len(res) != 1 || res[0].ID != "e" {
			t.Fatalf("expected ID=%q, got %+v", "e", res)
		}
	})

	t.Run("ReloadStripsTombstones", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		for i := range 20 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0, 0})
		}
		for i := range 5 {
			idx.Delete(i)
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[int](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		st := reloaded.Stats()
		if st.Len != 15 || st.Deleted != 0 {
			t.Fatalf("expected Len=15, Deleted=0, got %+v", st)
		}
	})

	t.Run("ReloadOverridesEfSearchAndSeed", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4, hnsw.WithEfSearch(10), hnsw.WithSeed(1))
		for i := range 10 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0, 0})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[int](&buf, hnsw.WithEfSearch(75), hnsw.WithSeed(99))
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		if reloaded.Stats().EfSearch != 75 {
			t.Fatalf("EfSearch = %d, want 75", reloaded.Stats().EfSearch)
		}
		res, err := reloaded.Search([]float32{3, 0, 0, 0}, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 3 || res[0].ID != 3 {
			t.Fatalf("unexpected search result: %+v", res)
		}
	})

	t.Run("CustomDistanceRoundTripWithOption", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithCustomDistance(e2eManhattan))
		e2eMustInsert(t, idx, 1, []float32{1, 1, 1})
		e2eMustInsert(t, idx, 2, []float32{5, 5, 5})
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[int](&buf, hnsw.WithCustomDistance(e2eManhattan))
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		if reloaded.Stats().Metric.String() != "custom" {
			t.Fatalf("Metric.String() = %q, want custom", reloaded.Stats().Metric.String())
		}
		res, err := reloaded.Search([]float32{0, 0, 0}, 1)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 1 || res[0].ID != 1 || !e2eApproxEqual(res[0].Distance, 3.0, 1e-5, 1e-5) {
			t.Fatalf("unexpected custom reload search: %+v", res)
		}
	})
}

// Tier 1 — Feature F8: Generic Key Types (5 cases)
func TestE2E_Tier1_F8_GenericKeys(t *testing.T) {
	t.Run("IntKeys", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		keys := []int{-10, 0, 42}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i), 0, 0})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[int](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for _, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing key %d", k)
			}
			if _, ok := reloaded.Get(k); !ok {
				t.Fatalf("Get(%d) false", k)
			}
		}
		res, err := reloaded.Search([]float32{2, 0, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != 42 {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("Int64Keys", func(t *testing.T) {
		idx := e2eMustNew[int64](t, 3)
		keys := []int64{1 << 50, -(1 << 45), 99}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i), 1, 0})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[int64](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for _, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing key %d", k)
			}
		}
		res, err := reloaded.Search([]float32{0, 1, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != keys[0] {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("Uint64Keys", func(t *testing.T) {
		idx := e2eMustNew[uint64](t, 3)
		keys := []uint64{1, 1<<63 | 7, 42}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{0, float32(i), 1})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[uint64](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for _, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing key %d", k)
			}
		}
		res, err := reloaded.Search([]float32{0, 1, 1}, 1)
		if err != nil || len(res) != 1 || res[0].ID != keys[1] {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("StringKeys", func(t *testing.T) {
		idx := e2eMustNew[string](t, 3)
		keys := []string{"alpha", "beta-2", "gamma/three"}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{0, 0, float32(i)})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[string](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for _, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing key %q", k)
			}
		}
		res, err := reloaded.Search([]float32{0, 0, 2}, 1)
		if err != nil || len(res) != 1 || res[0].ID != "gamma/three" {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("ComparableStructKeys", func(t *testing.T) {
		idx := e2eMustNew[e2eCompositeKey](t, 3)
		keys := []e2eCompositeKey{
			{Tenant: "acme", Bucket: 1, ID: 10},
			{Tenant: "globex", Bucket: 2, ID: 20},
		}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i * 10), 0, 0})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[e2eCompositeKey](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for _, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing struct key %+v", k)
			}
			if _, ok := reloaded.Get(k); !ok {
				t.Fatalf("Get(%+v) false", k)
			}
		}
		res, err := reloaded.Search([]float32{10, 0, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != keys[1] {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})
}

// Tier 1 — Feature F9: Package Distance Functions & Metric Strings (5 cases)
func TestE2E_Tier1_F9_DistanceFuncs(t *testing.T) {
	t.Run("EuclideanHelper", func(t *testing.T) {
		d, err := hnsw.Euclidean([]float32{1, 2, 3}, []float32{4, 6, 3})
		if err != nil || !e2eApproxEqual(d, 5.0, 1e-6, 1e-6) {
			t.Fatalf("Euclidean = (%v, %v), want (5, nil)", d, err)
		}
	})

	t.Run("SquaredEuclideanHelper", func(t *testing.T) {
		d, err := hnsw.SquaredEuclidean([]float32{1, 2, 3}, []float32{4, 6, 3})
		if err != nil || !e2eApproxEqual(d, 25.0, 1e-6, 1e-6) {
			t.Fatalf("SquaredEuclidean = (%v, %v), want (25, nil)", d, err)
		}
	})

	t.Run("CosineHelper", func(t *testing.T) {
		dSame, err := hnsw.Cosine([]float32{2, 0}, []float32{5, 0})
		if err != nil || !e2eApproxEqual(dSame, 0.0, 1e-6, 1e-6) {
			t.Fatalf("same dir Cosine = (%v, %v), want (0, nil)", dSame, err)
		}
		dPerp, err := hnsw.Cosine([]float32{2, 0}, []float32{0, 3})
		if err != nil || !e2eApproxEqual(dPerp, 1.0, 1e-6, 1e-6) {
			t.Fatalf("perp Cosine = (%v, %v), want (1, nil)", dPerp, err)
		}
		dOpp, err := hnsw.Cosine([]float32{2, 0}, []float32{-2, 0})
		if err != nil || !e2eApproxEqual(dOpp, 2.0, 1e-6, 1e-6) {
			t.Fatalf("opp Cosine = (%v, %v), want (2, nil)", dOpp, err)
		}
	})

	t.Run("DotProductHelper", func(t *testing.T) {
		d1, err := hnsw.DotProduct([]float32{2, -3, 4}, []float32{5, 6, 1})
		if err != nil || !e2eApproxEqual(d1, -4.0, 1e-6, 1e-6) {
			t.Fatalf("DotProduct = (%v, %v), want (-4, nil)", d1, err)
		}
		d2, err := hnsw.DotProduct([]float32{2, 3}, []float32{5, 6})
		if err != nil || !e2eApproxEqual(d2, 28.0, 1e-6, 1e-6) {
			t.Fatalf("DotProduct = (%v, %v), want (28, nil)", d2, err)
		}
	})

	t.Run("MetricStringValues", func(t *testing.T) {
		cases := []struct {
			m    hnsw.Metric
			want string
		}{
			{hnsw.MetricEuclidean, "euclidean"},
			{hnsw.MetricSquaredEuclidean, "squared_euclidean"},
			{hnsw.MetricCosine, "cosine"},
			{hnsw.MetricDotProduct, "dot_product"},
			{hnsw.Metric(99), "unknown"},
		}
		for _, tc := range cases {
			if got := tc.m.String(); got != tc.want {
				t.Fatalf("Metric(%d).String() = %q, want %q", tc.m, got, tc.want)
			}
		}
	})
}
