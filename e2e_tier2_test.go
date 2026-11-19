package hnsw_test

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/BryceWayne/hnsw"
)

// Tier 2 — Feature F1: Construction & Options Boundaries (5 cases)
func TestE2E_Tier2_F1_ConstructionEdges(t *testing.T) {
	t.Run("InvalidDimensions", func(t *testing.T) {
		for _, dim := range []int{0, -1, -1024} {
			_, err := hnsw.New[int](dim)
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidDimension, "New invalid dim")
		}
	})

	t.Run("MAndMmaxBounds", func(t *testing.T) {
		invalidOpts := [][]hnsw.Option{
			{hnsw.WithM(1)},
			{hnsw.WithM(65536)},
			{hnsw.WithM(40000)}, // implicit Mmax=80000 > 65535
			{hnsw.WithMmax(0)},
			{hnsw.WithMmax(65536)},
			{hnsw.WithM(16), hnsw.WithMmax(8)},
		}
		for _, opts := range invalidOpts {
			_, err := hnsw.New[int](4, opts...)
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "invalid M/Mmax")
		}

		idxMin := e2eMustNew[int](t, 4, hnsw.WithM(2), hnsw.WithMmax(2))
		if st := idxMin.Stats(); st.M != 2 || st.Mmax != 2 {
			t.Fatalf("unexpected min M/Mmax stats: %+v", st)
		}
		idxMax := e2eMustNew[int](t, 4, hnsw.WithM(65535), hnsw.WithMmax(65535))
		if st := idxMax.Stats(); st.M != 65535 || st.Mmax != 65535 {
			t.Fatalf("unexpected max M/Mmax stats: %+v", st)
		}
	})

	t.Run("EfBounds", func(t *testing.T) {
		for _, opts := range [][]hnsw.Option{
			{hnsw.WithEfConstruction(0)},
			{hnsw.WithEfConstruction(-5)},
			{hnsw.WithEfSearch(0)},
			{hnsw.WithEfSearch(-1)},
		} {
			_, err := hnsw.New[int](4, opts...)
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "invalid ef")
		}

		idx := e2eMustNew[int](t, 4, hnsw.WithEfConstruction(1), hnsw.WithEfSearch(1))
		if st := idx.Stats(); st.EfConstruction != 1 || st.EfSearch != 1 {
			t.Fatalf("unexpected ef=1 stats: %+v", st)
		}
	})

	t.Run("InvalidMetricAndNilCustomDist", func(t *testing.T) {
		_, err := hnsw.New[int](4, hnsw.WithMetric(hnsw.Metric(254)))
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "invalid metric")

		_, err = hnsw.New[int](4, hnsw.WithCustomDistance(nil))
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "nil custom dist")
	})

	t.Run("NilOptionAndExtremeSeeds", func(t *testing.T) {
		for _, seed := range []int64{0, -999} {
			idx := e2eMustNew[int](t, 4, nil, hnsw.WithSeed(seed), nil)
			e2eMustInsert(t, idx, 1, []float32{1, 2, 3, 4})
			res, err := idx.Search([]float32{1, 2, 3, 4}, 1)
			if err != nil || len(res) != 1 || res[0].ID != 1 {
				t.Fatalf("seed=%d search failed: res=%+v err=%v", seed, res, err)
			}
		}
	})
}

// Tier 2 — Feature F2: CRUD & Upsert Boundaries (5 cases)
func TestE2E_Tier2_F2_CRUDEdges(t *testing.T) {
	t.Run("InsertDimensionMismatch", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		for _, bad := range [][]float32{nil, {}, {1, 2, 3}, {1, 2, 3, 4, 5}} {
			err := idx.Insert(1, bad)
			e2eAssertErrorIs(t, err, hnsw.ErrDimensionMismatch, "Insert dim mismatch")
		}
		if idx.Len() != 0 {
			t.Fatalf("Len() = %d, want 0", idx.Len())
		}
	})

	t.Run("InsertNaNAndInf", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		badVecs := [][]float32{
			{float32(math.NaN()), 0, 0},
			{0, float32(math.Inf(1)), 0},
			{0, 0, float32(math.Inf(-1))},
		}
		for _, bad := range badVecs {
			err := idx.Insert(1, bad)
			e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, "Insert NaN/Inf")
		}
		if idx.Len() != 0 {
			t.Fatalf("Len() = %d, want 0", idx.Len())
		}
	})

	t.Run("ZeroVectorsAcrossAllMetrics", func(t *testing.T) {
		metrics := []hnsw.Metric{
			hnsw.MetricEuclidean,
			hnsw.MetricSquaredEuclidean,
			hnsw.MetricCosine,
			hnsw.MetricDotProduct,
		}
		zeroVec := []float32{0, 0, 0, 0}
		query := []float32{1, 0, 0, 0}
		for _, m := range metrics {
			idx := e2eMustNew[int](t, 4, hnsw.WithMetric(m))
			e2eMustInsert(t, idx, 1, zeroVec)
			got, ok := idx.Get(1)
			if !ok {
				t.Fatalf("metric %v Get(1) false", m)
			}
			e2eAssertVecEqual(t, got, zeroVec)

			res, err := idx.Search(query, 1)
			if err != nil || len(res) != 1 {
				t.Fatalf("metric %v Search failed: res=%+v err=%v", m, res, err)
			}
			switch m {
			case hnsw.MetricEuclidean, hnsw.MetricSquaredEuclidean:
				if !e2eApproxEqual(res[0].Distance, 1.0, 1e-6, 1e-6) {
					t.Fatalf("metric %v dist=%v, want 1.0", m, res[0].Distance)
				}
			case hnsw.MetricCosine:
				if res[0].Distance != 1.0 {
					t.Fatalf("MetricCosine zero vec dist=%v, want 1.0", res[0].Distance)
				}
				resZero, err := idx.Search(zeroVec, 1)
				if err != nil || len(resZero) != 1 || resZero[0].Distance != 1.0 {
					t.Fatalf("MetricCosine zero query search=(%+v, %v), want dist=1.0", resZero, err)
				}
			case hnsw.MetricDotProduct:
				if res[0].Distance != 0 || math.Signbit(float64(res[0].Distance)) {
					t.Fatalf("MetricDotProduct zero vec dist=%v (signbit=%v), want +0.0", res[0].Distance, math.Signbit(float64(res[0].Distance)))
				}
			}
		}
	})

	t.Run("GetAndContainsMissingAndDeleted", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		if vec, ok := idx.Get(1); ok || vec != nil {
			t.Fatalf("empty Get(1) = (%v, %v), want (nil, false)", vec, ok)
		}
		if idx.Contains(1) {
			t.Fatalf("empty Contains(1) = true")
		}
		e2eMustInsert(t, idx, 1, []float32{1, 2, 3})
		if idx.Contains(99) {
			t.Fatalf("unknown Contains(99) = true")
		}
		if !idx.Delete(1) {
			t.Fatalf("Delete(1) = false")
		}
		if vec, ok := idx.Get(1); ok || vec != nil {
			t.Fatalf("deleted Get(1) = (%v, %v), want (nil, false)", vec, ok)
		}
		if idx.Contains(1) {
			t.Fatalf("deleted Contains(1) = true")
		}
	})

	t.Run("UpsertAtSingleNodeIndex", func(t *testing.T) {
		idx := e2eMustNew[int](t, 2)
		e2eMustInsert(t, idx, 1, []float32{1, 0})
		e2eMustInsert(t, idx, 1, []float32{10, 0})
		if idx.Len() != 1 {
			t.Fatalf("Len() = %d, want 1", idx.Len())
		}
		got, ok := idx.Get(1)
		if !ok {
			t.Fatalf("Get(1) failed")
		}
		e2eAssertVecEqual(t, got, []float32{10, 0})
		res, err := idx.Search([]float32{10, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != 1 || res[0].Distance != 0 {
			t.Fatalf("unexpected Search after single-node upsert: res=%+v err=%v", res, err)
		}
	})
}

// Tier 2 — Feature F3: Metric & Odd Dimension Boundaries (5 cases)
func TestE2E_Tier2_F3_MetricEdges(t *testing.T) {
	t.Run("OddAndUnalignedDimsAllBuiltins", func(t *testing.T) {
		dims := []int{1, 3, 7, 8, 15, 127, 130}
		metrics := []hnsw.Metric{
			hnsw.MetricEuclidean,
			hnsw.MetricSquaredEuclidean,
			hnsw.MetricCosine,
			hnsw.MetricDotProduct,
		}
		for _, dim := range dims {
			vecs := e2eMakeUniformVecs(int64(dim*17+3), 10, dim)
			query := e2eMakeUniformVecs(int64(dim*31+9), 1, dim)[0]
			items := make([]hnsw.Item[int], len(vecs))
			for i, v := range vecs {
				items[i] = hnsw.Item[int]{ID: i, Vector: v}
			}
			for _, m := range metrics {
				idx := e2eMustNew[int](t, dim, hnsw.WithMetric(m), hnsw.WithEfSearch(10), hnsw.WithSeed(42))
				if err := idx.InsertBatch(items); err != nil {
					t.Fatalf("dim=%d metric=%v InsertBatch failed: %v", dim, m, err)
				}
				res, err := idx.Search(query, 3)
				if err != nil {
					t.Fatalf("dim=%d metric=%v Search failed: %v", dim, m, err)
				}
				oracle := e2eExactKNN(items, query, 3, e2eOracleDistFunc(t, m), nil)
				if len(res) != 3 {
					t.Fatalf("dim=%d metric=%v len=%d, want 3", dim, m, len(res))
				}
				if oracle[0].Distance+1e-6 < oracle[1].Distance && res[0].ID != oracle[0].ID {
					t.Fatalf("dim=%d metric=%v top-1 mismatch: got=%+v oracle=%+v", dim, m, res, oracle)
				}
				for i := range res {
					if !e2eApproxEqual(res[i].Distance, oracle[i].Distance, 1e-4, 1e-4) {
						t.Fatalf("dim=%d metric=%v res[%d] dist=%v, oracle=%v", dim, m, i, res[i].Distance, oracle[i].Distance)
					}
				}
			}
		}
	})

	t.Run("CustomDistanceSliceAppendCapSafety", func(t *testing.T) {
		appendingDist := func(a, b []float32) float32 {
			_ = append(a, 999)
			_ = append(b, 888)
			return e2eRefEuclidean(a, b)
		}
		idx := e2eMustNew[int](t, 5, hnsw.WithCustomDistance(appendingDist), hnsw.WithSeed(5))
		vecs := e2eMakeUniformVecs(5, 12, 5)
		for i, v := range vecs {
			e2eMustInsert(t, idx, i, v)
		}
		for i, v := range vecs {
			got, ok := idx.Get(i)
			if !ok {
				t.Fatalf("Get(%d) failed", i)
			}
			e2eAssertVecEqual(t, got, v)
		}
	})

	t.Run("CustomDistanceNegativeAndTiedScores", func(t *testing.T) {
		constNeg := func(a, b []float32) float32 {
			return -1.5
		}
		idx := e2eMustNew[int](t, 3, hnsw.WithCustomDistance(constNeg), hnsw.WithSeed(3))
		for i := range 8 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		res, err := idx.Search([]float32{0, 0, 0}, 4)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 4 {
			t.Fatalf("got %d results, want 4", len(res))
		}
		e2eAssertSortedDistances(t, res)
		for i, r := range res {
			if r.Distance != -1.5 {
				t.Fatalf("result[%d] distance = %v, want -1.5", i, r.Distance)
			}
		}
	})

	t.Run("CosineExtremeSubnormalAndLargeScales", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithMetric(hnsw.MetricCosine), hnsw.WithSeed(11))
		e2eMustInsert(t, idx, 1, []float32{1e-25, 0, 0})  // parallel
		e2eMustInsert(t, idx, 2, []float32{0, 1e20, 0})   // orthogonal
		e2eMustInsert(t, idx, 3, []float32{-1e20, 0, 0})  // opposite
		res, err := idx.Search([]float32{1e-25, 0, 0}, 3) // tiny query
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		e2eAssertSortedDistances(t, res)
		if len(res) != 3 || res[0].ID != 1 || res[1].ID != 2 || res[2].ID != 3 {
			t.Fatalf("unexpected extreme scale Cosine ranking: %+v", res)
		}
		if !e2eApproxEqual(res[0].Distance, 0, 1e-4, 1e-4) ||
			!e2eApproxEqual(res[1].Distance, 1, 1e-4, 1e-4) ||
			!e2eApproxEqual(res[2].Distance, 2, 1e-4, 1e-4) {
			t.Fatalf("unexpected extreme scale Cosine distances: %+v", res)
		}
	})

	t.Run("HeuristicDisabledSmallM", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithHeuristic(false), hnsw.WithM(2), hnsw.WithMmax(4), hnsw.WithSeed(21))
		for i := range 30 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		res, err := idx.Search([]float32{15, 0, 0}, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 5 {
			t.Fatalf("got %d results, want 5", len(res))
		}
		e2eAssertSortedDistances(t, res)
		if res[0].ID != 15 || res[0].Distance != 0 {
			t.Fatalf("expected top-1 {ID:15, Distance:0}, got %+v", res[0])
		}
		wantDists := []float32{0, 1, 1, 2, 2}
		wantIDs := map[int]bool{15: true, 14: true, 16: true, 13: true, 17: true}
		seenIDs := make(map[int]bool, 5)
		for i, r := range res {
			if r.Distance != wantDists[i] {
				t.Fatalf("res[%d].Distance = %v, want %v (full=%+v)", i, r.Distance, wantDists[i], res)
			}
			if !wantIDs[r.ID] || seenIDs[r.ID] {
				t.Fatalf("unexpected or duplicate ID %d in %+v", r.ID, res)
			}
			seenIDs[r.ID] = true
		}
	})
}

// Tier 2 — Feature F4: Search, Filter & Context Boundaries (5 cases)
func TestE2E_Tier2_F4_SearchEdges(t *testing.T) {
	t.Run("EmptyIndexSearchAndFilter", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		q := []float32{1, 2, 3, 4}
		res, err := idx.Search(q, 5)
		if err != nil || res == nil || len(res) != 0 {
			t.Fatalf("empty Search = (%#v, %v), want non-nil empty slice, nil", res, err)
		}
		fres, err := idx.SearchFilter(q, 5, func(id int) bool { return true })
		if err != nil || fres == nil || len(fres) != 0 {
			t.Fatalf("empty SearchFilter = (%#v, %v), want non-nil empty slice, nil", fres, err)
		}
	})

	t.Run("InvalidKAndQueryValidation", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		e2eMustInsert(t, idx, 1, []float32{1, 2, 3})

		for _, badK := range []int{0, -5} {
			_, err := idx.Search([]float32{1, 2, 3}, badK)
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidK, "invalid k")
		}
		_, err := idx.Search([]float32{1, 2}, 1)
		e2eAssertErrorIs(t, err, hnsw.ErrDimensionMismatch, "query dim mismatch")

		_, err = idx.Search([]float32{1, float32(math.NaN()), 3}, 1)
		e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, "query NaN")

		_, err = idx.Search([]float32{1, 2, float32(math.Inf(1))}, 1)
		e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, "query Inf")

		_, err = idx.Search([]float32{1, 2, 3}, 1, hnsw.WithSearchEf(0))
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "WithSearchEf(0)")
	})

	t.Run("KGreaterThanLenAndMaxInt", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		for i := range 5 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		res100, err := idx.Search([]float32{0, 0, 0}, 100)
		if err != nil || len(res100) != 5 {
			t.Fatalf("k=100 got (%d, %v), want (5, nil)", len(res100), err)
		}
		resMax, err := idx.Search([]float32{0, 0, 0}, math.MaxInt)
		if err != nil || len(resMax) != 5 {
			t.Fatalf("k=MaxInt got (%d, %v), want (5, nil)", len(resMax), err)
		}
	})

	t.Run("EfSearchOneAndEfLessThanK", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4, hnsw.WithEfSearch(1), hnsw.WithSeed(14))
		for i, v := range e2eMakeUniformVecs(14, 20, 4) {
			e2eMustInsert(t, idx, i, v)
		}
		res, err := idx.Search([]float32{0, 0, 0, 0}, 5, hnsw.WithSearchEf(1))
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 5 {
			t.Fatalf("got %d results, want 5 (clamped ef >= k)", len(res))
		}
		e2eAssertSortedDistances(t, res)
	})

	t.Run("FilterNilAllFalseAndCanceledContext", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3, hnsw.WithSeed(6))
		for i := range 10 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		q := []float32{3, 0, 0}
		resStd, err := idx.Search(q, 5, nil)
		if err != nil {
			t.Fatalf("Search with nil opt failed: %v", err)
		}
		resNilFilter, err := idx.SearchFilter(q, 5, nil)
		if err != nil {
			t.Fatalf("SearchFilter(nil) failed: %v", err)
		}
		if len(resStd) != len(resNilFilter) {
			t.Fatalf("nil filter len=%d != Search len=%d", len(resNilFilter), len(resStd))
		}
		for i := range resStd {
			if resStd[i].ID != resNilFilter[i].ID {
				t.Fatalf("nil filter[%d]=%v != Search[%d]=%v", i, resNilFilter[i], i, resStd[i])
			}
		}

		resNone, err := idx.SearchFilter(q, 5, func(id int) bool { return false })
		if err != nil || len(resNone) != 0 {
			t.Fatalf("all-false filter = (%+v, %v), want empty, nil", resNone, err)
		}

		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err = idx.Search(q, 3, hnsw.WithContext(canceledCtx))
		e2eAssertErrorIs(t, err, context.Canceled, "canceled context Search")

		expiredCtx, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancelExpired()
		_, err = idx.SearchFilter(q, 3, nil, hnsw.WithContext(expiredCtx))
		e2eAssertErrorIs(t, err, context.DeadlineExceeded, "expired context SearchFilter")
	})
}

// Tier 2 — Feature F5: Batch Operations Boundaries (5 cases)
func TestE2E_Tier2_F5_BatchEdges(t *testing.T) {
	t.Run("WithWorkersInvalidAndBoundaries", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		items := make([]hnsw.Item[int], 8)
		queries := make([][]float32, 8)
		for i := range 8 {
			v := []float32{float32(i), 1, 0}
			items[i] = hnsw.Item[int]{ID: i, Vector: v}
			queries[i] = v
		}

		for _, badW := range []int{0, -2} {
			err := idx.InsertBatch(items, hnsw.WithWorkers(badW))
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "InsertBatch invalid workers")
			_, err = idx.SearchBatch(queries, 1, hnsw.WithWorkers(badW))
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "SearchBatch invalid workers")
		}

		if err := idx.InsertBatch(items, hnsw.WithWorkers(1)); err != nil {
			t.Fatalf("InsertBatch workers=1 failed: %v", err)
		}
		res1, err := idx.SearchBatch(queries, 1, hnsw.WithWorkers(1))
		if err != nil {
			t.Fatalf("SearchBatch workers=1 failed: %v", err)
		}
		res100, err := idx.SearchBatch(queries, 1, hnsw.WithWorkers(100))
		if err != nil {
			t.Fatalf("SearchBatch workers=100 failed: %v", err)
		}
		for i := range queries {
			if len(res1[i]) != 1 || len(res100[i]) != 1 || res1[i][0].ID != i || res100[i][0].ID != i {
				t.Fatalf("query[%d] mismatch: w1=%+v w100=%+v", i, res1[i], res100[i])
			}
		}
	})

	t.Run("InsertBatchAtomicValidationFailure", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		for i := range 5 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0})
		}
		badDimBatch := []hnsw.Item[int]{
			{ID: 100, Vector: []float32{1, 2, 3}},
			{ID: 101, Vector: []float32{4, 5, 6}},
			{ID: 102, Vector: []float32{7, 8, 9}},
			{ID: 103, Vector: []float32{1, 2}}, // mismatch
		}
		err := idx.InsertBatch(badDimBatch)
		e2eAssertErrorIs(t, err, hnsw.ErrDimensionMismatch, "InsertBatch dim mismatch")
		if idx.Len() != 5 || idx.Contains(100) {
			t.Fatalf("InsertBatch mutated index on dim failure: Len=%d", idx.Len())
		}

		badNaNBatch := []hnsw.Item[int]{
			{ID: 104, Vector: []float32{1, 2, 3}},
			{ID: 105, Vector: []float32{float32(math.NaN()), 2, 3}},
		}
		err = idx.InsertBatch(badNaNBatch)
		e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, "InsertBatch NaN")
		if idx.Len() != 5 || idx.Contains(104) {
			t.Fatalf("InsertBatch mutated index on NaN failure: Len=%d", idx.Len())
		}
	})

	t.Run("SearchBatchEmptyIndexEmptyQueriesAndErrors", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		_, err := idx.SearchBatch(nil, 0)
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidK, "SearchBatch k=0")

		emptyQRes, err := idx.SearchBatch(nil, 3)
		if err != nil || len(emptyQRes) != 0 {
			t.Fatalf("SearchBatch(nil, 3) = (%+v, %v), want len=0, nil", emptyQRes, err)
		}

		res, err := idx.SearchBatch([][]float32{{1, 2, 3}, {4, 5, 6}}, 3)
		if err != nil || len(res) != 2 || len(res[0]) != 0 || len(res[1]) != 0 {
			t.Fatalf("SearchBatch on empty index = (%+v, %v)", res, err)
		}

		e2eMustInsert(t, idx, 1, []float32{1, 2, 3})
		_, err = idx.SearchBatch([][]float32{{1, 2, 3}, {1, 2}}, 1)
		e2eAssertErrorIs(t, err, hnsw.ErrDimensionMismatch, "SearchBatch dim mismatch")

		_, err = idx.SearchBatch([][]float32{{1, 2, float32(math.Inf(-1))}}, 1)
		e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, "SearchBatch Inf")
	})

	t.Run("BatchCanceledAndExpiredContext", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		items := []hnsw.Item[int]{
			{ID: 1, Vector: []float32{1, 0, 0}},
			{ID: 2, Vector: []float32{0, 1, 0}},
		}
		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		for _, workers := range []int{1, 4} {
			err := idx.InsertBatch(items, hnsw.WithContext(canceledCtx), hnsw.WithWorkers(workers))
			e2eAssertErrorIs(t, err, context.Canceled, "InsertBatch canceled")
			if idx.Len() != 0 {
				t.Fatalf("InsertBatch mutated on canceled ctx: Len=%d", idx.Len())
			}
			_, err = idx.SearchBatch([][]float32{{1, 0, 0}}, 1, hnsw.WithContext(canceledCtx), hnsw.WithWorkers(workers))
			e2eAssertErrorIs(t, err, context.Canceled, "SearchBatch canceled")
		}

		expiredCtx, cancelExp := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancelExp()
		err := idx.InsertBatch(items, hnsw.WithContext(expiredCtx))
		e2eAssertErrorIs(t, err, context.DeadlineExceeded, "InsertBatch expired")
		_, err = idx.SearchBatch([][]float32{{1, 0, 0}}, 1, hnsw.WithContext(expiredCtx))
		e2eAssertErrorIs(t, err, context.DeadlineExceeded, "SearchBatch expired")
	})

	t.Run("DeleteBatchSmallLargeAndAll", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4, hnsw.WithSeed(4))
		for i := range 30 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 1, 0, 0})
		}
		// Small batch with duplicates and missing ID
		r1 := idx.DeleteBatch([]int{0, 1, 1, 2, 3, 999})
		if r1 != 4 || idx.Len() != 26 {
			t.Fatalf("small DeleteBatch: r1=%d Len=%d, want 4, 26", r1, idx.Len())
		}
		// Larger batch
		batch2 := make([]int, 12)
		for i := range 12 {
			batch2[i] = i + 4
		}
		r2 := idx.DeleteBatch(batch2)
		if r2 != 12 || idx.Len() != 14 {
			t.Fatalf("larger DeleteBatch: r2=%d Len=%d, want 12, 14", r2, idx.Len())
		}
		// Delete remaining 14
		batch3 := make([]int, 14)
		for i := range 14 {
			batch3[i] = i + 16
		}
		r3 := idx.DeleteBatch(batch3)
		if r3 != 14 || idx.Len() != 0 || idx.Stats().EntryPointSet {
			t.Fatalf("final DeleteBatch: r3=%d Stats=%+v", r3, idx.Stats())
		}
	})
}

// Tier 2 — Feature F6: Delete, Compact & Clear Boundaries (5 cases)
func TestE2E_Tier2_F6_LifecycleEdges(t *testing.T) {
	t.Run("DeleteNonExistentAndDoubleDelete", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		if idx.Delete(1) {
			t.Fatalf("Delete on empty index returned true")
		}
		e2eMustInsert(t, idx, 1, []float32{1, 2, 3})
		if idx.Delete(999) {
			t.Fatalf("Delete(999) returned true")
		}
		if !idx.Delete(1) {
			t.Fatalf("first Delete(1) returned false")
		}
		if idx.Delete(1) {
			t.Fatalf("second Delete(1) returned true")
		}
		if idx.Stats().Deleted != 1 {
			t.Fatalf("Deleted = %d, want 1", idx.Stats().Deleted)
		}
	})

	t.Run("DeleteSingleNodeEntryPointAndReinsert", func(t *testing.T) {
		idx := e2eMustNew[int](t, 3)
		e2eMustInsert(t, idx, 1, []float32{1, 0, 0})
		if !idx.Delete(1) {
			t.Fatalf("Delete(1) failed")
		}
		if st := idx.Stats(); st.Len != 0 || st.EntryPointSet {
			t.Fatalf("unexpected stats after deleting sole node: %+v", st)
		}
		resEmpty, err := idx.Search([]float32{1, 0, 0}, 1)
		if err != nil || len(resEmpty) != 0 {
			t.Fatalf("search after sole-node delete = (%+v, %v)", resEmpty, err)
		}
		e2eMustInsert(t, idx, 2, []float32{0, 1, 0})
		if st := idx.Stats(); st.Len != 1 || !st.EntryPointSet {
			t.Fatalf("unexpected stats after reinsert: %+v", st)
		}
		res, err := idx.Search([]float32{0, 1, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != 2 {
			t.Fatalf("search after reinsert = (%+v, %v)", res, err)
		}
	})

	t.Run("DeleteMostNodesMultiLevelReelection", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4, hnsw.WithM(2), hnsw.WithMmax(4), hnsw.WithEfSearch(40), hnsw.WithSeed(42))
		vecs := e2eMakeUniformVecs(42, 40, 4)
		for i, v := range vecs {
			e2eMustInsert(t, idx, i, v)
		}
		before := idx.Stats()
		for i := range 30 {
			if !idx.Delete(i) {
				t.Fatalf("Delete(%d) failed", i)
			}
		}
		after := idx.Stats()
		if after.Len != 10 || !after.EntryPointSet || after.MaxLevel > before.MaxLevel {
			t.Fatalf("unexpected stats after 30 deletes: before=%+v after=%+v", before, after)
		}
		res, err := idx.Search(vecs[35], 10)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 10 {
			t.Fatalf("got %d reachable nodes, want 10", len(res))
		}
		e2eAssertSortedDistances(t, res)
		if res[0].ID != 35 {
			t.Fatalf("expected top-1 ID=35, got %+v", res[0])
		}
		for _, r := range res {
			if r.ID < 30 {
				t.Fatalf("deleted ID %d returned", r.ID)
			}
		}
	})

	t.Run("DeleteAllThenCompactReleasesCapacity", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		for i := range 30 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 0, 0, 0})
		}
		for i := range 30 {
			idx.Delete(i)
		}
		if err := idx.Compact(); err != nil {
			t.Fatalf("Compact failed: %v", err)
		}
		st := idx.Stats()
		if st.Len != 0 || st.Deleted != 0 || st.Capacity != 0 || st.EntryPointSet {
			t.Fatalf("expected empty compacted stats, got %+v", st)
		}
	})

	t.Run("RepeatedClearAndCompactCycles", func(t *testing.T) {
		idx := e2eMustNew[int](t, 4)
		for cycle := range 3 {
			for i := range 15 {
				e2eMustInsert(t, idx, cycle*100+i, []float32{float32(i), float32(cycle), 0, 0})
			}
			for i := range 5 {
				idx.Delete(cycle*100 + i)
			}
			if err := idx.Compact(); err != nil {
				t.Fatalf("cycle %d Compact failed: %v", cycle, err)
			}
			if idx.Len() != 10 || idx.Stats().Deleted != 0 {
				t.Fatalf("cycle %d unexpected post-compact stats: %+v", cycle, idx.Stats())
			}
			idx.Clear()
			if err := idx.Compact(); err != nil {
				t.Fatalf("cycle %d post-Clear Compact failed: %v", cycle, err)
			}
			st := idx.Stats()
			if st.Len != 0 || st.Deleted != 0 || st.Capacity != 0 || st.EntryPointSet {
				t.Fatalf("cycle %d post-Clear stats invalid: %+v", cycle, st)
			}
		}
	})
}

// Tier 2 — Feature F7: Persistence Error & Edge Boundaries (5 cases)
func TestE2E_Tier2_F7_PersistenceEdges(t *testing.T) {
	makeSerialized := func() []byte {
		idx := e2eMustNew[int](t, 4, hnsw.WithSeed(7))
		for i := range 10 {
			e2eMustInsert(t, idx, i, []float32{float32(i), 1, 2, 3})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		return buf.Bytes()
	}

	t.Run("TruncatedStreams", func(t *testing.T) {
		raw := makeSerialized()
		offsets := []int{0, 1, len(raw) / 4, len(raw) / 2, len(raw) - 1}
		for _, off := range offsets {
			_, err := hnsw.ReadFrom[int](bytes.NewReader(raw[:off]))
			e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "truncated stream")
		}
	})

	t.Run("CorruptedStreamAndTrailer", func(t *testing.T) {
		raw := makeSerialized()

		// Completely invalid / non-HNSW byte payloads.
		_, err := hnsw.ReadFrom[int](bytes.NewReader([]byte("NOT_AN_HNSW_INDEX_PAYLOAD_1234567890")))
		e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "non-HNSW payload")

		_, err = hnsw.ReadFrom[int](bytes.NewReader(make([]byte, len(raw))))
		e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "zeroed payload")

		flipAt := func(idxByte int, mask byte, label string) {
			corrupted := append([]byte(nil), raw...)
			corrupted[idxByte] ^= mask
			_, err := hnsw.ReadFrom[int](bytes.NewReader(corrupted))
			e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, label)
		}
		flipAt(0, 0xFF, "corrupted magic prefix")
		flipAt(len(raw)-1, 0xFF, "corrupted trailing checksum last byte")
		flipAt(len(raw)-4, 0xFF, "corrupted trailing checksum interior byte")
		flipAt(len(raw)/2, 0xFF, "corrupted payload interior midpoint")
		flipAt(len(raw)*3/4, 0xFF, "corrupted payload interior 3/4")
	})

	t.Run("WrongGenericKeyTypeOnLoad", func(t *testing.T) {
		rawInt := makeSerialized()
		_, err := hnsw.ReadFrom[string](bytes.NewReader(rawInt))
		e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "int loaded as string")

		_, err = hnsw.ReadFrom[int64](bytes.NewReader(rawInt))
		e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "int loaded as int64")

		idxStruct := e2eMustNew[e2eCompositeKey](t, 3)
		e2eMustInsert(t, idxStruct, e2eCompositeKey{Tenant: "t1", Bucket: 1, ID: 9}, []float32{1, 2, 3})
		var buf bytes.Buffer
		if _, err := idxStruct.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo struct failed: %v", err)
		}
		_, err = hnsw.ReadFrom[e2eAltCompositeKey](bytes.NewReader(buf.Bytes()))
		e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "structA loaded as structB")
	})

	t.Run("MetricAndCustomDistReloadConflicts", func(t *testing.T) {
		rawBuiltIn := makeSerialized()
		_, err := hnsw.ReadFrom[int](bytes.NewReader(rawBuiltIn), hnsw.WithMetric(hnsw.MetricCosine))
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "built-in metric conflict")

		_, err = hnsw.ReadFrom[int](bytes.NewReader(rawBuiltIn), hnsw.WithCustomDistance(e2eManhattan))
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "built-in loaded with custom dist")

		idxCustom := e2eMustNew[int](t, 3, hnsw.WithCustomDistance(e2eManhattan))
		e2eMustInsert(t, idxCustom, 1, []float32{1, 2, 3})
		var buf bytes.Buffer
		if _, err := idxCustom.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo custom failed: %v", err)
		}
		_, err = hnsw.ReadFrom[int](bytes.NewReader(buf.Bytes()))
		e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "custom loaded without WithCustomDistance")
	})

	t.Run("EmptyIndexAndMissingFileErrors", func(t *testing.T) {
		emptyInt := e2eMustNew[int](t, 4)
		var buf1 bytes.Buffer
		if _, err := emptyInt.WriteTo(&buf1); err != nil {
			t.Fatalf("empty WriteTo failed: %v", err)
		}
		reloadedEmpty, err := hnsw.ReadFrom[int](&buf1)
		if err != nil {
			t.Fatalf("empty ReadFrom failed: %v", err)
		}
		if st := reloadedEmpty.Stats(); st.Len != 0 || st.EntryPointSet {
			t.Fatalf("reloaded empty stats invalid: %+v", st)
		}

		delAllStr := e2eMustNew[string](t, 3)
		e2eMustInsert(t, delAllStr, "a", []float32{1, 2, 3})
		delAllStr.Delete("a")
		var buf2 bytes.Buffer
		if _, err := delAllStr.WriteTo(&buf2); err != nil {
			t.Fatalf("delete-all WriteTo failed: %v", err)
		}
		reloadedDelAll, err := hnsw.ReadFrom[string](&buf2)
		if err != nil {
			t.Fatalf("delete-all ReadFrom failed: %v", err)
		}
		if st := reloadedDelAll.Stats(); st.Len != 0 || st.Deleted != 0 || st.EntryPointSet {
			t.Fatalf("reloaded delete-all stats invalid: %+v", st)
		}

		tmpDir := t.TempDir()
		_, err = hnsw.LoadFile[int](filepath.Join(tmpDir, "missing.hnsw"))
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("LoadFile missing = %v, want os.ErrNotExist", err)
		}

		err = emptyInt.SaveFile(filepath.Join(tmpDir, "no-such-dir", "idx.hnsw"))
		if err == nil {
			t.Fatalf("SaveFile into nonexistent dir succeeded, want error")
		}
	})
}

// Tier 2 — Feature F8: Generic Key Boundary Values (5 cases)
func TestE2E_Tier2_F8_GenericKeyEdges(t *testing.T) {
	t.Run("SignedNumericBoundaryKeys", func(t *testing.T) {
		idx := e2eMustNew[int64](t, 2)
		keys := []int64{math.MinInt64, -1, 0, math.MaxInt64}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i), 0})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[int64](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for i, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing key %d", k)
			}
			got, ok := reloaded.Get(k)
			if !ok {
				t.Fatalf("Get(%d) false", k)
			}
			e2eAssertVecEqual(t, got, []float32{float32(i), 0})
		}
		res, err := reloaded.Search([]float32{0, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != math.MinInt64 {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("UnsignedNumericBoundaryKeys", func(t *testing.T) {
		idx := e2eMustNew[uint64](t, 2)
		keys := []uint64{0, 1, math.MaxUint64}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i * 5), 0})
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
				t.Fatalf("missing uint64 key %d", k)
			}
		}
		res, err := reloaded.Search([]float32{10, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != math.MaxUint64 {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("EmptyAndUnicodeStringKeys", func(t *testing.T) {
		idx := e2eMustNew[string](t, 2)
		longKey := strings.Repeat("x", 1024)
		keys := []string{"", "🚀", "null\x00byte", longKey}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i), 1})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[string](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for i, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing string key[%d]", i)
			}
			got, ok := reloaded.Get(k)
			if !ok {
				t.Fatalf("Get key[%d] false", i)
			}
			e2eAssertVecEqual(t, got, []float32{float32(i), 1})
		}
		res, err := reloaded.Search([]float32{2, 1}, 1)
		if err != nil || len(res) != 1 || res[0].ID != "null\x00byte" {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("FixedBytes16Keys", func(t *testing.T) {
		idx := e2eMustNew[[16]byte](t, 2)
		kZero := [16]byte{}
		var kMax [16]byte
		for i := range kMax {
			kMax[i] = 0xFF
		}
		kUUID := [16]byte{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 1, 2, 3, 4, 5, 6, 7, 8}
		keys := [][16]byte{kZero, kMax, kUUID}
		for i, k := range keys {
			e2eMustInsert(t, idx, k, []float32{float32(i), 2})
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[[16]byte](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		for _, k := range keys {
			if !reloaded.Contains(k) {
				t.Fatalf("missing [16]byte key %x", k)
			}
		}
		res, err := reloaded.Search([]float32{2, 2}, 1)
		if err != nil || len(res) != 1 || res[0].ID != kUUID {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})

	t.Run("ZeroValueAndNestedStructKeys", func(t *testing.T) {
		idx := e2eMustNew[e2eCompositeKey](t, 2)
		k0 := e2eCompositeKey{}
		k1 := e2eCompositeKey{Tenant: "t1", Bucket: 5, ID: 99}
		e2eMustInsert(t, idx, k0, []float32{0, 0})
		e2eMustInsert(t, idx, k1, []float32{5, 5})
		if !idx.Delete(k1) {
			t.Fatalf("Delete(k1) failed")
		}
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
		reloaded, err := hnsw.ReadFrom[e2eCompositeKey](&buf)
		if err != nil {
			t.Fatalf("ReadFrom failed: %v", err)
		}
		if reloaded.Len() != 1 || !reloaded.Contains(k0) || reloaded.Contains(k1) {
			t.Fatalf("unexpected reloaded struct state: Len=%d", reloaded.Len())
		}
		res, err := reloaded.Search([]float32{0, 0}, 1)
		if err != nil || len(res) != 1 || res[0].ID != k0 {
			t.Fatalf("Search failed: res=%+v err=%v", res, err)
		}
	})
}

// Tier 2 — Feature F9: Distance Helper & Sentinel Error Edges (6 cases)
func TestE2E_Tier2_F9_DistanceAndErrorEdges(t *testing.T) {
	type distFnEntry struct {
		name string
		fn   func(a, b []float32) (float32, error)
	}
	helpers := []distFnEntry{
		{"Euclidean", hnsw.Euclidean},
		{"SquaredEuclidean", hnsw.SquaredEuclidean},
		{"Cosine", hnsw.Cosine},
		{"DotProduct", hnsw.DotProduct},
	}

	t.Run("EmptyVectorsErrInvalidDimension", func(t *testing.T) {
		for _, h := range helpers {
			_, err := h.fn(nil, nil)
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidDimension, h.name+"(nil, nil)")
			_, err = h.fn([]float32{}, []float32{})
			e2eAssertErrorIs(t, err, hnsw.ErrInvalidDimension, h.name+"([], [])")
		}
	})

	t.Run("MismatchedLengthsErrDimensionMismatch", func(t *testing.T) {
		for _, h := range helpers {
			_, err := h.fn([]float32{1, 2}, []float32{1, 2, 3})
			e2eAssertErrorIs(t, err, hnsw.ErrDimensionMismatch, h.name+"(2, 3)")
			_, err = h.fn([]float32{}, []float32{1, 2, 3})
			e2eAssertErrorIs(t, err, hnsw.ErrDimensionMismatch, h.name+"(0, 3)")
		}
	})

	t.Run("NaNAndInfErrNaNOrInf", func(t *testing.T) {
		badVals := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
		for _, bad := range badVals {
			// Aligned position (index 0 in 8-float vec) and tail position (index 9 in 10-float vec)
			aAligned := make([]float32, 8)
			bAligned := make([]float32, 8)
			aAligned[0] = bad

			aTail := make([]float32, 10)
			bTail := make([]float32, 10)
			bTail[9] = bad

			for _, h := range helpers {
				_, err := h.fn(aAligned, bAligned)
				e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, h.name+" aligned bad in a")
				_, err = h.fn(aTail, bTail)
				e2eAssertErrorIs(t, err, hnsw.ErrNaNOrInf, h.name+" tail bad in b")
			}
		}
	})

	t.Run("CosineZeroAndExtremeScales", func(t *testing.T) {
		zeros := []float32{0, 0, 0}
		unit := []float32{1, 0, 0}
		d1, err := hnsw.Cosine(zeros, unit)
		if err != nil || d1 != 1.0 {
			t.Fatalf("Cosine(zeros, unit) = (%v, %v), want (1.0, nil)", d1, err)
		}
		d2, err := hnsw.Cosine(zeros, zeros)
		if err != nil || d2 != 1.0 {
			t.Fatalf("Cosine(zeros, zeros) = (%v, %v), want (1.0, nil)", d2, err)
		}

		tiny := []float32{1e-25, 2e-25, 0}
		dTiny, err := hnsw.Cosine(tiny, tiny)
		if err != nil || math.IsNaN(float64(dTiny)) || !e2eApproxEqual(dTiny, 0.0, 1e-4, 1e-4) {
			t.Fatalf("Cosine(tiny, tiny) = (%v, %v), want ~0", dTiny, err)
		}

		huge := []float32{1e20, 2e20, 0}
		dHuge, err := hnsw.Cosine(huge, huge)
		if err != nil || math.IsNaN(float64(dHuge)) || !e2eApproxEqual(dHuge, 0.0, 1e-4, 1e-4) {
			t.Fatalf("Cosine(huge, huge) = (%v, %v), want ~0", dHuge, err)
		}
	})

	t.Run("OddUnalignedDimsParity", func(t *testing.T) {
		dims := []int{1, 3, 7, 8, 15, 127, 130}
		for _, dim := range dims {
			vecs := e2eMakeUniformVecs(int64(dim*13+5), 2, dim+2)
			// Use offset [1 : 1+dim] to force unaligned slice backing pointers.
			a := vecs[0][1 : 1+dim]
			b := vecs[1][1 : 1+dim]

			gotEuc, err := hnsw.Euclidean(a, b)
			if err != nil || !e2eApproxEqual(gotEuc, e2eRefEuclidean(a, b), 1e-4, 1e-4) {
				t.Fatalf("dim=%d Euclidean got=%v err=%v ref=%v", dim, gotEuc, err, e2eRefEuclidean(a, b))
			}
			gotL2, err := hnsw.SquaredEuclidean(a, b)
			if err != nil || !e2eApproxEqual(gotL2, e2eRefL2Squared(a, b), 1e-4, 1e-4) {
				t.Fatalf("dim=%d SquaredEuclidean got=%v err=%v ref=%v", dim, gotL2, err, e2eRefL2Squared(a, b))
			}
			gotCos, err := hnsw.Cosine(a, b)
			if err != nil || !e2eApproxEqual(gotCos, e2eRefCosine(a, b), 1e-4, 1e-4) {
				t.Fatalf("dim=%d Cosine got=%v err=%v ref=%v", dim, gotCos, err, e2eRefCosine(a, b))
			}
			gotDot, err := hnsw.DotProduct(a, b)
			if err != nil || !e2eApproxEqual(gotDot, e2eRefDot(a, b), 1e-4, 1e-4) {
				t.Fatalf("dim=%d DotProduct got=%v err=%v ref=%v", dim, gotDot, err, e2eRefDot(a, b))
			}
		}
	})

	t.Run("SentinelErrorsAreDistinct", func(t *testing.T) {
		sentinels := []error{
			hnsw.ErrInvalidDimension,
			hnsw.ErrDimensionMismatch,
			hnsw.ErrInvalidConfig,
			hnsw.ErrInvalidK,
			hnsw.ErrNaNOrInf,
			hnsw.ErrNotFound,
			hnsw.ErrCorruptedIndex,
		}
		for i, a := range sentinels {
			for j, b := range sentinels {
				if i == j && !errors.Is(a, b) {
					t.Fatalf("sentinel[%d] did not match itself", i)
				}
				if i != j && errors.Is(a, b) {
					t.Fatalf("sentinel[%d] (%v) falsely matched sentinel[%d] (%v)", i, a, j, b)
				}
			}
		}
	})
}
