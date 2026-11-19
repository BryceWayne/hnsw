package hnsw_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/BryceWayne/hnsw"
)

func e2eRunMetricByKeyCase[K comparable](
	t *testing.T,
	metric hnsw.Metric,
	keys []K,
	vecs [][]float32,
	query []float32,
) {
	t.Helper()
	dim := len(query)
	idx := e2eMustNew[K](t, dim,
		hnsw.WithMetric(metric),
		hnsw.WithEfConstruction(64),
		hnsw.WithEfSearch(40),
		hnsw.WithSeed(101),
	)
	items := make([]hnsw.Item[K], len(keys))
	for i, k := range keys {
		items[i] = hnsw.Item[K]{ID: k, Vector: vecs[i]}
		e2eMustInsert(t, idx, k, vecs[i])
	}
	if idx.Len() != len(keys) {
		t.Fatalf("metric=%v Len()=%d, want %d", metric, idx.Len(), len(keys))
	}
	for i, k := range keys {
		if !idx.Contains(k) {
			t.Fatalf("metric=%v missing key %v", metric, k)
		}
		got, ok := idx.Get(k)
		if !ok {
			t.Fatalf("metric=%v Get(%v) false", metric, k)
		}
		e2eAssertVecEqual(t, got, vecs[i])
	}

	res, err := idx.Search(query, 5)
	if err != nil {
		t.Fatalf("metric=%v Search failed: %v", metric, err)
	}
	e2eAssertSortedDistances(t, res)
	oracle := e2eExactKNN(items, query, 5, e2eOracleDistFunc(t, metric), nil)
	if len(res) != 5 || res[0].ID != oracle[0].ID {
		t.Fatalf("metric=%v top-1 mismatch: got=%+v oracle=%+v", metric, res[0], oracle[0])
	}
	if !e2eApproxEqual(res[0].Distance, oracle[0].Distance, 1e-4, 1e-4) {
		t.Fatalf("metric=%v top-1 dist=%v, oracle=%v", metric, res[0].Distance, oracle[0].Distance)
	}
}

// 1. F3 x F8 x F2: Built-in metrics x generic key types x CRUD/Search.
func TestE2E_Tier3_MetricByKeyTypeMatrix(t *testing.T) {
	const count = 40
	const dim = 12
	vecs := e2eMakeUnitVecs(202, count, dim)
	query := e2eMakeUnitVecs(303, 1, dim)[0]

	metrics := []hnsw.Metric{
		hnsw.MetricEuclidean,
		hnsw.MetricSquaredEuclidean,
		hnsw.MetricCosine,
		hnsw.MetricDotProduct,
	}

	intKeys := make([]int, count)
	int64Keys := make([]int64, count)
	uint64Keys := make([]uint64, count)
	strKeys := make([]string, count)
	bytesKeys := make([][16]byte, count)
	structKeys := make([]e2eCompositeKey, count)
	for i := range count {
		intKeys[i] = i - 10
		int64Keys[i] = int64(i)*1_000_000_000 - 500
		uint64Keys[i] = uint64(i) + (1 << 62)
		strKeys[i] = fmt.Sprintf("doc-%03d", i)
		bytesKeys[i][0] = byte(i)
		bytesKeys[i][15] = byte(255 - i)
		structKeys[i] = e2eCompositeKey{Tenant: fmt.Sprintf("t%d", i%3), Bucket: uint32(i % 5), ID: int64(i)}
	}

	for _, m := range metrics {
		t.Run(m.String(), func(t *testing.T) {
			e2eRunMetricByKeyCase(t, m, intKeys, vecs, query)
			e2eRunMetricByKeyCase(t, m, int64Keys, vecs, query)
			e2eRunMetricByKeyCase(t, m, uint64Keys, vecs, query)
			e2eRunMetricByKeyCase(t, m, strKeys, vecs, query)
			e2eRunMetricByKeyCase(t, m, bytesKeys, vecs, query)
			e2eRunMetricByKeyCase(t, m, structKeys, vecs, query)
		})
	}
}

// 2. F4 x F5 x F6: Batch insert (with duplicate upserts) + Delete + DeleteBatch + SearchFilter + Compact.
func TestE2E_Tier3_BatchInsertDeleteCompactSearchFilter(t *testing.T) {
	const total = 180
	const dim = 16
	vecs := e2eMakeUniformVecs(404, total+10, dim)

	idx := e2eMustNew[int](t, dim, hnsw.WithEfConstruction(100), hnsw.WithEfSearch(64), hnsw.WithSeed(404))
	items := make([]hnsw.Item[int], 0, total+10)
	liveMap := make(map[int][]float32, total)
	for i := range total {
		items = append(items, hnsw.Item[int]{ID: i, Vector: vecs[i]})
		liveMap[i] = vecs[i]
	}
	// Duplicate IDs at end (last-write-wins)
	for j := range 10 {
		items = append(items, hnsw.Item[int]{ID: j, Vector: vecs[total+j]})
		liveMap[j] = vecs[total+j]
	}

	if err := idx.InsertBatch(items, hnsw.WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}
	if idx.Len() != total {
		t.Fatalf("Len() = %d, want %d", idx.Len(), total)
	}

	// Single Delete + <=8 DeleteBatch + >8 DeleteBatch
	if !idx.Delete(0) {
		t.Fatalf("Delete(0) failed")
	}
	delete(liveMap, 0)

	if removed := idx.DeleteBatch([]int{1, 2, 3, 4}); removed != 4 {
		t.Fatalf("<=8 DeleteBatch removed = %d, want 4", removed)
	}
	for id := 1; id <= 4; id++ {
		delete(liveMap, id)
	}

	largeDel := make([]int, 15)
	for i := range 15 {
		id := 10 + i
		largeDel[i] = id
		delete(liveMap, id)
	}
	if removed := idx.DeleteBatch(largeDel); removed != 15 {
		t.Fatalf(">8 DeleteBatch removed = %d, want 15", removed)
	}

	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact failed: %v", err)
	}
	if st := idx.Stats(); st.Len != total-20 || st.Deleted != 0 {
		t.Fatalf("unexpected stats after Compact: %+v", st)
	}

	liveItems := make([]hnsw.Item[int], 0, len(liveMap))
	for id, v := range liveMap {
		liveItems = append(liveItems, hnsw.Item[int]{ID: id, Vector: v})
	}
	filterFn := func(id int) bool { return id%2 == 0 }
	query := vecs[50]

	got, err := idx.SearchFilter(query, 10, filterFn)
	if err != nil {
		t.Fatalf("SearchFilter failed: %v", err)
	}
	e2eAssertSortedDistances(t, got)
	for _, r := range got {
		if !filterFn(r.ID) {
			t.Fatalf("filter predicate violated by ID %d", r.ID)
		}
		if _, exists := liveMap[r.ID]; !exists {
			t.Fatalf("deleted ID %d returned", r.ID)
		}
	}
	oracle := e2eExactKNN(liveItems, query, 10, e2eOracleDistFunc(t, hnsw.MetricEuclidean), filterFn)
	recall := e2eRecallAtK(got, oracle)
	if recall < 0.9 {
		t.Fatalf("filtered recall after DeleteBatch+Compact = %.2f, want >= 0.90", recall)
	}
}

// 3. F2 x F3 x F6 x F7 x F8: Persistence across metrics & keys with tombstones + post-reload mutations.
func TestE2E_Tier3_PersistenceMetricsKeysAndPostReloadMutations(t *testing.T) {
	const count = 80
	const dim = 16
	tmpDir := t.TempDir()

	verifyRoundTrip := func(
		metric hnsw.Metric,
		useFile bool,
	) {
		vecs := e2eMakeUnitVecs(505+int64(metric), count, dim)
		idx := e2eMustNew[e2eCompositeKey](t, dim,
			hnsw.WithMetric(metric),
			hnsw.WithM(12),
			hnsw.WithEfConstruction(80),
			hnsw.WithEfSearch(20),
			hnsw.WithSeed(505),
		)
		keys := make([]e2eCompositeKey, count)
		for i := range count {
			keys[i] = e2eCompositeKey{Tenant: "corp", Bucket: uint32(i % 4), ID: int64(i)}
			e2eMustInsert(t, idx, keys[i], vecs[i])
		}
		// Upsert 5 and delete 10 before persisting
		for i := range 5 {
			e2eMustInsert(t, idx, keys[i], vecs[(i+20)%count])
		}
		for i := 10; i < 20; i++ {
			idx.Delete(keys[i])
		}

		var reloaded *hnsw.Index[e2eCompositeKey]
		if useFile {
			path := filepath.Join(tmpDir, fmt.Sprintf("m_%d.hnsw", metric))
			if err := idx.SaveFile(path); err != nil {
				t.Fatalf("SaveFile failed: %v", err)
			}
			var err error
			reloaded, err = hnsw.LoadFile[e2eCompositeKey](path, hnsw.WithEfSearch(64))
			if err != nil {
				t.Fatalf("LoadFile failed: %v", err)
			}
		} else {
			var buf bytes.Buffer
			if _, err := idx.WriteTo(&buf); err != nil {
				t.Fatalf("WriteTo failed: %v", err)
			}
			var err error
			reloaded, err = hnsw.ReadFrom[e2eCompositeKey](&buf, hnsw.WithEfSearch(64))
			if err != nil {
				t.Fatalf("ReadFrom failed: %v", err)
			}
		}

		st := reloaded.Stats()
		if st.Len != count-10 || st.Deleted != 0 || st.EfSearch != 64 || st.Metric != metric || st.M != 12 {
			t.Fatalf("metric=%v reloaded stats mismatch: %+v", metric, st)
		}
		// Verify lossless Get for key[0] (upserted) and absence of key[10] (deleted)
		got0, ok := reloaded.Get(keys[0])
		if !ok {
			t.Fatalf("missing upserted key[0]")
		}
		e2eAssertVecEqual(t, got0, vecs[20])
		if reloaded.Contains(keys[10]) {
			t.Fatalf("deleted key[10] survived reload")
		}

		// Post-reload mutations: insert new, upsert existing, delete, compact, search
		extraKey := e2eCompositeKey{Tenant: "corp", Bucket: 9, ID: 9999}
		extraVec := vecs[30]
		e2eMustInsert(t, reloaded, extraKey, extraVec)
		if !reloaded.Delete(keys[25]) {
			t.Fatalf("post-reload Delete failed")
		}
		if err := reloaded.Compact(); err != nil {
			t.Fatalf("post-reload Compact failed: %v", err)
		}
		if reloaded.Len() != count-10 || reloaded.Stats().Deleted != 0 {
			t.Fatalf("post-reload mutate stats mismatch: %+v", reloaded.Stats())
		}
		res, err := reloaded.Search(extraVec, 3)
		if err != nil || len(res) == 0 || !reloaded.Contains(extraKey) {
			t.Fatalf("post-reload Search failed: res=%+v err=%v", res, err)
		}
	}

	verifyRoundTrip(hnsw.MetricEuclidean, false)
	verifyRoundTrip(hnsw.MetricSquaredEuclidean, true)
	verifyRoundTrip(hnsw.MetricCosine, false)
	verifyRoundTrip(hnsw.MetricDotProduct, true)
}

// 4. F2 x F4 x F5 x F6 x F7: Concurrent Read/Write/Compact/WriteTo under -race.
func TestE2E_Tier3_ConcurrentReadWriteCompactSnapshotRace(t *testing.T) {
	const dim = 16
	idx := e2eMustNew[int](t, dim,
		hnsw.WithMetric(hnsw.MetricCosine),
		hnsw.WithM(12),
		hnsw.WithEfConstruction(64),
		hnsw.WithEfSearch(32),
		hnsw.WithSeed(606),
	)
	seedVecs := e2eMakeUnitVecs(606, 64, dim)
	for i, v := range seedVecs {
		e2eMustInsert(t, idx, i, v)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 32)

	// 4 writer goroutines (inserts, upserts, deletes)
	for w := range 4 {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			wVecs := e2eMakeUnitVecs(int64(700+workerID), 30, dim)
			for step := range 30 {
				id := 100 + workerID*100 + step
				if err := idx.Insert(id, wVecs[step]); err != nil {
					errCh <- err
					return
				}
				if step%5 == 0 {
					// Upsert an existing key in this worker's range
					if err := idx.Insert(id, wVecs[(step+1)%30]); err != nil {
						errCh <- err
						return
					}
				}
				if step%7 == 0 {
					idx.Delete(id)
				}
			}
		}(w)
	}

	// 4 reader goroutines (Search, SearchFilter, SearchBatch, Get, Contains, Len, Stats)
	for r := range 4 {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			qVecs := e2eMakeUnitVecs(int64(800+readerID), 20, dim)
			for step := range 20 {
				q := qVecs[step]
				res, err := idx.Search(q, 5)
				if err != nil {
					errCh <- err
					return
				}
				for i := 1; i < len(res); i++ {
					if res[i].Distance < res[i-1].Distance {
						errCh <- fmt.Errorf("unsorted Search results")
						return
					}
				}
				if _, err := idx.SearchFilter(q, 3, func(id int) bool { return id%2 == 0 }); err != nil {
					errCh <- err
					return
				}
				if _, err := idx.SearchBatch([][]float32{q}, 3, hnsw.WithWorkers(2)); err != nil {
					errCh <- err
					return
				}
				_, _ = idx.Get(step)
				_ = idx.Contains(step)
				_ = idx.Len()
				_ = idx.Stats()
			}
		}(r)
	}

	// 1 snapshotter/compactor goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for step := range 4 {
			var snap bytes.Buffer
			if _, err := idx.WriteTo(&snap); err != nil {
				errCh <- err
				return
			}
			if _, err := hnsw.ReadFrom[int](&snap); err != nil {
				errCh <- err
				return
			}
			if step%2 == 0 {
				if err := idx.Compact(); err != nil {
					errCh <- err
					return
				}
			}
		}
	}()

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("concurrent operation error: %v", err)
	}

	var finalBuf bytes.Buffer
	if _, err := idx.WriteTo(&finalBuf); err != nil {
		t.Fatalf("final WriteTo failed: %v", err)
	}
	reloaded, err := hnsw.ReadFrom[int](&finalBuf)
	if err != nil {
		t.Fatalf("final ReadFrom failed: %v", err)
	}
	if reloaded.Len() != idx.Len() || reloaded.Stats().Deleted != 0 {
		t.Fatalf("final snapshot mismatch: reloaded=%+v orig=%+v", reloaded.Stats(), idx.Stats())
	}
}

// 5. F1 x F3 x F4 x F7: Custom distance + slice-cap guard + persistence + SearchFilter.
func TestE2E_Tier3_CustomDistancePersistenceAndFilter(t *testing.T) {
	const count = 60
	const dim = 8
	probingL1 := func(a, b []float32) float32 {
		_ = append(a, 12345)
		_ = append(b, 67890)
		return e2eManhattan(a, b)
	}

	idx := e2eMustNew[int](t, dim,
		hnsw.WithCustomDistance(probingL1),
		hnsw.WithEfConstruction(80),
		hnsw.WithEfSearch(40),
		hnsw.WithSeed(707),
	)
	vecs := e2eMakeUniformVecs(707, count, dim)
	items := make([]hnsw.Item[int], count)
	for i, v := range vecs {
		items[i] = hnsw.Item[int]{ID: i, Vector: v}
		e2eMustInsert(t, idx, i, v)
	}

	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}
	raw := buf.Bytes()

	// Negative cross-feature persistence contracts
	_, err := hnsw.ReadFrom[int](bytes.NewReader(raw))
	e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "custom reload without WithCustomDistance")

	_, err = hnsw.ReadFrom[int](bytes.NewReader(raw), hnsw.WithMetric(hnsw.MetricEuclidean))
	e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "custom reload with WithMetric")

	_, err = hnsw.ReadFrom[string](bytes.NewReader(raw), hnsw.WithCustomDistance(probingL1))
	e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "custom reload with wrong key type")

	// Valid reload
	reloaded, err := hnsw.ReadFrom[int](bytes.NewReader(raw), hnsw.WithCustomDistance(probingL1))
	if err != nil {
		t.Fatalf("valid custom ReadFrom failed: %v", err)
	}

	query := []float32{0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4}
	filterFn := func(id int) bool { return id%4 == 0 }
	got, err := reloaded.SearchFilter(query, 5, filterFn)
	if err != nil {
		t.Fatalf("reloaded SearchFilter failed: %v", err)
	}
	e2eAssertSortedDistances(t, got)
	oracle := e2eExactKNN(items, query, 5, e2eManhattan, filterFn)
	if recall := e2eRecallAtK(got, oracle); recall < 0.8 {
		t.Fatalf("custom distance filtered recall = %.2f, want >= 0.80", recall)
	}
	for i, v := range vecs {
		stored, ok := reloaded.Get(i)
		if !ok {
			t.Fatalf("missing key %d", i)
		}
		e2eAssertVecEqual(t, stored, v)
	}
}

// 6. F4 x F5: SearchBatch x WithWorkers x WithSearchEf x WithContext.
func TestE2E_Tier3_SearchBatchWorkersContextAndEf(t *testing.T) {
	const count = 250
	const queryCount = 25
	const dim = 16
	vecs, queries := e2eMakeClusteredDataset(808, count, queryCount, dim, 10, false)

	idx := e2eMustNew[int](t, dim,
		hnsw.WithM(16),
		hnsw.WithEfConstruction(96),
		hnsw.WithEfSearch(16),
		hnsw.WithSeed(808),
	)
	items := make([]hnsw.Item[int], count)
	for i, v := range vecs {
		items[i] = hnsw.Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items, hnsw.WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	resW1, err := idx.SearchBatch(queries, 10, hnsw.WithWorkers(1), hnsw.WithSearchEf(8))
	if err != nil {
		t.Fatalf("SearchBatch w=1 ef=8 failed: %v", err)
	}
	resW4LowEf, err := idx.SearchBatch(queries, 10, hnsw.WithWorkers(4), hnsw.WithSearchEf(8))
	if err != nil {
		t.Fatalf("SearchBatch w=4 ef=8 failed: %v", err)
	}
	resW4HighEf, err := idx.SearchBatch(queries, 10, hnsw.WithWorkers(4), hnsw.WithSearchEf(64))
	if err != nil {
		t.Fatalf("SearchBatch w=4 ef=64 failed: %v", err)
	}

	distFn := e2eOracleDistFunc(t, hnsw.MetricEuclidean)
	var sumLow, sumHigh float64
	for qi, q := range queries {
		e2eAssertSortedDistances(t, resW1[qi])
		e2eAssertSortedDistances(t, resW4LowEf[qi])
		e2eAssertSortedDistances(t, resW4HighEf[qi])
		if len(resW1[qi]) != len(resW4LowEf[qi]) {
			t.Fatalf("worker count altered result count at q=%d", qi)
		}
		for j := range resW1[qi] {
			if resW1[qi][j].ID != resW4LowEf[qi][j].ID {
				t.Fatalf("query[%d][%d] worker order mismatch: w1=%d w4=%d", qi, j, resW1[qi][j].ID, resW4LowEf[qi][j].ID)
			}
		}
		oracle := e2eExactKNN(items, q, 10, distFn, nil)
		sumLow += e2eRecallAtK(resW4LowEf[qi], oracle)
		sumHigh += e2eRecallAtK(resW4HighEf[qi], oracle)
	}
	avgLow := sumLow / float64(queryCount)
	avgHigh := sumHigh / float64(queryCount)
	if avgHigh < 0.95 || avgHigh < avgLow {
		t.Fatalf("expected high-ef recall >= 0.95 and >= low-ef: low=%.4f high=%.4f", avgLow, avgHigh)
	}

	timeoutCtx, cancelTimeout := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelTimeout()
	resCtx, err := idx.SearchBatch(queries, 5, hnsw.WithContext(timeoutCtx), hnsw.WithWorkers(2))
	if err != nil || len(resCtx) != queryCount {
		t.Fatalf("valid timeout SearchBatch failed: len=%d err=%v", len(resCtx), err)
	}

	canceledCtx, cancelNow := context.WithCancel(context.Background())
	cancelNow()
	_, err = idx.SearchBatch(queries, 5, hnsw.WithContext(canceledCtx))
	e2eAssertErrorIs(t, err, context.Canceled, "SearchBatch canceled context")

	_, err = idx.SearchBatch(queries, 5, hnsw.WithWorkers(0))
	e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "SearchBatch WithWorkers(0)")

	_, err = idx.SearchBatch(queries, 5, hnsw.WithSearchEf(0))
	e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "SearchBatch WithSearchEf(0)")
}

// 7. F1 x F3 x F4: Heuristic toggle x odd/unaligned dimensions x Recall@10.
func TestE2E_Tier3_HeuristicToggleOddDimsAndRecall(t *testing.T) {
	dims := []int{17, 33, 65}
	heuristics := []bool{true, false}
	const count = 300
	const queryCount = 20

	for _, dim := range dims {
		vecs, queries := e2eMakeClusteredDataset(int64(900+dim), count, queryCount, dim, 32, false)
		items := make([]hnsw.Item[int], count)
		for i, v := range vecs {
			items[i] = hnsw.Item[int]{ID: i, Vector: v}
		}
		distFn := e2eOracleDistFunc(t, hnsw.MetricEuclidean)
		for _, heur := range heuristics {
			t.Run(fmt.Sprintf("dim=%d_heur=%v", dim, heur), func(t *testing.T) {
				idx := e2eMustNew[int](t, dim,
					hnsw.WithM(16),
					hnsw.WithEfConstruction(80),
					hnsw.WithEfSearch(48),
					hnsw.WithHeuristic(heur),
					hnsw.WithSeed(909),
				)
				if err := idx.InsertBatch(items, hnsw.WithWorkers(1)); err != nil {
					t.Fatalf("InsertBatch failed: %v", err)
				}
				var recallSum float64
				for _, q := range queries {
					got, err := idx.Search(q, 10)
					if err != nil {
						t.Fatalf("Search failed: %v", err)
					}
					e2eAssertSortedDistances(t, got)
					oracle := e2eExactKNN(items, q, 10, distFn, nil)
					recallSum += e2eRecallAtK(got, oracle)
				}
				avg := recallSum / float64(queryCount)
				if avg < 0.95 {
					t.Fatalf("dim=%d heur=%v Recall@10 = %.4f < 0.95", dim, heur, avg)
				}
			})
		}
	}
}

// 8. F2 x F6: Clear + Reinsert overlapping keys + DeleteBatch + Compact + Stats.
func TestE2E_Tier3_ClearReinsertStatsAndCompact(t *testing.T) {
	const count = 100
	const dim = 8
	idx := e2eMustNew[int](t, dim, hnsw.WithM(8), hnsw.WithMetric(hnsw.MetricSquaredEuclidean), hnsw.WithSeed(1010))
	vecs1 := e2eMakeUniformVecs(1010, count, dim)
	items1 := make([]hnsw.Item[int], count)
	for i, v := range vecs1 {
		items1[i] = hnsw.Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items1); err != nil {
		t.Fatalf("first InsertBatch failed: %v", err)
	}
	delIDs := make([]int, 20)
	for i := range 20 {
		delIDs[i] = i
	}
	if removed := idx.DeleteBatch(delIDs); removed != 20 {
		t.Fatalf("DeleteBatch removed = %d, want 20", removed)
	}
	st1 := idx.Stats()
	if st1.Len != 80 || st1.Deleted != 20 || !st1.EntryPointSet || st1.MemoryBytes <= 0 {
		t.Fatalf("unexpected pre-Clear stats: %+v", st1)
	}

	idx.Clear()
	st2 := idx.Stats()
	if st2.Len != 0 || st2.Deleted != 0 || st2.Capacity != 0 || st2.EntryPointSet {
		t.Fatalf("unexpected post-Clear stats: %+v", st2)
	}
	emptyRes, err := idx.Search(vecs1[0], 5)
	if err != nil || len(emptyRes) != 0 {
		t.Fatalf("post-Clear Search = (%+v, %v), want empty", emptyRes, err)
	}

	// Reinsert distinct shifted vectors on overlapping keys 0..49
	vecs2 := e2eMakeUniformVecs(2020, 50, dim)
	items2 := make([]hnsw.Item[int], 50)
	for i, v := range vecs2 {
		shifted := make([]float32, dim)
		for j := range v {
			shifted[j] = v[j] + 100.0
		}
		items2[i] = hnsw.Item[int]{ID: i, Vector: shifted}
	}
	if err := idx.InsertBatch(items2); err != nil {
		t.Fatalf("second InsertBatch failed: %v", err)
	}
	if !idx.Delete(0) {
		t.Fatalf("post-Clear Delete(0) failed")
	}
	if err := idx.Compact(); err != nil {
		t.Fatalf("post-Clear Compact failed: %v", err)
	}
	if st3 := idx.Stats(); st3.Len != 49 || st3.Deleted != 0 || st3.Metric != hnsw.MetricSquaredEuclidean || st3.M != 8 {
		t.Fatalf("post-reinsert Compact stats mismatch: %+v", st3)
	}
	if idx.Contains(99) || idx.Contains(0) {
		t.Fatalf("stale/deleted keys present after Clear+Compact")
	}
	got1, ok := idx.Get(1)
	if !ok {
		t.Fatalf("Get(1) false after Clear+Reinsert")
	}
	e2eAssertVecEqual(t, got1, items2[1].Vector)
}

// 9. F3 x F4 x F9: Package distance helpers vs Index.Search distances (including unaligned subslices & zero Cosine & negative Dot).
func TestE2E_Tier3_PackageDistanceFuncsVsIndexSearch(t *testing.T) {
	const dim = 19
	rawA := e2eMakeUniformVecs(1111, 1, dim+3)[0]
	rawB := e2eMakeUniformVecs(2222, 1, dim+3)[0]
	rawQ := e2eMakeUniformVecs(3333, 1, dim+3)[0]
	// Unaligned subslice views
	a := rawA[1 : 1+dim]
	b := rawB[2 : 2+dim]
	q := rawQ[3 : 3+dim]
	zeros := make([]float32, dim)

	metrics := []hnsw.Metric{
		hnsw.MetricEuclidean,
		hnsw.MetricSquaredEuclidean,
		hnsw.MetricCosine,
		hnsw.MetricDotProduct,
	}
	for _, m := range metrics {
		idx := e2eMustNew[int](t, dim, hnsw.WithMetric(m), hnsw.WithEfSearch(32))
		e2eMustInsert(t, idx, 1, a)
		e2eMustInsert(t, idx, 2, b)
		e2eMustInsert(t, idx, 3, zeros)

		res, err := idx.Search(q, 3)
		if err != nil {
			t.Fatalf("metric=%v Search failed: %v", m, err)
		}
		oracleFn := e2eOracleDistFunc(t, m)
		expectedByID := map[int]float32{
			1: oracleFn(a, q),
			2: oracleFn(b, q),
			3: oracleFn(zeros, q),
		}
		for _, r := range res {
			want := expectedByID[r.ID]
			if !e2eApproxEqual(r.Distance, want, 1e-4, 1e-4) {
				t.Fatalf("metric=%v ID=%d index dist=%v != helper oracle=%v", m, r.ID, r.Distance, want)
			}
			if m == hnsw.MetricCosine && r.ID == 3 && r.Distance != 1.0 {
				t.Fatalf("MetricCosine zero vector dist=%v, want 1.0", r.Distance)
			}
			if m == hnsw.MetricDotProduct && r.ID == 3 && (r.Distance != 0 || math.Signbit(float64(r.Distance))) {
				t.Fatalf("MetricDotProduct zero vector dist=%v, want +0.0", r.Distance)
			}
		}
	}
}

// 10. F7 x F8 x F9: Negative persistence cross-feature matrix.
func TestE2E_Tier3_PersistenceNegativeOptionsAndCorruption(t *testing.T) {
	idxInt := e2eMustNew[int](t, 4, hnsw.WithMetric(hnsw.MetricEuclidean))
	e2eMustInsert(t, idxInt, 1, []float32{1, 2, 3, 4})
	var intBuf bytes.Buffer
	if _, err := idxInt.WriteTo(&intBuf); err != nil {
		t.Fatalf("WriteTo int failed: %v", err)
	}
	intBytes := intBuf.Bytes()

	// Conflicting built-in metric or custom distance -> ErrInvalidConfig
	_, err := hnsw.ReadFrom[int](bytes.NewReader(intBytes), hnsw.WithMetric(hnsw.MetricDotProduct))
	e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "conflicting WithMetric")

	_, err = hnsw.ReadFrom[int](bytes.NewReader(intBytes), hnsw.WithCustomDistance(e2eManhattan))
	e2eAssertErrorIs(t, err, hnsw.ErrInvalidConfig, "built-in stream with WithCustomDistance")

	// Wrong generic key type -> ErrCorruptedIndex
	_, err = hnsw.ReadFrom[int64](bytes.NewReader(intBytes))
	e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "int as int64")

	idxStruct := e2eMustNew[e2eCompositeKey](t, 4)
	e2eMustInsert(t, idxStruct, e2eCompositeKey{Tenant: "a", Bucket: 1, ID: 2}, []float32{1, 2, 3, 4})
	var structBuf bytes.Buffer
	if _, err := idxStruct.WriteTo(&structBuf); err != nil {
		t.Fatalf("WriteTo struct failed: %v", err)
	}
	_, err = hnsw.ReadFrom[e2eAltCompositeKey](bytes.NewReader(structBuf.Bytes()))
	e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "structA as structB")

	// Truncation & bitflip -> ErrCorruptedIndex
	_, err = hnsw.ReadFrom[int](bytes.NewReader(intBytes[:len(intBytes)/2]))
	e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "truncated stream")

	flipped := append([]byte(nil), intBytes...)
	flipped[len(flipped)-2] ^= 0x55
	_, err = hnsw.ReadFrom[int](bytes.NewReader(flipped))
	e2eAssertErrorIs(t, err, hnsw.ErrCorruptedIndex, "CRC bitflip")

	// Missing file -> os.ErrNotExist
	_, err = hnsw.LoadFile[int](filepath.Join(t.TempDir(), "does-not-exist.hnsw"))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("LoadFile missing = %v, want os.ErrNotExist", err)
	}
}
