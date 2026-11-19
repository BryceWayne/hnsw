package hnsw_test

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/BryceWayne/hnsw"
)

// 1. Semantic Document Retrieval (Cosine + string keys + SearchFilter + SaveFile/LoadFile)
func TestE2E_Tier4_SemanticDocumentRetrieval(t *testing.T) {
	const count = 600
	const queryCount = 30
	const dim = 48
	const k = 10

	unitVecs, queries := e2eMakeClusteredDataset(4001, count, queryCount, dim, 16, true)
	rng := rand.New(rand.NewSource(4002))

	idx := e2eMustNew[string](t, dim,
		hnsw.WithMetric(hnsw.MetricCosine),
		hnsw.WithM(16),
		hnsw.WithMmax(32),
		hnsw.WithEfConstruction(100),
		hnsw.WithEfSearch(64),
		hnsw.WithSeed(4001),
	)

	items := make([]hnsw.Item[string], count)
	docCategory := make(map[string]int, count)
	docVecs := make(map[string][]float32, count)
	for i := range count {
		scale := float32(0.5 + rng.Float64()*49.5)
		scaled := make([]float32, dim)
		for j := range dim {
			scaled[j] = unitVecs[i][j] * scale
		}
		key := fmt.Sprintf("doc-%04d", i)
		items[i] = hnsw.Item[string]{ID: key, Vector: scaled}
		docCategory[key] = i % 4 // 25% selectivity per category
		docVecs[key] = scaled
	}

	if err := idx.InsertBatch(items, hnsw.WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	oracleDist := e2eOracleDistFunc(t, hnsw.MetricCosine)
	checkRecall := func(targetIdx *hnsw.Index[string], live []hnsw.Item[string], filter func(string) bool, label string) {
		t.Helper()
		var recallSum float64
		for _, q := range queries {
			var got []hnsw.SearchResult[string]
			var err error
			if filter == nil {
				got, err = targetIdx.Search(q, k)
			} else {
				got, err = targetIdx.SearchFilter(q, k, filter)
			}
			if err != nil {
				t.Fatalf("%s search failed: %v", label, err)
			}
			e2eAssertSortedDistances(t, got)
			if filter != nil {
				for _, r := range got {
					if !filter(r.ID) {
						t.Fatalf("%s filter violated for %q", label, r.ID)
					}
				}
			}
			oracle := e2eExactKNN(live, q, k, oracleDist, filter)
			recallSum += e2eRecallAtK(got, oracle)
		}
		avg := recallSum / float64(len(queries))
		if avg < 0.95 {
			t.Fatalf("%s Recall@%d = %.4f < 0.95", label, k, avg)
		}
	}

	catZeroFilter := func(id string) bool { return docCategory[id] == 0 }
	checkRecall(idx, items, nil, "baseline")
	checkRecall(idx, items, catZeroFilter, "filtered")

	path := filepath.Join(t.TempDir(), "docs.hnsw")
	if err := idx.SaveFile(path); err != nil {
		t.Fatalf("SaveFile failed: %v", err)
	}
	reloaded, err := hnsw.LoadFile[string](path)
	if err != nil {
		t.Fatalf("LoadFile failed: %v", err)
	}
	for i := 0; i < count; i += 25 {
		key := fmt.Sprintf("doc-%04d", i)
		got, ok := reloaded.Get(key)
		if !ok {
			t.Fatalf("reloaded missing key %q", key)
		}
		e2eAssertVecEqual(t, got, docVecs[key])
	}
	checkRecall(reloaded, items, nil, "post-reload")

	// Upsert 20 docs and delete 40 docs
	newUnit, _ := e2eMakeClusteredDataset(4003, 20, 0, dim, 8, true)
	for i := range 20 {
		key := fmt.Sprintf("doc-%04d", i)
		e2eMustInsert(t, reloaded, key, newUnit[i])
		docVecs[key] = newUnit[i]
	}
	delKeys := make([]string, 40)
	for i := range 40 {
		key := fmt.Sprintf("doc-%04d", 100+i)
		delKeys[i] = key
		delete(docVecs, key)
	}
	if removed := reloaded.DeleteBatch(delKeys); removed != 40 {
		t.Fatalf("DeleteBatch removed = %d, want 40", removed)
	}
	surviving := make([]hnsw.Item[string], 0, len(docVecs))
	for id, v := range docVecs {
		surviving = append(surviving, hnsw.Item[string]{ID: id, Vector: v})
	}
	checkRecall(reloaded, surviving, catZeroFilter, "post-upsert-delete-filtered")
}

// 2. High-Dimensional Image/Patch Similarity + Live Churn + Compact
func TestE2E_Tier4_ImagePatchSimilarityLifecycle(t *testing.T) {
	const count = 800
	const queryCount = 30
	const dim = 32
	const k = 10

	vecs, queries := e2eMakeClusteredDataset(5001, count+80, queryCount, dim, 16, false)
	idx := e2eMustNew[int64](t, dim,
		hnsw.WithMetric(hnsw.MetricEuclidean),
		hnsw.WithM(16),
		hnsw.WithMmax(32),
		hnsw.WithEfConstruction(100),
		hnsw.WithEfSearch(64),
		hnsw.WithSeed(5001),
	)

	liveMap := make(map[int64][]float32, count)
	items := make([]hnsw.Item[int64], count)
	for i := range count {
		id := int64(i) + 10_000
		items[i] = hnsw.Item[int64]{ID: id, Vector: vecs[i]}
		liveMap[id] = vecs[i]
	}
	if err := idx.InsertBatch(items, hnsw.WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	oracleDist := e2eOracleDistFunc(t, hnsw.MetricEuclidean)
	assertRecall := func(live []hnsw.Item[int64], label string) {
		t.Helper()
		var sum float64
		for _, q := range queries {
			got, err := idx.Search(q, k)
			if err != nil {
				t.Fatalf("%s Search failed: %v", label, err)
			}
			e2eAssertSortedDistances(t, got)
			oracle := e2eExactKNN(live, q, k, oracleDist, nil)
			sum += e2eRecallAtK(got, oracle)
		}
		avg := sum / float64(len(queries))
		if avg < 0.95 {
			t.Fatalf("%s Recall@%d = %.4f < 0.95", label, k, avg)
		}
	}
	assertRecall(items, "baseline")

	// Upsert 80 patches (IDs 10000..10079) and DeleteBatch 120 patches (IDs 10080..10199)
	for i := range 80 {
		id := int64(i) + 10_000
		newVec := vecs[count+i]
		e2eMustInsert(t, idx, id, newVec)
		liveMap[id] = newVec
	}
	delIDs := make([]int64, 120)
	for i := range 120 {
		id := int64(80+i) + 10_000
		delIDs[i] = id
		delete(liveMap, id)
	}
	if removed := idx.DeleteBatch(delIDs); removed != 120 {
		t.Fatalf("DeleteBatch removed = %d, want 120", removed)
	}

	stBefore := idx.Stats()
	if stBefore.Len != 680 || stBefore.Deleted < 120 {
		t.Fatalf("unexpected stats before Compact: %+v", stBefore)
	}

	liveItems := make([]hnsw.Item[int64], 0, len(liveMap))
	for id, v := range liveMap {
		liveItems = append(liveItems, hnsw.Item[int64]{ID: id, Vector: v})
	}
	assertRecall(liveItems, "post-churn-pre-compact")

	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact failed: %v", err)
	}
	stAfter := idx.Stats()
	if stAfter.Len != 680 || stAfter.Deleted != 0 || stAfter.Capacity > stBefore.Capacity {
		t.Fatalf("unexpected stats after Compact: before=%+v after=%+v", stBefore, stAfter)
	}
	assertRecall(liveItems, "post-compact")
}

// 3. Recommendation Batch Inference (DotProduct + InsertBatch + SearchBatch + EfSearch + Context)
func TestE2E_Tier4_RecommendationBatchInference(t *testing.T) {
	const count = 600
	const queryCount = 30
	const dim = 24
	const k = 10

	vecs, rawQueries := e2eMakeClusteredDataset(6001, count, queryCount, dim, 12, true)
	queries := make([][]float32, queryCount)
	for i, q := range rawQueries {
		scaled := make([]float32, dim)
		scale := float32(1.0 + float64(i%5)*0.5)
		for j := range dim {
			scaled[j] = q[j] * scale
		}
		queries[i] = scaled
	}

	idx := e2eMustNew[uint64](t, dim,
		hnsw.WithMetric(hnsw.MetricDotProduct),
		hnsw.WithM(16),
		hnsw.WithMmax(32),
		hnsw.WithEfConstruction(100),
		hnsw.WithEfSearch(16),
		hnsw.WithSeed(6001),
	)

	items := make([]hnsw.Item[uint64], count)
	for i := range count {
		items[i] = hnsw.Item[uint64]{ID: uint64(i + 1), Vector: vecs[i]}
	}
	if err := idx.InsertBatch(items, hnsw.WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}

	resLow, err := idx.SearchBatch(queries, k, hnsw.WithWorkers(4))
	if err != nil {
		t.Fatalf("SearchBatch low ef failed: %v", err)
	}
	resHigh, err := idx.SearchBatch(queries, k, hnsw.WithWorkers(4), hnsw.WithSearchEf(64))
	if err != nil {
		t.Fatalf("SearchBatch high ef failed: %v", err)
	}

	oracleDist := e2eOracleDistFunc(t, hnsw.MetricDotProduct)
	var sumLow, sumHigh float64
	for qi, q := range queries {
		e2eAssertSortedDistances(t, resLow[qi])
		e2eAssertSortedDistances(t, resHigh[qi])
		for _, r := range resHigh[qi] {
			idxVec, ok := idx.Get(r.ID)
			if !ok {
				t.Fatalf("Get(%d) false", r.ID)
			}
			rawDot, err := hnsw.DotProduct(idxVec, q)
			if err != nil {
				t.Fatalf("DotProduct failed: %v", err)
			}
			wantDist := -rawDot
			if wantDist == 0 {
				wantDist = 0
			}
			if !e2eApproxEqual(r.Distance, wantDist, 1e-4, 1e-4) {
				t.Fatalf("q=%d ID=%d dist=%v != -DotProduct=%v", qi, r.ID, r.Distance, wantDist)
			}
		}
		oracle := e2eExactKNN(items, q, k, oracleDist, nil)
		sumLow += e2eRecallAtK(resLow[qi], oracle)
		sumHigh += e2eRecallAtK(resHigh[qi], oracle)
	}

	avgLow := sumLow / float64(queryCount)
	avgHigh := sumHigh / float64(queryCount)
	if avgHigh < 0.95 || avgHigh < avgLow {
		t.Fatalf("expected DotProduct Recall@10(ef=64) >= 0.95 and >= ef=16: low=%.4f high=%.4f", avgLow, avgHigh)
	}

	ctxOK, cancelOK := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelOK()
	if _, err := idx.SearchBatch(queries, k, hnsw.WithContext(ctxOK)); err != nil {
		t.Fatalf("SearchBatch valid timeout failed: %v", err)
	}

	ctxCancel, cancelNow := context.WithCancel(context.Background())
	cancelNow()
	_, err = idx.SearchBatch(queries, k, hnsw.WithContext(ctxCancel))
	e2eAssertErrorIs(t, err, context.Canceled, "SearchBatch canceled")
}

// 4. Multi-Tenant Catalog CRUD & Snapshot Recovery
func TestE2E_Tier4_MultiTenantCatalogCRUDAndRecovery(t *testing.T) {
	const tenants = 4
	const perTenant = 150
	const total = tenants * perTenant
	const queryCount = 24
	const dim = 24
	const k = 10

	vecs, queries := e2eMakeClusteredDataset(7001, total, queryCount, dim, 16, false)
	idx := e2eMustNew[e2eCompositeKey](t, dim,
		hnsw.WithMetric(hnsw.MetricSquaredEuclidean),
		hnsw.WithM(16),
		hnsw.WithMmax(32),
		hnsw.WithEfConstruction(96),
		hnsw.WithEfSearch(64),
		hnsw.WithSeed(7001),
	)

	allItems := make([]hnsw.Item[e2eCompositeKey], total)
	tenant0Keys := make([]e2eCompositeKey, 0, perTenant)
	survivingItems := make([]hnsw.Item[e2eCompositeKey], 0, total-perTenant)

	for i := range total {
		tenantIdx := i / perTenant
		key := e2eCompositeKey{
			Tenant: fmt.Sprintf("tenant-%d", tenantIdx),
			Bucket: uint32(i % 8),
			ID:     int64(i),
		}
		item := hnsw.Item[e2eCompositeKey]{ID: key, Vector: vecs[i]}
		allItems[i] = item
		if tenantIdx == 0 {
			tenant0Keys = append(tenant0Keys, key)
		} else {
			survivingItems = append(survivingItems, item)
		}
	}

	// Mix single Insert (first 30) and InsertBatch (remaining 570)
	for i := range 30 {
		e2eMustInsert(t, idx, allItems[i].ID, allItems[i].Vector)
	}
	if err := idx.InsertBatch(allItems[30:], hnsw.WithWorkers(4)); err != nil {
		t.Fatalf("InsertBatch failed: %v", err)
	}
	if idx.Len() != total {
		t.Fatalf("Len() = %d, want %d", idx.Len(), total)
	}

	oracleDist := e2eOracleDistFunc(t, hnsw.MetricSquaredEuclidean)
	tenant1Filter := func(id e2eCompositeKey) bool { return id.Tenant == "tenant-1" }

	var preRecallSum float64
	for _, q := range queries {
		got, err := idx.SearchFilter(q, k, tenant1Filter)
		if err != nil {
			t.Fatalf("pre-delete SearchFilter failed: %v", err)
		}
		oracle := e2eExactKNN(allItems, q, k, oracleDist, tenant1Filter)
		preRecallSum += e2eRecallAtK(got, oracle)
	}
	if avg := preRecallSum / float64(queryCount); avg < 0.95 {
		t.Fatalf("tenant-1 pre-delete Recall@10 = %.4f < 0.95", avg)
	}

	// Offboard tenant-0
	if removed := idx.DeleteBatch(tenant0Keys); removed != perTenant {
		t.Fatalf("DeleteBatch tenant-0 removed = %d, want %d", removed, perTenant)
	}

	var snap bytes.Buffer
	if _, err := idx.WriteTo(&snap); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	idx.Clear()
	if st := idx.Stats(); st.Len != 0 || st.Deleted != 0 || st.Capacity != 0 || st.EntryPointSet {
		t.Fatalf("original Clear stats invalid: %+v", st)
	}

	recovered, err := hnsw.ReadFrom[e2eCompositeKey](&snap)
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}
	if st := recovered.Stats(); st.Len != total-perTenant || st.Deleted != 0 {
		t.Fatalf("recovered stats mismatch: %+v", st)
	}
	for _, k0 := range tenant0Keys[:10] {
		if recovered.Contains(k0) {
			t.Fatalf("offboarded tenant-0 key %+v present in recovered index", k0)
		}
	}

	var postRecallSum float64
	for _, q := range queries {
		got, err := recovered.SearchFilter(q, k, tenant1Filter)
		if err != nil {
			t.Fatalf("post-recovery SearchFilter failed: %v", err)
		}
		oracle := e2eExactKNN(survivingItems, q, k, oracleDist, tenant1Filter)
		postRecallSum += e2eRecallAtK(got, oracle)
	}
	if avg := postRecallSum / float64(queryCount); avg < 0.95 {
		t.Fatalf("tenant-1 post-recovery Recall@10 = %.4f < 0.95", avg)
	}

	// Post-recovery lifecycle check
	extraKey := e2eCompositeKey{Tenant: "tenant-2", Bucket: 7, ID: 99999}
	e2eMustInsert(t, recovered, extraKey, vecs[0])
	if !recovered.Delete(extraKey) {
		t.Fatalf("post-recovery Delete failed")
	}
	if err := recovered.Compact(); err != nil {
		t.Fatalf("post-recovery Compact failed: %v", err)
	}
	if recovered.Len() != total-perTenant || recovered.Stats().Deleted != 0 {
		t.Fatalf("post-recovery Compact stats invalid: %+v", recovered.Stats())
	}
}

// 5. Concurrent Ingestion, Query, Deletion & Snapshot Stress (-race)
func TestE2E_Tier4_ConcurrentIngestionQueryDeletionStress(t *testing.T) {
	const seedCount = 200
	const queryCount = 30
	const dim = 24
	const k = 10
	const numWriters = 4
	const opsPerWriter = 60

	seedVecs, queries := e2eMakeClusteredDataset(8001, seedCount+numWriters*opsPerWriter, queryCount, dim, 12, false)
	idx := e2eMustNew[int](t, dim,
		hnsw.WithMetric(hnsw.MetricEuclidean),
		hnsw.WithM(12),
		hnsw.WithMmax(24),
		hnsw.WithEfConstruction(64),
		hnsw.WithEfSearch(48),
		hnsw.WithSeed(8001),
	)

	var gtMu sync.Mutex
	groundTruth := make(map[int][]float32, seedCount+numWriters*opsPerWriter)
	for i := range seedCount {
		e2eMustInsert(t, idx, i, seedVecs[i])
		groundTruth[i] = seedVecs[i]
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 32)

	// 4 writers operating on disjoint key partitions (seed IDs i%4==w and fresh IDs 1000+w*100+step)
	for w := range numWriters {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for step := range opsPerWriter {
				newID := 1000 + workerID*100 + step
				vec := seedVecs[seedCount+workerID*opsPerWriter+step]
				if err := idx.Insert(newID, vec); err != nil {
					errCh <- err
					return
				}
				gtMu.Lock()
				groundTruth[newID] = vec
				gtMu.Unlock()

				if step%4 == 0 {
					upsertID := workerID + (step/4)*numWriters
					if upsertID < seedCount {
						if err := idx.Insert(upsertID, vec); err != nil {
							errCh <- err
							return
						}
						gtMu.Lock()
						groundTruth[upsertID] = vec
						gtMu.Unlock()
					}
				}
				if step%5 == 0 {
					delID := 1000 + workerID*100 + step - 1
					if step > 0 {
						idx.Delete(delID)
						gtMu.Lock()
						delete(groundTruth, delID)
						gtMu.Unlock()
					}
				}
			}
		}(w)
	}

	// 4 concurrent reader goroutines
	for r := range 4 {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			for step := range 40 {
				q := queries[(readerID*7+step)%queryCount]
				res, err := idx.Search(q, k)
				if err != nil {
					errCh <- err
					return
				}
				for i, item := range res {
					d := float64(item.Distance)
					if math.IsNaN(d) || math.IsInf(d, 0) {
						errCh <- fmt.Errorf("non-finite distance in concurrent Search: %v", item.Distance)
						return
					}
					if i > 0 && item.Distance < res[i-1].Distance {
						errCh <- fmt.Errorf("unsorted concurrent Search at %d: %v < %v", i, item.Distance, res[i-1].Distance)
						return
					}
				}
				if _, err := idx.SearchFilter(q, 5, func(id int) bool { return id%2 == 0 }); err != nil {
					errCh <- err
					return
				}
				_, _ = idx.Get(step)
				_ = idx.Stats()
			}
		}(r)
	}

	// 1 concurrent snapshotter / compactor goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for step := range 5 {
			var snap bytes.Buffer
			if _, err := idx.WriteTo(&snap); err != nil {
				errCh <- err
				return
			}
			if _, err := hnsw.ReadFrom[int](&snap); err != nil {
				errCh <- err
				return
			}
			if step == 2 {
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
		t.Fatalf("concurrent stress error: %v", err)
	}

	var finalBuf bytes.Buffer
	if _, err := idx.WriteTo(&finalBuf); err != nil {
		t.Fatalf("final WriteTo failed: %v", err)
	}
	finalIdx, err := hnsw.ReadFrom[int](&finalBuf)
	if err != nil {
		t.Fatalf("final ReadFrom failed: %v", err)
	}
	if finalIdx.Len() != len(groundTruth) {
		t.Fatalf("final Len() = %d, groundTruth = %d", finalIdx.Len(), len(groundTruth))
	}

	liveItems := make([]hnsw.Item[int], 0, len(groundTruth))
	for id, v := range groundTruth {
		liveItems = append(liveItems, hnsw.Item[int]{ID: id, Vector: v})
	}
	oracleDist := e2eOracleDistFunc(t, hnsw.MetricEuclidean)
	var recallSum float64
	for _, q := range queries {
		got, err := finalIdx.Search(q, k)
		if err != nil {
			t.Fatalf("final Search failed: %v", err)
		}
		e2eAssertSortedDistances(t, got)
		oracle := e2eExactKNN(liveItems, q, k, oracleDist, nil)
		recallSum += e2eRecallAtK(got, oracle)
	}
	avg := recallSum / float64(queryCount)
	if avg < 0.95 {
		t.Fatalf("post-stress snapshot Recall@%d = %.4f < 0.95", k, avg)
	}
}
