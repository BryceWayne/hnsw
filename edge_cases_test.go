package hnsw

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

// failingWriter succeeds for the first limit bytes and then returns err.
type failingWriter struct {
	limit   int
	written int
	err     error
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.written >= w.limit {
		return 0, w.err
	}
	avail := w.limit - w.written
	if len(p) <= avail {
		w.written += len(p)
		return len(p), nil
	}
	w.written += avail
	return avail, w.err
}

// cancelAfterNCallsContext wraps context.Background and reports context.Canceled
// after Err() has been called triggerAt times.
type cancelAfterNCallsContext struct {
	context.Context
	calls     atomic.Int32
	triggerAt int32
}

func (c *cancelAfterNCallsContext) Err() error {
	if c.calls.Add(1) >= c.triggerAt {
		return context.Canceled
	}
	return nil
}

func TestSearchBatchContextCancellationPaths(t *testing.T) {
	idx, err := New[int](4, WithM(8), WithEfConstruction(32), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if err := idx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	queries := [][]float32{
		{0, 0, 0, 0},
		{1, 0, 0, 0},
		{2, 0, 0, 0},
		{3, 0, 0, 0},
	}

	// Single-worker path: context cancels after initial sc.apply() check.
	ctx1 := &cancelAfterNCallsContext{Context: context.Background(), triggerAt: 2}
	_, err = idx.SearchBatch(queries, 2, WithWorkers(1), WithContext(ctx1))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("single-worker SearchBatch err = %v, want context.Canceled", err)
	}

	// Multi-worker path: context cancels inside worker loop.
	ctxMany := &cancelAfterNCallsContext{Context: context.Background(), triggerAt: 2}
	_, err = idx.SearchBatch(queries, 2, WithWorkers(4), WithContext(ctxMany))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("multi-worker SearchBatch err = %v, want context.Canceled", err)
	}
}

func TestSearchCoreEuclideanUnderflowRescoreAndSort(t *testing.T) {
	idx, err := New[int](2, WithM(8), WithEfConstruction(32), WithMetric(MetricEuclidean), WithSeed(7))
	if err != nil {
		t.Fatal(err)
	}

	// Insert sub-1e-18 vectors so squared L2 distances underflow below 1e-36,
	// forcing euclideanFallback64 and needsSort insertion-sort reordering.
	vecs := map[int][]float32{
		1: {3e-19, 0},
		2: {1e-19, 0},
		3: {2e-19, 0},
		4: {5e-19, 0},
		5: {4e-19, 0},
	}
	for id, v := range vecs {
		if err := idx.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}

	res, err := idx.Search([]float32{0, 0}, 5, WithSearchEf(16))
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 5 {
		t.Fatalf("got %d results, want 5", len(res))
	}
	wantOrder := []int{2, 3, 1, 5, 4}
	for i, wantID := range wantOrder {
		if res[i].ID != wantID {
			t.Fatalf("res[%d].ID = %d, want %d (full=%+v)", i, res[i].ID, wantID, res)
		}
		if i > 0 && res[i].Distance < res[i-1].Distance {
			t.Fatalf("results not sorted ascending at %d: %v < %v", i, res[i].Distance, res[i-1].Distance)
		}
		if math.IsNaN(float64(res[i].Distance)) {
			t.Fatalf("unexpected NaN distance at %d", i)
		}
	}

	// Add a normal-scale vector so scratch.results[0].dist >= 1e-36 while smaller
	// results still have d < 1e-36, exercising the inline rescore + needsSort path.
	if err := idx.Insert(6, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	resMixed, err := idx.Search([]float32{0, 0}, 6, WithSearchEf(16))
	if err != nil {
		t.Fatal(err)
	}
	if len(resMixed) != 6 {
		t.Fatalf("got %d mixed results, want 6", len(resMixed))
	}
	wantMixed := []int{2, 3, 1, 5, 4, 6}
	for i, wantID := range wantMixed {
		if resMixed[i].ID != wantID {
			t.Fatalf("resMixed[%d].ID = %d, want %d (full=%+v)", i, resMixed[i].ID, wantID, resMixed)
		}
		if i > 0 && resMixed[i].Distance < resMixed[i-1].Distance {
			t.Fatalf("mixed results not sorted ascending at %d: %v < %v", i, resMixed[i].Distance, resMixed[i-1].Distance)
		}
	}
}

func TestScratchBeginMinCapacity(t *testing.T) {
	var s searchScratch
	s.begin(1)
	if len(s.visited) != 64 {
		t.Fatalf("len(s.visited) = %d, want 64", len(s.visited))
	}
	if s.epoch != 1 {
		t.Fatalf("s.epoch = %d, want 1", s.epoch)
	}
}

func TestCompactDeduplicatesAndDropsOutOfRangeNeighbors(t *testing.T) {
	idx, err := New[int](2, WithM(4), WithMmax(8), WithEfConstruction(16), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}

	// Insert enough nodes so at least one node gets an upper level (>0).
	for i := 0; i < 40; i++ {
		if err := idx.Insert(i, []float32{float32(i), float32(i % 5)}); err != nil {
			t.Fatal(err)
		}
	}

	// Delete a node so Compact performs work.
	if !idx.Delete(0) {
		t.Fatal("expected Delete(0) to succeed")
	}

	// Inject an out-of-range neighbor and a duplicate neighbor into layer 0
	// and an upper level (where present) to exercise Compact's defensive cleanup.
	total := idx.nextID.Load()
	injectedL0 := false
	injectedUpper := false
	for id := uint32(1); id < total; id++ {
		if idx.isDeleted(id) {
			continue
		}
		chunk, slot := idx.getChunk(id)
		cnt := int(chunk.l0Count[slot])
		if !injectedL0 && cnt >= 3 {
			base := slot * idx.mmax
			dupTarget := chunk.l0Edges[base]
			chunk.l0Edges[base+cnt-2] = total + 99 // out-of-range
			chunk.l0Edges[base+cnt-1] = dupTarget  // duplicate
			injectedL0 = true
		}
		if !injectedUpper && chunk.maxLevels[slot] > 0 && len(chunk.upperLevels[slot][0]) > 0 {
			dupTarget := chunk.upperLevels[slot][0][0]
			chunk.upperLevels[slot][0] = append(chunk.upperLevels[slot][0], total+77, dupTarget)
			injectedUpper = true
		}
		if injectedL0 && injectedUpper {
			break
		}
	}
	if !injectedL0 {
		t.Fatal("failed to find suitable node for l0 edge injection")
	}
	if !injectedUpper {
		t.Fatal("failed to find suitable node with upper level for edge injection")
	}

	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	// Verify compacted graph has no duplicate or out-of-range neighbors.
	newTotal := idx.nextID.Load()
	for id := uint32(0); id < newTotal; id++ {
		chunk, slot := idx.getChunk(id)
		seen := make(map[uint32]struct{})
		base := slot * idx.mmax
		for j := 0; j < int(chunk.l0Count[slot]); j++ {
			nb := chunk.l0Edges[base+j]
			if nb >= newTotal || nb == id {
				t.Fatalf("invalid l0 neighbor %d for node %d (total=%d)", nb, id, newTotal)
			}
			if _, dup := seen[nb]; dup {
				t.Fatalf("duplicate l0 neighbor %d for node %d", nb, id)
			}
			seen[nb] = struct{}{}
		}
		for l := 0; l < int(chunk.maxLevels[slot]); l++ {
			seenUpper := make(map[uint32]struct{})
			for _, nb := range chunk.upperLevels[slot][l] {
				if nb >= newTotal || nb == id {
					t.Fatalf("invalid upper neighbor %d for node %d level %d", nb, id, l)
				}
				if _, dup := seenUpper[nb]; dup {
					t.Fatalf("duplicate upper neighbor %d for node %d level %d", nb, id, l)
				}
				seenUpper[nb] = struct{}{}
			}
		}
	}
}

func TestWriteToEntryFallbackAndWriterErrors(t *testing.T) {
	idx, err := New[int](64, WithM(8), WithEfConstruction(32), WithSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	// Insert enough 64-dim vectors so serialized size exceeds serBufSize (64 KiB),
	// ensuring bufio.Writer flushes mid-stream and surfaces underlying writer errors.
	for i := 0; i < 400; i++ {
		vec := make([]float32, 64)
		vec[i%64] = float32(i + 1)
		if err := idx.Insert(i, vec); err != nil {
			t.Fatal(err)
		}
	}

	// Exercise WriteTo entry-point fallback when entryState is emptyEntry on a non-empty index.
	savedES := idx.entryState.Swap(emptyEntry)
	var fullBuf bytes.Buffer
	n, err := idx.WriteTo(&fullBuf)
	if err != nil {
		t.Fatalf("WriteTo with emptyEntry fallback failed: %v", err)
	}
	if n != int64(fullBuf.Len()) {
		t.Fatalf("WriteTo wrote %d bytes, buffer has %d", n, fullBuf.Len())
	}
	idx.entryState.Store(savedES)

	reloaded, err := ReadFrom[int](bytes.NewReader(fullBuf.Bytes()))
	if err != nil {
		t.Fatalf("ReadFrom after fallback WriteTo failed: %v", err)
	}
	if reloaded.Len() != idx.Len() {
		t.Fatalf("reloaded.Len() = %d, want %d", reloaded.Len(), idx.Len())
	}

	// Exercise WriteTo error propagation at multiple byte offsets past serBufSize.
	totalBytes := fullBuf.Len()
	if totalBytes <= serBufSize {
		t.Fatalf("expected serialized size > %d, got %d", serBufSize, totalBytes)
	}
	wantErr := errors.New("disk full")
	for _, limit := range []int{0, 100, serBufSize, totalBytes - 10, totalBytes - 2} {
		fw := &failingWriter{limit: limit, err: wantErr}
		_, err := idx.WriteTo(fw)
		if !errors.Is(err, wantErr) && !errors.Is(err, io.ErrShortWrite) {
			t.Fatalf("WriteTo(limit=%d) err = %v, want %v", limit, err, wantErr)
		}
	}
}

func TestWriteToGobEncodeErrorAndSaveFileErrors(t *testing.T) {
	type unexportedKey struct {
		x int
	}
	idx, err := New[unexportedKey](2, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Insert(unexportedKey{x: 1}, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err == nil {
		t.Fatal("expected gob encode error for unexported struct key, got nil")
	}

	// SaveFile should propagate WriteTo errors and clean up temporary file.
	tmpDir := t.TempDir()
	target := filepath.Join(tmpDir, "unexported.bin")
	if err := idx.SaveFile(target); err == nil {
		t.Fatal("expected SaveFile to fail when WriteTo fails")
	}
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected temp file cleanup, found %d files in %s", len(entries), tmpDir)
	}
}

func TestSearchBatchWithSearchEfAndMidSearchCancel(t *testing.T) {
	idx, err := New[int](4, WithM(8), WithEfConstruction(32), WithSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 64; i++ {
		if err := idx.Insert(i, []float32{float32(i), float32(i % 7), 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	queries := [][]float32{{0, 0, 0, 0}, {10, 3, 0, 0}, {20, 6, 0, 0}, {30, 2, 0, 0}}

	// Verify WithSearchEf override in SearchBatch.
	res, err := idx.SearchBatch(queries, 2, WithSearchEf(24), WithWorkers(2))
	if err != nil {
		t.Fatalf("SearchBatch with WithSearchEf failed: %v", err)
	}
	if len(res) != len(queries) {
		t.Fatalf("got %d result sets, want %d", len(res), len(queries))
	}

	// Cancel on the 3rd Err() call so sc.apply() and the loop check pass,
	// triggering cancellation inside searchValidated on the single-worker path.
	ctxSingle := &cancelAfterNCallsContext{Context: context.Background(), triggerAt: 3}
	if _, err := idx.SearchBatch(queries[:1], 2, WithWorkers(1), WithContext(ctxSingle)); !errors.Is(err, context.Canceled) {
		t.Fatalf("single-worker mid-search cancel err = %v, want context.Canceled", err)
	}

	// Cancel on the 3rd Err() call in multi-worker mode to hit searchValidated error handling.
	ctxMulti := &cancelAfterNCallsContext{Context: context.Background(), triggerAt: 3}
	if _, err := idx.SearchBatch(queries, 2, WithWorkers(2), WithContext(ctxMulti)); !errors.Is(err, context.Canceled) {
		t.Fatalf("multi-worker mid-search cancel err = %v, want context.Canceled", err)
	}
}

func TestSerializeChunkedHighDimVectors(t *testing.T) {
	const dim = 4100 // Exceeds 4096-float chunk buffer in WriteTo/ReadFrom.
	idx, err := New[int](dim, WithM(4), WithEfConstruction(16), WithSeed(9))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		v := make([]float32, dim)
		v[0] = float32(i + 1)
		v[dim-1] = float32(i + 10)
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}

	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo(dim=%d) error = %v", dim, err)
	}
	reloaded, err := ReadFrom[int](bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("ReadFrom(dim=%d) error = %v", dim, err)
	}
	for i := 0; i < 3; i++ {
		got, ok := reloaded.Get(i)
		if !ok || len(got) != dim || got[0] != float32(i+1) || got[dim-1] != float32(i+10) {
			t.Fatalf("reloaded.Get(%d) mismatch: ok=%v len=%d", i, ok, len(got))
		}
	}
}

func TestInternalHelpersCoverage(t *testing.T) {
	// cmpDistNode ordering and equality.
	cases := []struct {
		a, b distNode
		want int
	}{
		{distNode{id: 1, dist: 1.0}, distNode{id: 2, dist: 2.0}, -1},
		{distNode{id: 1, dist: 2.0}, distNode{id: 2, dist: 1.0}, 1},
		{distNode{id: 1, dist: 1.0}, distNode{id: 2, dist: 1.0}, -1},
		{distNode{id: 2, dist: 1.0}, distNode{id: 1, dist: 1.0}, 1},
		{distNode{id: 2, dist: 1.0}, distNode{id: 2, dist: 1.0}, 0},
	}
	for _, tc := range cases {
		if got := cmpDistNode(tc.a, tc.b); got != tc.want {
			t.Fatalf("cmpDistNode(%+v, %+v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}

	// distNodes for Euclidean and Cosine, plus copyNeighbors with zero-cap dst.
	for _, metric := range []Metric{MetricEuclidean, MetricCosine} {
		idx, err := New[int](2, WithM(4), WithEfConstruction(16), WithMetric(metric), WithSeed(42))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 20; i++ {
			if err := idx.Insert(i, []float32{float32(i + 1), float32((i % 3) + 1)}); err != nil {
				t.Fatal(err)
			}
		}
		if d := idx.distNodes(0, 1); d < 0 || math.IsNaN(float64(d)) {
			t.Fatalf("metric %v distNodes(0, 1) = %v", metric, d)
		}
		if nbrs := idx.copyNeighbors(0, 0, nil, true); len(nbrs) == 0 {
			t.Fatalf("metric %v copyNeighbors L0 returned empty", metric)
		}
		for id := uint32(0); id < idx.nextID.Load(); id++ {
			chunk, slot := idx.getChunk(id)
			if chunk.maxLevels[slot] > 0 {
				_ = idx.copyNeighbors(id, 1, nil, true)
				_ = idx.copyNeighbors(id, int(chunk.maxLevels[slot])+1, nil, true)
				break
			}
		}
	}

	// searchScratch.growVisited direct coverage.
	var s searchScratch
	s.begin(1)
	if !s.visit(200) || s.visit(200) {
		t.Fatal("expected visit(200) to grow visited table and deduplicate second call")
	}
}

func TestKeysAllAndSetEfSearch(t *testing.T) {
	idx, err := New[int](3, WithM(8), WithEfConstruction(32), WithEfSearch(16), WithMetric(MetricCosine), WithSeed(7))
	if err != nil {
		t.Fatal(err)
	}

	// Empty index checks.
	if got := idx.Keys(); len(got) != 0 {
		t.Fatalf("empty Keys() = %v, want empty", got)
	}
	for range idx.All() {
		t.Fatal("empty All() should yield zero items")
	}

	// SetEfSearch validation and persistence.
	if err := idx.SetEfSearch(0); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("SetEfSearch(0) err = %v, want ErrInvalidConfig", err)
	}
	if err := idx.SetEfSearch(-5); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("SetEfSearch(-5) err = %v, want ErrInvalidConfig", err)
	}
	if err := idx.SetEfSearch(64); err != nil {
		t.Fatalf("SetEfSearch(64) err = %v", err)
	}
	if got := idx.Stats().EfSearch; got != 64 {
		t.Fatalf("Stats().EfSearch = %d, want 64", got)
	}

	for i := 1; i <= 10; i++ {
		if err := idx.Insert(i, []float32{float32(i) * 2, 1, 0}); err != nil {
			t.Fatal(err)
		}
	}
	idx.Delete(5)

	keys := idx.Keys()
	slices.Sort(keys)
	wantKeys := []int{1, 2, 3, 4, 6, 7, 8, 9, 10}
	if !slices.Equal(keys, wantKeys) {
		t.Fatalf("Keys() = %v, want %v", keys, wantKeys)
	}

	// Verify All() yields independent lossless copies and supports early break.
	seen := make(map[int][]float32)
	for id, vec := range idx.All() {
		seen[id] = vec
		vec[0] = -999 // mutating yielded copy must not corrupt stored vector
	}
	if len(seen) != len(wantKeys) {
		t.Fatalf("All() yielded %d items, want %d", len(seen), len(wantKeys))
	}
	if got, ok := idx.Get(1); !ok || got[0] != 2 {
		t.Fatalf("stored vector mutated via All(): ok=%v vec=%v", ok, got)
	}

	count := 0
	for range idx.All() {
		count++
		if count == 3 {
			break
		}
	}
	if count != 3 {
		t.Fatalf("early break yielded %d items, want 3", count)
	}

	// Mutating inside All() must not deadlock and deleted keys are skipped.
	visited := 0
	for id := range idx.All() {
		visited++
		if visited == 1 {
			// Delete all remaining keys except id.
			for _, k := range wantKeys {
				if k != id {
					idx.Delete(k)
				}
			}
		}
	}
	if visited != 1 {
		t.Fatalf("All() after deleting remaining keys yielded %d items, want 1", visited)
	}

	// Verify SetEfSearch survives WriteTo -> ReadFrom.
	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	reloaded, err := ReadFrom[int](&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Stats().EfSearch; got != 64 {
		t.Fatalf("reloaded EfSearch = %d, want 64", got)
	}

	// Concurrent SetEfSearch + Search + Keys + All under -race.
	for i := 0; i < 32; i++ {
		if err := reloaded.Insert(i+100, []float32{float32(i + 1), 1, 2}); err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for i := 1; i <= 50; i++ {
			_ = reloaded.SetEfSearch(16 + (i % 64))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_, _ = reloaded.Search([]float32{1, 1, 2}, 5)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			_ = reloaded.Keys()
			for range reloaded.All() {
				break
			}
		}
	}()
	wg.Wait()
}

func TestGetAndContainsAllocs(t *testing.T) {
	if raceDetectorEnabled {
		t.Skip("allocation count disabled under -race")
	}
	idx, err := New[int](8, WithM(8), WithEfConstruction(32), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if err := idx.Insert(i, []float32{float32(i), 1, 2, 3, 4, 5, 6, 7}); err != nil {
			t.Fatal(err)
		}
	}

	containsAllocs := testing.AllocsPerRun(200, func() {
		if !idx.Contains(7) {
			t.Fatal("expected Contains(7) == true")
		}
	})
	if containsAllocs != 0 {
		t.Fatalf("Contains allocs/op = %.1f, want 0", containsAllocs)
	}

	getAllocs := testing.AllocsPerRun(200, func() {
		v, ok := idx.Get(7)
		if !ok || len(v) != 8 {
			t.Fatal("expected Get(7) to succeed")
		}
	})
	if getAllocs > 1 {
		t.Fatalf("Get allocs/op = %.1f, want <= 1", getAllocs)
	}
}

func TestSerializeChunkedFixedKeysAllTypes(t *testing.T) {
	// Test >512 keys to cross the 4096-byte key chunk boundary in writeFixedKeys/readFixedKeys.
	const n = 600
	idx, err := New[int](2, WithM(4), WithEfConstruction(16), WithSeed(5))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := idx.Insert(i, []float32{float32(i), float32(i % 13)}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}
	reloaded, err := ReadFrom[int](bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}
	if reloaded.Len() != n {
		t.Fatalf("reloaded.Len() = %d, want %d", reloaded.Len(), n)
	}
	for i := 0; i < n; i++ {
		if !reloaded.Contains(i) {
			t.Fatalf("reloaded missing key %d", i)
		}
	}

	// Verify truncated fixed-key stream returns ErrCorruptedIndex.
	trunc := buf.Bytes()[:120]
	if _, err := ReadFrom[int](bytes.NewReader(trunc)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("truncated fixed keys err = %v, want ErrCorruptedIndex", err)
	}
}

func TestDeleteBatchedRepairPreservesRecallAndGraph(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	idx, err := New[int](16, WithM(8), WithMmax(16), WithEfConstruction(64), WithSeed(99))
	if err != nil {
		t.Fatal(err)
	}
	const total = 300
	for i := 0; i < total; i++ {
		v := make([]float32, 16)
		for j := range v {
			v[j] = rng.Float32()
		}
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}

	// Single Delete + DeleteBatch across dense upper and layer-0 neighborhoods.
	for i := 0; i < 20; i++ {
		if !idx.Delete(i) {
			t.Fatalf("expected Delete(%d) == true", i)
		}
	}
	batchIDs := make([]int, 0, 50)
	for i := 20; i < 70; i++ {
		batchIDs = append(batchIDs, i)
	}
	if removed := idx.DeleteBatch(batchIDs); removed != 50 {
		t.Fatalf("DeleteBatch removed %d, want 50", removed)
	}
	if idx.Len() != total-70 {
		t.Fatalf("Len() = %d, want %d", idx.Len(), total-70)
	}

	// Every remaining node should be reachable by exact query.
	for i := 70; i < 120; i++ {
		vec, ok := idx.Get(i)
		if !ok {
			t.Fatalf("missing key %d", i)
		}
		res, err := idx.Search(vec, 1, WithSearchEf(64))
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 1 || res[0].ID != i {
			t.Fatalf("Search for live key %d returned %+v", i, res)
		}
	}
}

func BenchmarkDelete(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	const n = 1000
	const dim = 128
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()
		}
		vecs[i] = v
	}
	idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}
	items := make([]Item[int], n)
	for i := range items {
		items[i] = Item[int]{ID: i, Vector: vecs[i]}
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := i % n
		idx.Delete(id)
		b.StopTimer()
		if err := idx.Insert(id, vecs[id]); err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
	}
}

func BenchmarkGet(b *testing.B) {
	idx, err := New[int](128, WithM(16), WithEfConstruction(64), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}
	vec := make([]float32, 128)
	for i := 0; i < 1000; i++ {
		vec[0] = float32(i)
		if err := idx.Insert(i, vec); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if _, ok := idx.Get(i % 1000); !ok {
				b.Fatal("Get failed")
			}
			i++
		}
	})
}

func BenchmarkContains(b *testing.B) {
	idx, err := New[int](128, WithM(16), WithEfConstruction(64), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}
	vec := make([]float32, 128)
	for i := 0; i < 1000; i++ {
		vec[0] = float32(i)
		if err := idx.Insert(i, vec); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if !idx.Contains(i % 1000) {
				b.Fatal("Contains failed")
			}
			i++
		}
	})
}

func TestWithInitialCapacity(t *testing.T) {
	if _, err := New[int](8, WithInitialCapacity(-1)); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("WithInitialCapacity(-1) err = %v, want ErrInvalidConfig", err)
	}
	if _, err := New[int](8, WithInitialCapacity(maxArenaCapacity+1)); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("WithInitialCapacity(>max) err = %v, want ErrInvalidConfig", err)
	}

	idx, err := New[int](8, WithInitialCapacity(600), WithQuantization(QuantFloat16), WithMetric(MetricCosine), WithSeed(7))
	if err != nil {
		t.Fatalf("New with initial capacity failed: %v", err)
	}
	st := idx.Stats()
	if st.Len != 0 {
		t.Fatalf("Len = %d, want 0", st.Len)
	}
	// 600 rounds up to 3 chunks * 256 = 768 slots.
	if st.Capacity != 768 {
		t.Fatalf("Capacity = %d, want 768", st.Capacity)
	}
	if st.MemoryBytes <= 0 {
		t.Fatalf("MemoryBytes = %d, want > 0", st.MemoryBytes)
	}

	for i := 0; i < 600; i++ {
		if err := idx.Insert(i, []float32{float32(i + 1), 1, 0, 0, 0, 0, 0, 0}); err != nil {
			t.Fatalf("Insert(%d) failed: %v", i, err)
		}
	}
	stAfter := idx.Stats()
	if stAfter.Len != 600 || stAfter.Capacity != 768 {
		t.Fatalf("after 600 inserts: Len=%d Capacity=%d, want 600/768", stAfter.Len, stAfter.Capacity)
	}
}

func TestSearchFilterIntoZeroAllocs(t *testing.T) {
	if raceDetectorEnabled {
		t.Skip("allocation count disabled under -race")
	}

	idx, err := New[int](16, WithM(8), WithMmax(16), WithEfConstruction(64), WithEfSearch(32), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 500; i++ {
		v := make([]float32, 16)
		for j := range v {
			v[j] = rng.Float32()
		}
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}
	query := make([]float32, 16)
	for j := range query {
		query[j] = rng.Float32()
	}

	filter := func(id int) bool { return id%2 == 0 }
	want, err := idx.SearchFilter(query, 10, filter)
	if err != nil {
		t.Fatal(err)
	}

	dst := make([]SearchResult[int], 0, 10)
	got, err := idx.SearchFilterInto(query, 10, filter, dst)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d != len(want)=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%+v != want[%d]=%+v", i, got[i], i, want[i])
		}
	}

	// Empty index path with pre-allocated dst.
	emptyIdx, _ := New[int](16)
	emptyRes, err := emptyIdx.SearchFilterInto(query, 5, filter, dst)
	if err != nil || len(emptyRes) != 0 {
		t.Fatalf("empty SearchFilterInto = (%v, %v), want empty/nil", emptyRes, err)
	}

	// Steady-state 0 allocs/op verification.
	allocs := testing.AllocsPerRun(200, func() {
		res, err := idx.SearchFilterInto(query, 10, filter, dst)
		if err != nil || len(res) != 10 {
			t.Fatalf("SearchFilterInto failed: len=%d err=%v", len(res), err)
		}
	})
	if allocs != 0 {
		t.Fatalf("SearchFilterInto allocs/op = %.1f, want 0", allocs)
	}
}

func TestIndexStatsTombstoneRatio(t *testing.T) {
	if r := (IndexStats{}).TombstoneRatio(); r != 0 {
		t.Fatalf("empty TombstoneRatio = %v, want 0", r)
	}
	idx, err := New[int](4, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if err := idx.Insert(i, []float32{float32(i), 1, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	if r := idx.Stats().TombstoneRatio(); r != 0 {
		t.Fatalf("before delete TombstoneRatio = %v, want 0", r)
	}
	idx.Delete(0)
	if r := idx.Stats().TombstoneRatio(); math.Abs(r-0.25) > 1e-9 {
		t.Fatalf("after 1/4 delete TombstoneRatio = %v, want 0.25", r)
	}
	if err := idx.Compact(); err != nil {
		t.Fatal(err)
	}
	if r := idx.Stats().TombstoneRatio(); r != 0 {
		t.Fatalf("after Compact TombstoneRatio = %v, want 0", r)
	}
}
