package hnsw

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
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
