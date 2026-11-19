package hnsw

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/iotest"
)

type tier5GobKey struct {
	Tag string
	Seq uint64
}

type countdownContext struct {
	context.Context
	calls atomic.Int32
	limit int32
}

func newCountdownContext(limit int32) *countdownContext {
	return &countdownContext{
		Context: context.Background(),
		limit:   limit,
	}
}

func (c *countdownContext) Err() error {
	if c.calls.Add(1) > c.limit {
		return context.Canceled
	}
	return nil
}

type failAfterNWriter struct {
	limit   int
	written int
	err     error
}

func (w *failAfterNWriter) Write(p []byte) (int, error) {
	if w.written >= w.limit {
		return 0, w.err
	}
	rem := w.limit - w.written
	if len(p) > rem {
		w.written += rem
		return rem, w.err
	}
	w.written += len(p)
	return len(p), nil
}

func TestTier5_MidSearchContextCancellation(t *testing.T) {
	t.Run("gt64_iterations_in_loop", func(t *testing.T) {
		idx, err := New[int](4, WithM(8), WithMmax(16), WithEfConstruction(64), WithSeed(11))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 200; i++ {
			if err := idx.Insert(i, []float32{float32(i), float32(i % 7), 0, 0}); err != nil {
				t.Fatal(err)
			}
		}

		ctx, cancel := context.WithCancel(context.Background())
		var calls int
		_, err = idx.SearchFilter([]float32{100, 0, 0, 0}, 5, func(_ int) bool {
			calls++
			if calls == 2 {
				cancel()
			}
			return true
		}, WithSearchEf(200), WithContext(ctx))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("SearchFilter(>64 iters) err = %v, want context.Canceled (calls=%d)", err, calls)
		}
	})

	t.Run("lt64_iterations_post_loop", func(t *testing.T) {
		idx, err := New[int](4, WithM(4), WithMmax(8), WithEfConstruction(32), WithSeed(19))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 20; i++ {
			if err := idx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}

		ctx, cancel := context.WithCancel(context.Background())
		var calls int
		_, err = idx.SearchFilter([]float32{10, 0, 0, 0}, 5, func(_ int) bool {
			calls++
			if calls == 2 {
				cancel()
			}
			return true
		}, WithSearchEf(20), WithContext(ctx))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("SearchFilter(<64 iters) err = %v, want context.Canceled (calls=%d)", err, calls)
		}
	})
}

func TestTier5_ConcurrentSameKeyAndEmptyEntryInsert(t *testing.T) {
	t.Run("same_key_empty_index", func(t *testing.T) {
		const rounds = 50
		const workers = 16
		for round := 0; round < rounds; round++ {
			idx, err := New[int](4, WithM(4), WithMmax(8), WithSeed(int64(round+1)))
			if err != nil {
				t.Fatal(err)
			}
			start := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(workers)
			for g := 0; g < workers; g++ {
				go func(gid int) {
					defer wg.Done()
					<-start
					vec := []float32{float32(gid + 1), 1, 0, 0}
					if err := idx.Insert(42, vec); err != nil {
						t.Errorf("round %d gid %d Insert err = %v", round, gid, err)
					}
				}(g)
			}
			close(start)
			wg.Wait()

			if idx.Len() != 1 || !idx.Contains(42) {
				t.Fatalf("round %d: Len=%d Contains(42)=%v", round, idx.Len(), idx.Contains(42))
			}
			st := idx.Stats()
			if !st.EntryPointSet || st.Deleted < 0 {
				t.Fatalf("round %d: invalid stats %+v", round, st)
			}
			res, err := idx.Search([]float32{1, 1, 0, 0}, 1)
			if err != nil || len(res) != 1 || res[0].ID != 42 {
				t.Fatalf("round %d: Search = (%+v, %v)", round, res, err)
			}
		}
	})

	t.Run("distinct_keys_empty_index_entry_cas", func(t *testing.T) {
		const rounds = 50
		const workers = 16
		for round := 0; round < rounds; round++ {
			idx, err := New[int](4, WithM(4), WithMmax(8), WithSeed(int64(100+round)))
			if err != nil {
				t.Fatal(err)
			}
			start := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(workers)
			for g := 0; g < workers; g++ {
				go func(gid int) {
					defer wg.Done()
					<-start
					vec := []float32{float32(gid), float32(round), 0, 0}
					if err := idx.Insert(gid, vec); err != nil {
						t.Errorf("round %d gid %d Insert err = %v", round, gid, err)
					}
				}(g)
			}
			close(start)
			wg.Wait()

			if idx.Len() != workers || !idx.Stats().EntryPointSet {
				t.Fatalf("round %d: Len=%d stats=%+v", round, idx.Len(), idx.Stats())
			}
			for g := 0; g < workers; g++ {
				if !idx.Contains(g) {
					t.Fatalf("round %d: missing key %d", round, g)
				}
			}
		}
	})
}

func TestTier5_ArenaMultiChunkDeleteCompactAndMultiLevelEntryReelection(t *testing.T) {
	idx, err := New[int](4, WithM(2), WithMmax(4), WithEfConstruction(32), WithEfSearch(64), WithSeed(77))
	if err != nil {
		t.Fatal(err)
	}
	const total = 600
	for i := 0; i < total; i++ {
		if err := idx.Insert(i, []float32{float32(i), float32(i % 13), 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	st := idx.Stats()
	if st.Capacity != 768 || st.MaxLevel < 2 || !st.EntryPointSet {
		t.Fatalf("unexpected initial multi-chunk stats: %+v", st)
	}

	// Delete 300 nodes including upper-level entry points.
	for i := 0; i < 300; i++ {
		if !idx.Delete(i) {
			t.Fatalf("Delete(%d) returned false", i)
		}
		if idx.Len() != total-i-1 || !idx.Stats().EntryPointSet {
			t.Fatalf("after Delete(%d): Len=%d stats=%+v", i, idx.Len(), idx.Stats())
		}
	}

	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact failed: %v", err)
	}
	stAfter := idx.Stats()
	if stAfter.Capacity != 512 || stAfter.Deleted != 0 || stAfter.Len != 300 || !stAfter.EntryPointSet {
		t.Fatalf("unexpected stats after Compact: %+v", stAfter)
	}

	res, err := idx.Search([]float32{450, float32(450 % 13), 0, 0}, 1)
	if err != nil || len(res) != 1 || res[0].ID != 450 {
		t.Fatalf("Search(450) after Compact = (%+v, %v)", res, err)
	}

	// Delete remaining 300 nodes down to empty and verify entry-point re-election at every step.
	for i := 300; i < total; i++ {
		if !idx.Delete(i) {
			t.Fatalf("Delete(%d) returned false", i)
		}
		curSt := idx.Stats()
		wantEP := i < total-1
		if curSt.EntryPointSet != wantEP || curSt.Len != total-i-1 {
			t.Fatalf("Delete(%d) stats = %+v, want EntryPointSet=%v Len=%d", i, curSt, wantEP, total-i-1)
		}
	}

	// Re-insert into empty index to verify recovery from emptyEntry.
	if err := idx.Insert(999, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("re-insert failed: %v", err)
	}
	if idx.Len() != 1 || !idx.Stats().EntryPointSet {
		t.Fatalf("unexpected stats after re-insert: %+v", idx.Stats())
	}
	res, err = idx.Search([]float32{1, 2, 3, 4}, 1)
	if err != nil || len(res) != 1 || res[0].ID != 999 || res[0].Distance != 0 {
		t.Fatalf("Search after re-insert = (%+v, %v)", res, err)
	}
}

func TestTier5_DotProductOrthogonalAndLargeMHeuristic(t *testing.T) {
	t.Run("dot_product_orthogonal_zero", func(t *testing.T) {
		const dim = 16
		idx, err := New[int](dim, WithMetric(MetricDotProduct), WithM(4), WithMmax(8), WithEfConstruction(32), WithHeuristic(true), WithSeed(3))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < dim; i++ {
			v := make([]float32, dim)
			v[i] = 1
			if err := idx.Insert(i, v); err != nil {
				t.Fatal(err)
			}
		}

		// Explicitly verify distNodes hits d == 0 for orthogonal nodes.
		if d := idx.distNodes(0, 1); d != 0 || math.Signbit(float64(d)) {
			t.Fatalf("distNodes(0, 1) = %v (signbit=%v), want +0", d, math.Signbit(float64(d)))
		}

		q := make([]float32, dim)
		q[0] = 1
		res, err := idx.Search(q, 5, WithSearchEf(16))
		if err != nil || len(res) != 5 {
			t.Fatalf("Search orthogonal = (%+v, %v)", res, err)
		}
		if res[0].ID != 0 || res[0].Distance != -1 {
			t.Fatalf("top-1 = %+v, want ID=0 Distance=-1", res[0])
		}
		for i := 1; i < len(res); i++ {
			if res[i].Distance != 0 {
				t.Errorf("orthogonal res[%d] = %+v, want Distance=0", i, res[i])
			}
		}
	})

	t.Run("large_mmax_exceeds_stack_buf", func(t *testing.T) {
		idx, err := New[int](8, WithM(80), WithMmax(100), WithEfConstruction(150), WithSeed(5))
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(5))
		for i := 0; i < 150; i++ {
			v := make([]float32, 8)
			for j := range v {
				v[j] = rng.Float32()
			}
			if err := idx.Insert(i, v); err != nil {
				t.Fatal(err)
			}
		}
		q := []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
		res, err := idx.Search(q, 10)
		if err != nil || len(res) != 10 {
			t.Fatalf("Search(large Mmax) = (%+v, %v)", res, err)
		}
	})
}

func TestTier5_DistanceIdenticalExtremeScalesAndGuards(t *testing.T) {
	v := []float32{3.5, -2.25, 1.0, 0.5}
	if d, err := Euclidean(v, v); err != nil || d != 0 {
		t.Fatalf("Euclidean(v, v) = (%v, %v), want (0, nil)", d, err)
	}
	if d, err := SquaredEuclidean(v, v); err != nil || d != 0 {
		t.Fatalf("SquaredEuclidean(v, v) = (%v, %v), want (0, nil)", d, err)
	}

	// Extreme-scale Euclidean helper cases.
	hugeA := []float32{2e19, 0}
	hugeB := []float32{-2e19, 0}
	gotHuge, err := Euclidean(hugeA, hugeB)
	if err != nil || math.IsInf(float64(gotHuge), 0) || math.Abs(float64(gotHuge)-4e19) > 1e14 {
		t.Fatalf("Euclidean(2e19, -2e19) = (%v, %v), want ~4e19", gotHuge, err)
	}

	tinyA := []float32{1e-23, 0}
	zero := []float32{0, 0}
	gotTiny, err := Euclidean(tinyA, zero)
	if err != nil || gotTiny == 0 || math.Abs(float64(gotTiny)-1e-23) > 1e-28 {
		t.Fatalf("Euclidean(1e-23, 0) = (%v, %v), want ~1e-23", gotTiny, err)
	}

	subA := []float32{1e-44, 0}
	gotSub, err := Euclidean(subA, zero)
	if err != nil || gotSub == 0 || math.Abs(float64(gotSub)-1e-44) > 1e-45 {
		t.Fatalf("Euclidean(1e-44, 0) = (%v, %v), want ~1e-44", gotSub, err)
	}

	overA := []float32{3e38, 0}
	overB := []float32{-3e38, 0}
	gotOver, err := Euclidean(overA, overB)
	if err != nil || !math.IsInf(float64(gotOver), 1) {
		t.Fatalf("Euclidean(3e38, -3e38) = (%v, %v), want (+Inf, nil)", gotOver, err)
	}

	// Index MetricEuclidean extreme huge scale ranking (k=3 and k=2).
	idxHuge, err := New[int](2, WithMetric(MetricEuclidean), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	_ = idxHuge.Insert(1, []float32{2e19, 0})
	_ = idxHuge.Insert(2, []float32{0, 0})
	_ = idxHuge.Insert(3, []float32{-2e19, 0})
	resHuge, err := idxHuge.Search([]float32{-2e19, 0}, 3)
	if err != nil || len(resHuge) != 3 {
		t.Fatalf("huge Search(3) = (%+v, %v)", resHuge, err)
	}
	if resHuge[0].ID != 3 || resHuge[0].Distance != 0 ||
		resHuge[1].ID != 2 || math.Abs(float64(resHuge[1].Distance)-2e19) > 1e14 ||
		resHuge[2].ID != 1 || math.Abs(float64(resHuge[2].Distance)-4e19) > 1e14 {
		t.Fatalf("huge Search(3) ordering/distances = %+v", resHuge)
	}
	resHuge2, err := idxHuge.Search([]float32{-2e19, 0}, 2)
	if err != nil || len(resHuge2) != 2 || resHuge2[0].ID != 3 || resHuge2[1].ID != 2 {
		t.Fatalf("huge Search(2) = (%+v, %v), want IDs [3, 2]", resHuge2, err)
	}

	// Mixed-scale huge (>k +Inf items plus finite large items, k < N).
	_ = idxHuge.Insert(4, []float32{3e19, 0})    // dist 5e19 (+Inf^2)
	_ = idxHuge.Insert(5, []float32{-1.5e19, 0}) // dist 5e18 (finite 2.5e37^2)
	_ = idxHuge.Insert(6, []float32{1e19, 0})    // dist 3e19 (+Inf^2)
	resHugeMixed2, err := idxHuge.Search([]float32{-2e19, 0}, 2)
	if err != nil || len(resHugeMixed2) != 2 || resHugeMixed2[0].ID != 3 || resHugeMixed2[1].ID != 5 {
		t.Fatalf("mixed huge Search(2) = (%+v, %v), want IDs [3, 5]", resHugeMixed2, err)
	}
	resHugeMixed4, err := idxHuge.Search([]float32{-2e19, 0}, 4)
	if err != nil || len(resHugeMixed4) != 4 ||
		resHugeMixed4[0].ID != 3 || resHugeMixed4[1].ID != 5 || resHugeMixed4[2].ID != 2 || resHugeMixed4[3].ID != 6 {
		t.Fatalf("mixed huge Search(4) = (%+v, %v), want IDs [3, 5, 2, 6]", resHugeMixed4, err)
	}

	idxHugeInf, err := New[int](2, WithMetric(MetricEuclidean), WithSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	_ = idxHugeInf.Insert(1, []float32{5e19, 0})
	_ = idxHugeInf.Insert(2, []float32{4e19, 0})
	_ = idxHugeInf.Insert(3, []float32{3e19, 0})
	_ = idxHugeInf.Insert(4, []float32{2e19, 0})
	resHugeInf2, err := idxHugeInf.Search([]float32{0, 0}, 2)
	if err != nil || len(resHugeInf2) != 2 ||
		resHugeInf2[0].ID != 4 || math.Abs(float64(resHugeInf2[0].Distance)-2e19) > 1e14 ||
		resHugeInf2[1].ID != 3 || math.Abs(float64(resHugeInf2[1].Distance)-3e19) > 1e14 {
		t.Fatalf("all-+Inf huge Search(2) = (%+v, %v), want IDs [4, 3]", resHugeInf2, err)
	}

	// Index MetricEuclidean extreme tiny + mixed scale ranking.
	idxTiny, err := New[int](2, WithMetric(MetricEuclidean), WithSeed(2))
	if err != nil {
		t.Fatal(err)
	}
	_ = idxTiny.Insert(1, []float32{2e-23, 0})
	_ = idxTiny.Insert(2, []float32{1e-23, 0})
	_ = idxTiny.Insert(3, []float32{0, 0})
	resTiny, err := idxTiny.Search([]float32{0, 0}, 3)
	if err != nil || len(resTiny) != 3 ||
		resTiny[0].ID != 3 || resTiny[0].Distance != 0 ||
		resTiny[1].ID != 2 || math.Abs(float64(resTiny[1].Distance)-1e-23) > 1e-28 ||
		resTiny[2].ID != 1 || math.Abs(float64(resTiny[2].Distance)-2e-23) > 1e-28 {
		t.Fatalf("tiny Search(3) = (%+v, %v)", resTiny, err)
	}

	_ = idxTiny.Insert(4, []float32{10, 0})
	resMixed, err := idxTiny.Search([]float32{0, 0}, 4)
	if err != nil || len(resMixed) != 4 ||
		resMixed[0].ID != 3 || resMixed[1].ID != 2 || resMixed[2].ID != 1 || resMixed[3].ID != 4 {
		t.Fatalf("mixed tiny+normal Search(4) = (%+v, %v), want IDs [3,2,1,4]", resMixed, err)
	}
	resMixed2, err := idxTiny.Search([]float32{0, 0}, 2)
	if err != nil || len(resMixed2) != 2 ||
		resMixed2[0].ID != 3 || resMixed2[0].Distance != 0 ||
		resMixed2[1].ID != 2 || math.Abs(float64(resMixed2[1].Distance)-1e-23) > 1e-28 {
		t.Fatalf("mixed tiny+normal Search(2) = (%+v, %v), want IDs [3, 2] with distances [0, 1e-23]", resMixed2, err)
	}

	// clampCosine bounds and NaN.
	if got := clampCosine(-0.01); got != 0 {
		t.Errorf("clampCosine(-0.01) = %v, want 0", got)
	}
	if got := clampCosine(2.01); got != 2 {
		t.Errorf("clampCosine(2.01) = %v, want 2", got)
	}
	if got := clampCosine(float32(math.NaN())); got != 1.0 {
		t.Errorf("clampCosine(NaN) = %v, want 1.0", got)
	}

	// Empty and mismatched-length white-box guards.
	if got := l2Squared(nil, nil); got != 0 {
		t.Errorf("l2Squared(nil, nil) = %v, want 0", got)
	}
	if got := dotProduct(nil, nil); got != 0 {
		t.Errorf("dotProduct(nil, nil) = %v, want 0", got)
	}
	if got := l2Squared([]float32{1, 2}, []float32{1}); got != 0 {
		t.Errorf("l2Squared(short b) = %v, want 0", got)
	}
	if got := dotProduct([]float32{1, 2}, []float32{1}); got != 0 {
		t.Errorf("dotProduct(short b) = %v, want 0", got)
	}
	if got := l2SquaredGo(nil, nil); got != 0 {
		t.Errorf("l2SquaredGo(nil, nil) = %v, want 0", got)
	}
	if got := dotProductGo(nil, nil); got != 0 {
		t.Errorf("dotProductGo(nil, nil) = %v, want 0", got)
	}
}

func TestTier5_ScratchDynamicJumpAndInternalGuards(t *testing.T) {
	// scratch.visit dynamic >2x jump (scratch.go:56-58).
	s := newSearchScratch(0, 8)
	s.begin(1)
	if !s.visit(5) {
		t.Fatal("visit(5) returned false")
	}
	if !s.visit(200) || s.visit(200) {
		t.Fatal("visit(200) failed on >2x growth")
	}
	if s.visit(5) {
		t.Fatal("visit(5) lost after >2x growth")
	}

	idx, err := New[int](4, WithM(2), WithMmax(4), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		if err := idx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	// Out-of-bounds page/chunk guards.
	if c, slot := idx.getChunk(1 << 20); c != nil || slot != 0 {
		t.Fatalf("getChunk(1<<20) = (%v, %d), want (nil, 0)", c, slot)
	}
	if !idx.isDeleted(1 << 20) {
		t.Fatal("isDeleted(1<<20) = false, want true")
	}

	// Double markDeleted.
	if !idx.markDeleted(0) {
		t.Fatal("first markDeleted(0) returned false")
	}
	if idx.markDeleted(0) {
		t.Fatal("second markDeleted(0) returned true")
	}
	idx.liveCount.Add(-1)
	idx.tombstoneCount.Add(1)

	// setNeighbors truncation at level 0 (>mmax) and upper level (>m).
	idx.setNeighbors(1, 0, []uint32{2, 3, 4, 5, 2, 3}, false)
	if nbrs := idx.copyNeighbors(1, 0, nil, false); len(nbrs) != 4 {
		t.Fatalf("level 0 truncated len = %d, want 4", len(nbrs))
	}
	c1, s1 := idx.getChunk(1)
	c1.maxLevels[s1] = 1
	c1.upperLevels[s1] = make([][]uint32, 1)
	idx.setNeighbors(1, 1, []uint32{2, 3, 4, 5}, false)
	if nbrs := idx.copyNeighbors(1, 1, nil, false); len(nbrs) != 2 {
		t.Fatalf("level 1 truncated len = %d, want 2", len(nbrs))
	}

	// addConnection / removeEdge / selectNeighbors defensive guards.
	idx.addConnection(1, 1, 0, s, false)              // self-loop ignored
	idx.addConnection(0, 1, 0, s, false)              // deleted target ignored
	idx.addConnection(1, 2, maxNodeLevel+1, s, false) // lvl > maxLevels ignored
	idx.removeEdge(1<<20, 0, 0)                       // nil chunk ignored
	idx.removeEdge(1, 0, maxNodeLevel+1)              // lvl > maxLevels ignored

	cands := []distNode{
		{id: 1, dist: 0},
		{id: 0, dist: 1},
		{id: 2, dist: 1},
		{id: 2, dist: 1},
		{id: 3, dist: 2},
	}
	selHeap := idx.selectNeighborsFromHeap(1, cands, 4, s, nil)
	if len(selHeap) != 2 || selHeap[0] != 2 || selHeap[1] != 3 {
		t.Fatalf("selectNeighborsFromHeap = %v, want [2 3]", selHeap)
	}
	selIDs := idx.selectNeighborsFromIDs(1, []uint32{1, 0, 2, 2, 3}, 4, s, nil)
	if len(selIDs) != 2 || selIDs[0] != 2 || selIDs[1] != 3 {
		t.Fatalf("selectNeighborsFromIDs = %v, want [2 3]", selIDs)
	}

	// electEntryPoint when all internal slots are deleted.
	for id := uint32(1); id < idx.nextID.Load(); id++ {
		idx.markDeleted(id)
	}
	idx.electEntryPoint()
	if idx.entryState.Load() != emptyEntry {
		t.Fatalf("electEntryPoint on all-deleted = %d, want emptyEntry", idx.entryState.Load())
	}

	// Direct config.validate() defensive checks.
	badCfgs := []config{
		{m: 1, mmax: 2, mmaxSet: true, efConstruction: 10, efSearch: 10, metric: MetricEuclidean},
		{m: 4, mmax: 8, mmaxSet: true, efConstruction: 0, efSearch: 10, metric: MetricEuclidean},
		{m: 4, mmax: 8, mmaxSet: true, efConstruction: 10, efSearch: 0, metric: MetricEuclidean},
		{m: 4, mmax: 8, mmaxSet: true, efConstruction: 10, efSearch: 10, metric: metricCustom, customDist: nil},
		{m: 4, mmax: 8, mmaxSet: true, efConstruction: 10, efSearch: 10, metric: Metric(99)},
	}
	for i := range badCfgs {
		if err := badCfgs[i].validate(); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("badCfgs[%d].validate() = %v, want ErrInvalidConfig", i, err)
		}
	}
}

func TestTier5_SerializeWriteToAndSaveFileErrors(t *testing.T) {
	boom := errors.New("boom write")

	// Small index: failing at every byte offset triggers bw.Flush / header / keys / vec / graph / CRC errors.
	idxSmall, err := New[string](4, WithM(4), WithMmax(8), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	_ = idxSmall.Insert("alpha", []float32{1, 2, 3, 4})
	_ = idxSmall.Insert("beta", []float32{5, 6, 7, 8})
	var fullBuf bytes.Buffer
	totalBytes, err := idxSmall.WriteTo(&fullBuf)
	if err != nil {
		t.Fatal(err)
	}
	for cut := 0; cut <= int(totalBytes); cut += 7 {
		w := &failAfterNWriter{limit: cut, err: boom}
		if _, err := idxSmall.WriteTo(w); !errors.Is(err, boom) {
			t.Fatalf("small WriteTo(limit=%d) err = %v, want boom", cut, err)
		}
	}

	// Large (>64KB) index: triggers mid-stream mw.Write failures before bw.Flush.
	idxLarge, err := New[int](64, WithM(8), WithMmax(16), WithEfConstruction(32), WithSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 400; i++ {
		v := make([]float32, 64)
		v[i%64] = float32(i + 1)
		if err := idxLarge.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}
	fullBuf.Reset()
	largeBytes, err := idxLarge.WriteTo(&fullBuf)
	if err != nil {
		t.Fatal(err)
	}
	if largeBytes <= serBufSize {
		t.Fatalf("expected large serialized size > %d, got %d", serBufSize, largeBytes)
	}
	for _, cut := range []int{0, 10, 50, 70, 500, serBufSize, 70000, int(largeBytes) - 4} {
		w := &failAfterNWriter{limit: cut, err: boom}
		if _, err := idxLarge.WriteTo(w); !errors.Is(err, boom) {
			t.Fatalf("large WriteTo(limit=%d) err = %v, want boom", cut, err)
		}
	}

	// Direct writeKeys failure across every key codec.
	checkWriteKeysErr := func(name string, fn func(io.Writer) error) {
		t.Helper()
		if err := fn(&failAfterNWriter{limit: 0, err: boom}); !errors.Is(err, boom) {
			t.Errorf("writeKeys(%s) at 0 = %v, want boom", name, err)
		}
		if err := fn(&failAfterNWriter{limit: 4, err: boom}); !errors.Is(err, boom) {
			t.Errorf("writeKeys(%s) at 4 = %v, want boom", name, err)
		}
	}
	checkWriteKeysErr("int", func(w io.Writer) error {
		ix, _ := New[int](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		return ix.writeKeys(w, []uint32{0, 1}, keyKindInt)
	})
	checkWriteKeysErr("int64", func(w io.Writer) error {
		ix, _ := New[int64](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		return ix.writeKeys(w, []uint32{0, 1}, keyKindInt64)
	})
	checkWriteKeysErr("int32", func(w io.Writer) error {
		ix, _ := New[int32](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		return ix.writeKeys(w, []uint32{0, 1}, keyKindInt32)
	})
	checkWriteKeysErr("uint64", func(w io.Writer) error {
		ix, _ := New[uint64](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		return ix.writeKeys(w, []uint32{0, 1}, keyKindUint64)
	})
	checkWriteKeysErr("uint32", func(w io.Writer) error {
		ix, _ := New[uint32](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		return ix.writeKeys(w, []uint32{0, 1}, keyKindUint32)
	})
	checkWriteKeysErr("uint", func(w io.Writer) error {
		ix, _ := New[uint](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		return ix.writeKeys(w, []uint32{0, 1}, keyKindUint)
	})
	checkWriteKeysErr("int16", func(w io.Writer) error {
		ix, _ := New[int16](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		_ = ix.Insert(3, []float32{1, 1})
		return ix.writeKeys(w, []uint32{0, 1, 2}, keyKindInt16)
	})
	checkWriteKeysErr("uint16", func(w io.Writer) error {
		ix, _ := New[uint16](2, WithSeed(1))
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		_ = ix.Insert(3, []float32{1, 1})
		return ix.writeKeys(w, []uint32{0, 1, 2}, keyKindUint16)
	})
	checkWriteKeysErr("int8", func(w io.Writer) error {
		ix, _ := New[int8](2, WithSeed(1))
		for i := int8(0); i < 6; i++ {
			_ = ix.Insert(i, []float32{float32(i), 0})
		}
		return ix.writeKeys(w, []uint32{0, 1, 2, 3, 4, 5}, keyKindInt8)
	})
	checkWriteKeysErr("uint8", func(w io.Writer) error {
		ix, _ := New[uint8](2, WithSeed(1))
		for i := uint8(0); i < 6; i++ {
			_ = ix.Insert(i, []float32{float32(i), 0})
		}
		return ix.writeKeys(w, []uint32{0, 1, 2, 3, 4, 5}, keyKindUint8)
	})
	checkWriteKeysErr("string", func(w io.Writer) error {
		ix, _ := New[string](2, WithSeed(1))
		_ = ix.Insert("hello", []float32{1, 0})
		return ix.writeKeys(w, []uint32{0}, keyKindString)
	})
	checkWriteKeysErr("bytes16", func(w io.Writer) error {
		ix, _ := New[[16]byte](2, WithSeed(1))
		_ = ix.Insert([16]byte{1}, []float32{1, 0})
		return ix.writeKeys(w, []uint32{0}, keyKindBytes16)
	})
	checkWriteKeysErr("gob", func(w io.Writer) error {
		ix, _ := New[tier5GobKey](2, WithSeed(1))
		_ = ix.Insert(tier5GobKey{Tag: "a", Seq: 1}, []float32{1, 0})
		return ix.writeKeys(w, []uint32{0}, keyKindGob)
	})

	// SaveFile onto an existing directory triggers os.Rename error + temp cleanup.
	parentDir := t.TempDir()
	targetDir := filepath.Join(parentDir, "existing_dir")
	if err := os.Mkdir(targetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := idxSmall.SaveFile(targetDir); err == nil {
		t.Fatal("expected SaveFile onto existing directory to fail")
	}
	entries, err := os.ReadDir(parentDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".hnsw-tmp-") {
			t.Fatalf("leaked temp file %q after SaveFile rename failure", e.Name())
		}
	}

	// SaveFile into non-existent parent dir + LoadFile on directory.
	if err := idxSmall.SaveFile(filepath.Join(parentDir, "no_such_dir", "idx.bin")); err == nil {
		t.Fatal("expected SaveFile into missing dir to fail")
	}
	if _, err := LoadFile[string](targetDir); err == nil {
		t.Fatal("expected LoadFile on directory to fail")
	}

	// wrapCorrupt guards.
	if wrapCorrupt(nil) != nil {
		t.Fatal("wrapCorrupt(nil) != nil")
	}
	if got := wrapCorrupt(ErrCorruptedIndex); !errors.Is(got, ErrCorruptedIndex) || got != ErrCorruptedIndex {
		t.Fatalf("wrapCorrupt(ErrCorruptedIndex) = %v", got)
	}
}

func TestTier5_SerializeReadFromCorruptionsAndChunkedReader(t *testing.T) {
	// 1. Chunked 1-byte reader + options combinations.
	idx, err := New[int](4, WithM(4), WithMmax(8), WithEfConstruction(32), WithSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if err := idx.Insert(i, []float32{float32(i), float32(i % 3), 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	valid := buf.Bytes()

	loaded, err := ReadFrom[int](iotest.OneByteReader(bytes.NewReader(valid)), nil, WithSeed(99), WithEfSearch(77))
	if err != nil {
		t.Fatalf("ReadFrom OneByteReader failed: %v", err)
	}
	if loaded.Len() != 20 || loaded.Stats().EfSearch != 77 {
		t.Fatalf("unexpected loaded stats: %+v", loaded.Stats())
	}
	if _, err := ReadFrom[int](bytes.NewReader(valid), WithM(1)); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("ReadFrom WithM(1) = %v, want ErrInvalidConfig", err)
	}
	if _, err := ReadFrom[int](iotest.ErrReader(errors.New("io fail"))); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("ReadFrom ErrReader = %v, want ErrCorruptedIndex", err)
	}
	if _, err := ReadFrom[int](bytes.NewReader(valid), WithMetric(MetricEuclidean), WithCustomDistance(func(_, _ []float32) float32 { return 0 })); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("ReadFrom WithMetric+WithCustomDistance = %v, want ErrInvalidConfig", err)
	}

	// 2. Arena directory limit (dim <= 31) and payload limit.
	overArena := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint32(overArena[8:12], 4)
	binary.LittleEndian.PutUint32(overArena[40:44], uint32(dirL1Size*dirL2Size*chunkSize+1))
	if _, err := ReadFrom[int](bytes.NewReader(overArena)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("liveCount > arena limit = %v, want ErrCorruptedIndex", err)
	}

	overPayload := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint32(overPayload[8:12], 1024)
	binary.LittleEndian.PutUint32(overPayload[40:44], 3000000)
	if _, err := ReadFrom[int](bytes.NewReader(overPayload)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("liveCount*dim > MaxInt32 = %v, want ErrCorruptedIndex", err)
	}

	// 3. KeyTypeLen > maxSerKeyTypeLen & truncated keyTypeName.
	badKeyTypeLen := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint16(badKeyTypeLen[53:55], maxSerKeyTypeLen+1)
	if _, err := ReadFrom[int](bytes.NewReader(badKeyTypeLen)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("keyTypeLen > max = %v, want ErrCorruptedIndex", err)
	}
	truncKeyType := append([]byte(nil), valid[:57]...)
	binary.LittleEndian.PutUint16(truncKeyType[53:55], 100)
	if _, err := ReadFrom[int](bytes.NewReader(truncKeyType)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("truncated keyTypeName = %v, want ErrCorruptedIndex", err)
	}

	// 4. 70KB string key round-trip (> serBufSize) + string corruptions.
	bigKey := strings.Repeat("k", 70*1024)
	sIdx, _ := New[string](2, WithSeed(1))
	_ = sIdx.Insert(bigKey, []float32{1, 2})
	var sBuf bytes.Buffer
	if _, err := sIdx.WriteTo(&sBuf); err != nil {
		t.Fatal(err)
	}
	sValid := sBuf.Bytes()
	sLoaded, err := ReadFrom[string](bytes.NewReader(sValid))
	if err != nil || !sLoaded.Contains(bigKey) {
		t.Fatalf("70KB string key round-trip failed: err=%v contains=%v", err, sLoaded != nil && sLoaded.Contains(bigKey))
	}
	strKeyOff := 48 + 7 + len("string")
	// Truncate mid-70KB string payload.
	if _, err := ReadFrom[string](bytes.NewReader(sValid[:strKeyOff+10000])); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("truncated 70KB string = %v, want ErrCorruptedIndex", err)
	}
	// slen > maxSerStringKeyLen.
	tooLongStr := append([]byte(nil), sValid...)
	binary.LittleEndian.PutUint32(tooLongStr[strKeyOff:strKeyOff+4], maxSerStringKeyLen+1)
	if _, err := ReadFrom[string](bytes.NewReader(tooLongStr)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("slen > maxSerStringKeyLen = %v, want ErrCorruptedIndex", err)
	}

	// 5. keyKindGob corruptions (payloadLen > maxSerGobLen, truncated len/payload, invalid gob, wrong count).
	gIdx, _ := New[tier5GobKey](2, WithSeed(1))
	_ = gIdx.Insert(tier5GobKey{Tag: "a", Seq: 1}, []float32{1, 0})
	_ = gIdx.Insert(tier5GobKey{Tag: "b", Seq: 2}, []float32{0, 1})
	var gBuf bytes.Buffer
	if _, err := gIdx.WriteTo(&gBuf); err != nil {
		t.Fatal(err)
	}
	gValid := gBuf.Bytes()
	_, gTypeName := keyCodecFor[tier5GobKey]()
	gKeyOff := 48 + 7 + len(gTypeName)

	if _, err := ReadFrom[tier5GobKey](bytes.NewReader(gValid[:gKeyOff+4])); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("truncated gob len = %v, want ErrCorruptedIndex", err)
	}
	gobTooBig := append([]byte(nil), gValid...)
	binary.LittleEndian.PutUint64(gobTooBig[gKeyOff:gKeyOff+8], maxSerGobLen+1)
	if _, err := ReadFrom[tier5GobKey](bytes.NewReader(gobTooBig)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("gob payloadLen > max = %v, want ErrCorruptedIndex", err)
	}
	if _, err := ReadFrom[tier5GobKey](bytes.NewReader(gValid[:gKeyOff+12])); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("truncated gob payload = %v, want ErrCorruptedIndex", err)
	}
	gobBadBytes := append([]byte(nil), gValid...)
	gobBadBytes[gKeyOff+10] ^= 0xFF
	gobBadBytes[gKeyOff+11] ^= 0xFF
	if _, err := ReadFrom[tier5GobKey](bytes.NewReader(gobBadBytes)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("invalid gob bytes = %v, want ErrCorruptedIndex", err)
	}
	gobWrongCnt := append([]byte(nil), gValid...)
	binary.LittleEndian.PutUint32(gobWrongCnt[40:44], 1) // header liveCount=1 while gob encodes 2 keys
	if _, err := ReadFrom[tier5GobKey](bytes.NewReader(gobWrongCnt)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("gob wrong key count = %v, want ErrCorruptedIndex", err)
	}

	// 6. Truncated readKeys for every built-in codec.
	checkTruncReadKeys := func(name string, serialize func() []byte, readFn func([]byte) error) {
		t.Helper()
		raw := serialize()
		// Cut right inside the key section (after 48 + 7 + keyTypeLen + 1 byte).
		typeLen := int(binary.LittleEndian.Uint16(raw[53:55]))
		cut := 48 + 7 + typeLen + 1
		if name == "int8" || name == "uint8" {
			// 2 keys of 1 byte -> cut after first key (1 byte) so 2nd key hits EOF.
			cut = 48 + 7 + typeLen + 1
		}
		if err := readFn(raw[:cut]); !errors.Is(err, ErrCorruptedIndex) {
			t.Errorf("truncated readKeys(%s) = %v, want ErrCorruptedIndex", name, err)
		}
	}
	checkTruncReadKeys("int", func() []byte {
		ix, _ := New[int](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[int](bytes.NewReader(b)); return e })
	checkTruncReadKeys("int64", func() []byte {
		ix, _ := New[int64](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[int64](bytes.NewReader(b)); return e })
	checkTruncReadKeys("int32", func() []byte {
		ix, _ := New[int32](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[int32](bytes.NewReader(b)); return e })
	checkTruncReadKeys("uint64", func() []byte {
		ix, _ := New[uint64](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[uint64](bytes.NewReader(b)); return e })
	checkTruncReadKeys("uint32", func() []byte {
		ix, _ := New[uint32](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[uint32](bytes.NewReader(b)); return e })
	checkTruncReadKeys("uint", func() []byte {
		ix, _ := New[uint](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[uint](bytes.NewReader(b)); return e })
	checkTruncReadKeys("int16", func() []byte {
		ix, _ := New[int16](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[int16](bytes.NewReader(b)); return e })
	checkTruncReadKeys("uint16", func() []byte {
		ix, _ := New[uint16](2)
		_ = ix.Insert(1, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[uint16](bytes.NewReader(b)); return e })
	checkTruncReadKeys("int8", func() []byte {
		ix, _ := New[int8](2)
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[int8](bytes.NewReader(b)); return e })
	checkTruncReadKeys("uint8", func() []byte {
		ix, _ := New[uint8](2)
		_ = ix.Insert(1, []float32{1, 0})
		_ = ix.Insert(2, []float32{0, 1})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[uint8](bytes.NewReader(b)); return e })
	checkTruncReadKeys("bytes16", func() []byte {
		ix, _ := New[[16]byte](2)
		_ = ix.Insert([16]byte{1}, []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[[16]byte](bytes.NewReader(b)); return e })
	checkTruncReadKeys("string", func() []byte {
		ix, _ := New[string](2)
		_ = ix.Insert("ab", []float32{1, 0})
		var b bytes.Buffer
		_, _ = ix.WriteTo(&b)
		return b.Bytes()
	}, func(b []byte) error { _, e := ReadFrom[string](bytes.NewReader(b)); return e })

	// 7. Synthetic 2-level stream testing Inf floats, layer-0 & upper-layer corruptions, and entry point mismatch.
	buildTwoLevelStream := func() []byte {
		var payload bytes.Buffer
		var hdr [48]byte
		copy(hdr[0:4], serMagic[:])
		binary.LittleEndian.PutUint32(hdr[4:8], serVersion)
		binary.LittleEndian.PutUint32(hdr[8:12], 2) // dim=2
		hdr[12] = uint8(MetricEuclidean)
		hdr[13] = 1                                  // heuristic=true
		binary.LittleEndian.PutUint32(hdr[16:20], 2) // m=2
		binary.LittleEndian.PutUint32(hdr[20:24], 4) // mmax=4
		binary.LittleEndian.PutUint32(hdr[24:28], 16)
		binary.LittleEndian.PutUint32(hdr[28:32], 16)
		binary.LittleEndian.PutUint64(hdr[32:40], 1)
		binary.LittleEndian.PutUint32(hdr[40:44], 4) // liveCount=4
		binary.LittleEndian.PutUint32(hdr[44:48], 1) // maxLvl=1
		payload.Write(hdr[:])

		var tail [7]byte
		binary.LittleEndian.PutUint32(tail[0:4], 0) // entryPt=0
		tail[4] = keyKindUint32
		binary.LittleEndian.PutUint16(tail[5:7], uint16(len("uint32")))
		payload.Write(tail[:])
		payload.WriteString("uint32")

		// 4 uint32 keys: 10, 20, 30, 40
		var u32 [4]byte
		for i := uint32(1); i <= 4; i++ {
			binary.LittleEndian.PutUint32(u32[:], i*10)
			payload.Write(u32[:])
		}
		// 4 vectors of dim=2
		for i := 0; i < 4; i++ {
			binary.LittleEndian.PutUint32(u32[:], math.Float32bits(float32(i)))
			payload.Write(u32[:])
			binary.LittleEndian.PutUint32(u32[:], 0)
			payload.Write(u32[:])
		}
		// Graph adjacency for 4 nodes:
		// Node 0: numLevels=2, lvl0=[1], lvl1=[1]
		payload.WriteByte(2)
		binary.LittleEndian.PutUint32(u32[:], 1)
		payload.Write(u32[:])
		binary.LittleEndian.PutUint32(u32[:], 1)
		payload.Write(u32[:])
		binary.LittleEndian.PutUint32(u32[:], 1)
		payload.Write(u32[:])
		binary.LittleEndian.PutUint32(u32[:], 1)
		payload.Write(u32[:])
		// Node 1: numLevels=2, lvl0=[0], lvl1=[0]
		payload.WriteByte(2)
		binary.LittleEndian.PutUint32(u32[:], 1)
		payload.Write(u32[:])
		binary.LittleEndian.PutUint32(u32[:], 0)
		payload.Write(u32[:])
		binary.LittleEndian.PutUint32(u32[:], 1)
		payload.Write(u32[:])
		binary.LittleEndian.PutUint32(u32[:], 0)
		payload.Write(u32[:])
		// Nodes 2 and 3: numLevels=1, lvl0=[0]
		for _, nb := range []uint32{0, 0} {
			payload.WriteByte(1)
			binary.LittleEndian.PutUint32(u32[:], 1)
			payload.Write(u32[:])
			binary.LittleEndian.PutUint32(u32[:], nb)
			payload.Write(u32[:])
		}

		raw := payload.Bytes()
		binary.LittleEndian.PutUint32(u32[:], crc32.ChecksumIEEE(raw))
		raw = append(raw, u32[:]...)
		return raw
	}

	synValid := buildTwoLevelStream()
	if synLoaded, err := ReadFrom[uint32](bytes.NewReader(synValid)); err != nil || synLoaded.Len() != 4 {
		t.Fatalf("synthetic valid stream failed: err=%v len=%v", err, synLoaded)
	}

	// Offsets in synthetic stream:
	synVecOff := 48 + 7 + len("uint32") + 4*4
	synGraphOff := synVecOff + 4*2*4

	// +Inf float in vector.
	infVec := append([]byte(nil), synValid...)
	binary.LittleEndian.PutUint32(infVec[synVecOff:synVecOff+4], math.Float32bits(float32(math.Inf(1))))
	if _, err := ReadFrom[uint32](bytes.NewReader(infVec)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("+Inf vector = %v, want ErrCorruptedIndex", err)
	}

	// Layer-0 self-loop (node 0 lvl0 neighbor = 0).
	l0Self := append([]byte(nil), synValid...)
	binary.LittleEndian.PutUint32(l0Self[synGraphOff+5:synGraphOff+9], 0)
	if _, err := ReadFrom[uint32](bytes.NewReader(l0Self)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("l0 self-loop = %v, want ErrCorruptedIndex", err)
	}

	// Upper-layer deg > M (node 0 lvl1 deg = 3 > M=2, still < liveCount=4).
	upperDegOOB := append([]byte(nil), synValid...)
	binary.LittleEndian.PutUint32(upperDegOOB[synGraphOff+9:synGraphOff+13], 3)
	if _, err := ReadFrom[uint32](bytes.NewReader(upperDegOOB)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("upper deg > M = %v, want ErrCorruptedIndex", err)
	}

	// Upper-layer nb >= liveCount (node 0 lvl1 nb = 99).
	upperNbOOB := append([]byte(nil), synValid...)
	binary.LittleEndian.PutUint32(upperNbOOB[synGraphOff+13:synGraphOff+17], 99)
	if _, err := ReadFrom[uint32](bytes.NewReader(upperNbOOB)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("upper nb >= liveCount = %v, want ErrCorruptedIndex", err)
	}

	// Upper-layer self-loop (node 0 lvl1 nb = 0).
	upperSelf := append([]byte(nil), synValid...)
	binary.LittleEndian.PutUint32(upperSelf[synGraphOff+13:synGraphOff+17], 0)
	if _, err := ReadFrom[uint32](bytes.NewReader(upperSelf)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("upper self-loop = %v, want ErrCorruptedIndex", err)
	}

	// Entry point level mismatch (set entryPt = 2 which has maxLevel = 0 while maxLvl = 1).
	epMismatch := append([]byte(nil), synValid...)
	binary.LittleEndian.PutUint32(epMismatch[48:52], 2)
	binary.LittleEndian.PutUint32(epMismatch[len(epMismatch)-4:], crc32.ChecksumIEEE(epMismatch[:len(epMismatch)-4]))
	if _, err := ReadFrom[uint32](bytes.NewReader(epMismatch)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("entry point level mismatch = %v, want ErrCorruptedIndex", err)
	}

	// Truncated CRC32 trailer.
	if _, err := ReadFrom[uint32](bytes.NewReader(synValid[:len(synValid)-2])); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("truncated CRC32 = %v, want ErrCorruptedIndex", err)
	}

	// 8. 500-iteration deterministic mutation/truncation fuzz loop asserting zero panics.
	rng := rand.New(rand.NewSource(202611))
	for iter := 0; iter < 500; iter++ {
		mutated := append([]byte(nil), synValid...)
		if iter%3 == 0 {
			cut := rng.Intn(len(mutated))
			mutated = mutated[:cut]
		} else {
			flips := 1 + rng.Intn(4)
			for f := 0; f < flips; f++ {
				pos := rng.Intn(len(mutated))
				mutated[pos] ^= byte(1 + rng.Intn(255))
			}
			if iter%5 == 0 && len(mutated) > 4 {
				// Also test mutations with valid trailer CRC to probe deeper branches.
				binary.LittleEndian.PutUint32(mutated[len(mutated)-4:], crc32.ChecksumIEEE(mutated[:len(mutated)-4]))
			}
		}
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("fuzz iter %d panicked: %v", iter, r)
				}
			}()
			_, err := ReadFrom[uint32](bytes.NewReader(mutated))
			if err != nil && !errors.Is(err, ErrCorruptedIndex) && !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("fuzz iter %d unexpected err = %v", iter, err)
			}
		}()
	}
}

func TestTier5_BatchAndSearchMidContextCancellation(t *testing.T) {
	items := make([]Item[int], 200)
	for i := range items {
		items[i] = Item[int]{ID: i, Vector: []float32{float32(i), float32(i % 5), 0, 0}}
	}

	for _, workers := range []int{1, 4} {
		t.Run(fmt.Sprintf("InsertBatch_workers=%d", workers), func(t *testing.T) {
			idx, err := New[int](4, WithM(4), WithMmax(8), WithSeed(11))
			if err != nil {
				t.Fatal(err)
			}
			ctx := newCountdownContext(3)
			err = idx.InsertBatch(items, WithWorkers(workers), WithContext(ctx))
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("InsertBatch(workers=%d) err = %v, want context.Canceled", workers, err)
			}
			if idx.Len() >= len(items) {
				t.Fatalf("expected partial InsertBatch on cancel, got Len=%d", idx.Len())
			}
		})
	}

	idx, err := New[int](4, WithM(4), WithMmax(8), WithSeed(12))
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.InsertBatch(items); err != nil {
		t.Fatal(err)
	}

	queries := make([][]float32, 60)
	for i := range queries {
		queries[i] = []float32{float32(i), 0, 0, 0}
	}

	for _, workers := range []int{1, 4} {
		t.Run(fmt.Sprintf("SearchBatch_workers=%d", workers), func(t *testing.T) {
			ctx := newCountdownContext(4)
			_, err := idx.SearchBatch(queries, 5, WithWorkers(workers), WithContext(ctx))
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("SearchBatch(workers=%d) err = %v, want context.Canceled", workers, err)
			}
		})
	}

	// DeleteBatch edge cases: 0 removals on non-empty index, <=8 removals, >8 removals, delete-all.
	if removed := idx.DeleteBatch([]int{10000, 10001}); removed != 0 {
		t.Fatalf("DeleteBatch missing IDs = %d, want 0", removed)
	}
	if removed := idx.DeleteBatch([]int{0, 1, 2}); removed != 3 {
		t.Fatalf("DeleteBatch(<=8) = %d, want 3", removed)
	}
	bulk := make([]int, 20)
	for i := range bulk {
		bulk[i] = 10 + i
	}
	if removed := idx.DeleteBatch(bulk); removed != 20 {
		t.Fatalf("DeleteBatch(>8) = %d, want 20", removed)
	}

	// SearchFilter returning < k matches with small WithSearchEf.
	res, err := idx.SearchFilter([]float32{100, 0, 0, 0}, 10, func(id int) bool {
		return id == 150 || id == 151
	}, WithSearchEf(5))
	if err != nil {
		t.Fatalf("SearchFilter sparse failed: %v", err)
	}
	if len(res) > 2 {
		t.Fatalf("SearchFilter returned %d > 2 matching items: %+v", len(res), res)
	}

	// Delete all remaining items via DeleteBatch.
	all := make([]int, len(items))
	for i := range all {
		all[i] = i
	}
	if removed := idx.DeleteBatch(all); removed != len(items)-23 {
		t.Fatalf("DeleteBatch remaining = %d, want %d", removed, len(items)-23)
	}
	if idx.Len() != 0 || idx.Stats().EntryPointSet {
		t.Fatalf("expected empty index after DeleteBatch all: %+v", idx.Stats())
	}

	// Duplicate keys in InsertBatch (last-write-wins deduplication) and SearchBatch validation errors.
	if err := idx.InsertBatch(nil); err != nil {
		t.Fatalf("InsertBatch(nil) = %v", err)
	}
	if err := idx.InsertBatch([]Item[int]{
		{ID: 7, Vector: []float32{1, 0, 0, 0}},
		{ID: 7, Vector: []float32{9, 9, 9, 9}},
	}); err != nil {
		t.Fatalf("InsertBatch duplicate IDs failed: %v", err)
	}
	if gotV, ok := idx.Get(7); !ok || gotV[0] != 9 {
		t.Fatalf("last-write-wins Get(7) = (%v, %v), want [9 9 9 9]", gotV, ok)
	}
	if resEmpty, err := idx.SearchBatch(nil, 5); err != nil || len(resEmpty) != 0 {
		t.Fatalf("SearchBatch(nil) = (%v, %v)", resEmpty, err)
	}
	if _, err := idx.SearchBatch(queries[:1], 0); !errors.Is(err, ErrInvalidK) {
		t.Errorf("SearchBatch(k=0) = %v, want ErrInvalidK", err)
	}
	if _, err := idx.SearchBatch([][]float32{{1, 2}}, 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("SearchBatch(dim mismatch) = %v, want ErrDimensionMismatch", err)
	}
	if _, err := idx.SearchBatch([][]float32{{1, float32(math.NaN()), 0, 0}}, 1); !errors.Is(err, ErrNaNOrInf) {
		t.Errorf("SearchBatch(NaN) = %v, want ErrNaNOrInf", err)
	}
	if _, err := idx.SearchBatch(queries[:1], 1, WithWorkers(0)); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("SearchBatch(WithWorkers(0)) = %v, want ErrInvalidConfig", err)
	}
}

func TestTier5_CompactChunkShrinkAndCosineRoundTrip(t *testing.T) {
	idx, err := New[int](8, WithM(8), WithMmax(16), WithEfConstruction(64), WithMetric(MetricCosine), WithSeed(31))
	if err != nil {
		t.Fatal(err)
	}

	origVecs := make(map[int][]float32, 600)
	makeVec := func(seedVal int) []float32 {
		angle := float64(seedVal) * 0.37
		return []float32{
			float32(math.Cos(angle)) * float32(1+(seedVal%5)),
			float32(math.Sin(angle)) * float32(1+(seedVal%7)),
			float32(math.Cos(angle*1.7)) * 2.0,
			float32(math.Sin(angle*2.3)) * 3.0,
			float32((seedVal%11)-5) * 0.5,
			float32((seedVal%13)-6) * 0.4,
			1.5,
			0.75,
		}
	}
	for i := 0; i < 600; i++ {
		v := makeVec(i)
		origVecs[i] = v
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}
	if idx.Stats().Capacity != 768 {
		t.Fatalf("initial Capacity = %d, want 768", idx.Stats().Capacity)
	}

	// Upsert 50 vectors (creates 50 tombstones).
	for i := 0; i < 50; i++ {
		v := makeVec(i + 1000)
		origVecs[i] = v
		if err := idx.Insert(i, v); err != nil {
			t.Fatal(err)
		}
	}

	// Delete 450 vectors -> 150 live remaining (< 256 chunkSize).
	for i := 150; i < 600; i++ {
		if !idx.Delete(i) {
			t.Fatalf("Delete(%d) returned false", i)
		}
	}
	if err := idx.Compact(); err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	st := idx.Stats()
	if st.Capacity != 256 || st.Len != 150 || st.Deleted != 0 {
		t.Fatalf("post-Compact stats = %+v, want Capacity=256 Len=150 Deleted=0", st)
	}

	// Verify lossless unnormalized vector preservation and persistence round-trip.
	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	reloaded, err := ReadFrom[int](&buf)
	if err != nil {
		t.Fatal(err)
	}
	got0, ok := reloaded.Get(0)
	if !ok {
		t.Fatal("reloaded.Get(0) returned false")
	}
	for j, want := range origVecs[0] {
		if got0[j] != want {
			t.Fatalf("reloaded.Get(0)[%d] = %v, want lossless %v", j, got0[j], want)
		}
	}
	res, err := reloaded.Search(origVecs[0], 1)
	if err != nil || len(res) != 1 || res[0].ID != 0 || res[0].Distance > 1e-5 {
		t.Fatalf("reloaded Search(origVecs[0]) = (%+v, %v)", res, err)
	}
}
