package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	hnsw "github.com/BryceWayne/hnsw"
	coderhnsw "github.com/coder/hnsw"
	fogfish "github.com/fogfish/hnsw"
	fvec "github.com/fogfish/hnsw/vector"
	kvec "github.com/kshard/vector"
)

const (
	benchM              = 16
	benchMmax           = 32
	benchEfConstruction = 200
	benchEfSearch       = 64
	benchK              = 10
	benchNumQueries     = 100
)

type dataset struct {
	dim     int
	vectors [][]float32
	queries [][]float32
	truth   [][]int // exact top-k IDs per query (cosine distance)
}

func generateClusteredDataset(n, dim, numQueries int, seed int64) *dataset {
	rng := rand.New(rand.NewSource(seed))
	numClusters := 50
	if n < 500 {
		numClusters = 10
	}
	centers := make([][]float32, numClusters)
	for c := range centers {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalizeInPlace(v)
		centers[c] = v
	}

	// Scale per-coordinate Gaussian noise by 1/sqrt(dim) so angular cluster spread
	// stays dimension-invariant across dim=128, 768, and 1536 (realistic embedding manifold).
	noiseStd := 1.5 / math.Sqrt(float64(dim))

	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		c := centers[i%numClusters]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64()*noiseStd)
		}
		normalizeInPlace(v)
		vectors[i] = v
	}

	queries := make([][]float32, numQueries)
	for q := 0; q < numQueries; q++ {
		c := centers[q%numClusters]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64()*noiseStd)
		}
		normalizeInPlace(v)
		queries[q] = v
	}

	truth := make([][]int, numQueries)
	for q := 0; q < numQueries; q++ {
		truth[q] = bruteForceTopK(vectors, queries[q], benchK)
	}

	return &dataset{
		dim:     dim,
		vectors: vectors,
		queries: queries,
		truth:   truth,
	}
}

func normalizeInPlace(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

type scoredID struct {
	id   int
	dist float32
}

func bruteForceTopK(vectors [][]float32, query []float32, k int) []int {
	scores := make([]scoredID, len(vectors))
	for i, v := range vectors {
		d, _ := hnsw.Cosine(v, query)
		scores[i] = scoredID{id: i, dist: d}
	}
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].dist == scores[j].dist {
			return scores[i].id < scores[j].id
		}
		return scores[i].dist < scores[j].dist
	})
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = scores[i].id
	}
	return out
}

func computeRecall(got, truth []int) float64 {
	if len(truth) == 0 {
		return 1.0
	}
	set := make(map[int]struct{}, len(truth))
	for _, id := range truth {
		set[id] = struct{}{}
	}
	hits := 0
	for _, id := range got {
		if _, ok := set[id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(truth))
}

func buildBryceWayne(ds *dataset, quant hnsw.Quantization) *hnsw.Index[int] {
	idx, err := hnsw.New[int](
		ds.dim,
		hnsw.WithM(benchM),
		hnsw.WithMmax(benchMmax),
		hnsw.WithEfConstruction(benchEfConstruction),
		hnsw.WithEfSearch(benchEfSearch),
		hnsw.WithMetric(hnsw.MetricCosine),
		hnsw.WithQuantization(quant),
		hnsw.WithInitialCapacity(len(ds.vectors)),
		hnsw.WithSeed(42),
	)
	if err != nil {
		panic(err)
	}
	items := make([]hnsw.Item[int], len(ds.vectors))
	for i, v := range ds.vectors {
		items[i] = hnsw.Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items); err != nil {
		panic(err)
	}
	return idx
}

func buildCoder(ds *dataset) *coderhnsw.Graph[int] {
	g := coderhnsw.NewGraph[int]()
	g.M = benchM
	g.Ml = 0.25 // coder/hnsw documented default probability per layer
	// coder/hnsw uses EfSearch as construction beam width during Add.
	g.EfSearch = benchEfConstruction
	g.Distance = coderhnsw.CosineDistance
	g.Rng = rand.New(rand.NewSource(42))
	nodes := make([]coderhnsw.Node[int], len(ds.vectors))
	for i, v := range ds.vectors {
		nodes[i] = coderhnsw.MakeNode(i, v)
	}
	g.Add(nodes...)
	g.EfSearch = benchEfSearch
	return g
}

func buildFogfish(ds *dataset) *fogfish.HNSW[fvec.VF32] {
	surf := fvec.SurfaceVF32(kvec.Cosine())
	idx := fogfish.New[fvec.VF32](
		surf,
		fogfish.WithM(benchM),
		fogfish.WithM0(benchMmax),
		fogfish.WithEfConstruction(benchEfConstruction),
		fogfish.WithRandomSource(rand.NewSource(42)),
	)
	for i, v := range ds.vectors {
		idx.Insert(fvec.VF32{Key: uint32(i), Vec: v})
	}
	return idx
}

func reportPercentiles(b *testing.B, fn func(qIdx int), numQueries int) (p50us, p99us float64) {
	b.Helper()
	const samples = 200
	lat := make([]time.Duration, samples)
	for i := 0; i < samples; i++ {
		t0 := time.Now()
		fn(i % numQueries)
		lat[i] = time.Since(t0)
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	p50us = float64(lat[samples/2].Nanoseconds()) / 1000.0
	p99us = float64(lat[(samples*99)/100].Nanoseconds()) / 1000.0
	b.ReportMetric(p50us, "p50_us")
	b.ReportMetric(p99us, "p99_us")
	return p50us, p99us
}

// TestComparisonSanity verifies all compared engines build cleanly and achieve high Recall@10.
func TestComparisonSanity(t *testing.T) {
	ds := generateClusteredDataset(2000, 128, 50, 42)

	bwF32 := buildBryceWayne(ds, hnsw.QuantNone)
	bwF16 := buildBryceWayne(ds, hnsw.QuantFloat16)
	cg := buildCoder(ds)
	fg := buildFogfish(ds)

	eval := func(name string, minRecall float64, search func(q []float32) []int) {
		t.Helper()
		var sum float64
		for i, q := range ds.queries {
			sum += computeRecall(search(q), ds.truth[i])
		}
		recall := sum / float64(len(ds.queries))
		t.Logf("%-24s Recall@%d = %.4f", name, benchK, recall)
		if recall < minRecall {
			t.Errorf("%s recall@%d = %.4f, want >= %.2f", name, benchK, recall, minRecall)
		}
	}

	eval("BryceWayne/hnsw (f32)", 0.95, func(q []float32) []int {
		res, err := bwF32.Search(q, benchK)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int, len(res))
		for i, r := range res {
			out[i] = r.ID
		}
		return out
	})
	eval("BryceWayne/hnsw (fp16)", 0.95, func(q []float32) []int {
		res, err := bwF16.Search(q, benchK)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int, len(res))
		for i, r := range res {
			out[i] = r.ID
		}
		return out
	})
	// coder/hnsw v0.6.1 stops layer-0 search early whenever a candidate fails to
	// beat result.Min().dist, so we log its actual recall without asserting >=0.95.
	eval("coder/hnsw", 0.0, func(q []float32) []int {
		res := cg.Search(q, benchK)
		out := make([]int, len(res))
		for i, r := range res {
			out[i] = r.Key
		}
		return out
	})
	eval("fogfish/hnsw", 0.90, func(q []float32) []int {
		res := fg.Search(fvec.VF32{Vec: q}, benchK, benchEfSearch)
		out := make([]int, len(res))
		for i, r := range res {
			out[i] = int(r.Key)
		}
		return out
	})
}

func BenchmarkCompareSearch(b *testing.B) {
	configs := []struct {
		name string
		n    int
		dim  int
	}{
		{"10K_dim128", 10000, 128},
		{"10K_dim768", 10000, 768},
		{"5K_dim1536", 5000, 1536},
	}

	for _, cfg := range configs {
		ds := generateClusteredDataset(cfg.n, cfg.dim, benchNumQueries, 42)
		bwF32 := buildBryceWayne(ds, hnsw.QuantNone)
		bwF16 := buildBryceWayne(ds, hnsw.QuantFloat16)
		cg := buildCoder(ds)
		fg := buildFogfish(ds)

		calcRecall := func(search func(q []float32) []int) float64 {
			var sum float64
			for i, q := range ds.queries {
				sum += computeRecall(search(q), ds.truth[i])
			}
			return sum / float64(len(ds.queries))
		}

		bwF32Recall := calcRecall(func(q []float32) []int {
			res, _ := bwF32.Search(q, benchK)
			out := make([]int, len(res))
			for i, r := range res {
				out[i] = r.ID
			}
			return out
		})
		bwF16Recall := calcRecall(func(q []float32) []int {
			res, _ := bwF16.Search(q, benchK)
			out := make([]int, len(res))
			for i, r := range res {
				out[i] = r.ID
			}
			return out
		})
		cgRecall := calcRecall(func(q []float32) []int {
			res := cg.Search(q, benchK)
			out := make([]int, len(res))
			for i, r := range res {
				out[i] = r.Key
			}
			return out
		})
		fgRecall := calcRecall(func(q []float32) []int {
			res := fg.Search(fvec.VF32{Vec: q}, benchK, benchEfSearch)
			out := make([]int, len(res))
			for i, r := range res {
				out[i] = int(r.Key)
			}
			return out
		})

		b.Run(cfg.name+"/1T/BryceWayne_f32_SearchInto", func(b *testing.B) {
			dst := make([]hnsw.SearchResult[int], 0, benchK)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				dst, err = bwF32.SearchInto(ds.queries[i%benchNumQueries], benchK, dst)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(bwF32Recall, "recall@10")
			reportPercentiles(b, func(qi int) {
				dst, _ = bwF32.SearchInto(ds.queries[qi], benchK, dst)
			}, benchNumQueries)
		})

		b.Run(cfg.name+"/1T/BryceWayne_fp16_SearchInto", func(b *testing.B) {
			dst := make([]hnsw.SearchResult[int], 0, benchK)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				dst, err = bwF16.SearchInto(ds.queries[i%benchNumQueries], benchK, dst)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(bwF16Recall, "recall@10")
			reportPercentiles(b, func(qi int) {
				dst, _ = bwF16.SearchInto(ds.queries[qi], benchK, dst)
			}, benchNumQueries)
		})

		b.Run(cfg.name+"/1T/coder_hnsw", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = cg.Search(ds.queries[i%benchNumQueries], benchK)
			}
			b.StopTimer()
			b.ReportMetric(cgRecall, "recall@10")
			reportPercentiles(b, func(qi int) {
				_ = cg.Search(ds.queries[qi], benchK)
			}, benchNumQueries)
		})

		b.Run(cfg.name+"/1T/fogfish_hnsw", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = fg.Search(fvec.VF32{Vec: ds.queries[i%benchNumQueries]}, benchK, benchEfSearch)
			}
			b.StopTimer()
			b.ReportMetric(fgRecall, "recall@10")
			reportPercentiles(b, func(qi int) {
				_ = fg.Search(fvec.VF32{Vec: ds.queries[qi]}, benchK, benchEfSearch)
			}, benchNumQueries)
		})

		b.Run(cfg.name+"/1T/BruteForce_Exact", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = bruteForceTopK(ds.vectors, ds.queries[i%benchNumQueries], benchK)
			}
			b.StopTimer()
			b.ReportMetric(1.0, "recall@10")
		})

		b.Run(cfg.name+"/Parallel/BryceWayne_f32_SearchInto", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				dst := make([]hnsw.SearchResult[int], 0, benchK)
				i := 0
				for pb.Next() {
					var err error
					dst, err = bwF32.SearchInto(ds.queries[i%benchNumQueries], benchK, dst)
					if err != nil {
						b.Fatal(err)
					}
					i++
				}
			})
			b.StopTimer()
			b.ReportMetric(bwF32Recall, "recall@10")
		})

		b.Run(cfg.name+"/Parallel/BryceWayne_fp16_SearchInto", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				dst := make([]hnsw.SearchResult[int], 0, benchK)
				i := 0
				for pb.Next() {
					var err error
					dst, err = bwF16.SearchInto(ds.queries[i%benchNumQueries], benchK, dst)
					if err != nil {
						b.Fatal(err)
					}
					i++
				}
			})
			b.StopTimer()
			b.ReportMetric(bwF16Recall, "recall@10")
		})

		b.Run(cfg.name+"/Parallel/coder_hnsw", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					_ = cg.Search(ds.queries[i%benchNumQueries], benchK)
					i++
				}
			})
			b.StopTimer()
			b.ReportMetric(cgRecall, "recall@10")
		})

		b.Run(cfg.name+"/Parallel/fogfish_hnsw", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					_ = fg.Search(fvec.VF32{Vec: ds.queries[i%benchNumQueries]}, benchK, benchEfSearch)
					i++
				}
			})
			b.StopTimer()
			b.ReportMetric(fgRecall, "recall@10")
		})
	}
}

// BenchmarkCompareMixedReadWrite simulates a live production RAG workload:
// 80% concurrent reads and 20% concurrent inserts across all CPUs.
// Libraries without built-in write concurrency are guarded with sync.RWMutex.
func BenchmarkCompareMixedReadWrite(b *testing.B) {
	ds := generateClusteredDataset(5000, 128, benchNumQueries, 77)
	extra := generateClusteredDataset(2000, 128, 1, 99).vectors

	b.Run("BryceWayne_f32_NativeLocks", func(b *testing.B) {
		idx := buildBryceWayne(ds, hnsw.QuantNone)
		var nextID atomic.Int64
		nextID.Store(int64(len(ds.vectors)))
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			dst := make([]hnsw.SearchResult[int], 0, benchK)
			i := 0
			for pb.Next() {
				if i%5 == 0 {
					id := int(nextID.Add(1))
					if err := idx.Insert(id, extra[id%len(extra)]); err != nil {
						b.Fatal(err)
					}
				} else {
					var err error
					dst, err = idx.SearchInto(ds.queries[i%benchNumQueries], benchK, dst)
					if err != nil {
						b.Fatal(err)
					}
				}
				i++
			}
		})
	})

	b.Run("coder_hnsw_RWMutex", func(b *testing.B) {
		g := buildCoder(ds)
		var mu sync.RWMutex
		var nextID atomic.Int64
		nextID.Store(int64(len(ds.vectors)))
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%5 == 0 {
					id := int(nextID.Add(1))
					mu.Lock()
					g.Add(coderhnsw.MakeNode(id, extra[id%len(extra)]))
					mu.Unlock()
				} else {
					mu.RLock()
					_ = g.Search(ds.queries[i%benchNumQueries], benchK)
					mu.RUnlock()
				}
				i++
			}
		})
	})

	b.Run("fogfish_hnsw_RWMutex", func(b *testing.B) {
		fg := buildFogfish(ds)
		var mu sync.RWMutex
		var nextID atomic.Int64
		nextID.Store(int64(len(ds.vectors)))
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%5 == 0 {
					id := uint32(nextID.Add(1))
					mu.Lock()
					fg.Insert(fvec.VF32{Key: id, Vec: extra[int(id)%len(extra)]})
					mu.Unlock()
				} else {
					mu.RLock()
					_ = fg.Search(fvec.VF32{Vec: ds.queries[i%benchNumQueries]}, benchK, benchEfSearch)
					mu.RUnlock()
				}
				i++
			}
		})
	})
}

func measureHeapMB(build func() any) (buildTime time.Duration, heapMB float64) {
	runtime.GC()
	var m1, m2 runtime.MemStats
	runtime.ReadMemStats(&m1)
	t0 := time.Now()
	keep := build()
	buildTime = time.Since(t0)
	runtime.GC()
	runtime.ReadMemStats(&m2)
	runtime.KeepAlive(keep)
	if m2.HeapAlloc > m1.HeapAlloc {
		heapMB = float64(m2.HeapAlloc-m1.HeapAlloc) / (1024 * 1024)
	}
	return buildTime, heapMB
}

func cloneVectors(src [][]float32) [][]float32 {
	out := make([][]float32, len(src))
	for i, v := range src {
		c := make([]float32, len(v))
		copy(c, v)
		out[i] = c
	}
	return out
}

// TestBuildAndMemoryReport prints a reproducible table of build throughput and total owned
// heap usage (vectors + graph) on 5K x dim=768 vectors across all compared libraries.
func TestBuildAndMemoryReport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping build/memory report in -short mode")
	}
	ds := generateClusteredDataset(5000, 768, 20, 42)
	engines := []struct {
		name  string
		build func() any
	}{
		{"BryceWayne/hnsw (f32, InsertBatch)", func() any { return buildBryceWayne(ds, hnsw.QuantNone) }},
		{"BryceWayne/hnsw (fp16, InsertBatch)", func() any { return buildBryceWayne(ds, hnsw.QuantFloat16) }},
		{"coder/hnsw (Add, cloned vecs)", func() any {
			// coder/hnsw retains caller slice references without copying; clone so total
			// owned memory (vectors + graph) is measured apples-to-apples.
			owned := &dataset{dim: ds.dim, vectors: cloneVectors(ds.vectors)}
			return buildCoder(owned)
		}},
		{"fogfish/hnsw (Insert, cloned vecs)", func() any {
			owned := &dataset{dim: ds.dim, vectors: cloneVectors(ds.vectors)}
			return buildFogfish(owned)
		}},
	}

	t.Logf("\n%-38s | %10s | %12s | %12s", "Engine (5K x dim=768, Cosine)", "Build", "Vecs/sec", "Total Heap MB")
	for _, e := range engines {
		dur, mb := measureHeapMB(e.build)
		vps := float64(len(ds.vectors)) / dur.Seconds()
		t.Logf("%-38s | %10s | %12.0f | %12.2f", e.name, dur.Round(time.Millisecond), vps, mb)
	}
	_ = fmt.Sprintf("")
}
