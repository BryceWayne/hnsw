package hnsw

import (
	"bytes"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func generateClusteredDataset(numVecs, numQueries, dim, numClusters int, seed int64) (vecs, queries [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, numClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for j := range centers[c] {
			centers[c][j] = float32(rng.NormFloat64() * 3.0)
		}
	}

	vecs = make([][]float32, numVecs)
	for i := 0; i < numVecs; i++ {
		v := make([]float32, dim)
		c := centers[i%numClusters]
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64()*0.5)
		}
		vecs[i] = v
	}

	queries = make([][]float32, numQueries)
	for q := 0; q < numQueries; q++ {
		v := make([]float32, dim)
		c := centers[q%numClusters]
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64()*0.5)
		}
		queries[q] = v
	}
	return vecs, queries
}

func computeExactTopK(vecs, queries [][]float32, k int, distFn func(a, b []float32) (float32, error), filter func(id int) bool) ([][]int, error) {
	type idDist struct {
		id   int
		dist float32
	}
	truth := make([][]int, len(queries))
	dists := make([]idDist, 0, len(vecs))
	for qi, q := range queries {
		dists = dists[:0]
		for id, v := range vecs {
			if filter != nil && !filter(id) {
				continue
			}
			d, err := distFn(v, q)
			if err != nil {
				return nil, err
			}
			dists = append(dists, idDist{id: id, dist: d})
		}
		sort.Slice(dists, func(i, j int) bool {
			if dists[i].dist != dists[j].dist {
				return dists[i].dist < dists[j].dist
			}
			return dists[i].id < dists[j].id
		})
		limit := k
		if limit > len(dists) {
			limit = len(dists)
		}
		top := make([]int, limit)
		for i := 0; i < limit; i++ {
			top[i] = dists[i].id
		}
		truth[qi] = top
	}
	return truth, nil
}

func measureIndexRecall(idx *Index[int], queries [][]float32, truth [][]int, k, ef int, filter func(int) bool) (recall1, recallK float64, err error) {
	top1Hits := 0
	topKHits := 0
	for qi, q := range queries {
		var res []SearchResult[int]
		if filter == nil {
			res, err = idx.Search(q, k, WithSearchEf(ef))
		} else {
			res, err = idx.SearchFilter(q, k, filter, WithSearchEf(ef))
		}
		if err != nil {
			return 0, 0, err
		}
		oracle := truth[qi]
		if len(res) > 0 && len(oracle) > 0 && res[0].ID == oracle[0] {
			top1Hits++
		}
		oracleSet := make(map[int]struct{}, len(oracle))
		for _, id := range oracle {
			oracleSet[id] = struct{}{}
		}
		for _, r := range res {
			if _, ok := oracleSet[r.ID]; ok {
				topKHits++
			}
		}
	}
	recall1 = float64(top1Hits) / float64(len(queries))
	recallK = float64(topKHits) / float64(len(queries)*k)
	return recall1, recallK, nil
}

func BenchmarkRecallAndLatency(b *testing.B) {
	const (
		numVecs     = 10000
		numQueries  = 100
		dim         = 128
		numClusters = 32
		k           = 10
	)
	vecs, queries := generateClusteredDataset(numVecs, numQueries, dim, numClusters, 2026)
	items := make([]Item[int], numVecs)
	for i := 0; i < numVecs; i++ {
		items[i] = Item[int]{ID: i, Vector: vecs[i]}
	}

	metrics := []struct {
		name   string
		metric Metric
		distFn func(a, b []float32) (float32, error)
	}{
		{"MetricEuclidean", MetricEuclidean, Euclidean},
		{"MetricCosine", MetricCosine, Cosine},
	}

	efValues := []int{16, 32, 64, 128}

	for _, mc := range metrics {
		b.Run(mc.name, func(b *testing.B) {
			idx, err := New[int](dim,
				WithM(16),
				WithMmax(32),
				WithEfConstruction(200),
				WithMetric(mc.metric),
				WithSeed(2026),
			)
			if err != nil {
				b.Fatal(err)
			}
			if err := idx.InsertBatch(items); err != nil {
				b.Fatal(err)
			}
			truth, err := computeExactTopK(vecs, queries, k, mc.distFn, nil)
			if err != nil {
				b.Fatal(err)
			}

			for _, ef := range efValues {
				b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
					r1, r10, err := measureIndexRecall(idx, queries, truth, k, ef, nil)
					if err != nil {
						b.Fatal(err)
					}

					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err := idx.Search(queries[i%len(queries)], k, WithSearchEf(ef)); err != nil {
							b.Fatal(err)
						}
					}
					b.StopTimer()
					if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
						b.ReportMetric(float64(b.N)/elapsed, "queries/s")
					}
					b.ReportMetric(r1, "recall@1")
					b.ReportMetric(r10, "recall@10")
				})
			}
		})
	}
}

func BenchmarkSearchFilter(b *testing.B) {
	const (
		numVecs     = 10000
		numQueries  = 100
		dim         = 128
		numClusters = 32
		k           = 10
		ef          = 64
	)
	vecs, queries := generateClusteredDataset(numVecs, numQueries, dim, numClusters, 4242)
	items := make([]Item[int], numVecs)
	for i := 0; i < numVecs; i++ {
		items[i] = Item[int]{ID: i, Vector: vecs[i]}
	}

	idx, err := New[int](dim,
		WithM(16),
		WithMmax(32),
		WithEfConstruction(200),
		WithMetric(MetricEuclidean),
		WithSeed(4242),
	)
	if err != nil {
		b.Fatal(err)
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	filters := []struct {
		name string
		fn   func(int) bool
	}{
		{"selectivity=50pct", func(id int) bool { return id%2 == 0 }},
		{"selectivity=10pct", func(id int) bool { return id%10 == 0 }},
	}

	for _, fc := range filters {
		b.Run(fc.name, func(b *testing.B) {
			truth, err := computeExactTopK(vecs, queries, k, Euclidean, fc.fn)
			if err != nil {
				b.Fatal(err)
			}
			_, r10, err := measureIndexRecall(idx, queries, truth, k, ef, fc.fn)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := idx.SearchFilter(queries[i%len(queries)], k, fc.fn, WithSearchEf(ef)); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
				b.ReportMetric(float64(b.N)/elapsed, "queries/s")
			}
			b.ReportMetric(r10, "recall@10")
		})
	}
}

func BenchmarkPersistence(b *testing.B) {
	const (
		numVecs     = 10000
		dim         = 128
		numClusters = 32
	)
	vecs, _ := generateClusteredDataset(numVecs, 1, dim, numClusters, 777)
	items := make([]Item[int], numVecs)
	for i := 0; i < numVecs; i++ {
		items[i] = Item[int]{ID: i, Vector: vecs[i]}
	}

	idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(777))
	if err != nil {
		b.Fatal(err)
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	var serialized bytes.Buffer
	if _, err := idx.WriteTo(&serialized); err != nil {
		b.Fatal(err)
	}
	raw := serialized.Bytes()

	b.Run("WriteTo_10K_dim128", func(b *testing.B) {
		var buf bytes.Buffer
		buf.Grow(len(raw))
		b.SetBytes(int64(len(raw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf.Reset()
			if _, err := idx.WriteTo(&buf); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("ReadFrom_10K_dim128", func(b *testing.B) {
		r := bytes.NewReader(raw)
		b.SetBytes(int64(len(raw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r.Reset(raw)
			if _, err := ReadFrom[int](r); err != nil {
				b.Fatal(err)
			}
		}
	})
}
