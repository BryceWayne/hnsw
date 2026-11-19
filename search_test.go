package hnsw

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
)

func BenchmarkSearch(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	idx, err := New[int](128, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}

	items := make([]Item[int], 1000)
	for i := range items {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: vec}
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Search(query, 10); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
		b.ReportMetric(float64(b.N)/elapsed, "queries/s")
	}
}

func BenchmarkParallelSearch(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	idx, err := New[int](128, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}

	items := make([]Item[int], 1000)
	for i := range items {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: vec}
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := idx.Search(query, 10); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.StopTimer()
	if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
		b.ReportMetric(float64(b.N)/elapsed, "queries/s")
	}
}

func BenchmarkSearchComparison(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	sizes := []int{1000, 10000}
	dimensions := []int{32, 128, 512}

	for _, size := range sizes {
		for _, dim := range dimensions {
			b.Run(fmt.Sprintf("Size_%d_Dim_%d", size, dim), func(b *testing.B) {
				idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
				if err != nil {
					b.Fatal(err)
				}

				items := make([]Item[int], size)
				for i := 0; i < size; i++ {
					vec := make([]float32, dim)
					for j := range vec {
						vec[j] = rng.Float32()
					}
					items[i] = Item[int]{ID: i, Vector: vec}
				}
				if err := idx.InsertBatch(items); err != nil {
					b.Fatal(err)
				}

				query := make([]float32, dim)
				for i := range query {
					query[i] = rng.Float32()
				}

				b.ReportAllocs()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						if _, err := idx.Search(query, 10); err != nil {
							b.Fatal(err)
						}
					}
				})
				b.StopTimer()
				if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
					b.ReportMetric(float64(b.N)/elapsed, "queries/s")
				}
			})
		}
	}
}

func BenchmarkParallelSearchScaling(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	threads := []int{1, 2, 4, 8, 16}
	dim := 256
	size := 10000

	idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(64), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}
	items := make([]Item[int], size)
	for i := range items {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: vec}
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}

	queries := make([][]float32, 64)
	for i := range queries {
		queries[i] = make([]float32, dim)
		for j := range queries[i] {
			queries[i][j] = rng.Float32()
		}
	}
	origProcs := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(origProcs)

	for _, thread := range threads {
		b.Run(fmt.Sprintf("threads=%d", thread), func(b *testing.B) {
			runtime.GOMAXPROCS(thread)
			defer runtime.GOMAXPROCS(origProcs)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := idx.SearchBatch(queries, 10, WithWorkers(thread)); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
				b.ReportMetric(float64(b.N*len(queries))/elapsed, "queries/s")
			}
		})
	}
}
