package hnsw_test

import (
	"bytes"
	"fmt"
	"log"

	"github.com/BryceWayne/hnsw"
)

func ExampleNew() {
	idx, err := hnsw.New[string](
		3,
		hnsw.WithM(8),
		hnsw.WithMmax(16),
		hnsw.WithEfConstruction(64),
		hnsw.WithEfSearch(32),
		hnsw.WithMetric(hnsw.MetricEuclidean),
		hnsw.WithSeed(42),
	)
	if err != nil {
		log.Fatal(err)
	}

	st := idx.Stats()
	fmt.Printf("dim=%d m=%d mmax=%d metric=%s len=%d\n", idx.Dim(), st.M, st.Mmax, st.Metric, idx.Len())
	// Output:
	// dim=3 m=8 mmax=16 metric=euclidean len=0
}

func ExampleIndex_Search() {
	idx, err := hnsw.New[string](3, hnsw.WithM(8), hnsw.WithEfConstruction(64), hnsw.WithSeed(1))
	if err != nil {
		log.Fatal(err)
	}

	vectors := map[string][]float32{
		"origin": {0, 0, 0},
		"near":   {1, 0, 0},
		"mid":    {0, 2, 0},
		"far":    {0, 0, 5},
	}
	for id, vec := range vectors {
		if err := idx.Insert(id, vec); err != nil {
			log.Fatal(err)
		}
	}

	res, err := idx.Search([]float32{1, 0, 0}, 2, hnsw.WithSearchEf(16))
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range res {
		fmt.Printf("%s %.2f\n", r.ID, r.Distance)
	}
	// Output:
	// near 0.00
	// origin 1.00
}

func ExampleIndex_SearchFilter() {
	idx, err := hnsw.New[int](2, hnsw.WithM(8), hnsw.WithEfConstruction(64), hnsw.WithSeed(1))
	if err != nil {
		log.Fatal(err)
	}

	for id := 1; id <= 4; id++ {
		if err := idx.Insert(id, []float32{float32(id), 0}); err != nil {
			log.Fatal(err)
		}
	}

	res, err := idx.SearchFilter([]float32{0, 0}, 2, func(id int) bool {
		return id%2 == 0
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range res {
		fmt.Printf("%d %.1f\n", r.ID, r.Distance)
	}
	// Output:
	// 2 2.0
	// 4 4.0
}

func ExampleIndex_InsertBatch() {
	idx, err := hnsw.New[string](2, hnsw.WithM(8), hnsw.WithEfConstruction(64), hnsw.WithSeed(7))
	if err != nil {
		log.Fatal(err)
	}

	items := []hnsw.Item[string]{
		{ID: "a", Vector: []float32{0, 0}},
		{ID: "b", Vector: []float32{3, 4}},
		{ID: "c", Vector: []float32{6, 8}},
	}
	if err := idx.InsertBatch(items, hnsw.WithWorkers(2)); err != nil {
		log.Fatal(err)
	}

	queries := [][]float32{{0, 0}, {6, 8}}
	batch, err := idx.SearchBatch(queries, 1, hnsw.WithWorkers(2))
	if err != nil {
		log.Fatal(err)
	}
	for qi, res := range batch {
		fmt.Printf("q%d: %s %.1f\n", qi, res[0].ID, res[0].Distance)
	}
	// Output:
	// q0: a 0.0
	// q1: c 0.0
}

func ExampleIndex_Compact() {
	idx, err := hnsw.New[int](2, hnsw.WithM(8), hnsw.WithEfConstruction(64), hnsw.WithSeed(3))
	if err != nil {
		log.Fatal(err)
	}

	if err := idx.Insert(1, []float32{0, 0}); err != nil {
		log.Fatal(err)
	}
	if err := idx.Insert(2, []float32{1, 0}); err != nil {
		log.Fatal(err)
	}
	if err := idx.Insert(3, []float32{5, 0}); err != nil {
		log.Fatal(err)
	}

	deleted := idx.Delete(2)
	before := idx.Stats()
	if err := idx.Compact(); err != nil {
		log.Fatal(err)
	}
	after := idx.Stats()

	res, err := idx.Search([]float32{1, 0}, 1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("deleted=%v before=(len=%d,del=%d) after=(len=%d,del=%d) nearest=%d\n",
		deleted, before.Len, before.Deleted, after.Len, after.Deleted, res[0].ID)
	// Output:
	// deleted=true before=(len=2,del=1) after=(len=2,del=0) nearest=1
}

func ExampleIndex_WriteTo() {
	idx, err := hnsw.New[string](2, hnsw.WithMetric(hnsw.MetricSquaredEuclidean), hnsw.WithSeed(5))
	if err != nil {
		log.Fatal(err)
	}
	if err := idx.Insert("p1", []float32{1, 2}); err != nil {
		log.Fatal(err)
	}
	if err := idx.Insert("p2", []float32{4, 6}); err != nil {
		log.Fatal(err)
	}

	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		log.Fatal(err)
	}

	reloaded, err := hnsw.ReadFrom[string](&buf)
	if err != nil {
		log.Fatal(err)
	}
	res, err := reloaded.Search([]float32{1, 2}, 2)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("len=%d top=%s:%.1f second=%s:%.1f\n",
		reloaded.Len(), res[0].ID, res[0].Distance, res[1].ID, res[1].Distance)
	// Output:
	// len=2 top=p1:0.0 second=p2:25.0
}
