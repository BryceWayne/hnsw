package hnsw

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
)

func generateRandomUnitFloat32(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	if dim == 0 {
		return v
	}
	var sum float64
	for sum == 0 {
		for i := range v {
			x := rng.Float64()*2 - 1
			v[i] = float32(x)
			sum += float64(v[i]) * float64(v[i])
		}
	}
	scale := 1.0 / math.Sqrt(sum)
	for i := range v {
		v[i] = float32(float64(v[i]) * scale)
	}
	return v
}

func TestCosineEdgeCases(t *testing.T) {
	for _, dim := range []int{2, 13, 64} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			a := make([]float32, dim)
			b := make([]float32, dim)
			negA := make([]float32, dim)
			scaledA := make([]float32, dim)
			zeros := make([]float32, dim)

			a[0] = 1
			b[1] = 1
			negA[0] = -1
			scaledA[0] = 5

			if got, err := Cosine(a, a); err != nil || math.Abs(float64(got)) > 1e-6 {
				t.Errorf("same direction: got %v, err=%v; want 0", got, err)
			}
			if got, err := Cosine(a, scaledA); err != nil || math.Abs(float64(got)) > 1e-6 {
				t.Errorf("scaled parallel: got %v, err=%v; want 0", got, err)
			}
			if got, err := Cosine(a, negA); err != nil || math.Abs(float64(got-2)) > 1e-6 {
				t.Errorf("opposite direction: got %v, err=%v; want 2", got, err)
			}
			if got, err := Cosine(a, b); err != nil || math.Abs(float64(got-1)) > 1e-6 {
				t.Errorf("perpendicular: got %v, err=%v; want 1", got, err)
			}
			if got, err := Cosine(zeros, a); err != nil || got != 1.0 {
				t.Errorf("zero vector: got %v, err=%v; want 1.0, nil", got, err)
			}
			if got, err := Cosine(zeros, zeros); err != nil || got != 1.0 {
				t.Errorf("both zero vectors: got %v, err=%v; want 1.0, nil", got, err)
			}
		})
	}

	// Extreme float32 scaling (overflow / underflow guard)
	huge := []float32{1e20, 1e20}
	tiny := []float32{1e-25, 1e-25}
	if got, err := Cosine(huge, huge); err != nil || math.Abs(float64(got)) > 1e-5 {
		t.Errorf("huge vectors cosine = %v, err=%v; want 0", got, err)
	}
	if got, err := Cosine(tiny, tiny); err != nil || math.Abs(float64(got)) > 1e-5 {
		t.Errorf("tiny vectors cosine = %v, err=%v; want 0", got, err)
	}
}

func TestCosineIndexLosslessGetAndZeroVector(t *testing.T) {
	idx, err := New[string](4, WithMetric(MetricCosine), WithSeed(7))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Unnormalized vectors with distinct magnitudes
	rawA := []float32{30, 40, 0, 0}
	rawB := []float32{0, 0, 12, 5}
	zero := []float32{0, 0, 0, 0}

	if err := idx.Insert("a", rawA); err != nil {
		t.Fatal(err)
	}
	if err := idx.Insert("b", rawB); err != nil {
		t.Fatal(err)
	}
	if err := idx.Insert("zero", zero); err != nil {
		t.Fatal(err)
	}

	// Get must return exact unnormalized vector
	gotA, ok := idx.Get("a")
	if !ok {
		t.Fatal("Get(a) returned false")
	}
	for i := range rawA {
		if gotA[i] != rawA[i] {
			t.Fatalf("Get(a)[%d] = %v, want lossless %v", i, gotA[i], rawA[i])
		}
	}

	// Delete b, Compact, WriteTo/ReadFrom, and verify lossless Get(a) still holds
	idx.Delete("b")
	if err := idx.Compact(); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := idx.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := ReadFrom[string](&buf)
	if err != nil {
		t.Fatal(err)
	}
	gotALoaded, ok := loaded.Get("a")
	if !ok {
		t.Fatal("loaded.Get(a) returned false")
	}
	for i := range rawA {
		if gotALoaded[i] != rawA[i] {
			t.Fatalf("loaded.Get(a)[%d] = %v, want lossless %v", i, gotALoaded[i], rawA[i])
		}
	}

	// Query scaled parallel to rawA: must match "a" with distance ~0, and "zero" with distance 1.0
	res, err := loaded.Search([]float32{3, 4, 0, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != "a" || res[0].Distance > 1e-5 {
		t.Fatalf("unexpected cosine search results: %+v", res)
	}
	if res[1].ID != "zero" || math.Abs(float64(res[1].Distance-1.0)) > 1e-5 {
		t.Fatalf("zero vector cosine distance = %+v, want ID=zero Distance=1.0", res[1])
	}
}

func TestCosineExtremeScales(t *testing.T) {
	if got := clampCosine(float32(math.NaN())); got != 1.0 {
		t.Fatalf("clampCosine(NaN) = %v, want 1.0", got)
	}

	// Verify 1e-22 subnormal precision in Cosine helper.
	sA := []float32{1e-22, 0}
	sB := []float32{1e-22, 1e-22}
	gotSub, err := Cosine(sA, sB)
	if err != nil {
		t.Fatalf("Cosine(1e-22) error: %v", err)
	}
	wantSub := 1.0 - 1.0/math.Sqrt2
	if math.Abs(float64(gotSub)-wantSub) > 1e-5 {
		t.Fatalf("Cosine(1e-22) = %v, want %v", gotSub, wantSub)
	}

	for _, dim := range []int{4, 32} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			half := dim / 2
			makePair := func(scale float32) (base, opp, orth []float32) {
				base = make([]float32, dim)
				opp = make([]float32, dim)
				orth = make([]float32, dim)
				for i := 0; i < dim; i++ {
					base[i] = scale
					opp[i] = -scale
					if i < half {
						orth[i] = scale
					} else {
						orth[i] = -scale
					}
				}
				return base, opp, orth
			}

			scales := []struct {
				name  string
				scale float32
			}{
				{"1e-40", 1e-40},
				{"1e-25", 1e-25},
				{"1e-22", 1e-22},
				{"1e20", 1e20},
			}

			for _, sc := range scales {
				t.Run(sc.name, func(t *testing.T) {
					base, opp, orth := makePair(sc.scale)

					dSame, err := Cosine(base, base)
					if err != nil || math.IsNaN(float64(dSame)) || math.Abs(float64(dSame)) > 1e-5 {
						t.Fatalf("Cosine(same) = %v, err=%v; want 0", dSame, err)
					}
					dOpp, err := Cosine(base, opp)
					if err != nil || math.IsNaN(float64(dOpp)) || math.Abs(float64(dOpp-2.0)) > 1e-5 {
						t.Fatalf("Cosine(opp) = %v, err=%v; want 2", dOpp, err)
					}
					dOrth, err := Cosine(base, orth)
					if err != nil || math.IsNaN(float64(dOrth)) || math.Abs(float64(dOrth-1.0)) > 1e-5 {
						t.Fatalf("Cosine(orth) = %v, err=%v; want 1", dOrth, err)
					}

					idx, err := New[string](dim, WithMetric(MetricCosine), WithSeed(7))
					if err != nil {
						t.Fatal(err)
					}
					if err := idx.Insert("same", base); err != nil {
						t.Fatal(err)
					}
					if err := idx.Insert("orth", orth); err != nil {
						t.Fatal(err)
					}
					if err := idx.Insert("opp", opp); err != nil {
						t.Fatal(err)
					}

					res, err := idx.Search(base, 3)
					if err != nil {
						t.Fatalf("Search error: %v", err)
					}
					if len(res) != 3 {
						t.Fatalf("Search len = %d, want 3", len(res))
					}
					gotByID := make(map[string]float32, 3)
					for _, r := range res {
						if math.IsNaN(float64(r.Distance)) {
							t.Fatalf("Search returned NaN distance for %s: %+v", r.ID, res)
						}
						gotByID[r.ID] = r.Distance
					}
					if math.Abs(float64(gotByID["same"])) > 1e-5 {
						t.Errorf("Index same distance = %v, want 0 (res=%+v)", gotByID["same"], res)
					}
					if math.Abs(float64(gotByID["orth"]-1.0)) > 1e-5 {
						t.Errorf("Index orth distance = %v, want 1 (res=%+v)", gotByID["orth"], res)
					}
					if math.Abs(float64(gotByID["opp"]-2.0)) > 1e-5 {
						t.Errorf("Index opp distance = %v, want 2 (res=%+v)", gotByID["opp"], res)
					}
					if res[0].ID != "same" || res[1].ID != "orth" || res[2].ID != "opp" {
						t.Errorf("unexpected ranking: %+v", res)
					}
				})
			}
		})
	}
}

func TestCosineUnitSphereRecall(t *testing.T) {
	const (
		numVecs    = 400
		numQueries = 20
		dim        = 32
		k          = 10
	)
	rng := rand.New(rand.NewSource(99))
	idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(128), WithEfSearch(64), WithMetric(MetricCosine), WithSeed(99))
	if err != nil {
		t.Fatal(err)
	}
	vecs := make([][]float32, numVecs)
	items := make([]Item[int], numVecs)
	for i := 0; i < numVecs; i++ {
		vecs[i] = generateRandomUnitFloat32(rng, dim)
		items[i] = Item[int]{ID: i, Vector: vecs[i]}
	}
	if err := idx.InsertBatch(items); err != nil {
		t.Fatal(err)
	}

	type pair struct {
		id   int
		dist float32
	}
	hits := 0
	for q := 0; q < numQueries; q++ {
		qv := generateRandomUnitFloat32(rng, dim)
		dists := make([]pair, numVecs)
		for i := 0; i < numVecs; i++ {
			d, err := Cosine(vecs[i], qv)
			if err != nil {
				t.Fatal(err)
			}
			dists[i] = pair{id: i, dist: d}
		}
		sort.Slice(dists, func(i, j int) bool {
			if dists[i].dist != dists[j].dist {
				return dists[i].dist < dists[j].dist
			}
			return dists[i].id < dists[j].id
		})
		oracle := make(map[int]struct{}, k)
		for i := 0; i < k; i++ {
			oracle[dists[i].id] = struct{}{}
		}
		got, err := idx.Search(qv, k)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range got {
			if _, ok := oracle[r.ID]; ok {
				hits++
			}
		}
	}
	recall10 := float64(hits) / float64(numQueries*k)
	if recall10 < 0.90 {
		t.Fatalf("Cosine unit-sphere Recall@10 = %.4f, want >= 0.90", recall10)
	}
}
