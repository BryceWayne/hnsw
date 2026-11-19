package hnsw_test

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/BryceWayne/hnsw"
)

// e2eCompositeKey has exported fields as required by encoding/gob for struct keys.
type e2eCompositeKey struct {
	Tenant string
	Bucket uint32
	ID     int64
}

// e2eAltCompositeKey is a distinct struct type with identical fields for negative persistence checks.
type e2eAltCompositeKey struct {
	Tenant string
	Bucket uint32
	ID     int64
}

func e2eMustNew[K comparable](t *testing.T, dim int, opts ...hnsw.Option) *hnsw.Index[K] {
	t.Helper()
	idx, err := hnsw.New[K](dim, opts...)
	if err != nil {
		t.Fatalf("hnsw.New(%d) unexpected error: %v", dim, err)
	}
	return idx
}

func e2eMustInsert[K comparable](t *testing.T, idx *hnsw.Index[K], id K, vec []float32) {
	t.Helper()
	if err := idx.Insert(id, vec); err != nil {
		t.Fatalf("idx.Insert(%v) unexpected error: %v", id, err)
	}
}

func e2eAssertErrorIs(t *testing.T, err, target error, label string) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("%s: got err=%v, want errors.Is(..., %v)", label, err, target)
	}
}

func e2eApproxEqual(a, b, relTol, absTol float32) bool {
	diff := float32(math.Abs(float64(a) - float64(b)))
	if diff <= absTol {
		return true
	}
	scale := float32(math.Max(math.Abs(float64(a)), math.Abs(float64(b))))
	return diff <= relTol*scale
}

func e2eAssertVecEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("vector length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("vector element[%d] mismatch: got %g, want %g", i, got[i], want[i])
		}
	}
}

func e2eAssertSortedDistances[K comparable](t *testing.T, results []hnsw.SearchResult[K]) {
	t.Helper()
	for i, r := range results {
		d64 := float64(r.Distance)
		if math.IsNaN(d64) || math.IsInf(d64, 0) {
			t.Fatalf("result[%d] has non-finite distance: %v", i, r.Distance)
		}
		if i > 0 && r.Distance < results[i-1].Distance {
			t.Fatalf("results not sorted at index %d: %v < %v", i, r.Distance, results[i-1].Distance)
		}
	}
}

func e2eMakeUniformVecs(seed int64, count, dim int) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vecs := make([][]float32, count)
	for i := range count {
		v := make([]float32, dim)
		for j := range dim {
			v[j] = rng.Float32()*2 - 1
		}
		vecs[i] = v
	}
	return vecs
}

func e2eNormalizeVec64(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if sumSq == 0 {
		if len(out) > 0 {
			out[0] = 1
		}
		return out
	}
	inv := 1.0 / math.Sqrt(sumSq)
	for i, x := range v {
		out[i] = float32(float64(x) * inv)
	}
	return out
}

func e2eMakeUnitVecs(seed int64, count, dim int) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vecs := make([][]float32, count)
	for i := range count {
		raw := make([]float32, dim)
		for j := range dim {
			raw[j] = float32(rng.NormFloat64())
		}
		vecs[i] = e2eNormalizeVec64(raw)
	}
	return vecs
}

func e2eMakeClusteredDataset(seed int64, count, queryCount, dim, clusters int, normalize bool) ([][]float32, [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range clusters {
		center := make([]float32, dim)
		for j := range dim {
			center[j] = float32(rng.NormFloat64()) * 3.0
		}
		if normalize {
			center = e2eNormalizeVec64(center)
		}
		centers[c] = center
	}

	makeSample := func() []float32 {
		c := centers[rng.Intn(clusters)]
		v := make([]float32, dim)
		jitterScale := 0.25
		if normalize {
			jitterScale = 0.12
		}
		for j := range dim {
			v[j] = c[j] + float32(rng.NormFloat64()*jitterScale)
		}
		if normalize {
			v = e2eNormalizeVec64(v)
		}
		return v
	}

	vecs := make([][]float32, count)
	for i := range count {
		vecs[i] = makeSample()
	}
	queries := make([][]float32, queryCount)
	for i := range queryCount {
		queries[i] = makeSample()
	}
	return vecs, queries
}

func e2eRefEuclidean(a, b []float32) float32 {
	return float32(math.Sqrt(float64(e2eRefL2Squared(a, b))))
}

func e2eRefL2Squared(a, b []float32) float32 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	if sum < 0 {
		sum = 0
	}
	return float32(sum)
}

func e2eRefCosine(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if sim > 1.0 {
		sim = 1.0
	} else if sim < -1.0 {
		sim = -1.0
	}
	dist := 1.0 - sim
	if dist < 0 {
		dist = 0
	} else if dist > 2 {
		dist = 2
	}
	return float32(dist)
}

func e2eRefDot(a, b []float32) float32 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return float32(dot)
}

// e2eOracleDistFunc uses independent float64 reference helpers to match Index search ordering.
// Note: MetricDotProduct ranks by -e2eRefDot(a, b) with -0 normalized to 0.
func e2eOracleDistFunc(t *testing.T, metric hnsw.Metric) func(a, b []float32) float32 {
	t.Helper()
	return func(a, b []float32) float32 {
		if len(a) == 0 || len(a) != len(b) {
			t.Fatalf("oracle distance(%v) invalid dimensions: len(a)=%d, len(b)=%d", metric, len(a), len(b))
		}
		switch metric {
		case hnsw.MetricEuclidean:
			return e2eRefEuclidean(a, b)
		case hnsw.MetricSquaredEuclidean:
			return e2eRefL2Squared(a, b)
		case hnsw.MetricCosine:
			return e2eRefCosine(a, b)
		case hnsw.MetricDotProduct:
			d := -e2eRefDot(a, b)
			if d == 0 {
				d = 0
			}
			return d
		default:
			t.Fatalf("unsupported oracle metric: %v", metric)
			return 0
		}
	}
}

func e2eExactKNN[K comparable](
	items []hnsw.Item[K],
	query []float32,
	k int,
	distFn func(a, b []float32) float32,
	filter func(K) bool,
) []hnsw.SearchResult[K] {
	type scored struct {
		res hnsw.SearchResult[K]
		ord int
	}
	candidates := make([]scored, 0, len(items))
	for i, it := range items {
		if filter != nil && !filter(it.ID) {
			continue
		}
		candidates = append(candidates, scored{
			res: hnsw.SearchResult[K]{
				ID:       it.ID,
				Distance: distFn(it.Vector, query),
			},
			ord: i,
		})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].res.Distance != candidates[j].res.Distance {
			return candidates[i].res.Distance < candidates[j].res.Distance
		}
		return candidates[i].ord < candidates[j].ord
	})
	limit := k
	if len(candidates) < limit {
		limit = len(candidates)
	}
	out := make([]hnsw.SearchResult[K], limit)
	for i := range limit {
		out[i] = candidates[i].res
	}
	return out
}

func e2eRecallAtK[K comparable](got, oracle []hnsw.SearchResult[K]) float64 {
	if len(oracle) == 0 {
		if len(got) == 0 {
			return 1.0
		}
		return 0.0
	}
	wantSet := make(map[K]struct{}, len(oracle))
	for _, r := range oracle {
		wantSet[r.ID] = struct{}{}
	}
	hits := 0
	seen := make(map[K]struct{}, len(got))
	for _, r := range got {
		if _, dup := seen[r.ID]; dup {
			continue
		}
		seen[r.ID] = struct{}{}
		if _, ok := wantSet[r.ID]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(oracle))
}
