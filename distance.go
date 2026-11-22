package hnsw

import (
	"math"

	"github.com/BryceWayne/hnsw/internal/simd"
)

// Metric identifies a built-in distance function used by an Index.
type Metric uint8

const (
	// MetricEuclidean ranks vectors by Euclidean (L2) distance.
	MetricEuclidean Metric = iota

	// MetricSquaredEuclidean ranks vectors by squared Euclidean (L2^2) distance.
	MetricSquaredEuclidean

	// MetricCosine ranks vectors by cosine distance (1 - cosine similarity).
	MetricCosine

	// MetricDotProduct ranks vectors by negative inner product (-dot(a, b)) so higher dot products appear first.
	MetricDotProduct
)

const metricCustom Metric = 255

// String returns the human-readable name of the metric.
func (m Metric) String() string {
	switch m {
	case MetricEuclidean:
		return "euclidean"
	case MetricSquaredEuclidean:
		return "squared_euclidean"
	case MetricCosine:
		return "cosine"
	case MetricDotProduct:
		return "dot_product"
	case metricCustom:
		return "custom"
	default:
		return "unknown"
	}
}

// DistanceFunc computes a custom dissimilarity score between two equal-length float32 vectors.
// Smaller values indicate closer vectors.
type DistanceFunc func(a, b []float32) float32

// Euclidean returns the Euclidean (L2) distance between a and b, using a scaled float64
// fallback when squared sums would overflow or underflow float32.
//
// Returns [ErrInvalidDimension] if a is empty, [ErrDimensionMismatch] if len(a) != len(b),
// or [ErrNaNOrInf] if either vector contains NaN or ±Inf.
func Euclidean(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	return euclideanFast(a, b), nil
}

func euclideanFast(a, b []float32) float32 {
	d2 := simd.L2Squared(a, b)
	if d2 < 1e-36 || math.IsInf(float64(d2), 0) {
		return euclideanFallback64(a, b)
	}
	return float32(math.Sqrt(float64(d2)))
}

func euclideanFallback64(a, b []float32) float32 {
	var maxAbs float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs == 0 {
		return 0
	}
	inv := 1.0 / maxAbs
	var sum float64
	for i := range a {
		scaled := (float64(a[i]) - float64(b[i])) * inv
		sum += scaled * scaled
	}
	return float32(maxAbs * math.Sqrt(sum))
}

// SquaredEuclidean returns the squared Euclidean (L2^2) distance between a and b.
//
// Returns [ErrInvalidDimension] if a is empty, [ErrDimensionMismatch] if len(a) != len(b),
// or [ErrNaNOrInf] if either vector contains NaN or ±Inf.
func SquaredEuclidean(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	d2 := simd.L2Squared(a, b)
	if d2 <= 0 {
		return 0, nil
	}
	return d2, nil
}

// Cosine returns the cosine distance (1 - cosine similarity) between a and b, clamped to [0, 2].
// If either vector has zero norm, Cosine returns 1.0, nil. Extreme magnitudes fall back to
// scaled float64 arithmetic to avoid overflow or underflow.
//
// Returns [ErrInvalidDimension] if a is empty, [ErrDimensionMismatch] if len(a) != len(b),
// or [ErrNaNOrInf] if either vector contains NaN or ±Inf.
func Cosine(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	dotAB := simd.DotProduct(a, b)
	normSqA := simd.DotProduct(a, a)
	normSqB := simd.DotProduct(b, b)
	if normSqA < 1e-36 || normSqB < 1e-36 ||
		math.IsInf(float64(normSqA), 0) || math.IsInf(float64(normSqB), 0) ||
		dotAB != dotAB || math.IsInf(float64(dotAB), 0) {
		return cosineFallback64(a, b), nil
	}
	denom := float32(math.Sqrt(float64(normSqA)) * math.Sqrt(float64(normSqB)))
	if denom == 0 || denom != denom || math.IsInf(float64(denom), 0) {
		return cosineFallback64(a, b), nil
	}
	return clampCosine(float32(1.0) - dotAB/denom), nil
}

// DotProduct returns the raw inner product (sum of element-wise products) of a and b.
// Note that [MetricDotProduct] ranks index search results by -dot(a, b) so higher inner
// products appear first.
//
// Returns [ErrInvalidDimension] if a is empty, [ErrDimensionMismatch] if len(a) != len(b),
// or [ErrNaNOrInf] if either vector contains NaN or ±Inf.
func DotProduct(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	return simd.DotProduct(a, b), nil
}

func validateVectorFinite(v []float32) error {
	for _, x := range v {
		if x != x || math.IsInf(float64(x), 0) {
			return ErrNaNOrInf
		}
	}
	return nil
}

func validateVectorPair(a, b []float32) error {
	if len(a) != len(b) {
		return ErrDimensionMismatch
	}
	if len(a) == 0 {
		return ErrInvalidDimension
	}
	for i := range a {
		ai := a[i]
		bi := b[i]
		if ai != ai || math.IsInf(float64(ai), 0) || bi != bi || math.IsInf(float64(bi), 0) {
			return ErrNaNOrInf
		}
	}
	return nil
}

func clampCosine(d float32) float32 {
	if d != d {
		return 1.0
	}
	if d < 0 {
		return 0
	}
	if d > 2 {
		return 2
	}
	return d
}

func cosineFallback64(a, b []float32) float32 {
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
	return clampCosine(float32(1.0 - sim))
}

func computeInvNorm(v []float32) float32 {
	normSq := simd.DotProduct(v, v)
	if normSq > 0 && !math.IsInf(float64(normSq), 0) {
		return float32(1.0 / math.Sqrt(float64(normSq)))
	}
	var sum64 float64
	for _, x := range v {
		fx := float64(x)
		sum64 += fx * fx
	}
	if sum64 == 0 {
		return 0
	}
	return float32(1.0 / math.Sqrt(sum64))
}

func cosineWithInvNorms(a, b []float32, invA, invB float32) float32 {
	if invA == 0 || invB == 0 {
		return 1.0
	}
	if math.IsInf(float64(invA), 0) || math.IsInf(float64(invB), 0) ||
		invA > 1e15 || invB > 1e15 || invA < 1e-15 || invB < 1e-15 {
		return cosineFallback64(a, b)
	}
	dot := simd.DotProduct(a, b)
	if dot != dot || math.IsInf(float64(dot), 0) {
		return cosineFallback64(a, b)
	}
	return clampCosine(float32(1.0) - dot*invA*invB)
}
