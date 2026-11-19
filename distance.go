package hnsw

import "math"

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

// Euclidean returns the Euclidean (L2) distance between a and b.
func Euclidean(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	return euclideanFast(a, b), nil
}

func euclideanFast(a, b []float32) float32 {
	d2 := l2Squared(a, b)
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
func SquaredEuclidean(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	d2 := l2Squared(a, b)
	if d2 <= 0 {
		return 0, nil
	}
	return d2, nil
}

// Cosine returns the cosine distance (1 - cosine similarity) between a and b, clamped to [0, 2].
// If either vector has zero norm, Cosine returns 1.0, nil.
func Cosine(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	dotAB := dotProduct(a, b)
	normSqA := dotProduct(a, a)
	normSqB := dotProduct(b, b)
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

// DotProduct returns the inner product (sum of element-wise products) of a and b.
// Note that [MetricDotProduct] ranks index search results by -dot(a, b) so higher inner products appear first.
func DotProduct(a, b []float32) (float32, error) {
	if err := validateVectorPair(a, b); err != nil {
		return 0, err
	}
	return dotProduct(a, b), nil
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
	normSq := dotProduct(v, v)
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
	dot := dotProduct(a, b)
	if dot != dot || math.IsInf(float64(dot), 0) {
		return cosineFallback64(a, b)
	}
	return clampCosine(float32(1.0) - dot*invA*invB)
}

func l2SquaredGo(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+7 < n; i += 8 {
		a8 := a[i : i+8 : i+8]
		b8 := b[i : i+8 : i+8]
		d0 := a8[0] - b8[0]
		d1 := a8[1] - b8[1]
		d2 := a8[2] - b8[2]
		d3 := a8[3] - b8[3]
		d4 := a8[4] - b8[4]
		d5 := a8[5] - b8[5]
		d6 := a8[6] - b8[6]
		d7 := a8[7] - b8[7]
		s0 += d0*d0 + d4*d4
		s1 += d1*d1 + d5*d5
		s2 += d2*d2 + d6*d6
		s3 += d3*d3 + d7*d7
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func dotProductGo(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+7 < n; i += 8 {
		a8 := a[i : i+8 : i+8]
		b8 := b[i : i+8 : i+8]
		s0 += a8[0]*b8[0] + a8[4]*b8[4]
		s1 += a8[1]*b8[1] + a8[5]*b8[5]
		s2 += a8[2]*b8[2] + a8[6]*b8[6]
		s3 += a8[3]*b8[3] + a8[7]*b8[7]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
