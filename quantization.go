package hnsw

import (
	"math"

	"github.com/BryceWayne/hnsw/internal/simd"
)

// Quantization identifies the vector storage encoding used by an [Index].
type Quantization uint8

const (
	// QuantNone stores full 32-bit IEEE-754 float32 vectors (4 bytes per dimension).
	QuantNone Quantization = iota

	// QuantFloat16 stores IEEE-754 binary16 (half-precision) vectors (2 bytes per dimension),
	// halving vector arena memory and memory bandwidth with minimal recall impact.
	QuantFloat16
)

// String returns the human-readable name of the quantization mode.
func (q Quantization) String() string {
	switch q {
	case QuantNone:
		return "none"
	case QuantFloat16:
		return "float16"
	default:
		return "unknown"
	}
}

func cosineFallback64F16F32(a []uint16, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(simd.F16ToF32Table[a[i]])
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

func cosineFallback64F16F16(a, b []uint16) float32 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(simd.F16ToF32Table[a[i]])
		fb := float64(simd.F16ToF32Table[b[i]])
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

func cosineWithInvNormsF16F32(a []uint16, b []float32, invA, invB float32) float32 {
	if invA == 0 || invB == 0 {
		return 1.0
	}
	if math.IsInf(float64(invA), 0) || math.IsInf(float64(invB), 0) ||
		invA > 1e15 || invB > 1e15 || invA < 1e-15 || invB < 1e-15 {
		return cosineFallback64F16F32(a, b)
	}
	var dot float32
	if simd.HasFastF16() {
		dot = simd.DotProductF16F32Fast(a, b)
	} else {
		dot = simd.DotProductF16F32(a, b)
	}
	if dot != dot || math.IsInf(float64(dot), 0) {
		return cosineFallback64F16F32(a, b)
	}
	return clampCosine(float32(1.0) - dot*invA*invB)
}

func cosineWithInvNormsF16F16(a, b []uint16, invA, invB float32) float32 {
	if invA == 0 || invB == 0 {
		return 1.0
	}
	if math.IsInf(float64(invA), 0) || math.IsInf(float64(invB), 0) ||
		invA > 1e15 || invB > 1e15 || invA < 1e-15 || invB < 1e-15 {
		return cosineFallback64F16F16(a, b)
	}
	var dot float32
	if simd.HasFastF16() {
		dot = simd.DotProductF16F16Fast(a, b)
	} else {
		dot = simd.DotProductF16F16(a, b)
	}
	if dot != dot || math.IsInf(float64(dot), 0) {
		return cosineFallback64F16F16(a, b)
	}
	return clampCosine(float32(1.0) - dot*invA*invB)
}

func euclideanFallback64F16F32(a []uint16, b []float32) float32 {
	var maxAbs float64
	for i := range a {
		d := math.Abs(float64(simd.F16ToF32Table[a[i]]) - float64(b[i]))
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
		scaled := (float64(simd.F16ToF32Table[a[i]]) - float64(b[i])) * inv
		sum += scaled * scaled
	}
	return float32(maxAbs * math.Sqrt(sum))
}

func euclideanFastF16F32(a []uint16, b []float32) float32 {
	d2 := simd.L2SquaredF16F32(a, b)
	if d2 < 1e-36 || math.IsInf(float64(d2), 0) {
		return euclideanFallback64F16F32(a, b)
	}
	return float32(math.Sqrt(float64(d2)))
}

func computeInvNormF16(v []uint16) float32 {
	normSq := simd.DotProductF16F16(v, v)
	if normSq > 0 && !math.IsInf(float64(normSq), 0) {
		return float32(1.0 / math.Sqrt(float64(normSq)))
	}
	var sum64 float64
	for _, h := range v {
		fx := float64(simd.F16ToF32Table[h])
		sum64 += fx * fx
	}
	if sum64 == 0 {
		return 0
	}
	return float32(1.0 / math.Sqrt(sum64))
}
