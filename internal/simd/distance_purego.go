//go:build (!amd64 && !arm64) || purego

package simd

// L2Squared returns the squared Euclidean distance between a and b.
func L2Squared(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return L2SquaredGo(a, b[:n])
}

// DotProduct returns the inner product of a and b.
func DotProduct(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return DotProductGo(a, b[:n])
}

// L2SquaredF16F32 returns the squared Euclidean distance between binary16 vector a and float32 vector b.
func L2SquaredF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return L2SquaredF16F32Go(a, b[:n])
}

// DotProductF16F32 returns the inner product of binary16 vector a and float32 vector b.
func DotProductF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return DotProductF16F32Go(a, b[:n])
}

// L2SquaredF16F16 returns the squared Euclidean distance between binary16 vectors a and b.
func L2SquaredF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return L2SquaredF16F16Go(a, b[:n])
}

// DotProductF16F16 returns the inner product of binary16 vectors a and b.
func DotProductF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return DotProductF16F16Go(a, b[:n])
}
