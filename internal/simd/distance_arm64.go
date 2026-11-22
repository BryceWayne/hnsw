//go:build arm64 && !purego

package simd

//go:noescape
func l2SquaredNEON(a, b *float32, n int) float32

//go:noescape
func dotProductNEON(a, b *float32, n int) float32

// L2Squared returns the squared Euclidean distance between a and b using ARM64 NEON.
func L2Squared(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredNEON(&a[0], &b[0], n)
}

// DotProduct returns the inner product of a and b using ARM64 NEON.
func DotProduct(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductNEON(&a[0], &b[0], n)
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
