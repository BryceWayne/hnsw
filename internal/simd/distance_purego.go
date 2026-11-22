//go:build (!amd64 && !arm64) || purego

package simd

// HasFastF32 reports whether hardware-accelerated float32 SIMD kernels are active.
func HasFastF32() bool { return false }

// HasFastF16 reports whether hardware-accelerated binary16 SIMD kernels are active.
func HasFastF16() bool { return false }

// L2SquaredFast falls back to the pure-Go kernel on non-SIMD builds.
func L2SquaredFast(a, b []float32) float32 { return L2SquaredGo(a, b) }

// DotProductFast falls back to the pure-Go kernel on non-SIMD builds.
func DotProductFast(a, b []float32) float32 { return DotProductGo(a, b) }

// L2SquaredF16F32Fast falls back to the pure-Go kernel on non-SIMD builds.
func L2SquaredF16F32Fast(a []uint16, b []float32) float32 { return L2SquaredF16F32Go(a, b) }

// DotProductF16F32Fast falls back to the pure-Go kernel on non-SIMD builds.
func DotProductF16F32Fast(a []uint16, b []float32) float32 { return DotProductF16F32Go(a, b) }

// L2SquaredF16F16Fast falls back to the pure-Go kernel on non-SIMD builds.
func L2SquaredF16F16Fast(a, b []uint16) float32 { return L2SquaredF16F16Go(a, b) }

// DotProductF16F16Fast falls back to the pure-Go kernel on non-SIMD builds.
func DotProductF16F16Fast(a, b []uint16) float32 { return DotProductF16F16Go(a, b) }

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
