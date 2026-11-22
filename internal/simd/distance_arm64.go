//go:build arm64 && !purego

package simd

//go:noescape
func l2SquaredNEON(a, b *float32, n int) float32

//go:noescape
func dotProductNEON(a, b *float32, n int) float32

//go:noescape
func l2SquaredF16F32NEON(a *uint16, b *float32, n int) float32

//go:noescape
func dotProductF16F32NEON(a *uint16, b *float32, n int) float32

//go:noescape
func l2SquaredF16F16NEON(a, b *uint16, n int) float32

//go:noescape
func dotProductF16F16NEON(a, b *uint16, n int) float32

// HasFastF32 reports whether hardware-accelerated float32 SIMD kernels are active.
func HasFastF32() bool { return true }

// HasFastF16 reports whether hardware-accelerated binary16 SIMD kernels are active.
func HasFastF16() bool { return true }

// L2SquaredFast invokes the ARM64 NEON L2-squared kernel directly (inlinable wrapper).
func L2SquaredFast(a, b []float32) float32 {
	return l2SquaredNEON(&a[0], &b[0], len(a))
}

// DotProductFast invokes the ARM64 NEON dot-product kernel directly (inlinable wrapper).
func DotProductFast(a, b []float32) float32 {
	return dotProductNEON(&a[0], &b[0], len(a))
}

// L2SquaredF16F32Fast invokes the ARM64 NEON F16-F32 L2-squared kernel directly (inlinable wrapper).
func L2SquaredF16F32Fast(a []uint16, b []float32) float32 {
	return l2SquaredF16F32NEON(&a[0], &b[0], len(a))
}

// DotProductF16F32Fast invokes the ARM64 NEON F16-F32 dot-product kernel directly (inlinable wrapper).
func DotProductF16F32Fast(a []uint16, b []float32) float32 {
	return dotProductF16F32NEON(&a[0], &b[0], len(a))
}

// L2SquaredF16F16Fast invokes the ARM64 NEON F16-F16 L2-squared kernel directly (inlinable wrapper).
func L2SquaredF16F16Fast(a, b []uint16) float32 {
	return l2SquaredF16F16NEON(&a[0], &b[0], len(a))
}

// DotProductF16F16Fast invokes the ARM64 NEON F16-F16 dot-product kernel directly (inlinable wrapper).
func DotProductF16F16Fast(a, b []uint16) float32 {
	return dotProductF16F16NEON(&a[0], &b[0], len(a))
}

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

// L2SquaredF16F32 returns the squared Euclidean distance between binary16 vector a and float32 vector b using ARM64 NEON.
func L2SquaredF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredF16F32NEON(&a[0], &b[0], n)
}

// DotProductF16F32 returns the inner product of binary16 vector a and float32 vector b using ARM64 NEON.
func DotProductF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductF16F32NEON(&a[0], &b[0], n)
}

// L2SquaredF16F16 returns the squared Euclidean distance between binary16 vectors a and b using ARM64 NEON.
func L2SquaredF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredF16F16NEON(&a[0], &b[0], n)
}

// DotProductF16F16 returns the inner product of binary16 vectors a and b using ARM64 NEON.
func DotProductF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductF16F16NEON(&a[0], &b[0], n)
}
