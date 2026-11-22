//go:build amd64 && !purego

package simd

import "golang.org/x/sys/cpu"

var (
	useAVX2FMA     = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	useAVX2FMAF16C = useAVX2FMA && hasF16C()
)

func hasF16C() bool

//go:noescape
func l2SquaredAVX2(a, b *float32, n int) float32

//go:noescape
func dotProductAVX2(a, b *float32, n int) float32

//go:noescape
func l2SquaredF16F32AVX2(a *uint16, b *float32, n int) float32

//go:noescape
func dotProductF16F32AVX2(a *uint16, b *float32, n int) float32

//go:noescape
func l2SquaredF16F16AVX2(a, b *uint16, n int) float32

//go:noescape
func dotProductF16F16AVX2(a, b *uint16, n int) float32

// L2Squared returns the squared Euclidean distance between a and b.
func L2Squared(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMA {
		return l2SquaredAVX2(&a[0], &b[0], n)
	}
	return L2SquaredGo(a, b[:n])
}

// DotProduct returns the inner product of a and b.
func DotProduct(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMA {
		return dotProductAVX2(&a[0], &b[0], n)
	}
	return DotProductGo(a, b[:n])
}

// L2SquaredF16F32 returns the squared Euclidean distance between binary16 vector a and float32 vector b.
func L2SquaredF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return l2SquaredF16F32AVX2(&a[0], &b[0], n)
	}
	return L2SquaredF16F32Go(a, b[:n])
}

// DotProductF16F32 returns the inner product of binary16 vector a and float32 vector b.
func DotProductF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return dotProductF16F32AVX2(&a[0], &b[0], n)
	}
	return DotProductF16F32Go(a, b[:n])
}

// L2SquaredF16F16 returns the squared Euclidean distance between binary16 vectors a and b.
func L2SquaredF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return l2SquaredF16F16AVX2(&a[0], &b[0], n)
	}
	return L2SquaredF16F16Go(a, b[:n])
}

// DotProductF16F16 returns the inner product of binary16 vectors a and b.
func DotProductF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return dotProductF16F16AVX2(&a[0], &b[0], n)
	}
	return DotProductF16F16Go(a, b[:n])
}
