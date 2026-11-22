//go:build amd64 && !purego

package hnsw

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

func l2Squared(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMA {
		return l2SquaredAVX2(&a[0], &b[0], n)
	}
	return l2SquaredGo(a, b[:n])
}

func dotProduct(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMA {
		return dotProductAVX2(&a[0], &b[0], n)
	}
	return dotProductGo(a, b[:n])
}

func l2SquaredF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return l2SquaredF16F32AVX2(&a[0], &b[0], n)
	}
	return l2SquaredF16F32Go(a, b[:n])
}

func dotProductF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return dotProductF16F32AVX2(&a[0], &b[0], n)
	}
	return dotProductF16F32Go(a, b[:n])
}

func l2SquaredF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return l2SquaredF16F16AVX2(&a[0], &b[0], n)
	}
	return l2SquaredF16F16Go(a, b[:n])
}

func dotProductF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	if useAVX2FMAF16C {
		return dotProductF16F16AVX2(&a[0], &b[0], n)
	}
	return dotProductF16F16Go(a, b[:n])
}
