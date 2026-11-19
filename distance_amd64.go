//go:build amd64 && !purego

package hnsw

import "golang.org/x/sys/cpu"

var useAVX2FMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

//go:noescape
func l2SquaredAVX2(a, b *float32, n int) float32

//go:noescape
func dotProductAVX2(a, b *float32, n int) float32

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
