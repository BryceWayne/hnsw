//go:build amd64 && !purego && unix

package hnsw

import (
	"math"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

func TestAVX2GuardPageBounds(t *testing.T) {
	if !useAVX2FMA {
		t.Skip("AVX2+FMA not supported on this CPU")
	}

	pageSize := os.Getpagesize()
	memA, err := syscall.Mmap(-1, 0, 2*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatalf("mmap a failed: %v", err)
	}
	defer func() { _ = syscall.Munmap(memA) }()
	memB, err := syscall.Mmap(-1, 0, 2*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatalf("mmap b failed: %v", err)
	}
	defer func() { _ = syscall.Munmap(memB) }()

	if err := syscall.Mprotect(memA[pageSize:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect a failed: %v", err)
	}
	if err := syscall.Mprotect(memB[pageSize:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect b failed: %v", err)
	}

	dims := make([]int, 0, 37)
	for d := 1; d <= 33; d++ {
		dims = append(dims, d)
	}
	dims = append(dims, 63, 64, 65, 137)

	for _, dim := range dims {
		byteLen := dim * 4
		ptrA := unsafe.Pointer(&memA[pageSize-byteLen])
		ptrB := unsafe.Pointer(&memB[pageSize-byteLen])
		a := unsafe.Slice((*float32)(ptrA), dim)
		b := unsafe.Slice((*float32)(ptrB), dim)
		for i := 0; i < dim; i++ {
			a[i] = float32(i%7 + 1)
			b[i] = float32(i%5 + 1)
		}

		gotL2 := l2SquaredAVX2(&a[0], &b[0], dim)
		wantL2 := l2SquaredGo(a, b)
		if math.Abs(float64(gotL2-wantL2)) > 1e-3*math.Max(1, float64(wantL2)) {
			t.Fatalf("guard page dim=%d l2SquaredAVX2=%v, go=%v", dim, gotL2, wantL2)
		}

		gotDot := dotProductAVX2(&a[0], &b[0], dim)
		wantDot := dotProductGo(a, b)
		if math.Abs(float64(gotDot-wantDot)) > 1e-3*math.Max(1, math.Abs(float64(wantDot))) {
			t.Fatalf("guard page dim=%d dotProductAVX2=%v, go=%v", dim, gotDot, wantDot)
		}
	}
}
