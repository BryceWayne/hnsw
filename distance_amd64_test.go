//go:build amd64 && !purego

package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestAVX2KernelParity(t *testing.T) {
	if !useAVX2FMA {
		t.Skip("AVX2+FMA not supported on this CPU")
	}

	rng := rand.New(rand.NewSource(12345))
	dims := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 128, 384, 768, 1024, 1536}

	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			// Allocate extra elements for unaligned subslice testing (v[1 : dim+1]).
			raw1 := make([]float32, dim+3)
			raw2 := make([]float32, dim+5)
			for i := range raw1 {
				raw1[i] = float32(rng.Float64()*2 - 1)
			}
			for i := range raw2 {
				raw2[i] = float32(rng.Float64()*2 - 1)
			}

			for _, off := range []int{0, 1} {
				a := raw1[off : off+dim]
				b := raw2[off : off+dim]

				gotL2 := l2SquaredAVX2(&a[0], &b[0], dim)
				wantL2 := l2SquaredGo(a, b)
				tolL2 := 1e-4 * math.Max(1, float64(wantL2))
				if math.Abs(float64(gotL2-wantL2)) > tolL2 {
					t.Errorf("off=%d l2SquaredAVX2=%v, go=%v", off, gotL2, wantL2)
				}

				gotDot := dotProductAVX2(&a[0], &b[0], dim)
				wantDot := dotProductGo(a, b)
				tolDot := 1e-4 * math.Max(1, math.Abs(float64(wantDot)))
				if math.Abs(float64(gotDot-wantDot)) > tolDot {
					t.Errorf("off=%d dotProductAVX2=%v, go=%v", off, gotDot, wantDot)
				}
			}
		})
	}

	// Defensive n <= 0 check
	x := float32(1)
	if got := l2SquaredAVX2(&x, &x, 0); got != 0 {
		t.Errorf("l2SquaredAVX2(n=0) = %v, want 0", got)
	}
	if got := dotProductAVX2(&x, &x, 0); got != 0 {
		t.Errorf("dotProductAVX2(n=0) = %v, want 0", got)
	}
	if got := l2SquaredAVX2(&x, &x, -1); got != 0 {
		t.Errorf("l2SquaredAVX2(n=-1) = %v, want 0", got)
	}
	if got := dotProductAVX2(&x, &x, -1); got != 0 {
		t.Errorf("dotProductAVX2(n=-1) = %v, want 0", got)
	}
}

func TestTier5_AVX2AllTailsAndAlignments(t *testing.T) {
	if !useAVX2FMA {
		t.Skip("AVX2+FMA not supported on this CPU")
	}

	rng := rand.New(rand.NewSource(98765))
	const maxDim = 65
	rawA := make([]float32, maxDim+16)
	rawB := make([]float32, maxDim+16)
	for i := range rawA {
		rawA[i] = float32(rng.Float64()*2 - 1)
		rawB[i] = float32(rng.Float64()*2 - 1)
	}

	for dim := 1; dim <= maxDim; dim++ {
		for offA := 0; offA < 8; offA++ {
			for offB := 0; offB < 8; offB++ {
				a := rawA[offA : offA+dim]
				b := rawB[offB : offB+dim]

				gotL2 := l2SquaredAVX2(&a[0], &b[0], dim)
				wantL2 := l2SquaredGo(a, b)
				tolL2 := 1e-4 * math.Max(1, float64(wantL2))
				if math.Abs(float64(gotL2-wantL2)) > tolL2 {
					t.Fatalf("dim=%d off=(%d,%d) l2SquaredAVX2=%v, go=%v", dim, offA, offB, gotL2, wantL2)
				}

				gotDot := dotProductAVX2(&a[0], &b[0], dim)
				wantDot := dotProductGo(a, b)
				tolDot := 1e-4 * math.Max(1, math.Abs(float64(wantDot)))
				if math.Abs(float64(gotDot-wantDot)) > tolDot {
					t.Fatalf("dim=%d off=(%d,%d) dotProductAVX2=%v, go=%v", dim, offA, offB, gotDot, wantDot)
				}
			}
		}
	}
}

func TestTier5_AVX2FallbackToggle(t *testing.T) {
	prev := useAVX2FMA
	useAVX2FMA = false
	defer func() { useAVX2FMA = prev }()

	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	b := []float32{2, 4, 6, 8, 10, 12, 14, 16, 18}
	if got, want := l2Squared(a, b), l2SquaredGo(a, b); got != want {
		t.Fatalf("fallback l2Squared = %v, want %v", got, want)
	}
	if got, want := dotProduct(a, b), dotProductGo(a, b); got != want {
		t.Fatalf("fallback dotProduct = %v, want %v", got, want)
	}
}

func BenchmarkDistanceAVX2VsFallback(b *testing.B) {
	if !useAVX2FMA {
		b.Skip("AVX2+FMA not available")
	}
	rng := rand.New(rand.NewSource(42))
	for _, dim := range []int{32, 128, 384, 768, 1536} {
		v1 := make([]float32, dim)
		v2 := make([]float32, dim)
		for i := range v1 {
			v1[i] = float32(rng.Float64()*2 - 1)
			v2[i] = float32(rng.Float64()*2 - 1)
		}

		b.Run(fmt.Sprintf("L2Sq_dim=%d_AVX2", dim), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = l2SquaredAVX2(&v1[0], &v2[0], dim)
			}
		})
		b.Run(fmt.Sprintf("L2Sq_dim=%d_Go", dim), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = l2SquaredGo(v1, v2)
			}
		})
		b.Run(fmt.Sprintf("Dot_dim=%d_AVX2", dim), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = dotProductAVX2(&v1[0], &v2[0], dim)
			}
		})
		b.Run(fmt.Sprintf("Dot_dim=%d_Go", dim), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = dotProductGo(v1, v2)
			}
		})
	}
}
