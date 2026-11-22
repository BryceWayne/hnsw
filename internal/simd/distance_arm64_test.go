//go:build arm64 && !purego

package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestNEONKernelParity(t *testing.T) {
	rng := rand.New(rand.NewSource(12345))
	dims := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 384, 768, 1024, 1536}

	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			raw1 := make([]float32, dim+5)
			raw2 := make([]float32, dim+7)
			for i := range raw1 {
				raw1[i] = float32(rng.Float64()*2 - 1)
			}
			for i := range raw2 {
				raw2[i] = float32(rng.Float64()*2 - 1)
			}

			for _, off := range []int{0, 1, 2, 3} {
				a := raw1[off : off+dim]
				b := raw2[off : off+dim]

				gotL2 := l2SquaredNEON(&a[0], &b[0], dim)
				wantL2 := L2SquaredGo(a, b)
				tolL2 := 1e-4 * math.Max(1, float64(wantL2))
				if math.Abs(float64(gotL2-wantL2)) > tolL2 {
					t.Errorf("off=%d l2SquaredNEON=%v, go=%v", off, gotL2, wantL2)
				}

				gotDot := dotProductNEON(&a[0], &b[0], dim)
				wantDot := DotProductGo(a, b)
				tolDot := 1e-4 * math.Max(1, math.Abs(float64(wantDot)))
				if math.Abs(float64(gotDot-wantDot)) > tolDot {
					t.Errorf("off=%d dotProductNEON=%v, go=%v", off, gotDot, wantDot)
				}
			}
		})
	}

	x := float32(1)
	if got := l2SquaredNEON(&x, &x, 0); got != 0 {
		t.Errorf("l2SquaredNEON(n=0) = %v, want 0", got)
	}
	if got := dotProductNEON(&x, &x, 0); got != 0 {
		t.Errorf("dotProductNEON(n=0) = %v, want 0", got)
	}
	if got := l2SquaredNEON(&x, &x, -1); got != 0 {
		t.Errorf("l2SquaredNEON(n=-1) = %v, want 0", got)
	}
	if got := dotProductNEON(&x, &x, -1); got != 0 {
		t.Errorf("dotProductNEON(n=-1) = %v, want 0", got)
	}
}

func TestNEONF16KernelParity(t *testing.T) {
	rng := rand.New(rand.NewSource(54321))
	dims := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 384, 768, 1536}
	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			rawA := make([]uint16, dim+5)
			rawB := make([]uint16, dim+7)
			rawF := make([]float32, dim+7)
			for i := range rawA {
				rawA[i] = Float32ToFloat16Finite(float32(rng.Float64()*2 - 1))
			}
			for i := range rawB {
				f := float32(rng.Float64()*2 - 1)
				rawB[i] = Float32ToFloat16Finite(f)
				rawF[i] = f
			}

			for _, off := range []int{0, 1, 2, 3} {
				a := rawA[off : off+dim]
				b := rawB[off : off+dim]
				f := rawF[off : off+dim]

				gotL2F := l2SquaredF16F32NEON(&a[0], &f[0], dim)
				wantL2F := L2SquaredF16F32Go(a, f)
				if math.Abs(float64(gotL2F-wantL2F)) > 1e-4*math.Max(1, float64(wantL2F)) {
					t.Errorf("off=%d l2F16F32 neon=%v go=%v", off, gotL2F, wantL2F)
				}

				gotDotF := dotProductF16F32NEON(&a[0], &f[0], dim)
				wantDotF := DotProductF16F32Go(a, f)
				if math.Abs(float64(gotDotF-wantDotF)) > 1e-4*math.Max(1, math.Abs(float64(wantDotF))) {
					t.Errorf("off=%d dotF16F32 neon=%v go=%v", off, gotDotF, wantDotF)
				}

				gotL2H := l2SquaredF16F16NEON(&a[0], &b[0], dim)
				wantL2H := L2SquaredF16F16Go(a, b)
				if math.Abs(float64(gotL2H-wantL2H)) > 1e-4*math.Max(1, float64(wantL2H)) {
					t.Errorf("off=%d l2F16F16 neon=%v go=%v", off, gotL2H, wantL2H)
				}

				gotDotH := dotProductF16F16NEON(&a[0], &b[0], dim)
				wantDotH := DotProductF16F16Go(a, b)
				if math.Abs(float64(gotDotH-wantDotH)) > 1e-4*math.Max(1, math.Abs(float64(wantDotH))) {
					t.Errorf("off=%d dotF16F16 neon=%v go=%v", off, gotDotH, wantDotH)
				}
			}
		})
	}

	var h uint16 = 0x3C00
	var f float32 = 1.0
	if l2SquaredF16F32NEON(&h, &f, 0) != 0 || dotProductF16F32NEON(&h, &f, 0) != 0 ||
		l2SquaredF16F16NEON(&h, &h, 0) != 0 || dotProductF16F16NEON(&h, &h, 0) != 0 ||
		l2SquaredF16F32NEON(&h, &f, -1) != 0 || dotProductF16F32NEON(&h, &f, -1) != 0 ||
		l2SquaredF16F16NEON(&h, &h, -1) != 0 || dotProductF16F16NEON(&h, &h, -1) != 0 {
		t.Fatal("expected 0 for n<=0")
	}
}

func BenchmarkDistanceF16NEONVsFallback(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, dim := range []int{128, 768, 1536} {
		a := make([]uint16, dim)
		v := make([]float32, dim)
		for i := range a {
			a[i] = Float32ToFloat16Finite(float32(rng.Float64()*2 - 1))
			v[i] = float32(rng.Float64()*2 - 1)
		}
		b.Run(fmt.Sprintf("L2SqF16F32_dim=%d_NEON", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += l2SquaredF16F32NEON(&a[0], &v[0], dim)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("L2SqF16F32_dim=%d_Go", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += L2SquaredF16F32Go(a, v)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("DotF16F32_dim=%d_NEON", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += dotProductF16F32NEON(&a[0], &v[0], dim)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("DotF16F32_dim=%d_Go", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += DotProductF16F32Go(a, v)
			}
			_ = sink
		})
	}
}

func BenchmarkDistanceNEONVsFallback(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, dim := range []int{32, 128, 384, 768, 1536} {
		a := make([]float32, dim)
		v := make([]float32, dim)
		for i := range a {
			a[i] = float32(rng.Float64()*2 - 1)
			v[i] = float32(rng.Float64()*2 - 1)
		}
		b.Run(fmt.Sprintf("L2Sq_dim=%d_NEON", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += l2SquaredNEON(&a[0], &v[0], dim)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("L2Sq_dim=%d_Go", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += L2SquaredGo(a, v)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("Dot_dim=%d_NEON", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += dotProductNEON(&a[0], &v[0], dim)
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("Dot_dim=%d_Go", dim), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += DotProductGo(a, v)
			}
			_ = sink
		})
	}
}
