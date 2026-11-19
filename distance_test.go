package hnsw

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func scalarL2SquaredRef(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return sum
}

func scalarDotRef(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func TestMetricString(t *testing.T) {
	tests := []struct {
		m    Metric
		want string
	}{
		{MetricEuclidean, "euclidean"},
		{MetricSquaredEuclidean, "squared_euclidean"},
		{MetricCosine, "cosine"},
		{MetricDotProduct, "dot_product"},
		{metricCustom, "custom"},
		{Metric(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.m.String(); got != tt.want {
			t.Errorf("Metric(%d).String() = %q, want %q", tt.m, got, tt.want)
		}
	}
}

func TestDistanceHelpersValidationAndParity(t *testing.T) {
	nan := float32(math.NaN())
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))

	type fnCase struct {
		name string
		fn   func(a, b []float32) (float32, error)
	}
	funcs := []fnCase{
		{"Euclidean", Euclidean},
		{"SquaredEuclidean", SquaredEuclidean},
		{"Cosine", Cosine},
		{"DotProduct", DotProduct},
	}

	for _, fc := range funcs {
		t.Run(fc.name+"/errors", func(t *testing.T) {
			if _, err := fc.fn(nil, nil); !errors.Is(err, ErrInvalidDimension) {
				t.Errorf("empty: got err=%v, want ErrInvalidDimension", err)
			}
			if _, err := fc.fn([]float32{1, 2}, []float32{1, 2, 3}); !errors.Is(err, ErrDimensionMismatch) {
				t.Errorf("mismatch: got err=%v, want ErrDimensionMismatch", err)
			}
			if _, err := fc.fn([]float32{1, nan}, []float32{1, 2}); !errors.Is(err, ErrNaNOrInf) {
				t.Errorf("nan in a: got err=%v, want ErrNaNOrInf", err)
			}
			if _, err := fc.fn([]float32{1, 2}, []float32{posInf, 2}); !errors.Is(err, ErrNaNOrInf) {
				t.Errorf("+inf in b: got err=%v, want ErrNaNOrInf", err)
			}
			if _, err := fc.fn([]float32{1, negInf}, []float32{1, 2}); !errors.Is(err, ErrNaNOrInf) {
				t.Errorf("-inf in a: got err=%v, want ErrNaNOrInf", err)
			}
		})
	}

	rng := rand.New(rand.NewSource(42))
	dims := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 128, 384, 768, 1024, 1536}
	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			a := make([]float32, dim)
			b := make([]float32, dim)
			for i := 0; i < dim; i++ {
				a[i] = float32(rng.Float64()*4 - 2)
				b[i] = float32(rng.Float64()*4 - 2)
			}

			gotL2, err := Euclidean(a, b)
			if err != nil {
				t.Fatalf("Euclidean error: %v", err)
			}
			gotL2Sq, err := SquaredEuclidean(a, b)
			if err != nil {
				t.Fatalf("SquaredEuclidean error: %v", err)
			}
			gotDot, err := DotProduct(a, b)
			if err != nil {
				t.Fatalf("DotProduct error: %v", err)
			}
			gotCos, err := Cosine(a, b)
			if err != nil {
				t.Fatalf("Cosine error: %v", err)
			}

			wantL2Sq := scalarL2SquaredRef(a, b)
			wantL2 := math.Sqrt(wantL2Sq)
			wantDot := scalarDotRef(a, b)
			wantCos := float64(cosineFallback64(a, b))

			if math.Abs(float64(gotL2Sq)-wantL2Sq) > 1e-3*math.Max(1, wantL2Sq) {
				t.Errorf("SquaredEuclidean = %v, want %v", gotL2Sq, wantL2Sq)
			}
			if math.Abs(float64(gotL2)-wantL2) > 1e-4*math.Max(1, wantL2) {
				t.Errorf("Euclidean = %v, want %v", gotL2, wantL2)
			}
			if math.Abs(float64(gotDot)-wantDot) > 1e-3*math.Max(1, math.Abs(wantDot)) {
				t.Errorf("DotProduct = %v, want %v", gotDot, wantDot)
			}
			if math.Abs(float64(gotCos)-wantCos) > 1e-4 {
				t.Errorf("Cosine = %v, want %v", gotCos, wantCos)
			}
		})
	}
}
