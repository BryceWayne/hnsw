package hnsw

import (
	"math"
	"testing"
)

func TestEuclidean(t *testing.T) {
	tests := []struct {
		v1       Vector
		v2       Vector
		expected float64
	}{
		{Vector{0, 0}, Vector{1, 1}, math.Sqrt(2)},
		{Vector{1, 2, 3}, Vector{1, 2, 3}, 0},
		{Vector{0, 0, 0}, Vector{1, 1, 1}, math.Sqrt(3)},
	}

	for _, tt := range tests {
		got := Euclidean(tt.v1, tt.v2)
		if math.Abs(got-tt.expected) > 1e-10 {
			t.Errorf("Euclidean(%v, %v) = %v; want %v", tt.v1, tt.v2, got, tt.expected)
		}
	}
}

func TestCosine(t *testing.T) {
	tests := []struct {
		v1       Vector
		v2       Vector
		expected float64
	}{
		{Vector{1, 0}, Vector{1, 0}, 0},
		{Vector{1, 0}, Vector{0, 1}, 1},
		{Vector{1, 1}, Vector{1, 1}, 0},
	}

	for _, tt := range tests {
		got := Cosine(tt.v1, tt.v2)
		if math.Abs(got-tt.expected) > 1e-10 {
			t.Errorf("Cosine(%v, %v) = %v; want %v", tt.v1, tt.v2, got, tt.expected)
		}
	}
}
