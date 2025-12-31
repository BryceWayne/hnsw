//go:build amd64
// +build amd64

// distance_amd64.go
package hnsw

import (
    "golang.org/x/sys/cpu"
)

var useAVX2 = cpu.X86.HasAVX2

//go:noescape
func euclideanAVX2(v1, v2 Vector) float64

//go:noescape
func cosineAVX2(v1, v2 Vector) float64

// Computes Euclidean distance using SIMD when available
func Euclidean(v1, v2 Vector) float64 {
    if useAVX2 && len(v1) >= 8 {
        return euclideanAVX2(v1, v2)
    }
    return euclideanFallback(v1, v2)
}

// Computes cosine distance using SIMD when available
func Cosine(v1, v2 Vector) float64 {
    if useAVX2 && len(v1) >= 8 {
        return cosineAVX2(v1, v2)
    }
    return cosineFallback(v1, v2)
}
