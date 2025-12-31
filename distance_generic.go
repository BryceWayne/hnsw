//go:build !amd64
// +build !amd64

// distance_generic.go
package hnsw

func Euclidean(v1, v2 Vector) float64 {
    return euclideanFallback(v1, v2)
}

func Cosine(v1, v2 Vector) float64 {
    return cosineFallback(v1, v2)
}
