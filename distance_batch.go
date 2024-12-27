//go:build amd64
// +build amd64

// distance_batch.go
package hnsw

import "math"

// BatchEuclideanAVX2 calculates distances between query vector and multiple vectors
//
//go:noescape
func BatchEuclideanAVX2(query Vector, vectors []Vector, results []float64)

// BatchEuclideanAVX2Flat calculates distances between query vector and multiple vectors
//
//go:noescape
func BatchEuclideanAVX2Flat(query []float64, flatVectors []float64, dim int, results []float64)

// BatchEuclidean computes distances between query and multiple vectors
func BatchEuclidean(query Vector, vectors []Vector) []float64 {
    if len(vectors) == 0 {
        return []float64{}
    }

    // Flatten vectors into contiguous memory
    dim := len(query)
    flatData := make([]float64, len(vectors)*dim)
    for i, vec := range vectors {
        copy(flatData[i*dim:], vec)
    }

    results := make([]float64, len(vectors))
    if useAVX2 && dim >= 4 {
        BatchEuclideanAVX2Flat(query, flatData, dim, results)
        return results
    }
    return batchEuclideanFallback(query, vectors)
}

func batchEuclideanFallback(query Vector, vectors []Vector) []float64 {
    results := make([]float64, len(vectors))
    for i, vec := range vectors {
        var sum float64
        for j := 0; j < len(query); j++ {
            d := query[j] - vec[j]
            sum += d * d
        }
        results[i] = math.Sqrt(sum)
    }
    return results
}
