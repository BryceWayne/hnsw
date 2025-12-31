//go:build !amd64
// +build !amd64

// distance_batch_generic.go
package hnsw

func BatchEuclidean(query Vector, vectors []Vector) []float64 {
    return batchEuclideanFallback(query, vectors)
}

func BatchEuclideanFlat(query Vector, flatVectors []float64, dim int) []float64 {
    count := len(flatVectors) / dim
    results := make([]float64, count)
    for i := 0; i < count; i++ {
        offset := i * dim
        vec := Vector(flatVectors[offset : offset+dim])
        results[i] = euclideanFallback(query, vec)
    }
    return results
}
