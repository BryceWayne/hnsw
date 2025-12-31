//go:build amd64
// +build amd64

// distance_batch_amd64.go
package hnsw

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

// BatchEuclideanFlat computes distances using flat array for vectors
func BatchEuclideanFlat(query Vector, flatVectors []float64, dim int) []float64 {
    count := len(flatVectors) / dim
    results := make([]float64, count)

    if useAVX2 && dim >= 4 {
        BatchEuclideanAVX2Flat(query, flatVectors, dim, results)
        return results
    }

    // Fallback for flat vectors
    // Reconstruct temporary slice structure or just iterate
    // Iterating flat array is better
    for i := 0; i < count; i++ {
        offset := i * dim
        // We can't use euclideanFallback easily here because it expects Vector ([]float64)
        // So we just implement simple loop here or slice it
        vec := Vector(flatVectors[offset : offset+dim])
        results[i] = euclideanFallback(query, vec)
    }
    return results
}
