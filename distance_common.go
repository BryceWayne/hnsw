// distance_common.go
package hnsw

import "math"

// Fallback implementation for Euclidean distance
func euclideanFallback(v1, v2 Vector) float64 {
    var sum float64
    for i := 0; i < len(v1); i += 4 {
        if i+4 <= len(v1) {
            d0 := v1[i] - v2[i]
            d1 := v1[i+1] - v2[i+1]
            d2 := v1[i+2] - v2[i+2]
            d3 := v1[i+3] - v2[i+3]
            sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
        } else {
            for j := i; j < len(v1); j++ {
                d := v1[j] - v2[j]
                sum += d * d
            }
        }
    }
    return math.Sqrt(sum)
}

// Fallback implementation for Cosine distance
func cosineFallback(v1, v2 Vector) float64 {
    var dot, norm1, norm2 float64
    for i := 0; i < len(v1); i += 4 {
        if i+4 <= len(v1) {
            dot += v1[i]*v2[i] + v1[i+1]*v2[i+1] +
                v1[i+2]*v2[i+2] + v1[i+3]*v2[i+3]
            norm1 += v1[i]*v1[i] + v1[i+1]*v1[i+1] +
                v1[i+2]*v1[i+2] + v1[i+3]*v1[i+3]
            norm2 += v2[i]*v2[i] + v2[i+1]*v2[i+1] +
                v2[i+2]*v2[i+2] + v2[i+3]*v2[i+3]
        } else {
            for j := i; j < len(v1); j++ {
                dot += v1[j] * v2[j]
                norm1 += v1[j] * v1[j]
                norm2 += v2[j] * v2[j]
            }
        }
    }
    return 1 - dot/math.Sqrt(norm1*norm2)
}

// Fallback implementation for Batch Euclidean distance
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
