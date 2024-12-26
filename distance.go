package hnsw

import "math"

// Euclidean computes the Euclidean distance between two vectors
func Euclidean(v1, v2 Vector) float64 {
    sum := 0.0
    for i := range v1 {
        diff := v1[i] - v2[i]
        sum += diff * diff
    }
    return math.Sqrt(sum)
}

// Cosine computes the cosine distance between two vectors
func Cosine(v1, v2 Vector) float64 {
    dot := 0.0
    norm1 := 0.0
    norm2 := 0.0
    for i := range v1 {
        dot += v1[i] * v2[i]
        norm1 += v1[i] * v1[i]
        norm2 += v2[i] * v2[i]
    }
    return 1 - (dot / (math.Sqrt(norm1) * math.Sqrt(norm2)))
}
