package hnsw

// Vector represents a point in multi-dimensional space
type Vector []float64

// DistanceFunc defines a function that computes distance between two vectors
type DistanceFunc func(Vector, Vector) float64
