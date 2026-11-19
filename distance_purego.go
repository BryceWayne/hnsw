//go:build !amd64 || purego

package hnsw

func l2Squared(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredGo(a, b[:n])
}

func dotProduct(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductGo(a, b[:n])
}
