//go:build arm64 && !purego

package hnsw

//go:noescape
func l2SquaredNEON(a, b *float32, n int) float32

//go:noescape
func dotProductNEON(a, b *float32, n int) float32

func l2Squared(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredNEON(&a[0], &b[0], n)
}

func dotProduct(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductNEON(&a[0], &b[0], n)
}

func l2SquaredF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredF16F32Go(a, b[:n])
}

func dotProductF16F32(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductF16F32Go(a, b[:n])
}

func l2SquaredF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return l2SquaredF16F16Go(a, b[:n])
}

func dotProductF16F16(a, b []uint16) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	return dotProductF16F16Go(a, b[:n])
}
