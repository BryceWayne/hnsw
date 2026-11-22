package hnsw_test

import (
	"bytes"
	"testing"

	"github.com/BryceWayne/hnsw"
)

func FuzzReadFrom(f *testing.F) {
	// Seed with valid empty, int-keyed, string-keyed, and cosine serialized snapshots.
	if idx, err := hnsw.New[int](4, hnsw.WithSeed(1)); err == nil {
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err == nil {
			f.Add(buf.Bytes())
		}
		for i := 0; i < 8; i++ {
			_ = idx.Insert(i, []float32{float32(i), 1, 2, 3})
		}
		buf.Reset()
		if _, err := idx.WriteTo(&buf); err == nil {
			f.Add(buf.Bytes())
		}
	}

	if idx, err := hnsw.New[string](3, hnsw.WithMetric(hnsw.MetricCosine), hnsw.WithSeed(2)); err == nil {
		_ = idx.Insert("alpha", []float32{1, 0, 0})
		_ = idx.Insert("beta", []float32{0, 1, 0})
		var buf bytes.Buffer
		if _, err := idx.WriteTo(&buf); err == nil {
			f.Add(buf.Bytes())
		}
	}

	// Seed with malformed / truncated inputs.
	f.Add([]byte{})
	f.Add([]byte("HNSW\x01\x00\x00\x00"))
	f.Add([]byte("not an hnsw index"))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Neither int nor string deserialization may panic on arbitrary input.
		if idx, err := hnsw.ReadFrom[int](bytes.NewReader(data)); err == nil {
			if idx.Dim() <= 0 {
				t.Fatalf("deserialized invalid dim %d", idx.Dim())
			}
			if idx.Len() != len(idx.Keys()) {
				t.Fatalf("Len()=%d != len(Keys())=%d", idx.Len(), len(idx.Keys()))
			}
		}
		if idx, err := hnsw.ReadFrom[string](bytes.NewReader(data)); err == nil {
			if idx.Dim() <= 0 {
				t.Fatalf("deserialized invalid dim %d", idx.Dim())
			}
			if idx.Len() != len(idx.Keys()) {
				t.Fatalf("Len()=%d != len(Keys())=%d", idx.Len(), len(idx.Keys()))
			}
		}
	})
}
