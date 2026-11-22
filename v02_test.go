package hnsw

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestFloat16EncodingAndDecoding(t *testing.T) {
	if got := QuantNone.String(); got != "none" {
		t.Fatalf("QuantNone.String() = %q", got)
	}
	if got := QuantFloat16.String(); got != "float16" {
		t.Fatalf("QuantFloat16.String() = %q", got)
	}
	if got := Quantization(99).String(); got != "unknown" {
		t.Fatalf("Quantization(99).String() = %q", got)
	}

	// Exact round-trip for integers, powers of two, subnormals, signed zero, and saturation.
	exactVals := []float32{0, 1, -1, 0.5, -0.5, 2, 1024, 65504, -65504, 1.5, 0.25, 0.125}
	for _, v := range exactVals {
		h := float32ToFloat16Finite(v)
		if !isFloat16Finite(h) {
			t.Fatalf("float32ToFloat16Finite(%v) = 0x%04x (non-finite)", v, h)
		}
		got := f16ToF32Table[h]
		if got != v {
			t.Fatalf("roundtrip(%v) = %v (bits=0x%04x)", v, got, h)
		}
	}

	// Saturation of large finite float32s (> 65504) must stay finite (±65504).
	for _, v := range []float32{70000, 1e20, math.MaxFloat32, -70000, -1e20, -math.MaxFloat32} {
		h := float32ToFloat16Finite(v)
		if !isFloat16Finite(h) {
			t.Fatalf("expected finite saturation for %v, got 0x%04x", v, h)
		}
		got := f16ToF32Table[h]
		want := float32(65504)
		if v < 0 {
			want = -65504
		}
		if got != want {
			t.Fatalf("saturate(%v) = %v, want %v", v, got, want)
		}
	}

	// Tiny subnormal / underflow handling.
	for _, v := range []float32{1e-6, -1e-6, 1e-10, 1e-30} {
		h := float32ToFloat16Finite(v)
		if !isFloat16Finite(h) {
			t.Fatalf("tiny %v produced non-finite 0x%04x", v, h)
		}
	}

	// Special binary16 bit patterns: +Inf, -Inf, NaN, subnormal.
	if isFloat16Finite(0x7C00) || isFloat16Finite(0xFC00) || isFloat16Finite(0x7E00) {
		t.Fatal("expected Inf/NaN binary16 patterns to fail isFloat16Finite")
	}
	if !math.IsInf(float64(f16ToF32Table[0x7C00]), 1) {
		t.Fatalf("0x7C00 decoded to %v, want +Inf", f16ToF32Table[0x7C00])
	}
	if !math.IsNaN(float64(f16ToF32Table[0x7E00])) {
		t.Fatalf("0x7E00 decoded to %v, want NaN", f16ToF32Table[0x7E00])
	}
}

func TestSearchIntoZeroAllocsAndBehavior(t *testing.T) {
	idx, err := New[int](128, WithM(16), WithMmax(32), WithEfConstruction(64), WithEfSearch(32), WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()
	}

	// Empty index with non-nil dst should reuse dst[:0].
	dst := make([]SearchResult[int], 0, 16)
	res, err := idx.SearchInto(query, 10, dst)
	if err != nil {
		t.Fatalf("empty SearchInto err = %v", err)
	}
	if len(res) != 0 || cap(res) != 16 {
		t.Fatalf("empty SearchInto len=%d cap=%d, want 0/16", len(res), cap(res))
	}

	items := make([]Item[int], 200)
	for i := range items {
		v := make([]float32, 128)
		for j := range v {
			v[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items); err != nil {
		t.Fatal(err)
	}

	want, err := idx.Search(query, 10)
	if err != nil {
		t.Fatal(err)
	}
	got, err := idx.SearchInto(query, 10, dst)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("SearchInto len=%d != Search len=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SearchInto[%d]=%+v != Search[%d]=%+v", i, got[i], i, want[i])
		}
	}

	// Small dst should grow cleanly.
	smallDst := make([]SearchResult[int], 0, 2)
	gotGrow, err := idx.SearchInto(query, 10, smallDst)
	if err != nil || len(gotGrow) != 10 {
		t.Fatalf("SearchInto with small dst: len=%d err=%v", len(gotGrow), err)
	}

	if !raceDetectorEnabled {
		// Warm up scratchPool, then verify steady-state SearchInto is 0 allocs/op.
		_, _ = idx.SearchInto(query, 10, dst)
		allocs := testing.AllocsPerRun(200, func() {
			r, err := idx.SearchInto(query, 10, dst)
			if err != nil || len(r) != 10 {
				t.Fatalf("SearchInto err=%v len=%d", err, len(r))
			}
		})
		if allocs != 0 {
			t.Fatalf("SearchInto allocs/op = %.1f, want 0", allocs)
		}
	}
}

func TestWithMaxDistance(t *testing.T) {
	idx, err := New[int](2, WithM(8), WithEfConstruction(32), WithMetric(MetricEuclidean), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		if err := idx.Insert(i, []float32{float32(i), 0}); err != nil {
			t.Fatal(err)
		}
	}

	// NaN max distance must return ErrInvalidConfig.
	if _, err := idx.Search([]float32{0, 0}, 5, WithMaxDistance(float32(math.NaN()))); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("WithMaxDistance(NaN) err = %v, want ErrInvalidConfig", err)
	}

	// Euclidean true L2 distance threshold <= 2.5 -> IDs 0, 1, 2.
	res, err := idx.Search([]float32{0, 0}, 5, WithMaxDistance(2.5))
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 {
		t.Fatalf("got %d results (%+v), want 3", len(res), res)
	}
	for _, r := range res {
		if r.Distance > 2.5 {
			t.Fatalf("result %+v exceeded maxDist 2.5", r)
		}
	}

	// SearchInto + WithMaxDistance.
	dst := make([]SearchResult[int], 0, 5)
	resInto, err := idx.SearchInto([]float32{0, 0}, 5, dst, WithMaxDistance(1.5))
	if err != nil {
		t.Fatal(err)
	}
	if len(resInto) != 2 {
		t.Fatalf("SearchInto WithMaxDistance(1.5) got %d, want 2", len(resInto))
	}

	// SearchFilter + WithMaxDistance.
	resFilt, err := idx.SearchFilter([]float32{0, 0}, 5, func(id int) bool { return id%2 == 0 }, WithMaxDistance(2.5))
	if err != nil {
		t.Fatal(err)
	}
	if len(resFilt) != 2 || resFilt[0].ID != 0 || resFilt[1].ID != 2 {
		t.Fatalf("SearchFilter WithMaxDistance got %+v, want IDs [0, 2]", resFilt)
	}

	// SearchBatch + WithMaxDistance.
	batch, err := idx.SearchBatch([][]float32{{0, 0}, {5, 0}}, 5, WithMaxDistance(1.1), WithWorkers(2))
	if err != nil {
		t.Fatal(err)
	}
	if len(batch[0]) != 2 || len(batch[1]) != 2 {
		t.Fatalf("SearchBatch WithMaxDistance got lens (%d, %d), want (2, 2)", len(batch[0]), len(batch[1]))
	}

	// Underflow Euclidean rescore path with WithMaxDistance.
	tinyIdx, err := New[int](2, WithM(8), WithEfConstruction(32), WithMetric(MetricEuclidean), WithSeed(2))
	if err != nil {
		t.Fatal(err)
	}
	_ = tinyIdx.Insert(1, []float32{1e-19, 0})
	_ = tinyIdx.Insert(2, []float32{3e-19, 0})
	_ = tinyIdx.Insert(3, []float32{5e-19, 0})
	tinyRes, err := tinyIdx.Search([]float32{0, 0}, 3, WithMaxDistance(3.5e-19))
	if err != nil {
		t.Fatal(err)
	}
	if len(tinyRes) != 2 {
		t.Fatalf("tiny WithMaxDistance got %d (%+v), want 2", len(tinyRes), tinyRes)
	}
}

func TestQuantFloat16FullLifecycleAndPersistence(t *testing.T) {
	// Invalid quantization option.
	if _, err := New[int](4, WithQuantization(Quantization(99))); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("invalid quantization err = %v, want ErrInvalidConfig", err)
	}

	metrics := []Metric{MetricEuclidean, MetricSquaredEuclidean, MetricCosine, MetricDotProduct}
	for _, metric := range metrics {
		t.Run(metric.String(), func(t *testing.T) {
			const dim = 64
			const n = 200
			rng := rand.New(rand.NewSource(123))

			idxF16, err := New[int](dim, WithM(12), WithMmax(24), WithEfConstruction(64), WithEfSearch(32), WithMetric(metric), WithQuantization(QuantFloat16), WithSeed(42))
			if err != nil {
				t.Fatal(err)
			}
			idxF32, err := New[int](dim, WithM(12), WithMmax(24), WithEfConstruction(64), WithEfSearch(32), WithMetric(metric), WithQuantization(QuantNone), WithSeed(42))
			if err != nil {
				t.Fatal(err)
			}

			items := make([]Item[int], n)
			for i := range items {
				v := make([]float32, dim)
				for j := range v {
					v[j] = float32(rng.Float64()*2 - 1)
				}
				items[i] = Item[int]{ID: i, Vector: v}
			}
			if err := idxF16.InsertBatch(items); err != nil {
				t.Fatal(err)
			}
			if err := idxF32.InsertBatch(items); err != nil {
				t.Fatal(err)
			}

			st16 := idxF16.Stats()
			st32 := idxF32.Stats()
			if st16.Quantization != QuantFloat16 || st32.Quantization != QuantNone {
				t.Fatalf("Stats quantization mismatch: f16=%v f32=%v", st16.Quantization, st32.Quantization)
			}
			if st16.MemoryBytes >= st32.MemoryBytes {
				t.Fatalf("expected F16 MemoryBytes (%d) < F32 MemoryBytes (%d)", st16.MemoryBytes, st32.MemoryBytes)
			}

			// Verify Get decodes close to original.
			gotVec, ok := idxF16.Get(0)
			if !ok || len(gotVec) != dim {
				t.Fatalf("Get(0) failed: ok=%v len=%d", ok, len(gotVec))
			}
			for j := range gotVec {
				if math.Abs(float64(gotVec[j]-items[0].Vector[j])) > 1e-2 {
					t.Fatalf("Get(0)[%d] = %v, orig = %v", j, gotVec[j], items[0].Vector[j])
				}
			}

			// Verify self-search returns exact ID 0 as top-1 for non-dot metrics (and valid results for dot).
			res, err := idxF16.Search(items[0].Vector, 5)
			if err != nil || len(res) != 5 {
				t.Fatalf("Search failed: len=%d err=%v", len(res), err)
			}
			if metric != MetricDotProduct && res[0].ID != 0 {
				t.Fatalf("metric %s: self-query top-1 = %d, want 0", metric, res[0].ID)
			}

			// Delete + Compact lifecycle.
			for i := 0; i < 20; i++ {
				idxF16.Delete(i)
			}
			if err := idxF16.Compact(); err != nil {
				t.Fatalf("Compact failed: %v", err)
			}
			if idxF16.Len() != n-20 {
				t.Fatalf("after Compact Len() = %d, want %d", idxF16.Len(), n-20)
			}

			// Persistence v2 round-trip and smaller byte size than v1.
			var buf16, buf32 bytes.Buffer
			n16, err := idxF16.WriteTo(&buf16)
			if err != nil {
				t.Fatalf("F16 WriteTo failed: %v", err)
			}
			for i := 0; i < 20; i++ {
				idxF32.Delete(i)
			}
			n32, err := idxF32.WriteTo(&buf32)
			if err != nil {
				t.Fatalf("F32 WriteTo failed: %v", err)
			}
			if n16 >= n32 {
				t.Fatalf("expected F16 serialized bytes (%d) < F32 (%d)", n16, n32)
			}

			// Conflicting WithQuantization on ReadFrom must return ErrInvalidConfig.
			if _, err := ReadFrom[int](bytes.NewReader(buf16.Bytes()), WithQuantization(QuantNone)); !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("mismatched quantization ReadFrom err = %v, want ErrInvalidConfig", err)
			}

			reloaded, err := ReadFrom[int](bytes.NewReader(buf16.Bytes()))
			if err != nil {
				t.Fatalf("F16 ReadFrom failed: %v", err)
			}
			if reloaded.Stats().Quantization != QuantFloat16 || reloaded.Len() != idxF16.Len() {
				t.Fatalf("reloaded mismatch: quant=%v len=%d", reloaded.Stats().Quantization, reloaded.Len())
			}
			r2, err := reloaded.Search(items[25].Vector, 5)
			if err != nil || len(r2) != 5 {
				t.Fatalf("reloaded Search failed: len=%d err=%v", len(r2), err)
			}
			if metric != MetricDotProduct && r2[0].ID != 25 {
				t.Fatalf("reloaded metric %s: self-query top-1 = %d, want 25", metric, r2[0].ID)
			}
		})
	}

	// Custom distance + QuantFloat16.
	l1 := func(a, b []float32) float32 {
		var s float32
		for i := range a {
			d := a[i] - b[i]
			if d < 0 {
				d = -d
			}
			s += d
		}
		return s
	}
	cIdx, err := New[int](4, WithCustomDistance(l1), WithQuantization(QuantFloat16), WithM(4), WithEfConstruction(16), WithSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 30; i++ {
		if err := cIdx.Insert(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	if d := cIdx.distNodes(0, 5); math.Abs(float64(d-5)) > 0.01 {
		t.Fatalf("custom F16 distNodes(0,5) = %v, want 5", d)
	}
	cres, err := cIdx.Search([]float32{10, 0, 0, 0}, 1)
	if err != nil || len(cres) != 1 || cres[0].ID != 10 {
		t.Fatalf("custom F16 Search = %+v, err=%v", cres, err)
	}

	// Corrupt v2 header (bad quantization, non-zero reserved byte, non-finite FP16 payload).
	var v2Buf bytes.Buffer
	if _, err := cIdx.WriteTo(&v2Buf); err != nil {
		t.Fatal(err)
	}
	raw := v2Buf.Bytes()

	badQuant := append([]byte(nil), raw...)
	badQuant[14] = 99
	if _, err := ReadFrom[int](bytes.NewReader(badQuant), WithCustomDistance(l1)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("bad v2 quant err = %v, want ErrCorruptedIndex", err)
	}

	badRes := append([]byte(nil), raw...)
	badRes[15] = 1
	if _, err := ReadFrom[int](bytes.NewReader(badRes), WithCustomDistance(l1)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("bad v2 reserved err = %v, want ErrCorruptedIndex", err)
	}

	// Inject NaN (0x7E00) into first FP16 vector element (header 48 + tail 7 + keyTypeName + 30*8 keys).
	keyTypeLen := int(binary.LittleEndian.Uint16(raw[53:55]))
	vecStart := 48 + 7 + keyTypeLen + 30*8
	badNaN := append([]byte(nil), raw...)
	binary.LittleEndian.PutUint16(badNaN[vecStart:vecStart+2], 0x7E00)
	if _, err := ReadFrom[int](bytes.NewReader(badNaN), WithCustomDistance(l1)); !errors.Is(err, ErrCorruptedIndex) {
		t.Fatalf("non-finite F16 payload err = %v, want ErrCorruptedIndex", err)
	}
}

func TestFloat16EdgeCases(t *testing.T) {
	// Empty slice guards.
	if l2SquaredF16F32Go(nil, nil) != 0 || dotProductF16F32Go(nil, nil) != 0 ||
		l2SquaredF16F16Go(nil, nil) != 0 || dotProductF16F16Go(nil, nil) != 0 ||
		l2SquaredF16F32(nil, nil) != 0 || dotProductF16F32(nil, nil) != 0 ||
		l2SquaredF16F16(nil, nil) != 0 || dotProductF16F16(nil, nil) != 0 {
		t.Fatal("expected 0 for empty F16 slices")
	}

	// Zero-norm and extreme invNorm fallbacks.
	zeroH := []uint16{0, 0}
	oneH := []uint16{0x3C00, 0}
	if got := computeInvNormF16(zeroH); got != 0 {
		t.Fatalf("computeInvNormF16(zero) = %v, want 0", got)
	}
	if got := cosineWithInvNormsF16F32(zeroH, []float32{1, 0}, 0, 1); got != 1.0 {
		t.Fatalf("cosineWithInvNormsF16F32(zero) = %v, want 1.0", got)
	}
	if got := cosineWithInvNormsF16F16(zeroH, oneH, 0, 1); got != 1.0 {
		t.Fatalf("cosineWithInvNormsF16F16(zero) = %v, want 1.0", got)
	}
	if got := cosineWithInvNormsF16F32(oneH, []float32{1e20, 0}, 1, 1e-20); got > 1e-5 {
		t.Fatalf("cosineWithInvNormsF16F32(extreme) = %v, want ~0", got)
	}
	if got := cosineWithInvNormsF16F16(oneH, oneH, 1e20, 1); got > 1e-5 {
		t.Fatalf("cosineWithInvNormsF16F16(extreme) = %v, want ~0", got)
	}
	if got := cosineFallback64F16F32(zeroH, []float32{1, 0}); got != 1.0 {
		t.Fatalf("cosineFallback64F16F32(zero) = %v, want 1.0", got)
	}
	if got := cosineFallback64F16F16(zeroH, oneH); got != 1.0 {
		t.Fatalf("cosineFallback64F16F16(zero) = %v, want 1.0", got)
	}

	// Euclidean fallback & rescore paths for FP16.
	if got := euclideanFallback64F16F32(oneH, []float32{1, 0}); got != 0 {
		t.Fatalf("euclideanFallback64F16F32(identical) = %v, want 0", got)
	}
	if got := euclideanFastF16F32(oneH, []float32{1, 0}); got != 0 {
		t.Fatalf("euclideanFastF16F32(identical) = %v, want 0", got)
	}
	if got := euclideanFastF16F32(oneH, []float32{4, 4}); math.Abs(float64(got-5)) > 1e-5 {
		t.Fatalf("euclideanFastF16F32 = %v, want 5", got)
	}

	// Verify FP16 Euclidean rescore path inside Search.
	idx, err := New[int](2, WithM(4), WithEfConstruction(16), WithMetric(MetricEuclidean), WithQuantization(QuantFloat16), WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	_ = idx.Insert(1, []float32{0, 0})
	_ = idx.Insert(2, []float32{0, 0})
	_ = idx.Insert(3, []float32{1, 0})
	res, err := idx.Search([]float32{0, 0}, 3)
	if err != nil || len(res) != 3 {
		t.Fatalf("FP16 zero-distance Search err=%v len=%d", err, len(res))
	}
}

func BenchmarkSearchInto(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	idx, err := New[int](128, WithM(16), WithMmax(32), WithEfConstruction(100), WithSeed(42))
	if err != nil {
		b.Fatal(err)
	}
	items := make([]Item[int], 1000)
	for i := range items {
		v := make([]float32, 128)
		for j := range v {
			v[j] = rng.Float32()
		}
		items[i] = Item[int]{ID: i, Vector: v}
	}
	if err := idx.InsertBatch(items); err != nil {
		b.Fatal(err)
	}
	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()
	}
	dst := make([]SearchResult[int], 0, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.SearchInto(query, 10, dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantFloat16VsFloat32(b *testing.B) {
	for _, dim := range []int{768, 1536} {
		const n = 5000
		rng := rand.New(rand.NewSource(42))
		items := make([]Item[int], n)
		for i := range items {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(rng.Float64()*2 - 1)
			}
			items[i] = Item[int]{ID: i, Vector: v}
		}
		queries := make([][]float32, 32)
		for i := range queries {
			q := make([]float32, dim)
			for j := range q {
				q[j] = float32(rng.Float64()*2 - 1)
			}
			queries[i] = q
		}

		for _, q := range []Quantization{QuantNone, QuantFloat16} {
			idx, err := New[int](dim, WithM(16), WithMmax(32), WithEfConstruction(64), WithEfSearch(32), WithQuantization(q), WithSeed(42))
			if err != nil {
				b.Fatal(err)
			}
			if err := idx.InsertBatch(items); err != nil {
				b.Fatal(err)
			}

			b.Run(fmt.Sprintf("Search_dim=%d_%s", dim, q), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					dst := make([]SearchResult[int], 0, 10)
					qi := 0
					for pb.Next() {
						if _, err := idx.SearchInto(queries[qi%len(queries)], 10, dst); err != nil {
							b.Fatal(err)
						}
						qi++
					}
				})
				b.StopTimer()
				if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
					b.ReportMetric(float64(b.N)/elapsed, "queries/s")
				}
				b.ReportMetric(float64(idx.Stats().MemoryBytes)/(1024*1024), "MB_arena")
			})

			var buf bytes.Buffer
			if _, err := idx.WriteTo(&buf); err != nil {
				b.Fatal(err)
			}
			raw := buf.Bytes()
			b.Run(fmt.Sprintf("WriteTo_dim=%d_%s", dim, q), func(b *testing.B) {
				var out bytes.Buffer
				out.Grow(len(raw))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out.Reset()
					if _, err := idx.WriteTo(&out); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(len(raw))/(1024*1024), "MB_file")
			})
			b.Run(fmt.Sprintf("ReadFrom_dim=%d_%s", dim, q), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := ReadFrom[int](bytes.NewReader(raw)); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(len(raw))/(1024*1024), "MB_file")
			})
		}
	}
}
