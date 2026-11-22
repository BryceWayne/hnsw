package hnsw

import "math"

// Quantization identifies the vector storage encoding used by an [Index].
type Quantization uint8

const (
	// QuantNone stores full 32-bit IEEE-754 float32 vectors (4 bytes per dimension).
	QuantNone Quantization = iota

	// QuantFloat16 stores IEEE-754 binary16 (half-precision) vectors (2 bytes per dimension),
	// halving vector arena memory and memory bandwidth with minimal recall impact.
	QuantFloat16
)

// String returns the human-readable name of the quantization mode.
func (q Quantization) String() string {
	switch q {
	case QuantNone:
		return "none"
	case QuantFloat16:
		return "float16"
	default:
		return "unknown"
	}
}

// f16ToF32Table provides branchless O(1) lookup decoding from IEEE-754 binary16 to float32.
var f16ToF32Table [65536]float32

func init() {
	for i := 0; i < 65536; i++ {
		f16ToF32Table[i] = float16BitsToFloat32(uint16(i))
	}
}

// float16BitsToFloat32 converts an IEEE-754 binary16 bit pattern to float32.
func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x03FF)

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal binary16: normalize to float32.
		for (frac & 0x0400) == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x03FF
		f32Exp := (exp + (127 - 15)) << 23
		return math.Float32frombits(sign | f32Exp | (frac << 13))
	case 0x1F:
		if frac == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7FC00000 | (frac << 13))
	default:
		f32Exp := (exp + (127 - 15)) << 23
		return math.Float32frombits(sign | f32Exp | (frac << 13))
	}
}

// float32ToFloat16Finite converts a finite float32 to IEEE-754 binary16 using
// round-to-nearest-even, saturating magnitudes > 65504 to ±65504 (0x7BFF / 0xFBFF)
// so finite inputs never quantize to ±Inf.
func float32ToFloat16Finite(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x007FFFFF

	// Saturate large finite floats to max finite binary16 (65504.0 = 0x7BFF).
	if exp > 15 {
		return sign | 0x7BFF
	}
	// Subnormal or underflow in binary16.
	if exp < -14 {
		if exp < -25 {
			return sign
		}
		// Restore implicit leading 1 bit.
		mant := frac | 0x00800000
		shift := uint(-14 - exp + 13)
		roundBit := uint32(1) << (shift - 1)
		stickyMask := roundBit - 1
		halfMant := mant >> shift
		rem := mant & (roundBit | stickyMask)
		if rem > roundBit || (rem == roundBit && (halfMant&1) != 0) {
			halfMant++
		}
		return sign | uint16(halfMant)
	}

	// Normal binary16 with round-to-nearest-even.
	hExp := uint16(exp + 15)
	roundBit := uint32(1 << 12)
	rem := frac & 0x1FFF
	hFrac := frac >> 13
	if rem > roundBit || (rem == roundBit && (hFrac&1) != 0) {
		hFrac++
		if hFrac == 0x0400 {
			hFrac = 0
			hExp++
			if hExp >= 0x1F {
				return sign | 0x7BFF
			}
		}
	}
	return sign | (hExp << 10) | uint16(hFrac)
}

func isFloat16Finite(h uint16) bool {
	return (h & 0x7C00) != 0x7C00
}

func encodeFloat16Slice(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = float32ToFloat16Finite(v)
	}
}

func decodeFloat16Slice(dst []float32, src []uint16) {
	for i, h := range src {
		dst[i] = f16ToF32Table[h]
	}
}

func l2SquaredF16F32Go(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		d0 := f16ToF32Table[a[i]] - b[i]
		d1 := f16ToF32Table[a[i+1]] - b[i+1]
		d2 := f16ToF32Table[a[i+2]] - b[i+2]
		d3 := f16ToF32Table[a[i+3]] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		d := f16ToF32Table[a[i]] - b[i]
		sum += d * d
	}
	return sum
}

func dotProductF16F32Go(a []uint16, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		s0 += f16ToF32Table[a[i]] * b[i]
		s1 += f16ToF32Table[a[i+1]] * b[i+1]
		s2 += f16ToF32Table[a[i+2]] * b[i+2]
		s3 += f16ToF32Table[a[i+3]] * b[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += f16ToF32Table[a[i]] * b[i]
	}
	return sum
}

func l2SquaredF16F16Go(a, b []uint16) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		d0 := f16ToF32Table[a[i]] - f16ToF32Table[b[i]]
		d1 := f16ToF32Table[a[i+1]] - f16ToF32Table[b[i+1]]
		d2 := f16ToF32Table[a[i+2]] - f16ToF32Table[b[i+2]]
		d3 := f16ToF32Table[a[i+3]] - f16ToF32Table[b[i+3]]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		d := f16ToF32Table[a[i]] - f16ToF32Table[b[i]]
		sum += d * d
	}
	return sum
}

func dotProductF16F16Go(a, b []uint16) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		s0 += f16ToF32Table[a[i]] * f16ToF32Table[b[i]]
		s1 += f16ToF32Table[a[i+1]] * f16ToF32Table[b[i+1]]
		s2 += f16ToF32Table[a[i+2]] * f16ToF32Table[b[i+2]]
		s3 += f16ToF32Table[a[i+3]] * f16ToF32Table[b[i+3]]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += f16ToF32Table[a[i]] * f16ToF32Table[b[i]]
	}
	return sum
}

func cosineFallback64F16F32(a []uint16, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(f16ToF32Table[a[i]])
		fb := float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return clampCosine(float32(1.0 - sim))
}

func cosineFallback64F16F16(a, b []uint16) float32 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(f16ToF32Table[a[i]])
		fb := float64(f16ToF32Table[b[i]])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return clampCosine(float32(1.0 - sim))
}

func cosineWithInvNormsF16F32(a []uint16, b []float32, invA, invB float32) float32 {
	if invA == 0 || invB == 0 {
		return 1.0
	}
	if math.IsInf(float64(invA), 0) || math.IsInf(float64(invB), 0) ||
		invA > 1e15 || invB > 1e15 || invA < 1e-15 || invB < 1e-15 {
		return cosineFallback64F16F32(a, b)
	}
	dot := dotProductF16F32(a, b)
	if dot != dot || math.IsInf(float64(dot), 0) {
		return cosineFallback64F16F32(a, b)
	}
	return clampCosine(float32(1.0) - dot*invA*invB)
}

func cosineWithInvNormsF16F16(a, b []uint16, invA, invB float32) float32 {
	if invA == 0 || invB == 0 {
		return 1.0
	}
	if math.IsInf(float64(invA), 0) || math.IsInf(float64(invB), 0) ||
		invA > 1e15 || invB > 1e15 || invA < 1e-15 || invB < 1e-15 {
		return cosineFallback64F16F16(a, b)
	}
	dot := dotProductF16F16(a, b)
	if dot != dot || math.IsInf(float64(dot), 0) {
		return cosineFallback64F16F16(a, b)
	}
	return clampCosine(float32(1.0) - dot*invA*invB)
}

func euclideanFallback64F16F32(a []uint16, b []float32) float32 {
	var maxAbs float64
	for i := range a {
		d := math.Abs(float64(f16ToF32Table[a[i]]) - float64(b[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs == 0 {
		return 0
	}
	inv := 1.0 / maxAbs
	var sum float64
	for i := range a {
		scaled := (float64(f16ToF32Table[a[i]]) - float64(b[i])) * inv
		sum += scaled * scaled
	}
	return float32(maxAbs * math.Sqrt(sum))
}

func euclideanFastF16F32(a []uint16, b []float32) float32 {
	d2 := l2SquaredF16F32(a, b)
	if d2 < 1e-36 || math.IsInf(float64(d2), 0) {
		return euclideanFallback64F16F32(a, b)
	}
	return float32(math.Sqrt(float64(d2)))
}

func computeInvNormF16(v []uint16) float32 {
	normSq := dotProductF16F16(v, v)
	if normSq > 0 && !math.IsInf(float64(normSq), 0) {
		return float32(1.0 / math.Sqrt(float64(normSq)))
	}
	var sum64 float64
	for _, h := range v {
		fx := float64(f16ToF32Table[h])
		sum64 += fx * fx
	}
	if sum64 == 0 {
		return 0
	}
	return float32(1.0 / math.Sqrt(sum64))
}
