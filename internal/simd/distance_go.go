// Package simd implements architecture-accelerated (amd64 AVX2+FMA+F16C, arm64 NEON)
// and portable pure-Go float32 and IEEE-754 binary16 (FP16) distance kernels.
package simd

import "math"

// F16ToF32Table provides branchless O(1) lookup decoding from IEEE-754 binary16 to float32.
var F16ToF32Table [65536]float32

func init() {
	for i := 0; i < 65536; i++ {
		F16ToF32Table[i] = Float16BitsToFloat32(uint16(i))
	}
}

// Float16BitsToFloat32 converts an IEEE-754 binary16 bit pattern to float32.
func Float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x03FF)

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
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

// Float32ToFloat16Finite converts a finite float32 to IEEE-754 binary16 using
// round-to-nearest-even, saturating magnitudes > 65504 to ±65504 (0x7BFF / 0xFBFF).
func Float32ToFloat16Finite(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x007FFFFF

	if exp > 15 {
		return sign | 0x7BFF
	}
	if exp < -14 {
		if exp < -25 {
			return sign
		}
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

// IsFloat16Finite reports whether an IEEE-754 binary16 bit pattern is neither NaN nor ±Inf.
func IsFloat16Finite(h uint16) bool {
	return (h & 0x7C00) != 0x7C00
}

// EncodeFloat16Slice encodes finite float32 elements from src into binary16 values in dst.
func EncodeFloat16Slice(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = Float32ToFloat16Finite(v)
	}
}

// DecodeFloat16Slice decodes binary16 elements from src into float32 values in dst.
func DecodeFloat16Slice(dst []float32, src []uint16) {
	for i, h := range src {
		dst[i] = F16ToF32Table[h]
	}
}

// L2SquaredGo computes squared Euclidean distance using portable unrolled pure Go.
func L2SquaredGo(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+7 < n; i += 8 {
		a8 := a[i : i+8 : i+8]
		b8 := b[i : i+8 : i+8]
		d0 := a8[0] - b8[0]
		d1 := a8[1] - b8[1]
		d2 := a8[2] - b8[2]
		d3 := a8[3] - b8[3]
		d4 := a8[4] - b8[4]
		d5 := a8[5] - b8[5]
		d6 := a8[6] - b8[6]
		d7 := a8[7] - b8[7]
		s0 += d0*d0 + d4*d4
		s1 += d1*d1 + d5*d5
		s2 += d2*d2 + d6*d6
		s3 += d3*d3 + d7*d7
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// DotProductGo computes the inner product using portable unrolled pure Go.
func DotProductGo(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]
	b = b[:n]

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+7 < n; i += 8 {
		a8 := a[i : i+8 : i+8]
		b8 := b[i : i+8 : i+8]
		s0 += a8[0]*b8[0] + a8[4]*b8[4]
		s1 += a8[1]*b8[1] + a8[5]*b8[5]
		s2 += a8[2]*b8[2] + a8[6]*b8[6]
		s3 += a8[3]*b8[3] + a8[7]*b8[7]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// L2SquaredF16F32Go computes squared Euclidean distance between a binary16 vector and a float32 vector.
func L2SquaredF16F32Go(a []uint16, b []float32) float32 {
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
		d0 := F16ToF32Table[a[i]] - b[i]
		d1 := F16ToF32Table[a[i+1]] - b[i+1]
		d2 := F16ToF32Table[a[i+2]] - b[i+2]
		d3 := F16ToF32Table[a[i+3]] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		d := F16ToF32Table[a[i]] - b[i]
		sum += d * d
	}
	return sum
}

// DotProductF16F32Go computes the inner product between a binary16 vector and a float32 vector.
func DotProductF16F32Go(a []uint16, b []float32) float32 {
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
		s0 += F16ToF32Table[a[i]] * b[i]
		s1 += F16ToF32Table[a[i+1]] * b[i+1]
		s2 += F16ToF32Table[a[i+2]] * b[i+2]
		s3 += F16ToF32Table[a[i+3]] * b[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += F16ToF32Table[a[i]] * b[i]
	}
	return sum
}

// L2SquaredF16F16Go computes squared Euclidean distance between two binary16 vectors.
func L2SquaredF16F16Go(a, b []uint16) float32 {
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
		d0 := F16ToF32Table[a[i]] - F16ToF32Table[b[i]]
		d1 := F16ToF32Table[a[i+1]] - F16ToF32Table[b[i+1]]
		d2 := F16ToF32Table[a[i+2]] - F16ToF32Table[b[i+2]]
		d3 := F16ToF32Table[a[i+3]] - F16ToF32Table[b[i+3]]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		d := F16ToF32Table[a[i]] - F16ToF32Table[b[i]]
		sum += d * d
	}
	return sum
}

// DotProductF16F16Go computes the inner product between two binary16 vectors.
func DotProductF16F16Go(a, b []uint16) float32 {
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
		s0 += F16ToF32Table[a[i]] * F16ToF32Table[b[i]]
		s1 += F16ToF32Table[a[i+1]] * F16ToF32Table[b[i+1]]
		s2 += F16ToF32Table[a[i+2]] * F16ToF32Table[b[i+2]]
		s3 += F16ToF32Table[a[i+3]] * F16ToF32Table[b[i+3]]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += F16ToF32Table[a[i]] * F16ToF32Table[b[i]]
	}
	return sum
}
