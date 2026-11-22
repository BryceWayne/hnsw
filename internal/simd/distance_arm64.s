//go:build arm64 && !purego

#include "textflag.h"

// ARMv8-A NEON 4xfloat32 & FP16 widening vector encodings (verified via go tool objdump):
// FADD   Vd.4S, Vn.4S, Vm.4S : 0x4E20D400 | (Rm<<16) | (Rn<<5) | Rd
// FSUB   Vd.4S, Vn.4S, Vm.4S : 0x4EA0D400 | (Rm<<16) | (Rn<<5) | Rd
// FADDP  Vd.4S, Vn.4S, Vm.4S : 0x6E20D400 | (Rm<<16) | (Rn<<5) | Rd
// FCVTL  Vd.4S, Vn.4H        : 0x0E217800 | (Rn<<5) | Rd
// FCVTL2 Vd.4S, Vn.8H        : 0x4E217800 | (Rn<<5) | Rd
// FCVT   Sd, Hn              : 0x1EE24000 | (Rn<<5) | Rd
#define FADD_4S(d, n, m)  WORD $(0x4E20D400 | ((m)<<16) | ((n)<<5) | (d))
#define FSUB_4S(d, n, m)  WORD $(0x4EA0D400 | ((m)<<16) | ((n)<<5) | (d))
#define FADDP_4S(d, n, m) WORD $(0x6E20D400 | ((m)<<16) | ((n)<<5) | (d))
#define FCVTL_4S(d, n)    WORD $(0x0E217800 | ((n)<<5) | (d))
#define FCVTL2_4S(d, n)   WORD $(0x4E217800 | ((n)<<5) | (d))
#define FCVT_SH(d, n)     WORD $(0x1EE24000 | ((n)<<5) | (d))

// func l2SquaredNEON(a, b *float32, n int) float32
TEXT ·l2SquaredNEON(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2
	CMP	$0, R2
	BLE	l2_zero

	VEOR	V0.B16, V0.B16, V0.B16
	LSR	$4, R2, R3			// n / 16
	CBZ	R3, l2_loop4_check

	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

l2_loop16:
	VLD1.P	64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P	64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	FSUB_4S(4, 4, 8)
	FSUB_4S(5, 5, 9)
	FSUB_4S(6, 6, 10)
	FSUB_4S(7, 7, 11)
	VFMLA	V4.S4, V4.S4, V0.S4
	VFMLA	V5.S4, V5.S4, V1.S4
	VFMLA	V6.S4, V6.S4, V2.S4
	VFMLA	V7.S4, V7.S4, V3.S4
	SUB	$1, R3
	CBNZ	R3, l2_loop16

	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)

l2_loop4_check:
	AND	$15, R2, R3
	LSR	$2, R3, R3			// (n % 16) / 4
	CBZ	R3, l2_reduce

l2_loop4:
	VLD1.P	16(R0), [V4.S4]
	VLD1.P	16(R1), [V5.S4]
	FSUB_4S(4, 4, 5)
	VFMLA	V4.S4, V4.S4, V0.S4
	SUB	$1, R3
	CBNZ	R3, l2_loop4

l2_reduce:
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

	AND	$3, R2, R3			// n % 4
	CBZ	R3, l2_done

l2_tail:
	FMOVS.P	4(R0), F4
	FMOVS.P	4(R1), F5
	FSUBS	F5, F4, F4
	FMADDS	F4, F0, F4, F0
	SUB	$1, R3
	CBNZ	R3, l2_tail

l2_done:
	FMOVS	F0, ret+24(FP)
	RET

l2_zero:
	MOVW	ZR, ret+24(FP)
	RET

// func dotProductNEON(a, b *float32, n int) float32
TEXT ·dotProductNEON(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2
	CMP	$0, R2
	BLE	dot_zero

	VEOR	V0.B16, V0.B16, V0.B16
	LSR	$4, R2, R3			// n / 16
	CBZ	R3, dot_loop4_check

	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

dot_loop16:
	VLD1.P	64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P	64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VFMLA	V8.S4, V4.S4, V0.S4
	VFMLA	V9.S4, V5.S4, V1.S4
	VFMLA	V10.S4, V6.S4, V2.S4
	VFMLA	V11.S4, V7.S4, V3.S4
	SUB	$1, R3
	CBNZ	R3, dot_loop16

	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)

dot_loop4_check:
	AND	$15, R2, R3
	LSR	$2, R3, R3			// (n % 16) / 4
	CBZ	R3, dot_reduce

dot_loop4:
	VLD1.P	16(R0), [V4.S4]
	VLD1.P	16(R1), [V5.S4]
	VFMLA	V5.S4, V4.S4, V0.S4
	SUB	$1, R3
	CBNZ	R3, dot_loop4

dot_reduce:
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

	AND	$3, R2, R3			// n % 4
	CBZ	R3, dot_done

dot_tail:
	FMOVS.P	4(R0), F4
	FMOVS.P	4(R1), F5
	FMADDS	F5, F0, F4, F0
	SUB	$1, R3
	CBNZ	R3, dot_tail

dot_done:
	FMOVS	F0, ret+24(FP)
	RET

dot_zero:
	MOVW	ZR, ret+24(FP)
	RET

// func l2SquaredF16F32NEON(a *uint16, b *float32, n int) float32
TEXT ·l2SquaredF16F32NEON(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2
	CMP	$0, R2
	BLE	l2_f16f32_zero

	VEOR	V0.B16, V0.B16, V0.B16
	LSR	$4, R2, R3			// n / 16
	CBZ	R3, l2_f16f32_loop4_check

	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

l2_f16f32_loop16:
	VLD1.P	32(R0), [V12.H8, V13.H8]
	VLD1.P	64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	FCVTL_4S(4, 12)
	FCVTL2_4S(5, 12)
	FCVTL_4S(6, 13)
	FCVTL2_4S(7, 13)
	FSUB_4S(4, 4, 8)
	FSUB_4S(5, 5, 9)
	FSUB_4S(6, 6, 10)
	FSUB_4S(7, 7, 11)
	VFMLA	V4.S4, V4.S4, V0.S4
	VFMLA	V5.S4, V5.S4, V1.S4
	VFMLA	V6.S4, V6.S4, V2.S4
	VFMLA	V7.S4, V7.S4, V3.S4
	SUB	$1, R3
	CBNZ	R3, l2_f16f32_loop16

	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)

l2_f16f32_loop4_check:
	AND	$15, R2, R3
	LSR	$2, R3, R3			// (n % 16) / 4
	CBZ	R3, l2_f16f32_reduce

l2_f16f32_loop4:
	FMOVD.P	8(R0), F12
	VLD1.P	16(R1), [V5.S4]
	FCVTL_4S(4, 12)
	FSUB_4S(4, 4, 5)
	VFMLA	V4.S4, V4.S4, V0.S4
	SUB	$1, R3
	CBNZ	R3, l2_f16f32_loop4

l2_f16f32_reduce:
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

	AND	$3, R2, R3			// n % 4
	CBZ	R3, l2_f16f32_done

l2_f16f32_tail:
	MOVHU.P	2(R0), R4
	FMOVS	R4, F4
	FCVT_SH(4, 4)
	FMOVS.P	4(R1), F5
	FSUBS	F5, F4, F4
	FMADDS	F4, F0, F4, F0
	SUB	$1, R3
	CBNZ	R3, l2_f16f32_tail

l2_f16f32_done:
	FMOVS	F0, ret+24(FP)
	RET

l2_f16f32_zero:
	MOVW	ZR, ret+24(FP)
	RET

// func dotProductF16F32NEON(a *uint16, b *float32, n int) float32
TEXT ·dotProductF16F32NEON(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2
	CMP	$0, R2
	BLE	dot_f16f32_zero

	VEOR	V0.B16, V0.B16, V0.B16
	LSR	$4, R2, R3			// n / 16
	CBZ	R3, dot_f16f32_loop4_check

	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

dot_f16f32_loop16:
	VLD1.P	32(R0), [V12.H8, V13.H8]
	VLD1.P	64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	FCVTL_4S(4, 12)
	FCVTL2_4S(5, 12)
	FCVTL_4S(6, 13)
	FCVTL2_4S(7, 13)
	VFMLA	V8.S4, V4.S4, V0.S4
	VFMLA	V9.S4, V5.S4, V1.S4
	VFMLA	V10.S4, V6.S4, V2.S4
	VFMLA	V11.S4, V7.S4, V3.S4
	SUB	$1, R3
	CBNZ	R3, dot_f16f32_loop16

	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)

dot_f16f32_loop4_check:
	AND	$15, R2, R3
	LSR	$2, R3, R3			// (n % 16) / 4
	CBZ	R3, dot_f16f32_reduce

dot_f16f32_loop4:
	FMOVD.P	8(R0), F12
	VLD1.P	16(R1), [V5.S4]
	FCVTL_4S(4, 12)
	VFMLA	V5.S4, V4.S4, V0.S4
	SUB	$1, R3
	CBNZ	R3, dot_f16f32_loop4

dot_f16f32_reduce:
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

	AND	$3, R2, R3			// n % 4
	CBZ	R3, dot_f16f32_done

dot_f16f32_tail:
	MOVHU.P	2(R0), R4
	FMOVS	R4, F4
	FCVT_SH(4, 4)
	FMOVS.P	4(R1), F5
	FMADDS	F5, F0, F4, F0
	SUB	$1, R3
	CBNZ	R3, dot_f16f32_tail

dot_f16f32_done:
	FMOVS	F0, ret+24(FP)
	RET

dot_f16f32_zero:
	MOVW	ZR, ret+24(FP)
	RET

// func l2SquaredF16F16NEON(a, b *uint16, n int) float32
TEXT ·l2SquaredF16F16NEON(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2
	CMP	$0, R2
	BLE	l2_f16f16_zero

	VEOR	V0.B16, V0.B16, V0.B16
	LSR	$4, R2, R3			// n / 16
	CBZ	R3, l2_f16f16_loop4_check

	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

l2_f16f16_loop16:
	VLD1.P	32(R0), [V12.H8, V13.H8]
	VLD1.P	32(R1), [V14.H8, V15.H8]
	FCVTL_4S(4, 12)
	FCVTL2_4S(5, 12)
	FCVTL_4S(6, 13)
	FCVTL2_4S(7, 13)
	FCVTL_4S(8, 14)
	FCVTL2_4S(9, 14)
	FCVTL_4S(10, 15)
	FCVTL2_4S(11, 15)
	FSUB_4S(4, 4, 8)
	FSUB_4S(5, 5, 9)
	FSUB_4S(6, 6, 10)
	FSUB_4S(7, 7, 11)
	VFMLA	V4.S4, V4.S4, V0.S4
	VFMLA	V5.S4, V5.S4, V1.S4
	VFMLA	V6.S4, V6.S4, V2.S4
	VFMLA	V7.S4, V7.S4, V3.S4
	SUB	$1, R3
	CBNZ	R3, l2_f16f16_loop16

	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)

l2_f16f16_loop4_check:
	AND	$15, R2, R3
	LSR	$2, R3, R3			// (n % 16) / 4
	CBZ	R3, l2_f16f16_reduce

l2_f16f16_loop4:
	FMOVD.P	8(R0), F12
	FMOVD.P	8(R1), F13
	FCVTL_4S(4, 12)
	FCVTL_4S(5, 13)
	FSUB_4S(4, 4, 5)
	VFMLA	V4.S4, V4.S4, V0.S4
	SUB	$1, R3
	CBNZ	R3, l2_f16f16_loop4

l2_f16f16_reduce:
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

	AND	$3, R2, R3			// n % 4
	CBZ	R3, l2_f16f16_done

l2_f16f16_tail:
	MOVHU.P	2(R0), R4
	FMOVS	R4, F4
	FCVT_SH(4, 4)
	MOVHU.P	2(R1), R5
	FMOVS	R5, F5
	FCVT_SH(5, 5)
	FSUBS	F5, F4, F4
	FMADDS	F4, F0, F4, F0
	SUB	$1, R3
	CBNZ	R3, l2_f16f16_tail

l2_f16f16_done:
	FMOVS	F0, ret+24(FP)
	RET

l2_f16f16_zero:
	MOVW	ZR, ret+24(FP)
	RET

// func dotProductF16F16NEON(a, b *uint16, n int) float32
TEXT ·dotProductF16F16NEON(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2
	CMP	$0, R2
	BLE	dot_f16f16_zero

	VEOR	V0.B16, V0.B16, V0.B16
	LSR	$4, R2, R3			// n / 16
	CBZ	R3, dot_f16f16_loop4_check

	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

dot_f16f16_loop16:
	VLD1.P	32(R0), [V12.H8, V13.H8]
	VLD1.P	32(R1), [V14.H8, V15.H8]
	FCVTL_4S(4, 12)
	FCVTL2_4S(5, 12)
	FCVTL_4S(6, 13)
	FCVTL2_4S(7, 13)
	FCVTL_4S(8, 14)
	FCVTL2_4S(9, 14)
	FCVTL_4S(10, 15)
	FCVTL2_4S(11, 15)
	VFMLA	V8.S4, V4.S4, V0.S4
	VFMLA	V9.S4, V5.S4, V1.S4
	VFMLA	V10.S4, V6.S4, V2.S4
	VFMLA	V11.S4, V7.S4, V3.S4
	SUB	$1, R3
	CBNZ	R3, dot_f16f16_loop16

	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)

dot_f16f16_loop4_check:
	AND	$15, R2, R3
	LSR	$2, R3, R3			// (n % 16) / 4
	CBZ	R3, dot_f16f16_reduce

dot_f16f16_loop4:
	FMOVD.P	8(R0), F12
	FMOVD.P	8(R1), F13
	FCVTL_4S(4, 12)
	FCVTL_4S(5, 13)
	VFMLA	V5.S4, V4.S4, V0.S4
	SUB	$1, R3
	CBNZ	R3, dot_f16f16_loop4

dot_f16f16_reduce:
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

	AND	$3, R2, R3			// n % 4
	CBZ	R3, dot_f16f16_done

dot_f16f16_tail:
	MOVHU.P	2(R0), R4
	FMOVS	R4, F4
	FCVT_SH(4, 4)
	MOVHU.P	2(R1), R5
	FMOVS	R5, F5
	FCVT_SH(5, 5)
	FMADDS	F5, F0, F4, F0
	SUB	$1, R3
	CBNZ	R3, dot_f16f16_tail

dot_f16f16_done:
	FMOVS	F0, ret+24(FP)
	RET

dot_f16f16_zero:
	MOVW	ZR, ret+24(FP)
	RET
