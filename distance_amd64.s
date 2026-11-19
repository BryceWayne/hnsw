//go:build amd64 && !purego

#include "textflag.h"

// func l2SquaredAVX2(a, b *float32, n int) float32
TEXT ·l2SquaredAVX2(SB), NOSPLIT, $0-28
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	n+16(FP), BX
	TESTQ	BX, BX
	JLE	l2_zero

	VXORPS	Y0, Y0, Y0
	MOVQ	BX, CX
	SHRQ	$5, CX			// n / 32
	JZ	l2_loop8_check

	VXORPS	Y1, Y1, Y1
	VXORPS	Y2, Y2, Y2
	VXORPS	Y3, Y3, Y3

l2_loop32:
	VMOVUPS		(SI), Y4
	VMOVUPS		32(SI), Y5
	VMOVUPS		64(SI), Y6
	VMOVUPS		96(SI), Y7
	VSUBPS		(DI), Y4, Y4
	VSUBPS		32(DI), Y5, Y5
	VSUBPS		64(DI), Y6, Y6
	VSUBPS		96(DI), Y7, Y7
	VFMADD231PS	Y4, Y4, Y0
	VFMADD231PS	Y5, Y5, Y1
	VFMADD231PS	Y6, Y6, Y2
	VFMADD231PS	Y7, Y7, Y3
	ADDQ		$128, SI
	ADDQ		$128, DI
	DECQ		CX
	JNZ		l2_loop32

	VADDPS	Y1, Y0, Y0
	VADDPS	Y3, Y2, Y2
	VADDPS	Y2, Y0, Y0

l2_loop8_check:
	MOVQ	BX, CX
	ANDQ	$31, CX
	SHRQ	$3, CX			// (n % 32) / 8
	JZ	l2_reduce

l2_loop8:
	VMOVUPS		(SI), Y4
	VSUBPS		(DI), Y4, Y4
	VFMADD231PS	Y4, Y4, Y0
	ADDQ		$32, SI
	ADDQ		$32, DI
	DECQ		CX
	JNZ		l2_loop8

l2_reduce:
	VEXTRACTF128	$1, Y0, X1
	VADDPS		X1, X0, X0
	VSHUFPS		$0x4E, X0, X0, X1
	VADDPS		X1, X0, X0
	VSHUFPS		$0x55, X0, X0, X1
	VADDSS		X1, X0, X0

	ANDQ	$7, BX			// n % 8
	JZ	l2_done

l2_tail:
	VMOVSS		(SI), X1
	VSUBSS		(DI), X1, X1
	VFMADD231SS	X1, X1, X0
	ADDQ		$4, SI
	ADDQ		$4, DI
	DECQ		BX
	JNZ		l2_tail

l2_done:
	VMOVSS	X0, ret+24(FP)
	VZEROUPPER
	RET

l2_zero:
	MOVL	$0, ret+24(FP)
	RET

// func dotProductAVX2(a, b *float32, n int) float32
TEXT ·dotProductAVX2(SB), NOSPLIT, $0-28
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	n+16(FP), BX
	TESTQ	BX, BX
	JLE	dot_zero

	VXORPS	Y0, Y0, Y0
	MOVQ	BX, CX
	SHRQ	$5, CX			// n / 32
	JZ	dot_loop8_check

	VXORPS	Y1, Y1, Y1
	VXORPS	Y2, Y2, Y2
	VXORPS	Y3, Y3, Y3

dot_loop32:
	VMOVUPS		(SI), Y4
	VMOVUPS		32(SI), Y5
	VMOVUPS		64(SI), Y6
	VMOVUPS		96(SI), Y7
	VFMADD231PS	(DI), Y4, Y0
	VFMADD231PS	32(DI), Y5, Y1
	VFMADD231PS	64(DI), Y6, Y2
	VFMADD231PS	96(DI), Y7, Y3
	ADDQ		$128, SI
	ADDQ		$128, DI
	DECQ		CX
	JNZ		dot_loop32

	VADDPS	Y1, Y0, Y0
	VADDPS	Y3, Y2, Y2
	VADDPS	Y2, Y0, Y0

dot_loop8_check:
	MOVQ	BX, CX
	ANDQ	$31, CX
	SHRQ	$3, CX
	JZ	dot_reduce

dot_loop8:
	VMOVUPS		(SI), Y4
	VFMADD231PS	(DI), Y4, Y0
	ADDQ		$32, SI
	ADDQ		$32, DI
	DECQ		CX
	JNZ		dot_loop8

dot_reduce:
	VEXTRACTF128	$1, Y0, X1
	VADDPS		X1, X0, X0
	VSHUFPS		$0x4E, X0, X0, X1
	VADDPS		X1, X0, X0
	VSHUFPS		$0x55, X0, X0, X1
	VADDSS		X1, X0, X0

	ANDQ	$7, BX
	JZ	dot_done

dot_tail:
	VMOVSS		(SI), X1
	VFMADD231SS	(DI), X1, X0
	ADDQ		$4, SI
	ADDQ		$4, DI
	DECQ		BX
	JNZ		dot_tail

dot_done:
	VMOVSS	X0, ret+24(FP)
	VZEROUPPER
	RET

dot_zero:
	MOVL	$0, ret+24(FP)
	RET
