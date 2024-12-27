#include "textflag.h"

// func euclideanAVX2(v1, v2 []float64) float64
TEXT ·euclideanAVX2(SB), NOSPLIT, $0-48
    MOVQ    v1+0(FP), SI     // v1 slice
    MOVQ    v1_len+8(FP), BX // length
    MOVQ    v2+24(FP), DI    // v2 slice
    VXORPD  Y0, Y0, Y0       // sum = 0
    MOVQ    BX, CX
    SHRQ    $2, CX           // len/4 (process 4 doubles at a time)
    JZ      done_loop

euclidean_loop:
    VMOVUPD (SI), Y1         // load 4 doubles from v1
    VMOVUPD (DI), Y2         // load 4 doubles from v2
    VSUBPD  Y2, Y1, Y3       // diff = v1 - v2
    VMULPD  Y3, Y3, Y3       // square differences
    VADDPD  Y3, Y0, Y0       // add to sum
    ADDQ    $32, SI          // advance v1 pointer
    ADDQ    $32, DI          // advance v2 pointer
    DECQ    CX
    JNZ     euclidean_loop

done_loop:
    // Horizontal sum
    VEXTRACTF128 $1, Y0, X1
    VADDPD  X1, X0, X0
    MOVHLPS X0, X1
    ADDSD   X1, X0
    SQRTSD  X0, X0          // sqrt of sum
    
    MOVSD   X0, ret+40(FP)
    VZEROUPPER
    RET

// func cosineAVX2(v1, v2 []float64) float64
TEXT ·cosineAVX2(SB), NOSPLIT, $0-48
    MOVQ    v1+0(FP), SI     // v1 slice
    MOVQ    v1_len+8(FP), BX // length
    MOVQ    v2+24(FP), DI    // v2 slice
    VXORPD  Y0, Y0, Y0       // dot = 0
    VXORPD  Y1, Y1, Y1       // norm1 = 0
    VXORPD  Y2, Y2, Y2       // norm2 = 0
    MOVQ    BX, CX
    SHRQ    $2, CX           // len/4
    JZ      done_cosine

cosine_loop:
    VMOVUPD (SI), Y3         // load v1
    VMOVUPD (DI), Y4         // load v2
    VMULPD  Y3, Y4, Y5       // v1 * v2
    VADDPD  Y5, Y0, Y0       // add to dot
    VMULPD  Y3, Y3, Y5       // v1 * v1
    VADDPD  Y5, Y1, Y1       // add to norm1
    VMULPD  Y4, Y4, Y5       // v2 * v2
    VADDPD  Y5, Y2, Y2       // add to norm2
    ADDQ    $32, SI
    ADDQ    $32, DI
    DECQ    CX
    JNZ     cosine_loop

done_cosine:
    // Horizontal sums
    VEXTRACTF128 $1, Y0, X3
    VADDPD  X3, X0, X0
    MOVHLPS X0, X3
    ADDSD   X3, X0           // final dot

    VEXTRACTF128 $1, Y1, X3
    VADDPD  X3, X1, X1
    MOVHLPS X1, X3
    ADDSD   X3, X1           // final norm1

    VEXTRACTF128 $1, Y2, X3
    VADDPD  X3, X2, X2
    MOVHLPS X2, X3
    ADDSD   X3, X2           // final norm2
    
    // Calculate 1 - dot/(sqrt(norm1*norm2))
    MULSD   X2, X1           // norm1 * norm2
    SQRTSD  X1, X1           // sqrt(norm1 * norm2)
    DIVSD   X1, X0           // dot/sqrt(norm1*norm2)
    MOVSD   $1.0, X1
    SUBSD   X0, X1           // 1 - dot/sqrt(norm1*norm2)
    
    MOVSD   X1, ret+40(FP)
    VZEROUPPER
    RET
