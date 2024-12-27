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

// func BatchEuclideanAVX2(query Vector, vectors []Vector, results []float64)
TEXT ·BatchEuclideanAVX2(SB), NOSPLIT, $0-72
    MOVQ    query+0(FP), SI         // query data pointer
    MOVQ    query_len+8(FP), R8     // query length
    MOVQ    vectors+24(FP), DI      // vectors slice header
    MOVQ    vectors_len+32(FP), CX  // number of vectors
    MOVQ    results+48(FP), R9      // results data pointer
    
    XORQ    R10, R10               // vector index = 0

vector_loop:
    CMPQ    R10, CX                // if vector_index >= num_vectors, done
    JGE     done
    
    // Load vector pointer
    MOVQ    (DI)(R10*8), R11       // R11 = vectors[i]
    VXORPD  Y0, Y0, Y0             // sum = 0
    XORQ    R12, R12               // dim = 0
    
    // Process vector in chunks of 4 doubles
dim_loop:
    ADDQ    $32, R12               // Increment by 4 doubles (32 bytes)
    CMPQ    R12, R8                
    JG      finish_vector          // If we've processed all dimensions
    
    VMOVUPD -32(SI)(R12*1), Y1    // Load 4 doubles from query
    VMOVUPD -32(R11)(R12*1), Y2   // Load 4 doubles from vector
    VSUBPD  Y2, Y1, Y3            // Subtract
    VMULPD  Y3, Y3, Y3            // Square differences
    VADDPD  Y3, Y0, Y0            // Add to sum
    JMP     dim_loop

finish_vector:
    // Sum the lanes of Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD  X1, X0, X0
    MOVHLPS X0, X1
    ADDSD   X1, X0
    SQRTSD  X0, X0
    
    MOVSD   X0, (R9)(R10*8)       // Store result
    
    INCQ    R10                    // Next vector
    JMP     vector_loop

done:
    VZEROUPPER
    RET

// func BatchEuclideanAVX2Flat(query []float64, flatVectors []float64, dim int, results []float64)
TEXT ·BatchEuclideanAVX2Flat(SB), NOSPLIT, $0-56
    MOVQ query+0(FP), SI           // query ptr
    MOVQ flatVectors+24(FP), DI    // flatVectors ptr
    MOVQ dim+48(FP), R8            // dimension
    MOVQ results+56(FP), R9        // results ptr
    
    MOVQ flatVectors_len+32(FP), CX
    SHRQ $10, CX                   // divide by 1024 to get vector count
    
    XORQ R10, R10                  // vector index

vector_loop:
    CMPQ R10, CX
    JGE  done
    
    VXORPD Y0, Y0, Y0             // clear sum
    XORQ   R11, R11               // dimension counter
    
    MOVQ R10, R12
    SHLQ $10, R12                 // multiply by 1024 to get offset
    LEAQ (DI)(R12*8), R12         // current vector address

dim_loop:
    CMPQ R11, R8
    JGE  finish_vector
    
    VMOVUPD (SI)(R11*8), Y1       // load 4 query elements
    VMOVUPD (R12)(R11*8), Y2      // load 4 vector elements
    VSUBPD  Y2, Y1, Y3            // subtract
    VMULPD  Y3, Y3, Y3            // square
    VADDPD  Y3, Y0, Y0            // add to sum
    
    ADDQ $4, R11
    JMP  dim_loop

finish_vector:
    // Horizontal sum and sqrt
    VEXTRACTF128 $1, Y0, X1
    VADDPD  X1, X0, X0
    MOVHLPS X0, X1
    ADDSD   X1, X0
    SQRTSD  X0, X0
    
    MOVSD X0, (R9)(R10*8)
    
    INCQ R10
    JMP  vector_loop

done:
    VZEROUPPER
    RET
