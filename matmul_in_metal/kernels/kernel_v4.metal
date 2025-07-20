//
//  kernel_v4.metal
//  matmul_in_metal
//
//  Created by Tom Ludbrook on 17/07/2025.
//

#include <metal_stdlib>
using namespace metal;

/*
 * 1D Tiling
 * BM, BN, BK is the block dimension for each thread group.
 * TM is the thread level dim. Each thread computes TM values of C.
 */
template<const short BM, const short BN, const short BK, const short TM>
kernel void matmul_kernel_v4(device const float* A,
                             device const float* B,
                             device float* C,
                             constant ushort &M [[ buffer(3) ]],
                             constant ushort &N [[ buffer(4) ]],
                             constant ushort &K [[ buffer(5) ]],
                             ushort2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             ushort2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             ushort2 threads_per_threadgroup [[ threads_per_threadgroup ]])
{
    // threadgroup shape is with respect to BN dim
    const ushort thread_x = thread_position_in_threadgroup.x % BN;
    const ushort thread_y = thread_position_in_threadgroup.x / BN;
    
    // block position in grid.
    const ushort block_position_x = threadgroup_position_in_grid.x;
    const ushort block_position_y = threadgroup_position_in_grid.y;
    
    // in shared memory cache.
    threadgroup float AS[BM * BK];
    threadgroup float BS[BN * BK];
    
    // move the A, B pointers to the start position.
    A += block_position_y * BM * K;
    B += block_position_x * BN;
    C += block_position_y * BM * N + block_position_x * BN;
    
    // allocate thread-level cache (in register file) for accums for value of C.
    float C_T[TM] = {0.0};
    
    // align device to share memory loads to allow for memory coalescing.
    const int inner_col_A = thread_position_in_threadgroup.x % BK;
    const int inner_row_A = thread_position_in_threadgroup.x / BK;
    const int inner_col_B = thread_position_in_threadgroup.x % BN;
    const int inner_row_B = thread_position_in_threadgroup.x / BN;
    
    for (short block_index = 0; block_index < K; block_index += BK) {
        // row direction as stride=1, with respect to the simd-group,
        // allowing device memory coalescing.
        AS[inner_col_A + BK * inner_row_A] = A[inner_col_A + K * inner_row_A];
        BS[inner_col_B + BN * inner_row_B] = B[inner_col_B + N * inner_row_B];
            
        // syncs all therads in the threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        A += BK;
        B += BK * N;
        
        // dot_index moves along the cols of AS.
        // for each dot_index compute the partial accum for each tm.
        for (short dot_index = 0; dot_index < BK; dot_index++) {
            // keep B constant as we traverse in the tm dim, i.e along the rows of AS.
            float tempB = BS[thread_x + dot_index * BN];
            for (short tm = 0; tm < TM; tm++) {
                // spread threads by TM along the rows of AS.
                C_T[tm] += AS[(TM * thread_y + tm) * BK +  dot_index] * tempB;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int tm = 0; tm < TM; tm++)  {
        C[(TM * thread_y + tm) * N + thread_x] = C_T[tm];
    }
}

// allow me not to type the whole signature out.
typedef decltype(matmul_kernel_v4<64, 64, 8, 8>) kernel_t;

template [[host_name("matmul_kernel_v4_large")]] kernel kernel_t matmul_kernel_v4<64, 64, 8, 8>;
