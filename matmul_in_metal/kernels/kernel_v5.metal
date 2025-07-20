//
//  kernel_v5.metal
//  matmul_in_metal
//
//  Created by Tom Ludbrook on 17/07/2025.
//

#include <metal_stdlib>
using namespace metal;

/*
 * 2D tiling
 * Adding, TN, thread level dim. Each thread now compute (TN * TM) many values of C.
 */
template<const short BM, const short BN, const short BK, const short TM, const short TN>
kernel void matmul_kernel_v5(constant float* A,
                             constant float* B,
                             device float* C,
                             constant ushort &M [[ buffer(3) ]],
                             constant ushort &N [[ buffer(4) ]],
                             constant ushort &K [[ buffer(5) ]],
                             ushort2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             ushort2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             ushort2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    // spread each thread TN across BN dim
    const ushort thread_x = thread_position_in_threadgroup.x % (BN / TN);
    const ushort thread_y = thread_position_in_threadgroup.x / (BN / TN);
    
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

    // using keyword thread here just to be explict.
    thread float result_C[TM * TN] = {0.0};
    
    // when computing partial dotprod for (TM * TN)
    // we can reduce shared memory loads by (TN) by thread level caching
    // in the registers.
    thread float a_reg_cache[TM] = {0.0};
    thread float b_reg_cache[TN] = {0.0};
    
    const ushort inner_col_A = thread_position_in_threadgroup.x % BK;
    const ushort inner_row_A = thread_position_in_threadgroup.x / BK;
    const ushort inner_col_B = thread_position_in_threadgroup.x % BN;
    const ushort inner_row_B = thread_position_in_threadgroup.x / BN;
    
    const ushort stride_A = threads_per_threadgroup.x / BK;
    const ushort stride_B = threads_per_threadgroup.x / BN;

    for (short block_index = 0; block_index < K; block_index += BK) {
        // To achieve coalescing from device to shared memory we "wrap"
        // each threadgroups' simdgroups around the rows of A and B (row-major)
        
        for (int i = 0; i < BM; i += stride_A) {
            AS[inner_col_A + BK * (inner_row_A + i)] = A[inner_col_A + K * (inner_row_A + i)];
        }
        
        for (int i = 0; i < BK; i += stride_B) {
                BS[inner_col_B + BN * (inner_row_B + i)] = B[inner_col_B + N * (inner_row_B + i)];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // shift A and B
        A += BK;
        B += N * BK;
        
        for (short dot_index = 0; dot_index < BK; dot_index++) {
            // fill each cache.
            for (short i = 0; i < TM; i++)
                a_reg_cache[i] = AS[dot_index + BK * (i + thread_y * TM)];
            for (short i = 0; i < TN; i++)
                b_reg_cache[i] = BS[dot_index * BN + i + thread_x * TN];
            
            // compute the partial dotprod for the current squre.
            for (short tm = 0; tm < TM; tm++) {
                for (uint tn = 0; tn < TN; tn++) {
                    result_C[tm * TN + tn] += a_reg_cache[tm] * b_reg_cache[tn];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // write back to C.
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            C[(thread_y * TM + tm) * N + thread_x * TN + tn] = result_C[tm * TN + tn];
        }
    }
}

// allow me not to type the whole signature out.
typedef decltype(matmul_kernel_v5<128, 128, 8, 8, 8>) kernel_t;

template [[host_name("matmul_kernel_v5_large")]] kernel kernel_t matmul_kernel_v5<128, 128, 8, 8, 8>;
template [[host_name("matmul_kernel_v5_mid")]] kernel kernel_t matmul_kernel_v5<128, 64, 8, 8, 4>;

