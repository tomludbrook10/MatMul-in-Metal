//
//  opt_matmul.metal
//  matmul_in_metal
//
//  Created by Tom on 18/07/2025.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

template<const int BM, const int BN, const int BK>
kernel void opt_matmul(device const float* A,
                       device const float* B,
                       device float* C,
                       constant ushort &M [[ buffer(3) ]],
                       constant ushort &N [[ buffer(4) ]],
                       constant ushort &K [[ buffer(5) ]],
                       ushort2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                       ushort2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                       ushort2 threads_per_threadgroup [[ threads_per_threadgroup ]],
                       ushort simdgroup_index_in_threadgroup [[ simdgroup_index_in_threadgroup ]])
{
    // spread each thread TN across BN dim
    const ushort thread_x = thread_position_in_threadgroup.x % BN;
    const ushort thread_y = thread_position_in_threadgroup.x / BN;
    
    // block position in grid.
    const ushort block_position_x = threadgroup_position_in_grid.x;
    const ushort block_position_y = threadgroup_position_in_grid.y;
    
    // shared memory.
    threadgroup float AS[BM * BK];
    threadgroup float BS[BN * BK];
    
    A += block_position_y * BM * K;
    B += block_position_x * BN;
    C += block_position_y * BM * N + block_position_x * BM;
    
    const short inner_col_A = thread_position_in_threadgroup.x % BK;
    const short inner_row_A = thread_position_in_threadgroup.x / BK;
    const short inner_col_B = thread_position_in_threadgroup.x % (BK / 2);
    const short inner_row_B = thread_position_in_threadgroup.x / (8 / 2);
    
    const short stride_A = threads_per_threadgroup.x / BK;
    const short stride_B = threads_per_threadgroup.x / BK;
    
    //const short offset_simd_group_A = BK * (simdgroup_index_in_threadgroup * 8);
    //const short offset_simd_group_A = BK * (simdgroup_index_in_threadgroup * 8);
    
    simdgroup_float8x8 sgA;
    simdgroup_float8x8 sgB;
    simdgroup_float8x8 sgC;
    simdgroup_float8x8 sgD;
    
     // B, nee
    //const short b_col_offset = thread_position_in_threadgroup.x / (8 * BK);
    // const short offset_stride_B = threads_per_threadgroup.x / (8 * BK);
    
    
    
    for (short block_inx = 0; block_inx < K; block_inx += BK) {
        for (int i = 0; i < BM; i += stride_A)
            AS[inner_col_A + BK * (inner_row_A + i)] = A[inner_col_A + K * (inner_row_A + i)];
        
        BS[(inner_col_B * 2) + BK * inner_row_B] = B[(inner_col_B * 2) + + N * inner_row_B];
        BS[(inner_col_B * 2) + 1  + BK * inner_row_B] = B[(inner_col_B * 2) + 1 + + N * inner_row_B];
        
//        for (int i = 0; i < BN; i += stride_A) {
//            // note that BS is BN by BK matrix
//             //TODO: account for an offset stride jump.
//            BS[inner_col_B + BK * (inner_row_B + 0 + (8 * b_col_offset))] =
//                B[(inner_col_B + (8 * b_col_offset)) + N * (inner_row_B + 0)];
//        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        simdgroup_load(sgA, AS);
        simdgroup_load(sgB, BS);
        
       // simdgroup_multiply(sgD, sgA, sgB, );
        simdgroup_multiply_accumulate(sgD, sgA, sgB, sgC);
        
        A += BK;
        B += N * BK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        sgD = sgC;
    }
    // I'll have to use something else here. simdgroups operations only work everything is aligned.
    // this C of course it not.
    
   // float t = sgD[0][0];
    
    simdgroup_store(sgD, C);
}

// allow me not to type the whole signature out.
typedef decltype(opt_matmul<8, 8, 8>) kernel_t;

template [[host_name("opt_matmul_run")]] kernel kernel_t opt_matmul<8, 8 , 8>;

