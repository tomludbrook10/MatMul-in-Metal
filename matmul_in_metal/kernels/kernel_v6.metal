//
//  kernel_v6.metal
//  matmul_in_metal
//
//  Created by Tom on 17/07/2025.
//

#include <metal_stdlib>
using namespace metal;

/*
 * Vectorize loads from device memory to shared.
 * In addtion, we transpose AS such that loads are further vectorized from shared -> on thread memory.
 */
template<const short BM, const short BN, const short BK, const short TM, const short TN>
kernel void matmul_kernel_v6(device const float* A,
                             device const float* B,
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
    
    // Using a vectorize the load from device to shared memory.
    const int inner_col_A = thread_position_in_threadgroup.x % (BK / 4);
    const int inner_row_A = thread_position_in_threadgroup.x / (BK / 4);
    const int inner_col_B = thread_position_in_threadgroup.x % (BN / 4);
    const int inner_row_B = thread_position_in_threadgroup.x / (BN / 4);

    for (short block_index = 0; block_index < K; block_index += BK) {
        
        if (inner_col_A < BM) {
            // Tranposing AS while loading in, each thread loads in a vector row of 4, then places in a coloumn of AS.
              device const float4* tmp = reinterpret_cast<device const float4 *>(&A[inner_row_A * K + inner_col_A * 4]);
              // need to jump in the row by 4 from the vector loads.
             AS[(inner_col_A * 4 + 0) * BM + inner_row_A] = tmp->x;
             AS[(inner_col_A * 4 + 1) * BM + inner_row_A] = tmp->y;
             AS[(inner_col_A * 4 + 2) * BM + inner_row_A] = tmp->z;
             AS[(inner_col_A * 4 + 3) * BM + inner_row_A] = tmp->w;
            
            
        }
        
        if (inner_row_B < BK) {
            device const float4* tmp = reinterpret_cast<device const float4 *>(&B[inner_row_B * N + inner_col_B * 4]);
            reinterpret_cast<threadgroup float4 *>(&BS[inner_row_B * BN + inner_col_B * 4])[0] = tmp[0];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // shift A and B
        A += BK;
        B += N * BK;

        for (short dot_index = 0; dot_index < BK; dot_index++) {
            
            for (uint i = 0; i < TM; i++) {
                // change the load compared to kernel_5, too account for the transposed AS.
                a_reg_cache[i] = AS[dot_index * BM + TM * thread_y + i];
            }
            
            for (short i = 0; i < TN; i++) {
                b_reg_cache[i] = BS[dot_index * BN + i + thread_x * TN];
            }
            
            for (short tm = 0; tm < TM; tm++) {
                for (short tn = 0; tn < TN; tn++) {
                    result_C[tm * TN + tn] += a_reg_cache[tm] * b_reg_cache[tn];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int tm = 0; tm < TM; tm++) {
        // notice here, that we increment by 4.
        for (int tn = 0; tn < TN; tn+=4) {
            float4 tmp = reinterpret_cast<device float4 *>(&C[(thread_y * TM + tm) * N + thread_x * TN + tn])[0];
            tmp.x = result_C[tm * TN + tn];
            tmp.y = result_C[tm * TN + tn + 1];
            tmp.z = result_C[tm * TN + tn + 2];
            tmp.w = result_C[tm * TN + tn + 3];
            reinterpret_cast<device float4*>(&C[(thread_y * TM + tm) * N + thread_x * TN + tn])[0] = tmp;
        }
    }
}

// allow me not to type the whole signature out.
typedef decltype(matmul_kernel_v6<128, 128, 8, 8, 8>) kernel_t;

template [[host_name("matmul_kernel_v6_large")]] kernel kernel_t matmul_kernel_v6<128, 128, 8, 8, 8>;
template [[host_name("matmul_kernel_v6_mid")]] kernel kernel_t matmul_kernel_v6<128, 64, 8, 8, 4>;
