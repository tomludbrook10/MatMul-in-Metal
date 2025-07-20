//
//  kernel_v3.metal
//  matmul_in_metal
//
//  Created by Tom Ludbrook on 16/07/2025.
//

#include <metal_stdlib>
using namespace metal;

// Share Memory Cache-Blocking
/*
 * Each threadgroup loads BLOCKSIZE*BLOCKSIZE block of matrix A and B onto SMEM.
 * Then performs dotprod on the submatrix rows and cols for the given thread and adds to the accum.
 */

template<const short BLOCKSIZE>
kernel void matmul_kernel_v3(device const float* A,
                             device const float* B,
                             device float* C,
                             constant ushort &M [[ buffer(3) ]],
                             constant ushort &N [[ buffer(4) ]],
                             constant ushort &K [[ buffer(5) ]],
                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             uint2 threads_per_threadgroup [[ threads_per_threadgroup ]])
{
    // thread position in thread group.
    constexpr ushort simdgroup_size = 32;
    ushort thread_x = thread_position_in_threadgroup.x % simdgroup_size;
    ushort thread_y = thread_position_in_threadgroup.x / simdgroup_size;
    
    // block position in grid.
    const ushort block_position_x = threadgroup_position_in_grid.x;
    const ushort block_position_y = threadgroup_position_in_grid.y;
    
    // allocate memory on SMEM.
    threadgroup float AS[BLOCKSIZE * BLOCKSIZE];
    threadgroup float BS[BLOCKSIZE * BLOCKSIZE];
    
    // move the A, B pointers to the start position.
    A += block_position_y * BLOCKSIZE * K;
    B += block_position_x * BLOCKSIZE;
    C += block_position_y * BLOCKSIZE * N + block_position_x * BLOCKSIZE;
    
    float val = 0;
    for (short block_index = 0; block_index < K; block_index += BLOCKSIZE) {
        // Coalescing the memory load for BS submatrix.
        // each thread loads 2 values from device, one for A and B each.
        AS[thread_x + thread_y * BLOCKSIZE] = A[thread_x + K * thread_y];
        BS[thread_x + thread_y * BLOCKSIZE] = B[thread_x + N * thread_y];
            
        // wait untill all threads have loaded their respective part of AS and BS.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        
        for (short dot_index = 0; dot_index < BLOCKSIZE; dot_index++) {
            val += AS[BLOCKSIZE * thread_y + dot_index] * BS[BLOCKSIZE * dot_index + thread_x];
        }
        
        // We need to wait, untill all threads of computed their dotprod, untill we load
        // different values for AS and BS.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[thread_x + thread_y * N] = val;
}



typedef decltype(matmul_kernel_v3<32>) kernel_t;

template [[host_name("matmul_kernel_v3_32")]] kernel kernel_t matmul_kernel_v3<32>;
