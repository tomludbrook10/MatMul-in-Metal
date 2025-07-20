//
//  kernel_v1.metal
//  matmul_in_metal
//
//  Created by Tom Ludbrook on 15/07/2025.
//


#include <metal_stdlib>
using namespace metal;

/*
 * Naive matrix multiplcation
 * maping thread group(x,y) -> C(y,x)
 */
kernel void matmul_kernel_v1(device const float* A,
                             device const float* B,
                             device float* C,
                             constant ushort &M [[ buffer(3) ]],
                             constant ushort &N [[ buffer(4) ]],
                             device const ushort &K [[ buffer(5) ]],
                             ushort2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             ushort2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             ushort2 threads_per_threadgroup [[ threads_per_threadgroup ]])
{
    // simd group size for all m-series chips
    constexpr ushort simdgroup_size = 32;
    
    // thread position in thread group.
    ushort thread_x = thread_position_in_threadgroup.x % simdgroup_size;
    ushort thread_y = thread_position_in_threadgroup.x / simdgroup_size;
    
    // thread position in grid.
    ushort x = thread_y + (threadgroup_position_in_grid.x * simdgroup_size);
    ushort y = thread_x + (threadgroup_position_in_grid.y * simdgroup_size);
    
    if (x < N && y < M) {
        float val = 0;
        for (int k = 0; k < K; k++) {
            val += A[K * y + k] * B[N * k + x];
        }
        // write back.
        C[N * y + x] = val;
    }
}
