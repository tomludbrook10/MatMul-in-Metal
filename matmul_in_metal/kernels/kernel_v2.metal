//
//  kernel_v2.metal
//  matmul_in_metal
//
//  Created by Tom Ludbrook on 16/07/2025.
//

#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

//constant constexpr const ushort M [[ function_constant(0) ]];
//constant constexpr const ushort K [[function_constant(1)]];
//constant constexpr const ushort N [[ function_constant(2) ]];
/*
 * Naive matrix multiplcation with
 * maping threadgroup(x,y) -> C(x,y)
 * direct mapping allow global memory coalescing in the B input matrix.
 *
 */
//template<const int KK>
kernel void matmul_kernel_v2(device const float* A,
                             device const float* B,
                             device float* C,
                             constant ushort &M [[buffer(3)]],
                             constant ushort &K [[buffer(4)]],
                             constant ushort &N [[buffer(5)]],
                             ushort2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             ushort2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             ushort2 threads_per_threadgroup [[ threads_per_threadgroup ]])
{
    // simd group size for all m-series chips
    constexpr ushort simdgroup_size = 32;
    
    //constexpr ushort KK = 2048;
    
    // thread position in thread group.
    ushort thread_x = thread_position_in_threadgroup.x % simdgroup_size;
    ushort thread_y = thread_position_in_threadgroup.x / simdgroup_size;
    
    // thread position in grid.
    ushort x = thread_x + (threadgroup_position_in_grid.x * simdgroup_size);
    ushort y = thread_y + (threadgroup_position_in_grid.y * simdgroup_size);
    
    float val = 0;
    //#pragma clang loop unroll(full)
    for (int k = 0; k < K; k++) {
        val += A[K * y + k] * B[K * k + x];
    }
        // write back.
    C[K * y + x] = val;
}
