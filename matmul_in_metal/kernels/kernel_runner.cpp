//
//  kernel_runner.cpp
//  matmul_in_metal
//
//  Created by Tom on 15/07/2025.
//

#include "kernel_runner.hpp"

#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>
#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"
#include "msl_matmul.h"

#include <stdio.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define SIMDGROUP_SIZE 32
#define MAX_THREADGROUP_SIZE 1024

inline double get_gflops(const uint32_t M, const uint32_t K, const uint32_t N, const double time) {
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    double gflops =  (flops / time ) * 1e-9;
    return gflops;
}

void print_matmul(const std::string kernel_name,
                  const uint32_t M,
                  const uint32_t K,
                  const uint32_t N,
                  const double time,
                  const uint32_t run_its) {
    double time_in_secs = time / 1e6;
    double gflops = get_gflops(M, K, N, time_in_secs / run_its);
    printf("Kernel: %s, ran for a total of %d iterations, with average of %6.10f seconds per iteration, with GFLOPS: %6.10f \n",  kernel_name.c_str(), run_its, time_in_secs / run_its, gflops);
}

void basic_dispatcher(device_data &d_data,
                      matrix_data &m_data,
                      kernel_data &kernel_data) {
    auto kernel = d_data.shader_lib->newFunction(NS::String::string(kernel_data.kernel_name.c_str(),
                                                               NS::UTF8StringEncoding));
    if (!kernel) {
        std::fprintf(stderr, "Error: Failed to obtain kernel: %s", kernel_data.kernel_name.c_str());
        return;
    }
    
    // Create function constants
    NS::Error* error;
    auto compute_pipeline = d_data.device->newComputePipelineState(kernel, &error);
    if (!compute_pipeline || error) {
        std::fprintf(stderr, "Error: Failed to obtain get pipeline from device");
        return;
    }
    
    NS::UInteger thread_group_size = compute_pipeline->maxTotalThreadsPerThreadgroup();
    if (thread_group_size < kernel_data.block_dims.width) {
        std::printf("Warning: thread group size (%lu) exceeds the current kernels max thread group size (%lu), try reducing the occupancy", kernel_data.block_dims.width, thread_group_size);
        return;
    }
    
    auto uK = static_cast<uint16_t>(m_data.K);
    auto uM = static_cast<uint16_t>(m_data.M);
    auto uN = static_cast<uint16_t>(m_data.N);
    
    // loop for profiling the kernels.
    for (int i = 0; i < 10; i++)
    {
        // create commandQueue so cpu and dispatch command to the gpu
        auto command_buffer = d_data.command_queue->commandBuffer();
        auto command_encoder = command_buffer->computeCommandEncoder();
        
        NS::UInteger offset = 0;
        
        command_encoder->setComputePipelineState(compute_pipeline);
        command_encoder->setBuffer(d_data.buf_a, offset, 0);
        command_encoder->setBuffer(d_data.buf_b, offset, 1);
        command_encoder->setBuffer(d_data.buf_c, offset, 2);
        command_encoder->setBytes(&uM, sizeof(uint16_t), 3);
        command_encoder->setBytes(&uN, sizeof(uint16_t), 4);
        command_encoder->setBytes(&uK, sizeof(uint16_t), 5);
        
        for (int32_t i = 0; i < d_data.test_its; i++)
            command_encoder->dispatchThreadgroups(kernel_data.grid_dims, kernel_data.block_dims);
        
        command_encoder->endEncoding();
        
        // time the run.
        auto t0 = std::chrono::high_resolution_clock::now();
        command_buffer->commit();
        command_buffer->waitUntilCompleted(); // make sync
        auto t1 = std::chrono::high_resolution_clock::now();
        double micro = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        
        // print results
        print_matmul(kernel_data.kernel_name, m_data.M, m_data.K, m_data.N, micro, d_data.test_its);
        
        // copying contents of buf_c to C.
        float* src = static_cast<float*>(d_data.buf_c->contents());
        std::memcpy(m_data.C.data(), src, m_data.M * m_data.N * sizeof(float));
        
        // clean up.
        command_encoder->release();
        command_buffer->release();
    }
    compute_pipeline->release();
    kernel->release();
}


// naive
void matmul_in_metal::run_kernel_v1(device_data &d_data,
                                    matrix_data &m_data)
{
    
    MTL::Size grid_dim{CEIL_DIV(m_data.N, SIMDGROUP_SIZE), CEIL_DIV(m_data.M, SIMDGROUP_SIZE), 1};
    MTL::Size block_dim{MAX_THREADGROUP_SIZE, 1, 1};
    kernel_data k_data = { kernels::v1, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

// global memory coalescing.
void matmul_in_metal::run_kernel_v2(device_data &d_data,
                                    matrix_data &m_data)
{
    MTL::Size grid_dim{CEIL_DIV(m_data.N, SIMDGROUP_SIZE), CEIL_DIV(m_data.M, SIMDGROUP_SIZE), 1};
    MTL::Size block_dim{MAX_THREADGROUP_SIZE, 1, 1};
    kernel_data k_data = { kernels::v2, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

// cache blocking from device into shared memory.
void matmul_in_metal::run_kernel_v3(device_data &d_data,
                                    matrix_data &m_data)
{
    MTL::Size grid_dim{CEIL_DIV(m_data.N, SIMDGROUP_SIZE), CEIL_DIV(m_data.M, SIMDGROUP_SIZE), 1};
    MTL::Size block_dim{MAX_THREADGROUP_SIZE, 1, 1};
    kernel_data k_data = { kernels::v3_32, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

// 1D thread level tiling.
void matmul_in_metal::run_kernel_v4(device_data &d_data,
                                    matrix_data &m_data)
{
    const int32_t BN = 64, BM = 64, BK = 8, TM = 8;
    MTL::Size grid_dim{CEIL_DIV(m_data.N, BN), CEIL_DIV(m_data.M, BM), 1};
    MTL::Size block_dim{((BN * BM) / TM), 1, 1};
    
    // Currently kernel 4, isn't set up for any BN, BM, BK, TM.
    // For each BK, a thread loads in only one A and B into shared memory.
    // Thus if the threadgroupsize != BM*BK or BN*BK, then we won't fill our cache.
    // The fix is somewhat trival, but cluters the kernel which purpose is only educational.
    if (block_dim.width != (BN * BK) || block_dim.width != (BN * BK)) {
        return;
    }
    
    kernel_data k_data = { kernels::v4_large, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

// 2D thread level tiling.
void matmul_in_metal::run_kernel_v5(device_data &d_data,
                                    matrix_data &m_data)
{
    const int32_t BM = 128, BN = 64, TM = 8, TN = 4;
    
    MTL::Size grid_dim{CEIL_DIV(m_data.N, BN), CEIL_DIV(m_data.M, BM), 1};
    MTL::Size block_dim{((BN * BM) / (TM * TN)), 1, 1};
    
    //if (block_dim.width != (BN * BK) || block_dim.width != (BN * BK)) {
      //  return;
    //}
    
    kernel_data k_data = { kernels::v5_mid, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

void matmul_in_metal::run_kernel_v6(device_data &d_data,
                                    matrix_data &m_data)
{
    const int32_t BM = 128, BN = 64, TM = 8, TN = 4;
    
    MTL::Size grid_dim{CEIL_DIV(m_data.N, BN), CEIL_DIV(m_data.M, BM), 1};
    MTL::Size block_dim{((BN * BM) / (TM * TN)), 1, 1};
    
    //if (block_dim.width != (BN * BK) || block_dim.width != (BN * BK)) {
      //  return;
    //}
    
    kernel_data k_data = { kernels::v6, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

void matmul_in_metal::run_kernel_v7(device_data &d_data,
                                    matrix_data &m_data)
{
    MTL::Size grid_dim{2, 2, 1};
    MTL::Size block_dim{32, 1, 1};
    
    kernel_data k_data = { kernels::v7, grid_dim, block_dim};
    basic_dispatcher(d_data, m_data, k_data);
}

void matmul_in_metal::run_msl(void *device,
                              void *command_queue,
                              matrix_data &m_data,
                              std::vector<float> &C_truth)
{
    double val = matmul_in_metal::msl_matmul(device, command_queue, m_data, C_truth);
    print_matmul(kernels::msl, m_data.M, m_data.K, m_data.N, val, 1);
}

