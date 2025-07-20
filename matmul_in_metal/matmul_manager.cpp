//
//  matmul_manager.cpp
//  matmul_in_metal
//
//  Created by Tom on 07/07/2025.
//
#include "matmul_manager.hpp"
#include <algorithm>
#include "Metal/Metal.hpp"
#include "msl_matmul.h"
#include "kernels/kernel_runner.hpp"
#include <cstdlib>
#include <cmath>
#include "data.hpp"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

struct kernels;

inline void fill_vector_with_randoms(std::vector<float> &matrix) {
    
    std::generate(matrix.begin(), matrix.end(), []()
                  { return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX); });
}

// TODO: come back use correct guideline c++,
// for instance here, I should just parse the vector as const vector&
// or atleast you span
//inline void matrixToVectorize(const float* source, simd_float4* target, const uint size) {
//    if (size % 4 != 0) {
//        fprintf(stderr, "Error: this matrix cannot be vectorize");
//        return;
//    }
//    
//    for (uint i = 0; i < size * size / 4; i++) {
//        auto vec = simd_make_float4(source[i*4], source[i*4 + 1], source[i*4 + 2], source[i*4 + 3]);
//        target[i] = vec;
//    }
//}
//
//inline void vectorizeToMatrix(float* target, const simd_float4* source, const uint size) {
//    if (size % 4 != 0) {
//        fprintf(stderr, "Error: this matrix cannot be vectorize");
//        return;
//    }
//    
//    for (uint i = 0; i < size * size / 4; i++) {
//        target[i*4] = source[i].x;
//        target[i*4 + 1] = source[i].y;
//        target[i*4 + 2] = source[i].z;
//        target[i*4 + 3] = source[i].w;
//    }
//}

matmul_manager::matmul_manager(const uint32_t M, const uint32_t K, const uint32_t N)
: _matrix_data({ M, K, N, std::vector<float>(M * K), std::vector<float>(K * N), std::vector<float>(M * N)})
{
    auto device = MTL::CreateSystemDefaultDevice();
    
    if (!device) {
        fprintf(stderr, "Error: Failed to find device \n Exiting \n");
        std::exit(1);
    }
    
    auto shader_lib = device->newDefaultLibrary();
    auto command_queue = device->newCommandQueue();
    
    if (!shader_lib || !command_queue) {
        fprintf(stderr, "Error: Failed to construct matmul_manager \n Exiting \n");
        std::exit(1);
    }
    
    // populate with random vals.
    fill_vector_with_randoms(_matrix_data.A);
    fill_vector_with_randoms(_matrix_data.B);
    
    auto buf_a = device->newBuffer(_matrix_data.A.data(), M * K * sizeof(float), MTL::ResourceStorageModeShared);
    auto buf_b = device->newBuffer(_matrix_data.B.data(), K * N * sizeof(float), MTL::ResourceStorageModeShared);
    auto buf_c = device->newBuffer(_matrix_data.C.data(), M * N * sizeof(float), MTL::ResourceStorageModeShared);
    
    _device_data = device_data();
    _device_data.device = device;
    _device_data.shader_lib = shader_lib;
    _device_data.command_queue = command_queue;
    _device_data.buf_a = buf_a;
    _device_data.buf_b = buf_b;
    _device_data.buf_c = buf_c;
    _device_data.test_its = 10;
    
    // populate the true value for testing.
    C_truth = std::vector<float>(M * N);
    compute_msl();
}

void matmul_manager::compute_msl() {
    matmul_in_metal::run_msl(_device_data.device, _device_data.command_queue, _matrix_data, C_truth);
}

matmul_manager::~matmul_manager() {
    _device_data.command_queue->release();
    _device_data.shader_lib->release();
    _device_data.buf_a->release();
    _device_data.buf_b->release();
    _device_data.buf_c->release();
    _device_data.device->release();
}

void matmul_manager::run(const std::string kernel_name)
{
    printf("running kernel: %s ... \n", kernel_name.c_str());
    if (kernel_name == kernels::v1) {
        matmul_in_metal::run_kernel_v1(_device_data, _matrix_data);
    } else if (kernel_name == kernels::v2) {
        matmul_in_metal::run_kernel_v2(_device_data, _matrix_data);
    } else if (kernel_name == kernels::v3_32) {
        matmul_in_metal::run_kernel_v3(_device_data, _matrix_data);
    } else if (kernel_name == kernels::v4_large) {
        matmul_in_metal::run_kernel_v4(_device_data, _matrix_data);
    } else if (kernel_name == kernels::v5_large || kernel_name == kernels::v5_mid) {
        matmul_in_metal::run_kernel_v5(_device_data, _matrix_data);
    } else if (kernel_name == kernels::v6) {
        matmul_in_metal::run_kernel_v6(_device_data, _matrix_data);
    } else if (kernel_name == kernels::v7) {
        matmul_in_metal::run_kernel_v7(_device_data, _matrix_data);
    } else if (kernel_name == kernels::msl) {
        compute_msl();
    }
    
    if (!check_value()) {
        printf("%s, computed the incorrect value for C \n", kernel_name.c_str());
    }
}

void matmul_manager::profile(const std::string kernel_name)
{
    constexpr int its = 200;
    for (int i = 0; i < its; i++)
        run(kernel_name);
}

bool matmul_manager::check_value() {
    return C_truth == _matrix_data.C;
}


inline void print_matrix(std::vector<float> matrix, const uint32_t rows, const uint32_t cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::printf("%6.2f ", matrix[i*cols + j]);
        std::printf("\n");
    }
}

void matmul_manager::view_result()
{
    std::printf("A\n");
    print_matrix(_matrix_data.A, _matrix_data.M, _matrix_data.K);
    std::printf("\n------\n");
    std::printf("B\n");
    print_matrix(_matrix_data.B, _matrix_data.K, _matrix_data.N);
    std::printf("\n-------\n");
    std::printf("C\n");
    print_matrix(_matrix_data.C, _matrix_data.M, _matrix_data.N);
    std::printf("\n------\n");
}
