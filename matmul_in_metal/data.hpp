//
//  data.hpp
//  matmul_in_metal
//
//  Created by Tom on 16/07/2025.
//

#pragma once

#include <vector>
#include <cstdlib>
#include "Metal/Metal.hpp"
#include <string>

struct kernels {
    static constexpr std::string msl = "msl_kernel";
    static constexpr std::string v1 = "matmul_kernel_v1";
    static constexpr std::string v2 = "matmul_kernel_v2";
    static constexpr std::string v3_32 = "matmul_kernel_v3_32";
    static constexpr std::string v4_large = "matmul_kernel_v4_large";
    static constexpr std::string v5_large = "matmul_kernel_v5_large";
    static constexpr std::string v5_mid = "matmul_kernel_v5_mid";
    static constexpr std::string v7 = "opt_matmul_run";
    static constexpr std::string v6 = "matmul_kernel_v6_mid";
};


struct kernel_data {
    std::string kernel_name;
    MTL::Size grid_dims;
    MTL::Size block_dims;
};

struct matrix_data {
    const uint32_t M;
    const uint32_t K;
    const uint32_t N;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> C;
};

struct device_data {
    MTL::Device* device;
    MTL::Library* shader_lib;
    MTL::CommandQueue* command_queue;
    MTL::Buffer *buf_a;
    MTL::Buffer *buf_b;
    MTL::Buffer *buf_c;
    int32_t test_its;
};
