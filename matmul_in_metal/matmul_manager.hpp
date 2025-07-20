//
//  matmul_manager.hpp
//  matmul_in_metal
//
//  Created by Tom on 07/07/2025.
//

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include <string>
#include <simd/simd.h>
#include "data.hpp"

// send the seed

// TODO: Refactor to not use float*
// we can still use modern c++, and vector.data().

class matmul_manager {
public:
    // we also want be able set the thread group and grid size.
    explicit matmul_manager(const uint32_t M, const uint32_t K, const uint32_t N);
    ~matmul_manager();

    void run(const std::string kernel_name);
    void profile(const std::string kernel_name);
    void view_result();
    void compute_msl();
    bool check_value();
    
private:
    // To verify output.
    std::vector<float> C_truth;
    matrix_data _matrix_data;
    device_data _device_data;
};
