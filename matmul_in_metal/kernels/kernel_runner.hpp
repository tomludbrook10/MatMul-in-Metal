//
//  kernel_runner.h
//  matmul_in_metal
//
//  Created by Tom on 16/07/2025.
//

#ifndef kernel_runner_h
#define kernel_runner_h

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include <string>
#include <vector>
#include "data.hpp"

namespace matmul_in_metal {

void run_kernel_v1(device_data &d_data,
                   matrix_data &m_data);

void run_kernel_v2(device_data &d_data,
                   matrix_data &m_data);

void run_kernel_v3(device_data &d_data,
                   matrix_data &m_data);

void run_kernel_v4(device_data &d_data,
                   matrix_data &m_data);

void run_kernel_v5(device_data &d_data,
                   matrix_data &m_data);

void run_kernel_v6(device_data &d_data,
                   matrix_data &m_data);

void run_kernel_v7(device_data &d_data,
                   matrix_data &m_data);

void run_msl(void* device,
             void* command_queue,
             matrix_data &data,
             std::vector<float> &C_truth);
}

#endif /* kernel_runner_h */
