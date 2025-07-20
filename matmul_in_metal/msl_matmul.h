//
//  msl_matmul.hpp
//  matmul_in_metal
//
//  Created by Tom on 08/07/2025.
//

#pragma once

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include "data.hpp"
#include <vector>

namespace matmul_in_metal {
    double msl_matmul(void* device, void* commandQueue, matrix_data &data, std::vector<float> &C_truth);
}
