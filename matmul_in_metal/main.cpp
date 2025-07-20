//
//  main.cpp
//  matmul_in_metal
//
//  Created by Tom on 07/07/2025.
//
#include "mlt_impl.hpp"
#include "kernels/kernel_runner.hpp"
#include "matmul_manager.hpp"
#include <iostream>
#include "msl_matmul.h"
#include "data.hpp"
// send the seed

int main(int argc, const char * argv[]) {
    uint M = 16, K = 16, N = 16;
    
    if ((M > 32 && M % 32 != 0) ||
        (K > 32 && K % 32 != 0) ||
        (N > 32 && N % 32 != 0)) {
            fprintf(stderr, "If M, K, N are greater than 32, they must be divisible by 32 \n");
            return 1;
        }
    
    matmul_manager m_manager(M, K, N);
    m_manager.run(kernels::v7);
    m_manager.view_result();
    return 0;
}


