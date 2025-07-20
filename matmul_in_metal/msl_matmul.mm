//
//  msl_matmul.cpp
//  matmul_in_metal
//
//  Created by Tom on 08/07/2025.
//

#include "msl_matmul.h"
#include <iostream>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

double matmul_in_metal::msl_matmul(void *inputDevice, void* inputCommandQueue, matrix_data &data, std::vector<float> &C_truth)
{
    auto M = static_cast<NSUInteger>(data.M);
    auto K = static_cast<NSUInteger>(data.K);
    auto N = static_cast<NSUInteger>(data.N);
    
    if (!inputDevice) {
        fprintf(stderr, "Error: parsed a null ptr");
        return 0;
    }
    
    // need, cause I'm calling this in a loop.
    @autoreleasepool {
        
        auto device = (__bridge id<MTLDevice>)inputDevice;
        auto commandQueue = (__bridge id<MTLCommandQueue>)inputCommandQueue;
        
        if (!device) {
            fprintf(stderr, "Failed to cast the device to a Objective-C object");
            return 0;
        }
        
        MPSMatrixDescriptor *a_d = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:(K * sizeof(float))
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *b_d = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:(N * sizeof(float))   dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *c_d = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:(N * sizeof(float)) dataType:MPSDataTypeFloat32];
        
        id<MTLBuffer> buf_A = [device newBufferWithLength:(M * K * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_B = [device newBufferWithLength:(N * K * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_C = [device newBufferWithLength:(M * N * sizeof(float)) options:MTLResourceStorageModeShared];
        
        // Copy arrays into buffers.
        
        float* dst_a = static_cast<float*>(buf_A.contents);
        std::memcpy(dst_a, data.A.data(), M * K * sizeof(float));
        
        float* dst_b = static_cast<float*>(buf_B.contents);
        std::memcpy(dst_b, data.B.data(), N * K * sizeof(float));
        
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:buf_A descriptor:a_d];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:buf_B descriptor:b_d];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:buf_C descriptor:c_d];
        
        MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                          transposeLeft:false
                                                                         transposeRight:false
                                                                             resultRows:M
                                                                          resultColumns:N
                                                                        interiorColumns:K
                                                                                  alpha:1.0f
                                                                                   beta:0.0f];
        
        //id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
        [gemm encodeToCommandBuffer:cmd
                         leftMatrix:matA
                        rightMatrix:matB
                       resultMatrix:matC];
        
        // run and profile.
        auto t0 = std::chrono::high_resolution_clock::now();
        [cmd commit];
        [cmd waitUntilCompleted];
        auto t1 = std::chrono::high_resolution_clock::now();
        double mircoSecs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        
        // if we don't have square matrix this might cause issues/
        float* src = static_cast<float*>(buf_C.contents);
        std::memcpy(C_truth.data(), src, M * N * sizeof(float));
        
        return mircoSecs;
    }
    return 0;
}
