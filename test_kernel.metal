//
//  test_kernel.metal
//  matmul_in_metal
//
//  Created by Tom on 07/07/2025.
//

#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel_v6_nonve(device const float* A,
                             device const float* B,
                             device float* C,
                             constant uint &M [[ buffer(3) ]],
                             constant uint &N [[ buffer(4) ]],
                             constant uint &K [[ buffer(5) ]],
                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             uint2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 4;
    const uint BM = 64;
    const uint BN = 64;
    
    const uint blockX = thread_position_in_threadgroup.x % (BN / TN);
    const uint blockY = thread_position_in_threadgroup.x / (BN / TN);
    
    const uint blockCol = threadgroup_position_in_grid.x;
    const uint blockRow = threadgroup_position_in_grid.y;
    
    // compute blockTiles
    //const uint totalNumberTiles = BM * BN;
    //const uint numberOfThreads = totalNumberTiles / (TM * TN);
    
    // the .metal compiler seems to ignore asserts.
    //assert(numberOfThreads == threads_per_threadgroup.x);
    
    A += K * blockRow * BM;
    B += blockCol * BN;
    // TODO: author has put N here, I think it should be K.
        // So the stride is in the N, dim not K.
    C += blockCol * BN + N * blockRow * BM;
    
    threadgroup float AS[BM * BK];
    threadgroup float BS[BN * BK];
    
    thread float resultC[TM * TN] = {0.0}; // on thread memory
    thread float aCache[TM] = {0.0}; // temp cache during the dot product.
    thread float bCache[TN] = {0.0};
    
    // each thread, loads in four values, so we need skip by four.
    const uint innerColA = thread_position_in_threadgroup.x % (BK / 4);
    const uint innerRowA = thread_position_in_threadgroup.x / (BK / 4);
    const uint innerColB = thread_position_in_threadgroup.x % (BN / 4);
    const uint innerRowB = thread_position_in_threadgroup.x / (BN / 4);


    for (uint blockInx = 0; blockInx < K; blockInx +=BK) {
        
        
        if (innerRowA < BM) {
            // We tranpose AS while loading in, each thread loads in a vector row of 4, then places in a coloumn of AS.
            //float4 tmp = reinterpret_cast<device const float4 *>(&A[innerColA * 4 + K * innerRowA])[0];
            
            float4 tmp = float4(A[innerColA * 4 + K * innerRowA],
                                A[(innerColA * 4 + 1)+ K * innerRowA],
                                A[(innerColA * 4 + 2)+ K * innerRowA],
                                A[(innerColA * 4 + 3)+ K * innerRowA]);
            
            
            // need to jump in the row by 4 from the vector loads.
            AS[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
            AS[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
            AS[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
            AS[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
        }
        
        if (innerRowB < BK) {
            //float4 tmpB = reinterpret_cast<device const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
            //reinterpret_cast<threadgroup float4 *>(&BS[innerRowB * BN + innerColB * 4])[0] = tmpB;
            
            float4 tmp = float4(B[innerRowB * N + innerColB * 4],
                                B[innerRowB * N + innerColB * 4 + 1],
                                B[innerRowB * N + innerColB * 4 + 2],
                                B[innerRowB * N + innerColB * 4 + 3]);
            
            BS[innerRowB * BN + innerColB * 4] = tmp.x;
            BS[innerRowB * BN + innerColB * 4 + 1] = tmp.y;
            BS[innerRowB * BN + innerColB * 4 + 2] = tmp.z;
            BS[innerRowB * BN + innerColB * 4 + 3] = tmp.w;
        }
        
        threadgroup_barrier(mem_flags::mem_none);
        
        // shift A and B
        A += BK;
        B += N * BK;

        
        for (uint dotIndex = 0; dotIndex < BK; dotIndex++) {
            // fill in the thread memory to reduce shared memory access.
            for (uint i = 0; i < TM; i++) {
                aCache[i] = AS[dotIndex * BM + TM * blockY + i];
            }
            
            for (uint i = 0; i < TN; i++) {
                bCache[i] = BS[dotIndex * BN + i + blockX * TN];
            }
            
            for (uint resM = 0; resM < TM; resM++) {
                for (uint resN = 0; resN < TN; resN++) {
                    resultC[resM * TN + resN] += aCache[resM] * bCache[resN];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    for (uint resM = 0; resM < TM; resM++) {
        // notice here, that we increment by 4.
        for (uint resN = 0; resN < TN; resN+=4) {
            
            // first step kinda redunant.
            float4 tmp = reinterpret_cast<device float4*>(&C[(blockY * TM + resM) * N + blockX * TN + resN])[0];
            tmp.x = resultC[resM * TN + resN];
            tmp.y = resultC[resM * TN + resN + 1];
            tmp.z = resultC[resM * TN + resN + 2];
            tmp.w = resultC[resM * TN + resN + 3];
            reinterpret_cast<device float4*>(&C[(blockY * TM + resM) * N + blockX * TN + resN])[0] = tmp;
            
        }
    }
}


kernel void matmul_kernel_v6(device const float4* A,
                             device const float4* B,
                             device float4* C,
                             constant uint &M [[ buffer(3) ]],
                             constant uint &N [[ buffer(4) ]],
                             constant uint &K [[ buffer(5) ]],
                             uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                             uint2 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]],
                             uint2 threads_per_threadgroup [[ threads_per_threadgroup ]]) {
    
    // I need to add an assert, which ensure statically, ensure that number of threadgroup
    // #thread_in_threadgroup == BK * BN / 4. If this is false, then we have to have
    // different GMEM -> SMEM algorithm, we could just add a loop for this
    // cause we are always either 2, 4 many thread too less or more.
    // this would work for uneven number of thread as well.
    
    
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 4;
    const uint BM = 64;
    const uint BN = 64;
    
    const uint blockX = thread_position_in_threadgroup.x % (BN / TN);
    const uint blockY = thread_position_in_threadgroup.x / (BN / TN);
    
    const uint blockCol = threadgroup_position_in_grid.x;
    const uint blockRow = threadgroup_position_in_grid.y;
    
    // compute blockTiles
    //const uint totalNumberTiles = BM * BN;
    //const uint numberOfThreads = totalNumberTiles / (TM * TN);
    
    // the .metal compiler seems to ignore asserts.
    //assert(numberOfThreads == threads_per_threadgroup.x);
    
    A += (K * blockRow * BM) / 4;
    B += (blockCol * BN) / 4;
    // TODO: author has put N here, I think it should be K.
        // So the stride is in the N, dim not K.
    C += (blockCol * BN + N * blockRow * BM) / 4;
    
    threadgroup float AS[BM * BK];
    threadgroup float BS[BN * BK];
    
    thread float resultC[TM * TN] = {0.0}; // on thread memory
    thread float aCache[TM] = {0.0}; // temp cache during the dot product.
    thread float bCache[TN] = {0.0};
    
    // each thread, loads in four values, so we need skip by four.
    const uint innerColA = thread_position_in_threadgroup.x % (BK / 4);
    const uint innerRowA = thread_position_in_threadgroup.x / (BK / 4);
    const uint innerColB = thread_position_in_threadgroup.x % (BN / 4);
    const uint innerRowB = thread_position_in_threadgroup.x / (BN / 4);


    for (uint blockInx = 0; blockInx < K; blockInx += BK) {
        
        
        if (innerRowA < BM) {
            // We tranpose AS while loading in, each thread loads in a vector row of 4, then places in a coloumn of AS.
            //float4 tmp = reinterpret_cast<device const float4 *>(&A[innerColA * 4 + K * innerRowA])[0];
            
            float4 tmp = A[innerColA + (K / 4) * innerRowA];
            // need to jump in the row by 4 from the vector loads.
            AS[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
            AS[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
            AS[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
            AS[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
        }
        
        if (innerRowB < BK) {
            //float4 tmpB = reinterpret_cast<device const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
            //reinterpret_cast<threadgroup float4 *>(&BS[innerRowB * BN + innerColB * 4])[0] = tmpB;
            float4 tmp = B[innerRowB * (N / 4) + innerColB];
            BS[innerRowB * BN + innerColB * 4] = tmp.x;
            BS[innerRowB * BN + innerColB * 4 + 1] = tmp.y;
            BS[innerRowB * BN + innerColB * 4 + 2] = tmp.z;
            BS[innerRowB * BN + innerColB * 4 + 3] = tmp.w;
        }
        
        threadgroup_barrier(mem_flags::mem_none);
        
        // shift A and B
        A += (BK / 4);
        B += (N * BK / 4);

        for (uint dotIndex = 0; dotIndex < BK; dotIndex++) {
            // fill in the thread memory to reduce shared memory access.
            for (uint i = 0; i < TM; i++) {
                aCache[i] = AS[dotIndex * BM + TM * blockY + i];
            }
            
            for (uint i = 0; i < TN; i++) {
                bCache[i] = BS[dotIndex * BN + i + blockX * TN];
            }
            
            for (uint resM = 0; resM < TM; resM++) {
                for (uint resN = 0; resN < TN; resN++) {
                    resultC[resM * TN + resN] += aCache[resM] * bCache[resN];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    for (uint resM = 0; resM < TM; resM++) {
        // notice here, that we increment by 4.
        for (uint resN = 0; resN < TN / 4; resN++) {
        
            float4 tmp = float4(resultC[resM * TN + (resN * 4)],
                                resultC[resM * TN + (resN * 4) + 1],
                                resultC[resM * TN + (resN * 4) + 2],
                                resultC[resM * TN + (resN * 4) + 3]);
            C[(blockY * TM + resM) * (N / 4) + blockX * (TN / 4) + resN] = tmp;
        }
    }
}
