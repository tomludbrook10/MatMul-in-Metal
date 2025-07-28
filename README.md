# MatMul-in-Metal

The first six kernels are a Metal port from https://siboehm.com/articles/22/CUDA-MMM. Currently, kernel v6 achieves 1550GFLOPS on the m1 pro GPU and is memory bound. 

### Next Steps 

I'm currently working on the final kernel when I find the time. The current idea is to exploit the SIMD-GROUP atomics
(6.9.2 of https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) to achieve SIMD-GROUP level parallelism. 
