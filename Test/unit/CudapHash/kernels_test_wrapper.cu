#include "kernels.cuh"

#include "kernels_test_wrapper.h"
#include <cuda_runtime.h>
#include <stdexcept>

// Wrapper function for testing dctMatrixKernel
void testDctMatrixKernel(float* h_T, int N, float invSqrtN, float sqrtTwoDivN) {
    float* d_T;

    cudaMalloc(&d_T, N * N * sizeof(float));
    cudaMemcpy(d_T, h_T, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(N);
    dim3 threads(N);
    dctMatrixKernel << <blocks, threads >> > (d_T, invSqrtN, sqrtTwoDivN);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_T);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_T, d_T, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_T);
}