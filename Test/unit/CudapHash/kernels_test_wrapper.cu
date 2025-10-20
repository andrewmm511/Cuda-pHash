#include "kernels.cuh"
#include "kernels_test_wrapper.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Map test structures to actual kernel structures
struct GpuData {
    unsigned char* decodedPtr = nullptr;
    float* resizedPtr = nullptr;
    size_t originalWidth;
    size_t originalHeight;
};

struct alignas(16) pHash {
    uint64_t words[2] = { 0, 0 };
};

// Helper function to copy TestGpuData to GpuData
GpuData convertToGpuData(const TestGpuData& testData) {
    GpuData data;
    data.decodedPtr = testData.decodedPtr;
    data.resizedPtr = testData.resizedPtr;
    data.originalWidth = testData.originalWidth;
    data.originalHeight = testData.originalHeight;
    return data;
}

// Wrapper function for testing dctMatrixKernel
void testDctMatrixKernel(float* h_T, int N, float invSqrtN, float sqrtTwoDivN) {
    float* d_T;

    cudaMalloc(&d_T, N * N * sizeof(float));
    cudaMemcpy(d_T, h_T, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(N);
    dim3 threads(N);
    dctMatrixKernel<<<blocks, threads>>>(d_T, invSqrtN, sqrtTwoDivN);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_T);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_T, d_T, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_T);
}

// Wrapper function for testing batchBicubicResizeKernel
void testBatchBicubicResizeKernel(
    const TestGpuData* h_gpuDataArray,
    int batchSize,
    size_t outSize,
    const unsigned char** h_inputImages,
    float** h_outputImages)
{
    GpuData* d_gpuDataArray;
    cudaMalloc(&d_gpuDataArray, batchSize * sizeof(GpuData));

    std::vector<GpuData> tempGpuData(batchSize);
    std::vector<unsigned char*> d_inputPtrs(batchSize);
    std::vector<float*> d_outputPtrs(batchSize);

    for (int i = 0; i < batchSize; i++) {
        size_t inputSize = h_gpuDataArray[i].originalWidth * h_gpuDataArray[i].originalHeight;
        cudaMalloc(&d_inputPtrs[i], inputSize * sizeof(unsigned char));
        cudaMalloc(&d_outputPtrs[i], outSize * outSize * sizeof(float));

        cudaMemcpy(d_inputPtrs[i], h_inputImages[i], inputSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        tempGpuData[i] = convertToGpuData(h_gpuDataArray[i]);
        tempGpuData[i].decodedPtr = d_inputPtrs[i];
        tempGpuData[i].resizedPtr = d_outputPtrs[i];
    }

    cudaMemcpy(d_gpuDataArray, tempGpuData.data(), batchSize * sizeof(GpuData), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((outSize + threads.x - 1) / threads.x,
                (outSize + threads.y - 1) / threads.y,
                batchSize);

    batchBicubicResizeKernel<<<blocks, threads>>>(d_gpuDataArray, outSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        for (int i = 0; i < batchSize; i++) {
            cudaFree(d_inputPtrs[i]);
            cudaFree(d_outputPtrs[i]);
        }
        cudaFree(d_gpuDataArray);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < batchSize; i++) {
        cudaMemcpy(h_outputImages[i], d_outputPtrs[i], outSize * outSize * sizeof(float), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < batchSize; i++) {
        cudaFree(d_inputPtrs[i]);
        cudaFree(d_outputPtrs[i]);
    }
    cudaFree(d_gpuDataArray);
}

// Wrapper function for testing medianThresholdKernel
void testMedianThresholdKernel(
    const float** h_imgs,
    int cropSize,
    int stride,
    TestpHash* h_outHashes,
    int batchSize
) {
    float** d_imgs;
    cudaMalloc(&d_imgs, batchSize * sizeof(float*));

    std::vector<float*> d_imgPtrs(batchSize);
    for (int i = 0; i < batchSize; i++) {
        cudaMalloc(&d_imgPtrs[i], stride * cropSize * sizeof(float));
        cudaMemcpy(d_imgPtrs[i], h_imgs[i], stride * cropSize * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_imgs, d_imgPtrs.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice);

    pHash* d_outHashes;
    cudaMalloc(&d_outHashes, batchSize * sizeof(pHash));

    int threadsPerBlock = 256; // Should be at least cropSize*cropSize
    int sharedMemSize = threadsPerBlock * sizeof(float);

    medianThresholdKernel<<<batchSize, threadsPerBlock, sharedMemSize>>>(
        (const float* const*)d_imgs, cropSize, stride, d_outHashes, batchSize
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        for (int i = 0; i < batchSize; i++) cudaFree(d_imgPtrs[i]);
        cudaFree(d_imgs);
        cudaFree(d_outHashes);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_outHashes, d_outHashes, batchSize * sizeof(pHash), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++) cudaFree(d_imgPtrs[i]);
    cudaFree(d_imgs);
    cudaFree(d_outHashes);
}

// Wrapper function for testing computeLSHKeysKernel
void testComputeLSHKeysKernel(
    const size_t* h_offsets,
    const TestpHash* h_hashes,
    int n,
    const int* h_bitPositions,
    int bitsPerTable,
    int tableIndex,
    uint64_t* h_keys,
    int* h_idx
) {
    size_t* d_offsets;
    pHash* d_hashes;
    int* d_bitPositions;
    uint64_t* d_keys;
    int* d_idx;

    cudaMalloc(&d_offsets, n * sizeof(size_t));
    cudaMalloc(&d_hashes, n * sizeof(pHash));
    int bitPosSize = (tableIndex + 1) * bitsPerTable;
    cudaMalloc(&d_bitPositions, bitPosSize * sizeof(int));
    cudaMalloc(&d_keys, n * sizeof(uint64_t));
    cudaMalloc(&d_idx, n * sizeof(int));

    cudaMemcpy(d_offsets, h_offsets, n * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashes, h_hashes, n * sizeof(pHash), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitPositions, h_bitPositions, bitPosSize * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    computeLSHKeysKernel<<<blocks, threadsPerBlock>>>(
        d_offsets, d_hashes, n, d_bitPositions, bitsPerTable, tableIndex, d_keys, d_idx
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_offsets);
        cudaFree(d_hashes);
        cudaFree(d_bitPositions);
        cudaFree(d_keys);
        cudaFree(d_idx);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_keys, d_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx, d_idx, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_offsets);
    cudaFree(d_hashes);
    cudaFree(d_bitPositions);
    cudaFree(d_keys);
    cudaFree(d_idx);
}

// Wrapper function for testing findPairsKernel
void testFindPairsKernel(
    const uint64_t* h_keys,
    const int* h_idx,
    int n,
    const size_t* h_offsets,
    const TestpHash* h_hashes,
    int threshold,
    TestEdge* h_outEdges,
    int* h_edgeCount,
    int maxEdges,
    int maxPairsPerBucket
) {
    uint64_t* d_keys;
    int* d_idx;
    size_t* d_offsets;
    pHash* d_hashes;
    Edge* d_outEdges;
    int* d_edgeCount;

    cudaMalloc(&d_keys, n * sizeof(uint64_t));
    cudaMalloc(&d_idx, n * sizeof(int));
    cudaMalloc(&d_offsets, n * sizeof(size_t));
    cudaMalloc(&d_hashes, n * sizeof(pHash));
    cudaMalloc(&d_outEdges, maxEdges * sizeof(Edge));
    cudaMalloc(&d_edgeCount, sizeof(int));

    cudaMemcpy(d_keys, h_keys, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, n * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashes, h_hashes, n * sizeof(pHash), cudaMemcpyHostToDevice);
    cudaMemset(d_edgeCount, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = (blocks > 1024) ? 1024 : blocks; // Cap at reasonable number

    findPairsKernel<<<blocks, threadsPerBlock>>>(
        d_keys, d_idx, n, d_offsets, d_hashes, threshold,
        d_outEdges, d_edgeCount, maxEdges, maxPairsPerBucket
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_keys);
        cudaFree(d_idx);
        cudaFree(d_offsets);
        cudaFree(d_hashes);
        cudaFree(d_outEdges);
        cudaFree(d_edgeCount);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_edgeCount, d_edgeCount, sizeof(int), cudaMemcpyDeviceToHost);
    int actualEdges = (*h_edgeCount > maxEdges) ? maxEdges : *h_edgeCount;
    cudaMemcpy(h_outEdges, d_outEdges, actualEdges * sizeof(Edge), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_idx);
    cudaFree(d_offsets);
    cudaFree(d_hashes);
    cudaFree(d_outEdges);
    cudaFree(d_edgeCount);
}