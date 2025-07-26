#include "kernels.cuh"
#include "../include/phash_cuda.cuh"
#include "../cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

__device__ inline int clamp(int x, int min, int max) {
    return x < min ? min : (x > max ? max : x);
}

__device__ inline float cubicWeight(float x, float a = -0.5f) {
    x = fabsf(x);
    if (x <= 1.0f) { return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f; }
    else if (x < 2.0f) { return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a; }
    else { return 0.0f; }
}

__global__ void batchBicubicResizeKernel(const GpuData* data, size_t outSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outSize || y >= outSize) { return; }

    const unsigned char* d_in = data[blockIdx.z].decodedPtr;
    float* d_out = data[blockIdx.z].resizedPtr;
    size_t inWidth = data[blockIdx.z].originalWidth;
    size_t inHeight = data[blockIdx.z].originalHeight;

    float u = (x + 0.5f) / outSize * inWidth;
    float v = (y + 0.5f) / outSize * inHeight;

    int x0 = floorf(u) - 1;
    int y0 = floorf(v) - 1;

    float weightX[4], weightY[4];
#pragma unroll
    for (int i = 0; i < 4; i++) { weightX[i] = cubicWeight(u - (x0 + i)); }
#pragma unroll
    for (int j = 0; j < 4; j++) { weightY[j] = cubicWeight(v - (y0 + j)); }

    float sum = 0.0f;
    float totalWeight = 0.0f;

#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int xi = clamp(x0 + i, 0, static_cast<int>(inWidth) - 1);
            int yj = clamp(y0 + j, 0, static_cast<int>(inHeight) - 1);

            float weight = weightX[i] * weightY[j];

            sum += weight * d_in[yj * inWidth + xi];
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0f) { sum /= totalWeight; }

    d_out[y * outSize + x] = sum;
}

__global__ void dctMatrixKernel(float* T, float invSqrtN, float sqrtTwoDivN) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int N = blockDim.x;

    float normFactor = (i == 0) ? invSqrtN : sqrtTwoDivN;
    T[i * N + j] = normFactor * cosf(((2.0f * j + 1.0f) / (2.0f * N)) * static_cast<float>(i) * PI);
}

__global__ void medianThresholdKernel(const float* const* __restrict__ d_imgs, int cropSize, int stride, pHash* __restrict__ d_outHashes, int batchSize) {
    extern __shared__ float sdata[];

    int imgIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (imgIdx >= batchSize) { return; }

    if (tid == 0) {
        d_outHashes[imgIdx].words[0] = 0ULL;
        d_outHashes[imgIdx].words[1] = 0ULL;
    }
    __syncthreads();

    const int pixels = cropSize * cropSize;

    float val = 0.0f;
    if (tid < pixels) {
        int row = tid / cropSize;
        int col = tid % cropSize;
        if (row != 0 || col != 0) {
            val = d_imgs[imgIdx][row * stride + col];
        }
    }
    sdata[tid] = val;
    __syncthreads();

    for (int k = 2; k <= pixels; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && ixj < pixels && tid < pixels) {
                bool ascending = ((tid & k) == 0);
                float a = sdata[tid];
                float b = sdata[ixj];
                if ((ascending && a > b) || (!ascending && a < b)) {
                    sdata[tid] = b;
                    sdata[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float median;
        if (pixels & 1) {
            median = sdata[pixels / 2];
        }
        else {
            median = 0.5f * (sdata[pixels / 2 - 1] + sdata[pixels / 2]);
        }
        sdata[0] = median;
    }
    __syncthreads();
    float median = sdata[0];

    if (tid < pixels && val > median) {
        if (tid < 64) {
            atomicOr(reinterpret_cast<unsigned long long*>(&d_outHashes[imgIdx].words[0]), 1ULL << tid);
        }
        else {
            atomicOr(reinterpret_cast<unsigned long long*>(&d_outHashes[imgIdx].words[1]), 1ULL << (tid - 64));
        }
    }
}

__device__ __forceinline__ int hammingDistance128(const uint64_t* wA, const uint64_t* wB) {
    uint64_t x0 = wA[0] ^ wB[0];
    uint64_t x1 = wA[1] ^ wB[1];
    return __popcll(x0) + __popcll(x1);
}

__global__ void computeLSHKeysKernel(const size_t* __restrict__ d_offsets, const pHash* __restrict__ d_hashes, int n, const int* __restrict__ d_bitPositions, int bitsPerTable, int tableIndex, uint64_t* __restrict__ d_keys, int* __restrict__ d_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }

    size_t hashOffset = d_offsets[i];
    pHash h = d_hashes[hashOffset];

    uint64_t key = 0;
    const int offsetInBitPos = tableIndex * bitsPerTable;
    for (int b = 0; b < bitsPerTable; b++) {
        int bitPos = d_bitPositions[offsetInBitPos + b];
        uint64_t bit = (h.words[bitPos / 64] >> (bitPos % 64)) & 1ULL;
        key |= (bit << b);
    }

    d_keys[i] = key;
    d_idx[i] = i;
}

__global__ void findPairsKernel(const uint64_t* __restrict__ d_keys, const int* __restrict__ d_idx, int n, const size_t* __restrict__ d_offsets, const pHash* __restrict__ d_hashes, int threshold, Edge* __restrict__ d_outEdges, int* __restrict__ d_edgeCount, int maxEdges, int maxPairsPerBucket) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int pos = tid; pos < n; pos += totalThreads) {
        uint64_t currentKey = d_keys[pos];

        int runStart = pos;
        while (runStart > 0 && d_keys[runStart - 1] == currentKey) {
            runStart--;
        }

        int runEnd = pos + 1;
        while (runEnd < n && d_keys[runEnd] == currentKey) {
            runEnd++;
        }

        int runSize = runEnd - runStart;

        if (pos != runStart) { continue; }

        long long totalPairs = (1ll * runSize * (runSize - 1)) / 2;
        long long pairsToProcess = totalPairs;

        if (maxPairsPerBucket > 0 && totalPairs > maxPairsPerBucket) {
            pairsToProcess = maxPairsPerBucket;
        }

        long long step = totalPairs / pairsToProcess;
        if (step < 1) { step = 1; }

        for (long long sampleIdx = 0; sampleIdx < pairsToProcess; sampleIdx++) {
            long long pairIdx = sampleIdx * step;
            if (pairIdx >= totalPairs) { break; }

            int i = static_cast<int>((sqrtf(8.0f * pairIdx + 1.0f) - 1.0f) * 0.5f);
            int j = static_cast<int>(pairIdx - (1ll * i * (i + 1) / 2)) + i + 1;

            if (i >= runSize || j >= runSize) { continue; }

            int idx1 = runStart + i;
            int idx2 = runStart + j;

            int img1_idx = d_idx[idx1];
            int img2_idx = d_idx[idx2];

            const pHash h1 = d_hashes[d_offsets[img1_idx]];
            const pHash h2 = d_hashes[d_offsets[img2_idx]];

            uint64_t x0 = h1.words[0] ^ h2.words[0];
            uint64_t x1 = h1.words[1] ^ h2.words[1];
            int dist = __popcll(x0) + __popcll(x1);

            if (dist < threshold) {
                int i_final = (img1_idx < img2_idx) ? img1_idx : img2_idx;
                int j_final = (img1_idx < img2_idx) ? img2_idx : img1_idx;

                int edgeIdx = atomicAdd(d_edgeCount, 1);
                if (edgeIdx < maxEdges) {
                    d_outEdges[edgeIdx] = { i_final, j_final, dist };
                }
            }
        }
    }
}