#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>

struct GpuData;
struct pHash;

struct LSHKeyPair {
    uint64_t key;
    int imageIdx;
    int table;
};

struct LSHKeyPairComparator {
    __host__ __device__ bool operator()(const LSHKeyPair& a, const LSHKeyPair& b) const {
        if (a.table != b.table) { return a.table < b.table; }
        return a.key < b.key;
    }
};

struct Edge {
    int i;
    int j;
    int dist;

    bool operator<(const Edge& other) const {
        return std::tie(i, j) < std::tie(other.i, other.j);
    }
    bool operator==(const Edge& other) const {
        return i == other.i && j == other.j;
    }
};

// Kernel declarations
__global__ void batchBicubicResizeKernel(const GpuData* data, size_t outSize);
__global__ void dctMatrixKernel(float* T, float invSqrtN, float sqrtTwoDivN);
__global__ void medianThresholdKernel(const float* const* __restrict__ d_imgs, int cropSize, int stride, pHash* __restrict__ d_outHashes, int batchSize);
__global__ void findPairsKernel(const uint64_t* __restrict__ d_keys, const int* __restrict__ d_idx, int n, const size_t* __restrict__ d_offsets, const pHash* __restrict__ d_hashes, int threshold, Edge* __restrict__ d_outEdges, int* __restrict__ d_edgeCount, int maxEdges, int maxPairsPerBucket = 10000);
__global__ void computeLSHKeysKernel(const size_t* __restrict__ d_offsets, const pHash* __restrict__ d_hashes, int n, const int* __restrict__ d_bitPositions, int bitsPerTable, int tableIndex, uint64_t* __restrict__ d_keys, int* __restrict__ d_idx);

#endif // KERNELS_CUH