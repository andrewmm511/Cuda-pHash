#ifndef TEST_CUDA_WRAPPERS_H
#define TEST_CUDA_WRAPPERS_H

#include <cstdint>

// Mirror structures from the main codebase for testing
struct TestGpuData {
    unsigned char* decodedPtr = nullptr;
    float* resizedPtr = nullptr;
    size_t originalWidth;
    size_t originalHeight;
};

struct alignas(16) TestpHash {
    uint64_t words[2] = { 0, 0 };
};

struct TestEdge {
    int i;
    int j;
    int dist;
};

#ifdef __cplusplus
extern "C" {
#endif

    // Wrapper function declarations
    void testDctMatrixKernel(float* h_T, int N, float invSqrtN, float sqrtTwoDivN);

    void testBatchBicubicResizeKernel(
        const TestGpuData* h_gpuDataArray,
        int batchSize,
        size_t outSize,
        const unsigned char** h_inputImages,
        float** h_outputImages
    );

    void testMedianThresholdKernel(
        const float** h_imgs,
        int cropSize,
        int stride,
        TestpHash* h_outHashes,
        int batchSize
    );

    void testComputeLSHKeysKernel(
        const size_t* h_offsets,
        const TestpHash* h_hashes,
        int n,
        const int* h_bitPositions,
        int bitsPerTable,
        int tableIndex,
        uint64_t* h_keys,
        int* h_idx
    );

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
    );

#ifdef __cplusplus
}
#endif

#endif // TEST_CUDA_WRAPPERS_H