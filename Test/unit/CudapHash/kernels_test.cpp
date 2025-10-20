#include <gtest/gtest.h>

#include "kernels_test_wrapper.h"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcountll __popcnt
#endif
#include <corecrt_math_defines.h>

namespace Kernels {

// Helper function to verify if a matrix is approximately orthogonal
bool isOrthogonal(const std::vector<float>& matrix, int N, float tolerance = 1e-4f) {
    std::vector<float> product(N * N, 0.0f);

    // Multiply matrix by its transpose
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += matrix[i * N + k] * matrix[j * N + k];
            }
            product[i * N + j] = sum;
        }
    }

    // Check if product is identity matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            if (std::abs(product[i * N + j] - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// DCT Kernel Tests
// ============================================================================

class DctKernelTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(DctKernelTest, BasicDCTMatrix) {
    const int N = 8;
    std::vector<float> T(N * N, 0.0f);

    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    ASSERT_NO_THROW(testDctMatrixKernel(T.data(), N, invSqrtN, sqrtTwoDivN));

    // Verify first row (all elements should be invSqrtN)
    for (int j = 0; j < N; j++) {
        EXPECT_NEAR(T[j], invSqrtN, 1e-5);
    }

    // Verify second row alternates signs with proper cosine values
    for (int j = 0; j < N; j++) {
        float expected = sqrtTwoDivN * std::cos((2.0f * j + 1.0f) * M_PI / (2.0f * N));
        EXPECT_NEAR(T[N + j], expected, 1e-5);
    }
}

TEST_F(DctKernelTest, OrthogonalityProperty) {
    const int N = 8;
    std::vector<float> T(N * N, 0.0f);

    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    ASSERT_NO_THROW(testDctMatrixKernel(T.data(), N, invSqrtN, sqrtTwoDivN));

    // DCT matrix should be orthogonal (T * T^T = I)
    EXPECT_TRUE(isOrthogonal(T, N));
}

TEST_F(DctKernelTest, LargerMatrix32x32) {
    const int N = 32;
    std::vector<float> T(N * N, 0.0f);

    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    ASSERT_NO_THROW(testDctMatrixKernel(T.data(), N, invSqrtN, sqrtTwoDivN));

    // Check first row values
    for (int j = 0; j < N; j++) {
        EXPECT_NEAR(T[j], invSqrtN, 1e-5);
    }

    // Check orthogonality for larger matrix
    EXPECT_TRUE(isOrthogonal(T, N, 1e-3f));
}

TEST_F(DctKernelTest, SmallMatrix4x4) {
    const int N = 4;
    std::vector<float> T(N * N, 0.0f);

    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    ASSERT_NO_THROW(testDctMatrixKernel(T.data(), N, invSqrtN, sqrtTwoDivN));

    // For a 4x4 matrix, we can verify exact values
    EXPECT_NEAR(T[0], 0.5f, 1e-5);  // 1/sqrt(4) = 0.5
    EXPECT_NEAR(T[1], 0.5f, 1e-5);
    EXPECT_NEAR(T[2], 0.5f, 1e-5);
    EXPECT_NEAR(T[3], 0.5f, 1e-5);
}

// ============================================================================
// Batch Bicubic Resize Kernel Tests
// ============================================================================

class BicubicResizeTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper to create test image with gradient
    std::vector<unsigned char> createGradientImage(size_t width, size_t height) {
        std::vector<unsigned char> image(width * height);
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                image[y * width + x] = static_cast<unsigned char>((x * 255) / width);
            }
        }
        return image;
    }

    // Helper to create checkerboard pattern
    std::vector<unsigned char> createCheckerboard(size_t width, size_t height, size_t squareSize) {
        std::vector<unsigned char> image(width * height);
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                bool isWhite = ((x / squareSize) + (y / squareSize)) % 2 == 0;
                image[y * width + x] = isWhite ? 255 : 0;
            }
        }
        return image;
    }
};

TEST_F(BicubicResizeTest, BasicResize) {
    const size_t inWidth = 16, inHeight = 16;
    const size_t outSize = 8;
    const int batchSize = 1;

    // Create test data
    auto inputImage = createGradientImage(inWidth, inHeight);
    std::vector<float> outputImage(outSize * outSize);

    TestGpuData gpuData;
    gpuData.originalWidth = inWidth;
    gpuData.originalHeight = inHeight;

    const unsigned char* inputPtr = inputImage.data();
    float* outputPtr = outputImage.data();

    ASSERT_NO_THROW(testBatchBicubicResizeKernel(
        &gpuData, batchSize, outSize, &inputPtr, &outputPtr
    ));

    // Verify output is not all zeros
    bool hasNonZero = false;
    for (float val : outputImage) {
        if (val != 0.0f) {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero);
}

TEST_F(BicubicResizeTest, BatchProcessing) {
    const size_t inWidth = 32, inHeight = 32;
    const size_t outSize = 16;
    const int batchSize = 3;

    // Create test data for batch
    std::vector<std::vector<unsigned char>> inputImages;
    std::vector<std::vector<float>> outputImages(batchSize);
    std::vector<const unsigned char*> inputPtrs;
    std::vector<float*> outputPtrs;
    std::vector<TestGpuData> gpuDataArray(batchSize);

    for (int i = 0; i < batchSize; i++) {
        inputImages.push_back(createGradientImage(inWidth, inHeight));
        outputImages[i].resize(outSize * outSize);
        inputPtrs.push_back(inputImages[i].data());
        outputPtrs.push_back(outputImages[i].data());

        gpuDataArray[i].originalWidth = inWidth;
        gpuDataArray[i].originalHeight = inHeight;
    }

    ASSERT_NO_THROW(testBatchBicubicResizeKernel(
        gpuDataArray.data(), batchSize, outSize,
        inputPtrs.data(), outputPtrs.data()
    ));

    // Verify all batches have output
    for (int i = 0; i < batchSize; i++) {
        bool hasNonZero = false;
        for (float val : outputImages[i]) {
            if (val != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        EXPECT_TRUE(hasNonZero) << "Batch " << i << " has no non-zero values";
    }
}

TEST_F(BicubicResizeTest, NonSquareInput) {
    const size_t inWidth = 20, inHeight = 10;
    const size_t outSize = 8;
    const int batchSize = 1;

    auto inputImage = createGradientImage(inWidth, inHeight);
    std::vector<float> outputImage(outSize * outSize);

    TestGpuData gpuData;
    gpuData.originalWidth = inWidth;
    gpuData.originalHeight = inHeight;

    const unsigned char* inputPtr = inputImage.data();
    float* outputPtr = outputImage.data();

    ASSERT_NO_THROW(testBatchBicubicResizeKernel(
        &gpuData, batchSize, outSize, &inputPtr, &outputPtr
    ));

    // Output should still be square
    EXPECT_EQ(outputImage.size(), outSize * outSize);
}

TEST_F(BicubicResizeTest, EdgeCase1x1Output) {
    const size_t inWidth = 10, inHeight = 10;
    const size_t outSize = 1;
    const int batchSize = 1;

    auto inputImage = createGradientImage(inWidth, inHeight);
    std::vector<float> outputImage(outSize * outSize);

    TestGpuData gpuData;
    gpuData.originalWidth = inWidth;
    gpuData.originalHeight = inHeight;

    const unsigned char* inputPtr = inputImage.data();
    float* outputPtr = outputImage.data();

    ASSERT_NO_THROW(testBatchBicubicResizeKernel(
        &gpuData, batchSize, outSize, &inputPtr, &outputPtr
    ));

    // Should produce a single pixel
    EXPECT_EQ(outputImage.size(), 1u);
    EXPECT_GE(outputImage[0], 0.0f);
    EXPECT_LE(outputImage[0], 255.0f);
}

TEST_F(BicubicResizeTest, CheckerboardPattern) {
    const size_t inWidth = 64, inHeight = 64;
    const size_t outSize = 32;
    const int batchSize = 1;

    auto inputImage = createCheckerboard(inWidth, inHeight, 8);
    std::vector<float> outputImage(outSize * outSize);

    TestGpuData gpuData;
    gpuData.originalWidth = inWidth;
    gpuData.originalHeight = inHeight;

    const unsigned char* inputPtr = inputImage.data();
    float* outputPtr = outputImage.data();

    ASSERT_NO_THROW(testBatchBicubicResizeKernel(
        &gpuData, batchSize, outSize, &inputPtr, &outputPtr
    ));

    // Check that we have both light and dark pixels
    float minVal = *std::min_element(outputImage.begin(), outputImage.end());
    float maxVal = *std::max_element(outputImage.begin(), outputImage.end());

    EXPECT_LT(minVal, 128.0f);  // Should have dark pixels
    EXPECT_GT(maxVal, 128.0f);  // Should have light pixels
}

// ============================================================================
// Median Threshold Kernel Tests
// ============================================================================

class MedianThresholdTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper to create test DCT data
    std::vector<float> createTestDCT(int size, float fillValue = 0.0f) {
        std::vector<float> dct(size * size, fillValue);
        return dct;
    }

    // Count the number of set bits in a hash
    int countBits(const TestpHash& hash) {
        return __builtin_popcountll(hash.words[0]) +
               __builtin_popcountll(hash.words[1]);
    }
};

TEST_F(MedianThresholdTest, AllZeroInput) {
    const int cropSize = 8;
    const int stride = 32;
    const int batchSize = 1;

    auto dctData = createTestDCT(stride, 0.0f);
    const float* dctPtr = dctData.data();
    TestpHash outputHash;

    ASSERT_NO_THROW(testMedianThresholdKernel(
        &dctPtr, cropSize, stride, &outputHash, batchSize
    ));

    // With all zeros (except DC component), hash should be all zeros
    EXPECT_EQ(outputHash.words[0], 0ULL);
    EXPECT_EQ(outputHash.words[1], 0ULL);
}

TEST_F(MedianThresholdTest, HalfHighHalfLow) {
    const int cropSize = 8;
    const int stride = 32;
    const int batchSize = 1;

    auto dctData = createTestDCT(stride, 0.0f);
    // Set half the values to high, half to low
    for (int i = 0; i < cropSize; i++) {
        for (int j = 0; j < cropSize; j++) {
            if (i * cropSize + j < 32) {
                dctData[i * stride + j] = 100.0f;
            } else {
                dctData[i * stride + j] = -100.0f;
            }
        }
    }

    const float* dctPtr = dctData.data();
    TestpHash outputHash;

    ASSERT_NO_THROW(testMedianThresholdKernel(
        &dctPtr, cropSize, stride, &outputHash, batchSize
    ));

    // Should have roughly half bits set
    int bitCount = countBits(outputHash);
    EXPECT_GT(bitCount, 20);  // Should have some bits set
    EXPECT_LT(bitCount, 44);  // But not all bits
}

TEST_F(MedianThresholdTest, AlternatingPattern) {
    const int cropSize = 8;
    const int stride = 32;
    const int batchSize = 1;

    auto dctData = createTestDCT(stride, 0.0f);
    // Create alternating pattern
    for (int i = 0; i < cropSize; i++) {
        for (int j = 0; j < cropSize; j++) {
            dctData[i * stride + j] = ((i + j) % 2 == 0) ? 100.0f : -100.0f;
        }
    }

    const float* dctPtr = dctData.data();
    TestpHash outputHash;

    ASSERT_NO_THROW(testMedianThresholdKernel(
        &dctPtr, cropSize, stride, &outputHash, batchSize
    ));

    // Should have roughly half bits set
    int bitCount = countBits(outputHash);
    EXPECT_GT(bitCount, 20);
    EXPECT_LT(bitCount, 44);
}

TEST_F(MedianThresholdTest, BatchProcessing) {
    const int cropSize = 8;
    const int stride = 32;
    const int batchSize = 4;

    std::vector<std::vector<float>> dctDataArray(batchSize);
    std::vector<const float*> dctPtrs;

    for (int i = 0; i < batchSize; i++) {
        dctDataArray[i] = createTestDCT(stride, static_cast<float>(i * 10));
        dctPtrs.push_back(dctDataArray[i].data());
    }

    std::vector<TestpHash> outputHashes(batchSize);

    ASSERT_NO_THROW(testMedianThresholdKernel(
        dctPtrs.data(), cropSize, stride, outputHashes.data(), batchSize
    ));

    // Each batch should produce different hash
    for (int i = 1; i < batchSize; i++) {
        bool isDifferent = (outputHashes[i].words[0] != outputHashes[0].words[0]) ||
                          (outputHashes[i].words[1] != outputHashes[0].words[1]);
        EXPECT_TRUE(isDifferent) << "Batch " << i << " has same hash as batch 0";
    }
}

TEST_F(MedianThresholdTest, DCComponentSkipped) {
    const int cropSize = 8;
    const int stride = 32;
    const int batchSize = 1;

    auto dctData = createTestDCT(stride, 50.0f);
    // Set DC component (0,0) to extreme value
    dctData[0] = 1000000.0f;

    const float* dctPtr = dctData.data();
    TestpHash outputHash;

    ASSERT_NO_THROW(testMedianThresholdKernel(
        &dctPtr, cropSize, stride, &outputHash, batchSize
    ));

    // Despite extreme DC value, hash should be based on other components
    // All other components are the same, so should get specific pattern
    EXPECT_NE(outputHash.words[0], 0xFFFFFFFFFFFFFFFFULL);
}

// ============================================================================
// Compute LSH Keys Kernel Tests
// ============================================================================

class ComputeLSHKeysTest : public ::testing::Test {
protected:
    void SetUp() override {}

    // Helper to create test hash
    TestpHash createTestHash(uint64_t word0, uint64_t word1) {
        TestpHash hash;
        hash.words[0] = word0;
        hash.words[1] = word1;
        return hash;
    }
};

TEST_F(ComputeLSHKeysTest, BasicKeyExtraction) {
    const int n = 1;
    const int bitsPerTable = 8;
    const int tableIndex = 0;

    std::vector<size_t> offsets = {0};
    auto hash = createTestHash(0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL);
    std::vector<int> bitPositions;
    for (int i = 0; i < bitsPerTable; i++) {
        bitPositions.push_back(i);
    }

    uint64_t key;
    int idx;

    ASSERT_NO_THROW(testComputeLSHKeysKernel(
        offsets.data(), &hash, n, bitPositions.data(),
        bitsPerTable, tableIndex, &key, &idx
    ));

    // First 8 bits of 0xAA... should be 10101010 = 0xAA
    EXPECT_EQ(key, 0xAAULL);
    EXPECT_EQ(idx, 0);
}

TEST_F(ComputeLSHKeysTest, MultipleHashes) {
    const int n = 3;
    const int bitsPerTable = 4;
    const int tableIndex = 0;

    std::vector<size_t> offsets = {0, 1, 2};
    std::vector<TestpHash> hashes = {
        createTestHash(0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL),
        createTestHash(0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL),
        createTestHash(0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL)
    };

    std::vector<int> bitPositions = {0, 1, 2, 3};
    std::vector<uint64_t> keys(n);
    std::vector<int> indices(n);

    ASSERT_NO_THROW(testComputeLSHKeysKernel(
        offsets.data(), hashes.data(), n, bitPositions.data(),
        bitsPerTable, tableIndex, keys.data(), indices.data()
    ));

    // First hash: first 4 bits of 0xFF... = 1111 = 15
    EXPECT_EQ(keys[0], 15ULL);
    // Second hash: first 4 bits of 0x00... = 0000 = 0
    EXPECT_EQ(keys[1], 0ULL);
    // Third hash: first 4 bits of 0xF0... = 0000 = 0
    EXPECT_EQ(keys[2], 0ULL);

    // Indices should be 0, 1, 2
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(indices[i], i);
    }
}

TEST_F(ComputeLSHKeysTest, CrossWordBoundary) {
    const int n = 1;
    const int bitsPerTable = 8;
    const int tableIndex = 0;

    std::vector<size_t> offsets = {0};
    auto hash = createTestHash(0xFFFFFFFFFFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL);

    // Select bits that cross word boundary (bits 60-67)
    std::vector<int> bitPositions;
    for (int i = 60; i < 68; i++) {
        bitPositions.push_back(i);
    }

    uint64_t key;
    int idx;

    ASSERT_NO_THROW(testComputeLSHKeysKernel(
        offsets.data(), &hash, n, bitPositions.data(),
        bitsPerTable, tableIndex, &key, &idx
    ));

    // Bits 60-63 from word[0] = 1111, bits 64-67 from word[1] = 1010
    // Combined = 10101111 = 0xAF
    EXPECT_EQ(key, 0xAFULL);
}

TEST_F(ComputeLSHKeysTest, DifferentTables) {
    const int n = 1;
    const int bitsPerTable = 4;

    std::vector<size_t> offsets = {0};
    auto hash = createTestHash(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);

    // Create bit positions for multiple tables
    std::vector<int> bitPositions;
    for (int t = 0; t < 3; t++) {
        for (int b = 0; b < bitsPerTable; b++) {
            bitPositions.push_back(t * bitsPerTable + b);
        }
    }

    std::vector<uint64_t> keys(3);
    std::vector<int> indices(3);

    // Test different tables
    for (int table = 0; table < 3; table++) {
        ASSERT_NO_THROW(testComputeLSHKeysKernel(
            offsets.data(), &hash, n, bitPositions.data(),
            bitsPerTable, table, &keys[table], &indices[table]
        ));
    }

    // Each table should extract different bits and produce different keys
    EXPECT_NE(keys[0], keys[1]);
    EXPECT_NE(keys[1], keys[2]);
    EXPECT_NE(keys[0], keys[2]);
}

// ============================================================================
// Find Pairs Kernel Tests
// ============================================================================

class FindPairsTest : public ::testing::Test {
protected:
    void SetUp() override {}

    int hammingDistance(const TestpHash& a, const TestpHash& b) {
        uint64_t xor0 = a.words[0] ^ b.words[0];
        uint64_t xor1 = a.words[1] ^ b.words[1];
        return __builtin_popcountll(xor0) + __builtin_popcountll(xor1);
    }
};

TEST_F(FindPairsTest, NoPairsFound) {
    const int n = 3;
    const int threshold = 5;
    const int maxEdges = 10;
    const int maxPairsPerBucket = 10;

    // All have same key (to trigger comparison)
    std::vector<uint64_t> keys = {1, 1, 1};
    std::vector<int> indices = {0, 1, 2};
    std::vector<size_t> offsets = {0, 1, 2};

    // Create hashes with large Hamming distance
    std::vector<TestpHash> hashes = {
        {0x0000000000000000ULL, 0x0000000000000000ULL},
        {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
        {0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL}
    };

    std::vector<TestEdge> edges(maxEdges);
    int edgeCount = 0;

    ASSERT_NO_THROW(testFindPairsKernel(
        keys.data(), indices.data(), n, offsets.data(), hashes.data(),
        threshold, edges.data(), &edgeCount, maxEdges, maxPairsPerBucket
    ));

    // No pairs should be found (all distances > threshold)
    EXPECT_EQ(edgeCount, 0);
}

TEST_F(FindPairsTest, SinglePairFound) {
    const int n = 2;
    const int threshold = 10;
    const int maxEdges = 10;
    const int maxPairsPerBucket = 10;

    std::vector<uint64_t> keys = {1, 1};  // Same key
    std::vector<int> indices = {0, 1};
    std::vector<size_t> offsets = {0, 1};

    // Create similar hashes (distance = 1)
    std::vector<TestpHash> hashes = {
        {0x0000000000000000ULL, 0x0000000000000000ULL},
        {0x0000000000000001ULL, 0x0000000000000000ULL}
    };

    std::vector<TestEdge> edges(maxEdges);
    int edgeCount = 0;

    ASSERT_NO_THROW(testFindPairsKernel(
        keys.data(), indices.data(), n, offsets.data(), hashes.data(),
        threshold, edges.data(), &edgeCount, maxEdges, maxPairsPerBucket
    ));

    EXPECT_EQ(edgeCount, 1);
    EXPECT_EQ(edges[0].i, 0);
    EXPECT_EQ(edges[0].j, 1);
    EXPECT_EQ(edges[0].dist, 1);
}

TEST_F(FindPairsTest, MultiplePairsInBucket) {
    const int n = 4;
    const int threshold = 20;
    const int maxEdges = 10;
    const int maxPairsPerBucket = 10;

    // All in same bucket
    std::vector<uint64_t> keys = {5, 5, 5, 5};
    std::vector<int> indices = {0, 1, 2, 3};
    std::vector<size_t> offsets = {0, 1, 2, 3};

    // Create hashes with various small distances
    std::vector<TestpHash> hashes = {
        {0x0000000000000000ULL, 0x0000000000000000ULL},
        {0x0000000000000001ULL, 0x0000000000000000ULL},
        {0x0000000000000003ULL, 0x0000000000000000ULL},
        {0x0000000000000007ULL, 0x0000000000000000ULL}
    };

    std::vector<TestEdge> edges(maxEdges);
    int edgeCount = 0;

    ASSERT_NO_THROW(testFindPairsKernel(
        keys.data(), indices.data(), n, offsets.data(), hashes.data(),
        threshold, edges.data(), &edgeCount, maxEdges, maxPairsPerBucket
    ));

    // Should find multiple pairs
    EXPECT_GT(edgeCount, 0);
    EXPECT_LE(edgeCount, 6);  // Maximum possible pairs from 4 items

    // Verify all found pairs have distance < threshold
    for (int i = 0; i < edgeCount; i++) {
        int dist = hammingDistance(hashes[edges[i].i], hashes[edges[i].j]);
        EXPECT_LT(dist, threshold);
        EXPECT_EQ(edges[i].dist, dist);
    }
}

TEST_F(FindPairsTest, MaxPairsPerBucketLimit) {
    const int n = 10;
    const int threshold = 128;  // Accept all pairs
    const int maxEdges = 100;
    const int maxPairsPerBucket = 5;  // Limit pairs per bucket

    // All in same bucket
    std::vector<uint64_t> keys(n, 42);
    std::vector<int> indices(n);
    std::vector<size_t> offsets(n);
    std::vector<TestpHash> hashes(n);

    for (int i = 0; i < n; i++) {
        indices[i] = i;
        offsets[i] = i;
        hashes[i] = {static_cast<uint64_t>(i), 0ULL};
    }

    std::vector<TestEdge> edges(maxEdges);
    int edgeCount = 0;

    ASSERT_NO_THROW(testFindPairsKernel(
        keys.data(), indices.data(), n, offsets.data(), hashes.data(),
        threshold, edges.data(), &edgeCount, maxEdges, maxPairsPerBucket
    ));

    // Should be limited by maxPairsPerBucket
    // With 10 items, there are 45 possible pairs, but we limit to 5
    EXPECT_LE(edgeCount, maxPairsPerBucket);
}

TEST_F(FindPairsTest, MultipleBuckets) {
    const int n = 6;
    const int threshold = 10;
    const int maxEdges = 20;
    const int maxPairsPerBucket = 10;

    // Three different buckets
    std::vector<uint64_t> keys = {1, 1, 2, 2, 3, 3};
    std::vector<int> indices = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> offsets = {0, 1, 2, 3, 4, 5};

    // Similar hashes within each bucket
    std::vector<TestpHash> hashes = {
        {0x0000000000000000ULL, 0x0000000000000000ULL},  // Bucket 1
        {0x0000000000000001ULL, 0x0000000000000000ULL},  // Bucket 1
        {0x1000000000000000ULL, 0x0000000000000000ULL},  // Bucket 2
        {0x1000000000000001ULL, 0x0000000000000000ULL},  // Bucket 2
        {0x2000000000000000ULL, 0x0000000000000000ULL},  // Bucket 3
        {0x2000000000000001ULL, 0x0000000000000000ULL}   // Bucket 3
    };

    std::vector<TestEdge> edges(maxEdges);
    int edgeCount = 0;

    ASSERT_NO_THROW(testFindPairsKernel(
        keys.data(), indices.data(), n, offsets.data(), hashes.data(),
        threshold, edges.data(), &edgeCount, maxEdges, maxPairsPerBucket
    ));

    // Should find one pair per bucket = 3 pairs total
    EXPECT_EQ(edgeCount, 3);

    // Verify pairs are correct
    bool foundPair01 = false, foundPair23 = false, foundPair45 = false;
    for (int i = 0; i < edgeCount; i++) {
        if ((edges[i].i == 0 && edges[i].j == 1) ||
            (edges[i].i == 1 && edges[i].j == 0)) foundPair01 = true;
        if ((edges[i].i == 2 && edges[i].j == 3) ||
            (edges[i].i == 3 && edges[i].j == 2)) foundPair23 = true;
        if ((edges[i].i == 4 && edges[i].j == 5) ||
            (edges[i].i == 5 && edges[i].j == 4)) foundPair45 = true;
    }
    EXPECT_TRUE(foundPair01);
    EXPECT_TRUE(foundPair23);
    EXPECT_TRUE(foundPair45);
}

TEST_F(FindPairsTest, EdgeOrdering) {
    const int n = 3;
    const int threshold = 10;
    const int maxEdges = 10;
    const int maxPairsPerBucket = 10;

    std::vector<uint64_t> keys = {1, 1, 1};
    std::vector<int> indices = {2, 0, 1};  // Shuffled indices
    std::vector<size_t> offsets = {0, 1, 2};

    std::vector<TestpHash> hashes = {
        {0x0000000000000000ULL, 0x0000000000000000ULL},
        {0x0000000000000001ULL, 0x0000000000000000ULL},
        {0x0000000000000002ULL, 0x0000000000000000ULL}
    };

    std::vector<TestEdge> edges(maxEdges);
    int edgeCount = 0;

    ASSERT_NO_THROW(testFindPairsKernel(
        keys.data(), indices.data(), n, offsets.data(), hashes.data(),
        threshold, edges.data(), &edgeCount, maxEdges, maxPairsPerBucket
    ));

    // Verify edge indices are always ordered (i < j)
    for (int i = 0; i < edgeCount; i++) {
        EXPECT_LT(edges[i].i, edges[i].j) << "Edge " << i << " not properly ordered";
    }
}

} // namespace Kernels