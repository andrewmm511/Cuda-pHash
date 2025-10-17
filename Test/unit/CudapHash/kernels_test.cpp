#include <gtest/gtest.h>

#include "kernels_test_wrapper.h"

#include <vector>
#include <cmath>
#include <stdexcept>

namespace CudapHash {

class DctKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // TODO
    }
};

TEST_F(DctKernelTest, BasicDCTMatrix) {
    const int N = 8;
    std::vector<float> T(N * N, 0.0f);

    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    // Call the wrapper function
    ASSERT_NO_THROW(testDctMatrixKernel(T.data(), N, invSqrtN, sqrtTwoDivN));

    // Verify the results
    // Check first row (all elements should be invSqrtN)
    for (int j = 0; j < N; j++) {
        EXPECT_NEAR(T[j], invSqrtN, 1e-5);
    }

    // Check specific known values or properties
    // For example, verify orthogonality, specific DCT values, etc.
}

TEST_F(DctKernelTest, LargerMatrix) {
    const int N = 32;
    std::vector<float> T(N * N, 0.0f);

    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    ASSERT_NO_THROW(testDctMatrixKernel(T.data(), N, invSqrtN, sqrtTwoDivN));

    // Add verification logic here
}

// Test for error conditions if applicable
TEST_F(DctKernelTest, HandlesNullPointer) {
    const int N = 8;
    float invSqrtN = 1.0f / std::sqrt(static_cast<float>(N));
    float sqrtTwoDivN = std::sqrt(2.0f / N);

    // This should handle the null pointer gracefully in wrapper
    // Modify wrapper to check for null and throw exception?
    EXPECT_THROW(testDctMatrixKernel(nullptr, N, invSqrtN, sqrtTwoDivN), std::runtime_error);
}

} // CudapHash namespace