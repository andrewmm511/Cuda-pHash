#ifndef TEST_CUDA_WRAPPERS_H
#define TEST_CUDA_WRAPPERS_H

#ifdef __cplusplus
extern "C" {
#endif

	// Wrapper function declarations
	void testDctMatrixKernel(float* h_T, int N, float invSqrtN, float sqrtTwoDivN);

#ifdef __cplusplus
}
#endif

#endif // TEST_CUDA_WRAPPERS_H