#pragma once

#include "phash_base.h"

#include <vector>
#include <string>

#ifdef BUILDING_DLL
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif

// Forward-declare cublasHandle_t
struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

/**
 * @brief A light-weight struct for GPU image data.
 */
struct GpuImage {
    float* data = nullptr; ///< Pointer to GPU buffer
    size_t pitch = 0;      ///< Pitch in bytes
    int    width = 0;      ///< Image width
    int    height = 0;     ///< Image height
};

/**
 * @brief A CUDA-based pHash calculator.
 */
class API_EXPORT CudaPhash : public PhashBase {
public:
    /**
     * @brief Constructor
     * @param hashSize        Size of the final hash dimension (e.g. 8)
     * @param highFreqFactor  Factor of oversampling for DCT (e.g. 4 => 32x32)
     * @param batchSize       Number of images to process per batch (default = 5000)
     */
    CudaPhash(int hashSize, int highFreqFactor, int batchSize = 5000);

    /**
     * @brief Destructor
     */
    ~CudaPhash();

    /**
     * @brief Sets an external cuBLAS handle if desired.
     */
    void setHandle(const cublasHandle_t& handle);

    /**
     * @brief Computes pHashes of a list of images on disk, using m_batchSize images per batch.
     * @param imagePaths Paths to the images on disk.
     * @return A vector of pHash results (each pHash is a vector of 32-bit words).
     */
    std::vector<std::vector<uint32_t>> phash(const std::vector<std::string>& imagePaths);

private:
    int              m_hashSize;
    int              m_highFreqFactor;
    int              m_imgSize;
    int              m_batchSize;
    cublasHandle_t   m_handle;

    float* d_T;
    float* d_TT;
    float* d_tmp_batch;
    size_t d_tmp_batch_capacity;
    float** d_A_array;
    float** d_tmp_array;
    float** d_Aout_array;
    const float** d_T_array;
    const float** d_TT_array;
    size_t d_array_capacity;
    cudaStream_t m_stream;

    /**
     * @brief Performs batched DCT on a list of GPU images (all must have same pitch).
     * @param gpuImages Images to process via batched DCT.
     */
    void batchDct(std::vector<GpuImage>& gpuImages);

    /**
     * @brief Computes bit-packed mean-threshold hashes for a set of float images that are
     *        (m_hashSize x m_hashSize).
     * @param inputs GpuImage array.
     * @return Hash results as a vector of unsigned int (bit-packed).
     */
    std::vector<unsigned int> computeMeanThresholdHash(const std::vector<GpuImage>& inputs);
};