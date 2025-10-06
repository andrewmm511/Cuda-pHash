#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>
#include <nvjpeg.h>
#include <cublas_v2.h>
#include "progress_tracker.hpp"

class WorkQueue;
class MemoryManager;

struct GpuData {
    unsigned char* decodedPtr = nullptr;
    float* resizedPtr = nullptr;
    size_t originalWidth;
    size_t originalHeight;
};

struct alignas(16) pHash {
    uint64_t words[2] = { 0, 0 };
};

struct Image {
    size_t id;
    std::string path;
    size_t fileSize = 0;
    unsigned char* rawFileData = nullptr;
    GpuData gpuData;
    size_t hashOffset = static_cast<size_t>(-1);
    pHash hash;
    std::string mostSimilarImage;
};

/**
 * @brief A CUDA-based pHash calculator.
 */
class CudaPhash {
public:
    using ProgressCallback = std::function<void(const ProgressInfo&)>;

    /**
     * @brief Constructor
     * @param hashSize        Size of the final hash dimension (e.g. 8)
     * @param highFreqFactor  Factor of oversampling for DCT (e.g. 4 => 32x32)
     * @param batchSize       Number of images to process per batch (default = 500)
     * @param threads         Number of threads for file I/O (-1 for auto)
     * @param prefetchFactor  Queue size multiplier for prefetching
     * @param logLevel        Logging verbosity level
     * @param progressCb      Optional callback for progress updates
     */
    CudaPhash(int hashSize, int highFreqFactor, int batchSize = 500, int threads = -1,
              int prefetchFactor = 8, int logLevel = 1U, ProgressCallback progressCb = nullptr);

    /**
     * @brief Destructor
     */
    ~CudaPhash();

    /**
     * @brief Sets an external cuBLAS handle if desired.
     * @param handle The cuBLAS handle to use
     */
    void setHandle(const cublasHandle_t& handle);
    
    /**
     * @brief Computes and returns raw pHashes of a list of images.
     * @param imagePaths Paths to the images on disk.
     * @return A vector of pHash results.
     */
    std::vector<pHash> computeHashes(const std::vector<std::string>& imagePaths);

    /**
    * @brief Detects pairs of images below a certain pHash similarity threshold.
    * @param imagePaths Paths to the images on disk.
    * @param threshold The Hamming distance threshold for duplicates.
    * @param numTables The number of LSH tables to use.
    * @param bitsPerTable The number of bits per LSH table.
    * @return A vector of images that are duplicates.
    */
    std::vector<Image> findDuplicatesGPU(const std::vector<std::string>& imagePaths, int threshold, int numTables = 32, int bitsPerTable = 8);

private:
    // Configuration
    const int m_hashSize;
    const int m_highFreqFactor;
    const int m_imgSize;
    const int m_batchSize;
    int m_threads;
    const int m_prefetchFactor;
    const float m_alpha = 1.0f;
    const float m_beta = 0.0f;

    // CUDA handles
    cublasHandle_t m_handle;
    std::unique_ptr<MemoryManager> m_memMgr;
    nvjpegHandle_t m_nvjHandle;
    nvjpegJpegState_t m_nvjSingleDecoderState;
    nvjpegJpegState_t m_nvjDecoderState;

    // Device memory
    GpuData* d_gpuDataVec = nullptr;
    float* d_T = nullptr;
    float* d_TT = nullptr;
    float* d_tmpBatch = nullptr;
    const float** d_TArray = nullptr;
    const float** d_TTArray = nullptr;
    float** d_AArray = nullptr;
    float** d_tmpArray = nullptr;
    float** d_AoutArray = nullptr;

    // Streams
    cudaStream_t m_decodeStream;
    cudaStream_t m_resizeStream;
    cudaStream_t m_hashStream;

    // Hash stage memory
    const float** d_hashImgPtrs = nullptr;
    pHash* d_hashes = nullptr;

    // Progress tracking
    ProgressCallback m_progressCallback;
    std::unique_ptr<ProgressTracker> m_progressTracker;

    // Pipeline methods
    std::vector<Image> runPipeline(const std::vector<std::string>& imagePaths);
    void reader(WorkQueue& inQueue, WorkQueue& outQueue, int threadId);
    void decoder(WorkQueue& inQueue, WorkQueue& outQueue);
    void resizer(WorkQueue& inQueue, WorkQueue& outQueue);
    void hasher(WorkQueue& outQueue);
};
