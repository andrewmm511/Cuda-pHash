#include "../include/phash_cuda.cuh"
#include "logger.hpp"
#include "work_queue.hpp"
#include "cuda_utils.hpp"
#include "kernels.cuh"
#include "memory_manager.cuh"

// System headers
#include <algorithm>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvjpeg.h>

// Thrust headers
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#ifdef _WIN32
#include <io.h>
#define stat _stat64
#endif

CudaPhash::CudaPhash(int hashSize, int highFreqFactor, int batchSize, int threads, int prefetchFactor, int logLevel, ProgressCallback progressCb)
    : m_hashSize(hashSize),
    m_highFreqFactor(highFreqFactor),
    m_imgSize(hashSize* highFreqFactor),
    m_batchSize(batchSize),
    m_threads(threads),
    m_prefetchFactor(prefetchFactor),
    m_handle(nullptr),
    m_memMgr(std::make_unique<MemoryManager>()),
    m_progressCallback(progressCb),
    m_progressTracker(nullptr) {
    if (hashSize < 5 || hashSize > 11) { throw std::invalid_argument("hashSize must be between 5 and 11!"); }
    if (highFreqFactor < 1 || highFreqFactor > 8) { throw std::invalid_argument("highFreqFactor must be between 1 and 8!"); }

    logger::init(static_cast<logger::Level>(logLevel));

    if (threads < 1) {
        m_threads = std::min<int>(32, static_cast<int>(std::thread::hardware_concurrency()));
        DEBUG("Main", "Setting concurrency to ", m_threads, " threads.");
    }

    nvjpegStatus_t status = nvjpegCreateExV2(NVJPEG_BACKEND_GPU_HYBRID, m_memMgr->getNvjDevAllocator(), m_memMgr->getNvjPinnedAllocator(), 0, &m_nvjHandle);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ERROR_L("Main", "nvjpegCreateExV2 failed with status: ", status);
        throw std::runtime_error("nvJPEG handle creation failed");
    }

    status = nvjpegJpegStateCreate(m_nvjHandle, &m_nvjSingleDecoderState);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ERROR_L(__func__, "nvjpegJpegStateCreate failed with status: ", status);
        throw std::runtime_error("nvJPEG state creation failed");
    }

    status = nvjpegJpegStateCreate(m_nvjHandle, &m_nvjDecoderState);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ERROR_L(__func__, "nvjpegJpegStateCreate failed with status: ", status);
        throw std::runtime_error("nvJPEG state creation failed");
    }
    nvjpegDecodeBatchedInitialize(m_nvjHandle, m_nvjDecoderState, m_batchSize, m_threads, NVJPEG_OUTPUT_Y);

    CUDA_CHECK(cudaMalloc(&d_gpuDataVec, m_batchSize * sizeof(GpuData)));

    CUDA_CHECK(cudaStreamCreate(&m_decodeStream));
    CUDA_CHECK(cudaStreamCreate(&m_resizeStream));
    CUDA_CHECK(cudaStreamCreate(&m_hashStream));

    CUDA_CHECK(cudaMalloc(&d_T, m_imgSize * m_imgSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_TT, m_imgSize * m_imgSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AArray, m_batchSize * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_tmpArray, m_batchSize * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_AoutArray, m_batchSize * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_TArray, m_batchSize * sizeof(const float*)));
    CUDA_CHECK(cudaMalloc(&d_TTArray, m_batchSize * sizeof(const float*)));
    CUDA_CHECK(cudaMalloc(&d_tmpBatch, m_batchSize * static_cast<size_t>(m_imgSize) * m_imgSize * sizeof(float)));

    std::vector<const float*> h_T_ptrs(m_batchSize, d_T);
    std::vector<const float*> h_TT_ptrs(m_batchSize, d_TT);
    CUDA_CHECK(cudaMemcpy(d_TArray, h_T_ptrs.data(), m_batchSize * sizeof(const float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_TTArray, h_TT_ptrs.data(), m_batchSize * sizeof(const float*), cudaMemcpyHostToDevice));

    dctMatrixKernel << <m_imgSize, m_imgSize >> > (d_T, std::sqrt(1.0f / static_cast<float>(m_imgSize)), std::sqrt(2.0f / static_cast<float>(m_imgSize)));

    cublasCreate(&m_handle);
    cublasSetStream(m_handle, m_hashStream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgeam(m_handle, CUBLAS_OP_T, CUBLAS_OP_N, m_imgSize, m_imgSize, &alpha, d_T, m_imgSize, &beta, d_T, m_imgSize, d_TT, m_imgSize);

    cublasSetMathMode(m_handle, CUBLAS_TF32_TENSOR_OP_MATH);
}

CudaPhash::~CudaPhash() {
    if (m_handle) { cublasDestroy(m_handle); }

    if (d_gpuDataVec) { cudaFree(d_gpuDataVec); }
    if (d_T) { cudaFree(d_T); }
    if (d_TT) { cudaFree(d_TT); }
    if (d_tmpBatch) { cudaFree(d_tmpBatch); }
    if (d_TArray) { cudaFree(d_TArray); }
    if (d_TTArray) { cudaFree(d_TTArray); }
    if (d_AArray) { cudaFree(d_AArray); }
    if (d_tmpArray) { cudaFree(d_tmpArray); }
    if (d_AoutArray) { cudaFree(d_AoutArray); }
    if (d_hashImgPtrs) { cudaFree(d_hashImgPtrs); }
    if (d_hashes) { cudaFree(d_hashes); }

    cudaStreamDestroy(m_decodeStream);
    cudaStreamDestroy(m_resizeStream);
    cudaStreamDestroy(m_hashStream);
    nvjpegDestroy(m_nvjHandle);
}

void CudaPhash::setHandle(const cublasHandle_t& handle)
{
    m_handle = handle;
}

void CudaPhash::reader(WorkQueue& inQueue, WorkQueue& outQueue, int threadId) {
    const std::string id = __func__ + std::to_string(threadId);

    int nComponents = 0;
    nvjpegChromaSubsampling_t sampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    while (true) {
        if (inQueue.empty()) {
            INFO(id, "Completed work, exiting.");
            return;
        }
        Image* img = inQueue.pop();

        std::ifstream file(img->path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            WARN(id, "Could not open file: ", img->path);

            if (m_progressTracker) m_progressTracker->updateFailed(1);
            continue;
        }

        std::streamsize fileSize = file.tellg();
        if (fileSize <= 0) {
            DEBUG(id, "Empty or invalid file: ", img->path);
            file.close();

            if (m_progressTracker) m_progressTracker->updateFailed(1);
            continue;
        }
        img->fileSize = static_cast<size_t>(fileSize);

        unsigned char* pinnedMem = m_memMgr->Allocate<unsigned char>(MemoryManager::PINNED_POOL, img->fileSize);
        img->rawFileData = pinnedMem;

        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(pinnedMem), img->fileSize);
        file.close();

        nvjpegStatus_t info = nvjpegGetImageInfo(m_nvjHandle, pinnedMem, img->fileSize, &nComponents, &sampling, widths, heights);

        if (info != NVJPEG_STATUS_SUCCESS) {
            ERROR_L(id, "nvjpegGetImageInfo failed for ", img->path);
            m_memMgr->Free<unsigned char>(MemoryManager::PINNED_POOL, pinnedMem);
            img->rawFileData = nullptr;

            if (m_progressTracker) m_progressTracker->updateFailed(1);
            continue;
        }

        img->gpuData.originalWidth = widths[0];
        img->gpuData.originalHeight = heights[0];

        outQueue.push(img);

        if (m_progressTracker) m_progressTracker->update(PipelineStage::Read, 1);
    }
}

void CudaPhash::decoder(WorkQueue& inQueue, WorkQueue& outQueue) {
    nvjpegStatus_t status;
    nvjpegStatus_t batchStatus = NVJPEG_STATUS_SUCCESS;

    while (true) {
        std::vector<Image*> imgs = inQueue.popMax(m_batchSize);

        if (imgs.empty() && inQueue.isSentinel() && inQueue.empty()) {
            INFO(__func__, "Completed work, exiting.");
            break;
        }

        INFO(__func__, "Decoding ", imgs.size(), " images. Queue: ", inQueue.size());

        if (imgs.size() != static_cast<size_t>(m_batchSize) || batchStatus != NVJPEG_STATUS_SUCCESS) {
            INFO(__func__, "Reinitializing nvJPEG for batch of ", imgs.size(), " images");

            status = nvjpegDecodeBatchedInitialize(m_nvjHandle, m_nvjDecoderState, static_cast<int>(imgs.size()), m_threads, NVJPEG_OUTPUT_Y);
            if (status != NVJPEG_STATUS_SUCCESS) {
                ERROR_L(__func__, "nvjpegDecodeBatchedInitialize failed with status: ", status);
                throw std::runtime_error("nvJPEG batched initialization failed");
            }
            batchStatus = NVJPEG_STATUS_SUCCESS;
        }

        std::vector<const unsigned char*> rawPointers(imgs.size());
        std::vector<size_t> rawLengths(imgs.size());
        std::vector<nvjpegImage_t> outImages(imgs.size());

        size_t totalSize = 0;
        for (size_t i = 0; i < imgs.size(); ++i) {
            Image* img = imgs[i];
            rawPointers[i] = img->rawFileData;
            rawLengths[i] = img->fileSize;

            size_t decodeSize = img->gpuData.originalWidth * img->gpuData.originalHeight;
            totalSize += decodeSize;
            unsigned char* dMem = m_memMgr->Allocate<unsigned char>(MemoryManager::DEVICE_POOL, decodeSize);
            img->gpuData.decodedPtr = dMem;

            memset(&outImages[i], 0, sizeof(nvjpegImage_t));
            outImages[i].channel[0] = dMem;
            outImages[i].pitch[0] = img->gpuData.originalWidth;
        }

        DEBUG(__func__, "Allocated ", totalSize / (1024 * 1024), " MB for decoded images.");

        batchStatus = nvjpegDecodeBatched(m_nvjHandle, m_nvjDecoderState, rawPointers.data(), rawLengths.data(), outImages.data(), m_decodeStream);
        cudaStreamSynchronize(m_decodeStream);

        if (batchStatus != NVJPEG_STATUS_SUCCESS) {
            WARN(__func__, "nvjpegDecodeBatched failed with code: ", batchStatus, ". Attempting to decode batch individually.");

            std::vector<Image*> successfulImages;
            successfulImages.reserve(imgs.size());

            for (auto* img : imgs) {
                DEBUG(__func__, "Attempting to decode image ", img->path, " individually.");
                nvjpegImage_t singleOut;
                memset(&singleOut, 0, sizeof(nvjpegImage_t));
                singleOut.channel[0] = img->gpuData.decodedPtr;
                singleOut.pitch[0] = img->gpuData.originalWidth;

                nvjpegStatus_t singleDecode = nvjpegDecode(m_nvjHandle, m_nvjSingleDecoderState, img->rawFileData, img->fileSize, NVJPEG_OUTPUT_Y, &singleOut, m_decodeStream);

                if (singleDecode != NVJPEG_STATUS_SUCCESS) {
                    WARN(__func__, "Failed to decode image ", img->path, ". Skipping. Error code: ", singleDecode);
                    m_memMgr->Free(MemoryManager::DEVICE_POOL, img->gpuData.decodedPtr);
                    img->gpuData.decodedPtr = nullptr;

                    if (m_progressTracker) m_progressTracker->updateFailed(1);
                }
                else {
                    DEBUG(__func__, "SUCCESSFUL -- ", img->path);
                    successfulImages.push_back(img);
                }
            }

            cudaStreamSynchronize(m_decodeStream);

            INFO(__func__, "Individual decode succeeded for ", successfulImages.size(), " of ", imgs.size(), " images.");

            if (!successfulImages.empty()) {
                outQueue.pushMany(successfulImages);
                if (m_progressTracker) m_progressTracker->update(PipelineStage::Decode, successfulImages.size());
            }

        }
        else {
            INFO(__func__, "Decoded ", imgs.size(), " images.");
            outQueue.pushMany(imgs);

            if (m_progressTracker) m_progressTracker->update(PipelineStage::Decode, imgs.size());
        }

        for (auto& img : imgs) {
            m_memMgr->Free(MemoryManager::PINNED_POOL, img->rawFileData);
            img->rawFileData = nullptr;
        }
    }
    outQueue.setSentinel();
}

void CudaPhash::resizer(WorkQueue& inQueue, WorkQueue& outQueue) {
    const std::string id = "Resizer";
    dim3 block(32, 16, 1);

    while (true) {
        std::vector<Image*> imgs = inQueue.popMax(m_batchSize);

        if (inQueue.isSentinel() && imgs.empty() && inQueue.empty()) {
            INFO(id, "Completed work, exiting.");
            break;
        }

        INFO(id, "Resizing ", imgs.size(), " images.");

        size_t batchSize = imgs.size();

        std::vector<GpuData> gpuDataVec(batchSize);
        for (size_t i = 0; i < batchSize; ++i) {
            imgs[i]->gpuData.resizedPtr = m_memMgr->Allocate<float>(MemoryManager::DEVICE_POOL, m_imgSize * m_imgSize);
            gpuDataVec[i] = imgs[i]->gpuData;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_gpuDataVec, gpuDataVec.data(), batchSize * sizeof(GpuData), cudaMemcpyHostToDevice, m_resizeStream));

        dim3 grid((static_cast<int>(m_imgSize) + block.x - 1) / block.x, (static_cast<int>(m_imgSize) + block.y - 1) / block.y, static_cast<int>(batchSize));
        batchBicubicResizeKernel <<<grid, block, 1, m_resizeStream >>> (d_gpuDataVec, m_imgSize);

        cudaStreamSynchronize(m_resizeStream);

        for (auto& img : imgs) {
            m_memMgr->Free(MemoryManager::DEVICE_POOL, img->gpuData.decodedPtr);
            img->gpuData.decodedPtr = nullptr;
        }

        outQueue.pushMany(imgs);

        if (m_progressTracker) m_progressTracker->update(PipelineStage::Resize, imgs.size());

        INFO(id, "Resized ", imgs.size(), " images.");
    }

    outQueue.setSentinel();
}

static int nextPow2OrMax1024(int x) {
    if (x > 1024) { return 1024; }
    int r = 1;
    while (r < x && r < 1024) { r <<= 1; }
    return r;
}

void CudaPhash::hasher(WorkQueue& inQueue) {
    int blockSize = nextPow2OrMax1024(m_hashSize * m_hashSize);

    size_t idx = 0;
    while (true) {
        std::vector<Image*> imgs = inQueue.popMax(m_batchSize);

        if (imgs.empty() && inQueue.isSentinel()) {
            DEBUG(__func__, "Completed work, exiting.");
            break;
        }

        size_t batchSize = imgs.size();
        const int N = m_imgSize;

        INFO(__func__, "Hashing ", batchSize, " images.");

        std::vector<const float*> h_A_array(batchSize);
        std::vector<float*> h_tmp_array(batchSize);
        std::vector<float*> h_Aout_array(batchSize);
        std::vector<const float*> h_T_array(batchSize, d_T);
        std::vector<const float*> h_TT_array(batchSize, d_TT);

        for (size_t i = 0; i < batchSize; ++i) {
            h_A_array[i] = imgs[i]->gpuData.resizedPtr;
            h_tmp_array[i] = d_tmpBatch + i * static_cast<size_t>(N) * N;
            h_Aout_array[i] = imgs[i]->gpuData.resizedPtr;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_AArray, h_A_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream));
        CUDA_CHECK(cudaMemcpyAsync(d_tmpArray, h_tmp_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream));
        CUDA_CHECK(cudaMemcpyAsync(d_AoutArray, h_Aout_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream));
        CUDA_CHECK(cudaMemcpyAsync(d_TArray, h_T_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream));
        CUDA_CHECK(cudaMemcpyAsync(d_TTArray, h_TT_array.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream));

        cublasSgemmBatched(m_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &m_alpha, d_TArray, N, d_AArray, N, &m_beta, d_tmpArray, N, static_cast<int>(batchSize));
        cublasSgemmBatched(m_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &m_alpha, d_tmpArray, N, d_TTArray, N, &m_beta, d_AoutArray, N, static_cast<int>(batchSize));

        std::vector<const float*> h_imgPtrs(batchSize, nullptr);
        for (size_t i = 0; i < batchSize; ++i) { h_imgPtrs[i] = imgs[i]->gpuData.resizedPtr; }

        CUDA_CHECK(cudaMemcpyAsync(d_hashImgPtrs + idx, h_imgPtrs.data(), batchSize * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream));

        medianThresholdKernel << <static_cast<int>(batchSize), blockSize, blockSize * sizeof(float), m_hashStream >> > (
            d_hashImgPtrs + idx, m_hashSize, m_imgSize, d_hashes + idx, static_cast<int>(batchSize));

        for (size_t i = 0; i < batchSize; ++i) { imgs[i]->hashOffset = idx + i; }

        cudaStreamSynchronize(m_hashStream);

        for (auto& img : imgs) {
            m_memMgr->Free(MemoryManager::DEVICE_POOL, img->gpuData.resizedPtr);
            img->gpuData.resizedPtr = nullptr;
        }

        idx += imgs.size();

        if (m_progressTracker) m_progressTracker->update(PipelineStage::Hash, imgs.size());

        INFO(__func__, "Hashed ", imgs.size(), " images.");
    }
}

std::vector<Image> CudaPhash::runPipeline(const std::vector<std::string>& imagePaths) {
    INFO("Main", "Starting pHash computation on ", imagePaths.size(), " images.");

    std::vector<Image> imgs(imagePaths.size());
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        imgs[i].path = imagePaths[i];
    }

    // Initialize progress tracker with the callback from constructor
    m_progressTracker = std::make_unique<ProgressTracker>(imgs.size(), m_progressCallback);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hashImgPtrs), imgs.size() * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_hashes, imgs.size() * sizeof(pHash)));

    m_memMgr->InitializePools();

    WorkQueue inputQueue(imgs.size(), "input");
    WorkQueue decodeQueue(m_batchSize * m_prefetchFactor, "decode");
    WorkQueue resizeQueue(imgs.size(), "resize");
    WorkQueue hashQueue(imgs.size(), "hash");

    for (auto& img : imgs) {
        inputQueue.push(&img);
    }

    std::vector<std::thread> readers;
    for (int i = 0; i < m_threads; ++i) {
        readers.emplace_back(&CudaPhash::reader, this, std::ref(inputQueue), std::ref(decodeQueue), i);
    }

    std::thread decoder(&CudaPhash::decoder, this, std::ref(decodeQueue), std::ref(resizeQueue));
    std::thread resizer(&CudaPhash::resizer, this, std::ref(resizeQueue), std::ref(hashQueue));
    std::thread hasher(&CudaPhash::hasher, this, std::ref(hashQueue));

    for (auto& t : readers) {
        if (t.joinable()) { t.join(); }
    }
    decodeQueue.setSentinel();

    decoder.join();
    resizer.join();
    hasher.join();

    cudaStreamSynchronize(m_decodeStream);
    cudaStreamSynchronize(m_resizeStream);
    cudaStreamSynchronize(m_hashStream);

    auto end = std::ranges::remove_if(imgs, [](const Image& img) { return img.hashOffset == static_cast<size_t>(-1); }).begin();
    imgs.erase(end, imgs.end());

    if (m_progressTracker) m_progressTracker->forceUpdate();

    return imgs;
}

std::vector<pHash> CudaPhash::computeHashes(const std::vector<std::string>& imagePaths) {
    if (imagePaths.empty()) { return {}; }

    std::vector<Image> imgs = runPipeline(imagePaths);

    DEBUG(__func__, "Copying and bitpacking perceptual hashes");

    std::vector<pHash> h_hashes(imgs.size());
    CUDA_CHECK(cudaMemcpy(h_hashes.data(), d_hashes, imgs.size() * sizeof(pHash), cudaMemcpyDeviceToHost));

    INFO(__func__, "Flatten start");
    for (size_t i = 0; i < imgs.size(); ++i) {
        imgs[i].hash = h_hashes[imgs[i].hashOffset];
    }
    INFO(__func__, "Flatten Stop");

    std::vector<pHash> result;
    result.reserve(imgs.size());
    for (const auto& img : imgs) {
        result.push_back(img.hash);
    }

    return result;
}

std::vector<Image> CudaPhash::findDuplicatesGPU(const std::vector<std::string>& imagePaths, int threshold, int numTables, int bitsPerTable) {
    if (imagePaths.empty()) { return {}; }

    std::vector<Image> imgs = runPipeline(imagePaths);
    size_t n = imgs.size();
    if (n == 0) { return imgs; }

    DEBUG(__func__, "Generated ", imgs.size(), " hashes. Finding similar images...");

    std::vector<size_t> h_offsets(n);
    for (size_t i = 0; i < n; i++) { h_offsets[i] = imgs[i].hashOffset; }

    size_t* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, n * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));

    std::mt19937 rng(DEFAULT_RNG_SEED);
    std::vector<int> h_bitPositions(numTables * bitsPerTable);
    for (int t = 0; t < numTables; t++) {
        for (int b = 0; b < bitsPerTable; b++) { h_bitPositions[t * bitsPerTable + b] = rng() % 128; }
    }

    int* d_bitPositions = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bitPositions, h_bitPositions.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_bitPositions, h_bitPositions.data(), h_bitPositions.size() * sizeof(int), cudaMemcpyHostToDevice));

    uint64_t* d_keys = nullptr;
    int* d_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_idx, n * sizeof(int)));

    const size_t MAX_EDGES = 2ull * n * MAX_EDGES_FACTOR;
    Edge* d_allEdges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_allEdges, MAX_EDGES * sizeof(Edge)));

    int* d_edgeCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edgeCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edgeCount, 0, sizeof(int)));

    DEBUG(__func__, "Finished allocations.");

    int maxPairsPerBucket = std::max(10000, static_cast<int>(n / 10));

    {
        const int blockSize = 256;
        const int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);

        for (int t = 0; t < numTables; t++) {
            computeLSHKeysKernel << <gridSize, blockSize >> > (d_offsets, d_hashes, static_cast<int>(n), d_bitPositions, bitsPerTable, t, d_keys, d_idx);
            cudaDeviceSynchronize();

            thrust::sort_by_key(thrust::device, d_keys, d_keys + n, d_idx);

            int pairGridSize = std::min(65535, static_cast<int>((n + blockSize - 1) / blockSize));
            findPairsKernel << <pairGridSize, blockSize >> > (d_keys, d_idx, static_cast<int>(n), d_offsets, d_hashes, threshold, d_allEdges, d_edgeCount, static_cast<int>(MAX_EDGES), maxPairsPerBucket);
            cudaDeviceSynchronize();
        }
    }

    DEBUG(__func__, "Finished kernel calls.");

    int h_edgeCount;
    CUDA_CHECK(cudaMemcpy(&h_edgeCount, d_edgeCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_edgeCount < 1) {
        cudaFree(d_offsets);
        cudaFree(d_bitPositions);
        cudaFree(d_keys);
        cudaFree(d_idx);
        cudaFree(d_allEdges);
        cudaFree(d_edgeCount);
        return {};
    }

    std::vector<Edge> edges(h_edgeCount);
    CUDA_CHECK(cudaMemcpy(edges.data(), d_allEdges, h_edgeCount * sizeof(Edge), cudaMemcpyDeviceToHost));

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    DEBUG(__func__, "Found ", edges.size(), " edges.");

    std::vector<std::pair<int, int>> edgePairs;
    edgePairs.reserve(edges.size());
    for (const auto& e : edges) { edgePairs.emplace_back(e.i, e.j); }

    DEBUG(__func__, "GPU-based LSH pass produced ", edgePairs.size(), " edges.");

    std::vector<int> toDeleteIdx = coverByHighestDegree(edgePairs, static_cast<int>(n));

    DEBUG(__func__, "Covering by highest degree, ", toDeleteIdx.size(), " images to delete.");

    std::vector<Image> toDelete;
    toDelete.reserve(toDeleteIdx.size());
    std::vector<bool> markDel(n, false);
    for (auto idx : toDeleteIdx) { markDel[idx] = true; }

    std::vector<int> bestDist(n, threshold + 1);
    std::vector<int> bestNeighborIdx(n, -1);

    for (const auto& e : edges) {
        if (markDel[e.i]) {
            if (e.dist < bestDist[e.i]) {
                bestDist[e.i] = e.dist;
                bestNeighborIdx[e.i] = e.j;
            }
        }
        if (markDel[e.j]) {
            if (e.dist < bestDist[e.j]) {
                bestDist[e.j] = e.dist;
                bestNeighborIdx[e.j] = e.i;
            }
        }
    }

    for (int i = 0; i < static_cast<int>(n); i++) {
        if (markDel[i] && bestNeighborIdx[i] != -1) { imgs[i].mostSimilarImage = imgs[bestNeighborIdx[i]].path; }
    }

    DEBUG(__func__, "Building toDelete vector.");

    for (int i = 0; i < static_cast<int>(n); i++) {
        if (markDel[i]) {
            toDelete.push_back(std::move(imgs[i]));
        }
    }

    cudaFree(d_offsets);
    cudaFree(d_bitPositions);
    cudaFree(d_keys);
    cudaFree(d_idx);
    cudaFree(d_allEdges);
    cudaFree(d_edgeCount);

    DEBUG(__func__, "Returning ", toDelete.size(), " images for deletion.");
    return toDelete;
}
