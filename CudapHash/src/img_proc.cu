//#include <cuda_runtime_api.h>
//#include <nvjpeg.h>
//
//#include <thread>
//#include <mutex>
//#include <condition_variable>
//#include <queue>
//#include <vector>
//#include <string>
//#include <iostream>
//#include <fstream>
//#include <filesystem>
//#include <algorithm>
//#include <execution>
//#include <chrono> 
//
//#include "memory_manager.cuh"
//#include "img_proc.cuh"
//#include "../logger.hpp"
//
///**
// * @brief Clamp a value between a min and max value.
// */
//__device__ int clamp(int x, int min, int max) {
//    return x < min ? min : (x > max ? max : x);
//}
//
///**
// * @brief Cubic interpolation function for bicubic resampling.
// *
// * @param x The distance from the target position.
// * @param a Free parameter, typically -0.5 for balance between smoothness and sharpness.
// * @return The weight given by the cubic function.
// */
//__device__ float cubicWeight(float x, float a = -0.5f) {
//    x = fabsf(x);
//    if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
//    else if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
//    else return 0.0f;
//}
//
///**
// * @brief CUDA kernel to resize batch of images using bicubic interpolation.
// *
// * @param d_inputs   Array of device pointers, one per image (batch).
// * @param inWidths   Array of input image widths.
// * @param inHeights  Array of input image heights.
// * @param d_outputs  Array of device pointers for output float images, one per batch.
// * @param outWidth   Common output width.
// * @param outHeight  Common output height.
// */
//__global__ void batchBicubicResizeKernel(const unsigned char* const* d_inputs,
//    const int* inWidths,
//    const int* inHeights,
//    float* const* d_outputs,
//    int outWidth,
//    int outHeight)
//{
//    // Pixel coordinates in output image
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x >= outWidth || y >= outHeight) return;
//
//    // Pointers & dimensions for image
//    const unsigned char* d_in = d_inputs[blockIdx.z];
//    float* d_out = d_outputs[blockIdx.z];
//    int inWidth = inWidths[blockIdx.z];
//    int inHeight = inHeights[blockIdx.z];
//
//    // Normalized coords in source image
//    float u = (x + 0.5f) / outWidth * inWidth;
//    float v = (y + 0.5f) / outHeight * inHeight;
//
//    // Top-left pixel of 4x4 neighborhood
//    int x0 = floorf(u) - 1;
//    int y0 = floorf(v) - 1;
//
//    // Accumulate weighted sum
//    float sum = 0.0f;
//    float totalWeight = 0.0f;
//
//    for (int j = 0; j < 4; j++) {
//        for (int i = 0; i < 4; i++) {
//            int xi = clamp(x0 + i, 0, inWidth - 1);
//            int yj = clamp(y0 + j, 0, inHeight - 1);
//
//            float weight = cubicWeight(u - xi) * cubicWeight(v - yj);
//
//            // Add the weighted pixel value
//            sum += weight * d_in[yj * inWidth + xi];
//            totalWeight += weight;
//        }
//    }
//
//    // Normalize result
//    if (totalWeight > 0.0f) sum /= totalWeight;
//
//    d_out[y * outWidth + x] = sum;
//}
//
//
///**
// * @brief Host function to launch the batched bicubic resize kernel for all images in a batch.
// *
// * @param d_inputVec  Vector of device pointers to each decoded (grayscale) input image.
// * @param widths      Vector of input widths.
// * @param heights     Vector of input heights.
// * @param d_outputVec Vector of device pointers to each output image.
// * @param outWidth    (common) desired output width.
// * @param outHeight   (common) desired output height.
// * @param resources   The persistent CUDA resources to use.
// */
//void bicubicResize(const std::vector<unsigned char*>& d_inputVec,
//    const std::vector<int>& widths,
//    const std::vector<int>& heights,
//    const std::vector<float*>& d_outputVec,
//    int outWidth,
//    int outHeight,
//    CudaResources& resources)
//{
//    size_t batchSize = d_inputVec.size();
//    if (batchSize == 0) return;
//
//    resources.ensureCapacity(batchSize);
//
//    // Copy host vectors to device arrays
//    cudaMemcpyAsync(resources.d_inputsArray, d_inputVec.data(), batchSize * sizeof(unsigned char*),
//        cudaMemcpyHostToDevice, resources.stream);
//    cudaMemcpyAsync(resources.d_outputsArray, d_outputVec.data(), batchSize * sizeof(float*),
//        cudaMemcpyHostToDevice, resources.stream);
//    cudaMemcpyAsync(resources.d_inWidths, widths.data(), batchSize * sizeof(int),
//        cudaMemcpyHostToDevice, resources.stream);
//    cudaMemcpyAsync(resources.d_inHeights, heights.data(), batchSize * sizeof(int),
//        cudaMemcpyHostToDevice, resources.stream);
//
//    // 3D grid: 2D for output resolution, 1D in z for batch dimension
//    dim3 block(16, 16, 1);
//    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y, batchSize);
//
//    batchBicubicResizeKernel << <grid, block, 0, resources.stream >> > (
//        resources.d_inputsArray,
//        resources.d_inWidths,
//        resources.d_inHeights,
//        resources.d_outputsArray,
//        outWidth,
//        outHeight
//        );
//}
//
//
//// ------------------------------
//// Thread-Safe Queue
//// ------------------------------
//template<typename T>
//class ThreadSafeQueue {
//private:
//    std::queue<T> m_queue;
//    std::mutex m_mutex;
//    std::condition_variable m_cond;
//    bool sentinel = false;
//public:
//    void push(const T& item) {
//        {
//            std::unique_lock<std::mutex> lock(m_mutex);
//            m_queue.push(item);
//        }
//        m_cond.notify_one();
//    }
//
//    T pop() {
//        std::unique_lock<std::mutex> lock(m_mutex);
//        m_cond.wait(lock, [this] { return !m_queue.empty(); });
//        T val = m_queue.front();
//        m_queue.pop();
//        return val;
//    }
//
//    std::vector<T> popMax(size_t maxItems) {
//        std::vector<T> items;
//        {
//            std::unique_lock<std::mutex> lock(m_mutex);
//            while (!m_queue.empty() && items.size() < maxItems) {
//                items.push_back(m_queue.front());
//                m_queue.pop();
//            }
//        }
//        return items;
//    }
//
//    bool empty() {
//        std::unique_lock<std::mutex> lock(m_mutex);
//        return m_queue.empty();
//    }
//
//    bool isSentinel() {
//        std::unique_lock<std::mutex> lock(m_mutex);
//        return sentinel;
//    }
//
//    void setSentinel() {
//        std::unique_lock<std::mutex> lock(m_mutex);
//        sentinel = true;
//    }
//
//    size_t size() {
//        std::unique_lock<std::mutex> lock(m_mutex);
//        return m_queue.size();
//    }
//};
//
//
//// ------------------------------
//// Global Pipeline Parameters
//// ------------------------------
//static const int BATCH_DECODE_SIZE = 128;
//static const int BATCH_RESIZE_SIZE = 128;
//
//static const int NUM_READER_THREADS = 4;
//static const int NUM_DECODE_THREADS = 1;
//static const int NUM_RESIZE_THREADS = 1;
//
//
//// -------------------------------------------------------
//// readImages Stage (From inputQueue to decodeQueue)
//// -------------------------------------------------------
//void readImages(MemoryManager* memMgr,
//    nvjpegHandle_t nvjHandle,
//    ThreadSafeQueue<Image*>* inputQueue,
//    ThreadSafeQueue<Image*>* decodeQueue,
//    int threadId)
//{
//    std::string threadName = "[Reader " + std::to_string(threadId) + "] ";
//    while (true) {
//        // No more images to read, exit
//        if (inputQueue->empty()) return;
//
//        Image* img = inputQueue->pop();
//
//        std::ifstream file(img->path, std::ios::binary | std::ios::ate);
//        if (!file.is_open()) {
//            logger::error(threadName, "Could not open file: ", img->path);
//            continue; // Skip adding to queue
//        }
//
//        // Get file size from current position
//        std::streamsize file_size = file.tellg();
//        if (file_size <= 0) {
//            logger::debug(threadName, "Empty or invalid file: ", img->path);
//            file.close();
//            continue;
//        }
//        img->file_size = static_cast<int>(file_size);
//
//        // Allocate pinned memory for the raw file
//        unsigned char* pinnedMem = memMgr->Allocate<unsigned char>(MemoryManager::RAW_FILE_POOL, img->file_size);
//        img->rawFileData = pinnedMem;
//
//        // Read file data from disk (seek back to beginning)
//        file.seekg(0, std::ios::beg);
//        file.read(reinterpret_cast<char*>(pinnedMem), img->file_size);
//        file.close();
//
//        // Use nvJPEG to get image info (width, height)
//        int nComponents = 0;
//        nvjpegChromaSubsampling_t subsampling;
//        int widths[NVJPEG_MAX_COMPONENT];
//        int heights[NVJPEG_MAX_COMPONENT];
//
//        nvjpegStatus_t infoRet = nvjpegGetImageInfo(
//            nvjHandle, pinnedMem, img->file_size,
//            &nComponents, &subsampling,
//            widths, heights
//        );
//
//        if (infoRet != NVJPEG_STATUS_SUCCESS) {
//            logger::error(threadName, "nvjpegGetImageInfo failed for ", img->path);
//            memMgr->Free<unsigned char>(MemoryManager::RAW_FILE_POOL, pinnedMem);
//            img->rawFileData = nullptr;
//            continue;
//        }
//
//        img->originalWidth = *std::max_element(widths, widths + nComponents);
//        img->originalHeight = *std::max_element(heights, heights + nComponents);
//
//        // Push image pointer to decode queue
//        decodeQueue->push(img);
//
//        logger::debug(threadName, "Read+Info for: ", img->path);
//    }
//}
//
//
//// -------------------------------------------------------
//// decodeImages Stage (From decodeQueue to resizeQueue)
//// -------------------------------------------------------
//void decodeImages(MemoryManager* memMgr,
//    nvjpegHandle_t nvjHandle,
//    ThreadSafeQueue<Image*>* decodeQueue,
//    ThreadSafeQueue<Image*>* resizeQueue,
//    int threadId)
//{
//    std::string threadName = "[Decoder " + std::to_string(threadId) + "] ";
//
//    // Create persistent CUDA resources
//    cudaStream_t stream;
//    cudaEvent_t event;
//    cudaStreamCreate(&stream);
//    cudaEventCreate(&event);
//
//    nvjpegJpegState_t nvjDecoderState;
//    nvjpegStatus_t status = nvjpegJpegStateCreate(nvjHandle, &nvjDecoderState);
//    if (status != NVJPEG_STATUS_SUCCESS) {
//        logger::error(threadName, "nvjpegJpegStateCreate failed with status: ", status);
//        throw std::runtime_error("nvJPEG state creation failed");
//    }
//    int initializeSize = 0;
//
//    std::vector<Image*> batch;
//
//    // Sleep for a bit to allow reader threads to start
//    std::this_thread::sleep_for(std::chrono::milliseconds(5));
//
//    while (true) {
//        batch.clear();
//
//        // Collect images for the batch
//        std::vector<Image*> imgs = decodeQueue->popMax(BATCH_DECODE_SIZE);
//        for (auto& img : imgs) batch.push_back(img);
//
//        // If we have no more images to process and all readers are done
//        if (decodeQueue->isSentinel() && batch.empty() && decodeQueue->empty()) {
//            logger::info(threadName, "Sentinels received and no images left => exiting.");
//            return;
//        }
//
//        if (batch.empty()) continue;
//
//        logger::info(threadName, "Decoding ", batch.size(), " images, queue size: ", decodeQueue->size());
//
//        // Check if we need to reinitialize
//        if (batch.size() != initializeSize) {
//            logger::info(threadName, "Reinitializing nvJPEG for batch of ", batch.size(), " images");
//
//            status = nvjpegDecodeBatchedInitialize(nvjHandle, nvjDecoderState, batch.size(), 8, NVJPEG_OUTPUT_Y);
//            if (status != NVJPEG_STATUS_SUCCESS) {
//                logger::error(threadName, "nvjpegDecodeBatchedInitialize failed with status: ", status);
//                // Continue anyway - don't throw here
//            }
//            else {
//                initializeSize = batch.size();
//                logger::info(threadName, "Reinitialized successfully");
//            }
//        }
//
//        // Build array of raw pointers for batch decode
//        std::vector<const unsigned char*> rawPointers(batch.size());
//        std::vector<size_t> rawLengths(batch.size());
//        std::vector<nvjpegImage_t> out_images(batch.size());
//
//        for (size_t i = 0; i < batch.size(); ++i) {
//            Image* img = batch[i];
//            rawPointers[i] = img->rawFileData;
//            rawLengths[i] = img->file_size;
//
//            size_t decodeSize = (size_t)img->originalWidth * (size_t)img->originalHeight; // 1 channel
//            unsigned char* dMem = memMgr->Allocate<unsigned char>(MemoryManager::DECODED_IMAGE_POOL, decodeSize);
//            img->DecodedData = dMem;
//
//            // Setup nvjpegImage_t for a single channel
//            memset(&out_images[i], 0, sizeof(nvjpegImage_t));
//            out_images[i].channel[0] = dMem; // grayscale
//            out_images[i].pitch[0] = img->originalWidth;
//        }
//
//        nvjpegStatus_t decRet = nvjpegDecodeBatched(
//            nvjHandle,
//            nvjDecoderState,
//            rawPointers.data(),
//            rawLengths.data(),
//            out_images.data(),
//            stream
//        );
//
//        // Error handling
//        std::vector<Image*> failedImages;
//        if (decRet != NVJPEG_STATUS_SUCCESS) {
//            logger::error(threadName, "nvjpegDecodeBatched failed with code: ", decRet);
//
//            for (auto& img : batch) {
//                if (img->DecodedData) {
//                    memMgr->Free(MemoryManager::DECODED_IMAGE_POOL, img->DecodedData);
//                    img->DecodedData = nullptr;
//                }
//                failedImages.push_back(img);
//            }
//            batch.clear();
//        }
//
//        // Record event and synchronize
//        cudaEventRecord(event, stream);
//        cudaEventSynchronize(event);
//
//        // Free raw file data
//        for (auto& img : batch) {
//            memMgr->Free(MemoryManager::RAW_FILE_POOL, img->rawFileData);
//            img->rawFileData = nullptr;
//        }
//
//        logger::info(threadName, "Decoded batch of ", batch.size(), " images.");
//
//        // Push images to resize queue
//        for (auto& img : batch) resizeQueue->push(img);
//
//        // Handle failed images
//        for (auto& img : failedImages) resizeQueue->push(img);
//
//        // Check if this was the final batch
//        if (decodeQueue->isSentinel() && decodeQueue->empty()) {
//            logger::info(threadName, "Processed final batch, exiting.");
//            return;
//        }
//    }
//
//    // sync and destroy
//    cudaStreamSynchronize(stream);
//    cudaStreamDestroy(stream);
//    cudaEventDestroy(event);
//    nvjpegJpegStateDestroy(nvjDecoderState);
//    nvjpegDestroy(nvjHandle);
//}
//
//
//// -------------------------------------------------------
//// resizeImages Stage (From resizeQueue to final result)
//// -------------------------------------------------------
//void resizeImages(MemoryManager* memMgr,
//    ThreadSafeQueue<Image*>* resizeQueue,
//    int targetWidth,
//    int targetHeight,
//    int threadId)
//{
//    std::string threadName = "[Resizer " + std::to_string(threadId) + "] ";
//
//    // Create persistent CUDA resources
//    CudaResources resources(BATCH_RESIZE_SIZE);
//
//    std::vector<Image*> batch;
//    batch.reserve(BATCH_RESIZE_SIZE);
//
//    std::this_thread::sleep_for(std::chrono::milliseconds(10));
//
//    while (true) {
//        batch.clear();
//        std::vector<Image*> imgs = resizeQueue->popMax(BATCH_RESIZE_SIZE);
//
//        for (auto& img : imgs) batch.push_back(img);
//
//        // If we have no more images to process and all decoders are done
//        if (resizeQueue->isSentinel() && batch.empty() && resizeQueue->empty()) {
//            logger::debug(threadName, "Sentinel received and no images left = > exiting.");
//            return;
//        }
//
//        if (batch.empty()) {
//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
//            continue;
//        }
//
//        logger::debug(threadName, "Starting resize of", batch.size(), " images, queue size: ", resizeQueue->size());
//
//        std::vector<unsigned char*> d_inputVec(batch.size());
//        std::vector<float*> d_outputVec(batch.size());
//        std::vector<int> widths(batch.size());
//        std::vector<int> heights(batch.size());
//
//        for (size_t i = 0; i < batch.size(); ++i) {
//            Image* img = batch[i];
//            d_inputVec[i] = img->DecodedData;
//            widths[i] = img->originalWidth;
//            heights[i] = img->originalHeight;
//
//            float* resizedMem = memMgr->Allocate<float>(MemoryManager::RESIZED_IMAGE_POOL);
//            img->resizedData = resizedMem;
//            d_outputVec[i] = resizedMem;
//        }
//
//        // Perform bicubic resize using persistent resources
//        bicubicResize(d_inputVec,
//            widths,
//            heights,
//            d_outputVec,
//            targetWidth,
//            targetHeight,
//            resources);
//
//        // Record and synchronize
//        cudaEventRecord(resources.event, resources.stream);
//        cudaEventSynchronize(resources.event);
//
//        // Mark final image shape and free decoded data
//        for (auto& img : batch) {
//            img->resizedWidth = targetWidth;
//            img->resizedHeight = targetHeight;
//
//            memMgr->Free(MemoryManager::DECODED_IMAGE_POOL, img->DecodedData);
//            img->DecodedData = nullptr;
//        }
//
//        logger::info(threadName, "Resized batch of ", batch.size(), " images.");
//    }
//}
//
//
//// -------------------------------------------------------
//// Helper: Suggest Memory Pool Sizes
//// -------------------------------------------------------
//void suggestPoolSizes(size_t& rawPoolSize,
//    size_t& decodedPoolSize,
//    size_t& resizedPoolSize,
//    size_t resizedBlockSize,
//    size_t numImages)
//{
//    // Query current free memory
//    size_t freeMem, totalMem;
//    cudaMemGetInfo(&freeMem, &totalMem);
//
//    // Reserve 20% VRAM headroom to avoid over-allocation
//    size_t usableMem = static_cast<size_t>(freeMem * 0.8);
//
//    // Estimate JPEG compression ratio (~10:1)
//    constexpr double COMPRESSION_RATIO = 10.0;
//
//    // Estimate total memory needed in each stage
//    size_t estimatedRawSize = (numImages * resizedBlockSize) / COMPRESSION_RATIO;   // Raw JPEGs
//    size_t estimatedDecodedSize = numImages * resizedBlockSize * COMPRESSION_RATIO; // Decoded images
//    size_t estimatedResizedSize = numImages * resizedBlockSize * sizeof(float);     // Resized float images
//
//    // Assign weights to each stage
//    constexpr double RAW_WEIGHT = 0.20;      // Raw pool should be small
//    constexpr double DECODED_WEIGHT = 0.80;  // Decoded images are large
//
//    // Compute ideal allocation using weighted distribution
//    size_t weightedRawPool = static_cast<size_t>(usableMem * RAW_WEIGHT);
//    size_t weightedDecodedPool = static_cast<size_t>(usableMem * DECODED_WEIGHT);
//
//    // Ensure each pool has at least its estimated requirement, if possible
//    rawPoolSize = std::min(weightedRawPool, estimatedRawSize);
//    decodedPoolSize = std::min(weightedDecodedPool, estimatedDecodedSize);
//    resizedPoolSize = estimatedResizedSize;  // This must always fit all images
//
//    // If there's excess VRAM available, distribute it dynamically:
//    size_t remainingMem = usableMem - (rawPoolSize + decodedPoolSize + resizedPoolSize);
//
//    if (remainingMem > 0) {
//        decodedPoolSize += static_cast<size_t>(remainingMem * 0.75);
//        rawPoolSize += static_cast<size_t>(remainingMem * 0.25);
//    }
//}
//
//
//// -------------------------------------------------------
//// process_images Main Entry
//// -------------------------------------------------------
//std::vector<Image> process_images(const std::vector<std::string>& imgPaths, int targetSize, bool verbose)
//{
//    // Initialize the logger with verbosity setting
//    logger::set_verbose(verbose);
//
//    using clock = std::chrono::high_resolution_clock;
//    auto globalStart = clock::now();
//
//    // Define target dimensions based on the provided size
//    const int TARGET_WIDTH = targetSize;
//    const int TARGET_HEIGHT = targetSize;
//
//    // 1. Create and initialize an Image struct for each path
//    size_t totalFiles = imgPaths.size();
//    std::vector<Image> images(totalFiles);
//
//    // Get file sizes for each image
//    std::for_each(std::execution::par, images.begin(), images.end(),
//        [&](Image& img) {
//            img.path = imgPaths[&img - &images[0]];
//
//            std::error_code ec;
//            auto fileSize = std::filesystem::file_size(img.path, ec);
//            if (ec) {
//                logger::error("Failed getting file size for ", img.path, ": ", ec.message());
//                img.file_size = 0;
//            }
//            else img.file_size = static_cast<int>(fileSize);
//        });
//
//    // 2. Suggest memory pool sizes
//    size_t rawPoolSize, decodedPoolSize, resizedPoolSize;
//    size_t resizedBlockSize = (size_t)TARGET_WIDTH * TARGET_HEIGHT * sizeof(float); // grayscale float
//    suggestPoolSizes(rawPoolSize, decodedPoolSize, resizedPoolSize, resizedBlockSize, totalFiles);
//
//    // Create MemoryManager
//    std::shared_ptr<MemoryManager> memMgr = std::make_shared<MemoryManager>(rawPoolSize, decodedPoolSize, resizedPoolSize, resizedBlockSize);
//
//    for (auto& img : images) img.memMgr = memMgr;
//
//    // 3. Initialize nvJPEG
//    nvjpegHandle_t nvjHandle;
//
//    memMgr->initNvJpegPools(
//        1024 * 1024 * 850,  // 850 MB for nvJPEG device memory
//        1024 * 1024 * 300   // 300 MB for nvJPEG pinned memory
//    );
//
//    nvjpegStatus_t status = nvjpegCreateExV2(
//        NVJPEG_BACKEND_GPU_HYBRID,
//        memMgr->getNvJpegDevAllocator(),
//        memMgr->getNvJpegPinnedAllocator(),
//        0, // flags
//        &nvjHandle
//    );
//
//    if (status != NVJPEG_STATUS_SUCCESS) {
//        logger::error("[Main] nvjpegCreateExV2 failed with status: ", status);
//        throw std::runtime_error("nvJPEG initialization failed");
//    }
//
//    // 4. Create Stage Queues
//    ThreadSafeQueue<Image*> inputQueue;   // images not yet read
//    ThreadSafeQueue<Image*> decodeQueue;  // images ready for decoding
//    ThreadSafeQueue<Image*> resizeQueue;  // images ready for resizing
//
//    // Push all valid images to input queue
//    for (auto& img : images) {
//        // Skip images with invalid file size
//        if (img.file_size > 0) inputQueue.push(&img);
//        else logger::warn("Skipping ", img.path, " due to file size error");
//    }
//
//    // 5. Create Reader Threads
//    std::vector<std::thread> readers;
//    readers.reserve(NUM_READER_THREADS);
//    for (int i = 0; i < NUM_READER_THREADS; ++i) {
//        readers.emplace_back(readImages,
//            memMgr.get(),
//            nvjHandle,
//            &inputQueue,
//            &decodeQueue,
//            i);
//    }
//
//    // 6. Create Decoder Threads
//    std::vector<std::thread> decoders;
//    decoders.reserve(NUM_DECODE_THREADS);
//    for (int i = 0; i < NUM_DECODE_THREADS; ++i) {
//        decoders.emplace_back(decodeImages,
//            memMgr.get(),
//            nvjHandle,
//            &decodeQueue,
//            &resizeQueue,
//            i);
//    }
//
//    // 7. Create Resizer Threads
//    std::vector<std::thread> resizers;
//    resizers.reserve(NUM_RESIZE_THREADS);
//    for (int i = 0; i < NUM_RESIZE_THREADS; ++i) {
//        resizers.emplace_back(resizeImages,
//            memMgr.get(),
//            &resizeQueue,
//            TARGET_WIDTH,
//            TARGET_HEIGHT,
//            i);
//    }
//
//    // 8. Join all threads
//    for (auto& t : readers) { if (t.joinable())  t.join(); }
//
//    // Readers are done, set decode queue sentinel
//    decodeQueue.setSentinel();
//
//    for (auto& t : decoders) { if (t.joinable())  t.join(); }
//
//    // Decoders are done, set resize queue sentinel
//    resizeQueue.setSentinel();
//
//    for (auto& t : resizers) { if (t.joinable())  t.join(); }
//
//    // 9. Destroy nvJPEG objects
//    nvjpegDestroy(nvjHandle);
//
//    logger::info("[INFO] Total pipeline time: ", std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - globalStart).count(), "ms");
//
//    // 10. Return final vector
//    // Each Image now has { path, resizedWidth=targetSize, resizedHeight=targetSize, resizedData=(GPU pointer) }
//    return images;
//}
