#include "phash_cuda.cuh"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <opencv2/opencv.hpp>
#include <future>
#include <cmath>

#ifndef PI
#define PI 3.14159265358979323846f
#endif

/**
 * @brief Safely computes the next power of two for x, up to max of 1024.
 * @param x Input value.
 * @return Next power of two or 1024 if x exceeds 1024.
 */
static int nextPow2OrMax1024(int x)
{
    if (x > 1024)
        return 1024;
    int r = 1;
    while (r < x && r < 1024)
        r <<= 1;
    return r;
}

/**
 * @brief CUDA kernel that copies 8-bit host data to GPU float data.
 * @param d_in    8-bit source data (uint2-based).
 * @param d_out   Destination float array.
 * @param width   Image width.
 * @param height  Image height.
 */
__global__ void host8uToFloatKernel(const uint2 *__restrict__ d_in, float *__restrict__ d_out, int width, int height)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPix = width * height;
    int startPix = threadId * 8;
    if (startPix >= totalPix)
        return;

    uint2 data = d_in[threadId];
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&data);
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int pixIdx = startPix + i;
        if (pixIdx < totalPix)
            d_out[pixIdx] = static_cast<float>(bytes[i]);
    }
}

/**
 * @brief CUDA kernel that computes mean thresholds and bit-packs results into 32-bit words.
 * @param d_imgs      Array of pointers to float image data.
 * @param pitch       Pitch in float elements.
 * @param d_outHashes Device buffer for bit-packed hashes (32 bits per 32 pixels).
 * @param batchSize   Number of images in batch.
 * @param width       Image width.
 * @param height      Image height.
 */
__global__ void meanBitpackKernel(const float *const *__restrict__ d_imgs, size_t pitch, unsigned int *__restrict__ d_outHashes, int batchSize, int width, int height)
{
    extern __shared__ float sdata[]; // for partial sums
    int imgIdx = blockIdx.x;
    if (imgIdx >= batchSize)
        return;

    int totalPixels = width * height;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (tid < totalPixels)
    {
        int row = tid / width;
        int col = tid % width;
        val = d_imgs[imgIdx][row * pitch + col];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // sdata[0] holds the sum of all pixels.
    float threshold = sdata[0] / static_cast<float>(totalPixels);
    __syncthreads();

    // Compare each pixel to the threshold and build a ballot
    bool bitSet = (val > threshold);
    unsigned int mask = __ballot_sync(0xFFFFFFFF, bitSet);

    // Each warp writes a 32-bit word. warpId = tid / 32.
    int warpId = tid >> 5;
    int wordsPerImage = (totalPixels + 31) / 32;

    if ((tid & 31) == 0 && warpId < wordsPerImage)
        d_outHashes[imgIdx * wordsPerImage + warpId] = mask;
}

/**
 * @brief Kernel to build the DCT matrix T (N x N) in global memory.
 * @param T             Destination DCT matrix in device memory.
 * @param invSqrtN      1 / sqrt(N).
 * @param sqrtTwoDivN   sqrt(2 / N).
 */
__global__ void dctMatKernel(float *T, float invSqrtN, float sqrtTwoDivN)
{
    int i = blockIdx.x, j = threadIdx.x, N = blockDim.x;
    float normFactor = (i == 0) ? invSqrtN : sqrtTwoDivN;
    T[i * N + j] = normFactor * cosf(((2.0f * j + 1.0f) / (2.0f * N)) * (float)i * PI);
}

CudaPhash::CudaPhash(int hashSize, int highFreqFactor, int batchSize)
    : m_hashSize(hashSize), m_highFreqFactor(highFreqFactor), m_imgSize(hashSize * highFreqFactor), m_batchSize(batchSize), m_handle(nullptr), d_T(nullptr), d_TT(nullptr), d_tmp_batch(nullptr), d_tmp_batch_capacity(0), d_A_array(nullptr), d_tmp_array(nullptr), d_Aout_array(nullptr), d_T_array(nullptr), d_TT_array(nullptr), d_array_capacity(0)
{
    cudaStreamCreate(&m_stream);
    cudaMalloc(&d_T, m_imgSize * m_imgSize * sizeof(float));
    cudaMalloc(&d_TT, m_imgSize * m_imgSize * sizeof(float));

    dctMatKernel<<<m_imgSize, m_imgSize>>>(d_T,
                                           std::sqrt(1.0f / static_cast<float>(m_imgSize)),
                                           std::sqrt(2.0f / static_cast<float>(m_imgSize)));

    cublasCreate(&m_handle);
    cublasSetStream(m_handle, m_stream);
    cublasSetMathMode(m_handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgeam(m_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m_imgSize, m_imgSize,
                &alpha, d_T, m_imgSize,
                &beta, d_T, m_imgSize,
                d_TT, m_imgSize);
}

CudaPhash::~CudaPhash()
{
    if (m_handle)
        cublasDestroy(m_handle);
    cudaFree(d_T);
    cudaFree(d_TT);

    if (d_tmp_batch)
        cudaFree(d_tmp_batch);
    if (d_A_array)
        cudaFree(d_A_array);
    if (d_tmp_array)
        cudaFree(d_tmp_array);
    if (d_Aout_array)
        cudaFree(d_Aout_array);
    if (d_T_array)
        cudaFree(d_T_array);
    if (d_TT_array)
        cudaFree(d_TT_array);

    cudaStreamDestroy(m_stream);
}

void CudaPhash::setHandle(const cublasHandle_t &handle)
{
    m_handle = handle;
}

/**
 * @brief Performs batched DCT on a list of GPU images (all must have same pitch).
 * @param gpuImages Images to process via batched DCT.
 */
void CudaPhash::batchDct(std::vector<GpuImage> &gpuImages)
{
    size_t batchSize = gpuImages.size();
    if (batchSize == 0)
        return;
    int N = m_imgSize;
    size_t pitchInFloats = gpuImages[0].pitch / sizeof(float);

    size_t neededElems = batchSize * static_cast<size_t>(N) * N;
    if (neededElems > d_tmp_batch_capacity)
    {
        if (d_tmp_batch)
            cudaFree(d_tmp_batch);
        cudaMalloc(&d_tmp_batch, neededElems * sizeof(float));
        d_tmp_batch_capacity = neededElems;
    }

    if (batchSize > d_array_capacity)
    {
        if (d_A_array)
            cudaFree(d_A_array);
        if (d_tmp_array)
            cudaFree(d_tmp_array);
        if (d_Aout_array)
            cudaFree(d_Aout_array);
        if (d_T_array)
            cudaFree(d_T_array);
        if (d_TT_array)
            cudaFree(d_TT_array);
        cudaMalloc(&d_A_array, batchSize * sizeof(float *));
        cudaMalloc(&d_tmp_array, batchSize * sizeof(float *));
        cudaMalloc(&d_Aout_array, batchSize * sizeof(float *));
        cudaMalloc(&d_T_array, batchSize * sizeof(const float *));
        cudaMalloc(&d_TT_array, batchSize * sizeof(const float *));
        d_array_capacity = batchSize;
    }

    std::vector<const float *> h_A_array(batchSize);
    std::vector<float *> h_tmp_array(batchSize);
    std::vector<float *> h_Aout_array(batchSize);
    std::vector<const float *> h_T_array(batchSize, d_T);
    std::vector<const float *> h_TT_array(batchSize, d_TT);

    for (size_t i = 0; i < batchSize; ++i)
    {
        h_A_array[i] = gpuImages[i].data;
        h_Aout_array[i] = gpuImages[i].data;
        h_tmp_array[i] = d_tmp_batch + i * static_cast<size_t>(N) * N;
    }

    cudaMemcpyAsync(d_A_array, h_A_array.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice, m_stream);
    cudaMemcpyAsync(d_tmp_array, h_tmp_array.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice, m_stream);
    cudaMemcpyAsync(d_Aout_array, h_Aout_array.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice, m_stream);
    cudaMemcpyAsync(d_T_array, h_T_array.data(), batchSize * sizeof(const float *), cudaMemcpyHostToDevice, m_stream);
    cudaMemcpyAsync(d_TT_array, h_TT_array.data(), batchSize * sizeof(const float *), cudaMemcpyHostToDevice, m_stream);

    const float alpha = 1.0f, beta = 0.0f;

    // D = T * A
    cublasSgemmBatched(m_handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N, N, N,
                       &alpha,
                       d_T_array, N,
                       d_A_array, static_cast<int>(pitchInFloats),
                       &beta,
                       d_tmp_array, N,
                       static_cast<int>(batchSize));

    // D = (T * A) * T^T
    cublasSgemmBatched(m_handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N, N, N,
                       &alpha,
                       d_tmp_array, N,
                       d_TT_array, N,
                       &beta,
                       d_Aout_array, static_cast<int>(pitchInFloats),
                       static_cast<int>(batchSize));
}

/**
 * @brief Computes bit-packed mean-threshold hashes for a set of float images that are (m_hashSize x m_hashSize).
 * @param inputs GpuImage array.
 * @return Hash results as a vector of unsigned int (bit-packed).
 */
std::vector<unsigned int> CudaPhash::computeMeanThresholdHash(const std::vector<GpuImage> &inputs)
{
    int batchSize = static_cast<int>(inputs.size());
    if (batchSize == 0)
        return {};

    int totalPixels = inputs[0].width * inputs[0].height;
    int wordsPerImage = (totalPixels + 31) / 32;

    std::vector<const float *> h_imgPtrs(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        h_imgPtrs[i] = inputs[i].data;
    }

    const float **d_imgPtrs = nullptr;
    cudaMalloc(&d_imgPtrs, batchSize * sizeof(float *));
    cudaMemcpyAsync(d_imgPtrs, h_imgPtrs.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice, m_stream);

    unsigned int *d_outHashes = nullptr;
    cudaMalloc(&d_outHashes, batchSize * wordsPerImage * sizeof(unsigned int));

    int blockSize = nextPow2OrMax1024(totalPixels);
    dim3 grid(batchSize);
    meanBitpackKernel<<<grid, blockSize, blockSize * sizeof(float), m_stream>>>(
        d_imgPtrs, inputs[0].pitch / sizeof(float),
        d_outHashes, batchSize,
        inputs[0].width,
        inputs[0].height);

    std::vector<unsigned int> h_allHashes(batchSize * wordsPerImage);
    cudaMemcpyAsync(h_allHashes.data(), d_outHashes, batchSize * wordsPerImage * sizeof(unsigned int), cudaMemcpyDeviceToHost, m_stream);

    cudaFree(d_imgPtrs);
    cudaFree(d_outHashes);
    return h_allHashes;
}

/**
 * @brief Helper struct for double-buffering memory + associated device buffer.
 */
struct HostDeviceBuffer
{
    unsigned char *pinnedHost = nullptr; ///< Pinned (mapped) host buffer
    unsigned char *d_pinned = nullptr;   ///< Device pointer alias
    float *d_baseF = nullptr;            ///< Device buffer for float images
    size_t capacity = 0;                 ///< Capacity in number of 8-bit pixels
};

/**
 * @brief Allocates pinned host memory and corresponding device buffers.
 * @param buf            Buffer struct to fill.
 * @param pixelsPerBatch Number of 8-bit pixels to allocate.
 */
static void allocateHostDeviceBuffer(HostDeviceBuffer &buf, size_t pixelsPerBatch)
{
    cudaHostAlloc(reinterpret_cast<void **>(&buf.pinnedHost),
                  pixelsPerBatch * sizeof(unsigned char),
                  cudaHostAllocMapped);
    cudaHostGetDevicePointer(reinterpret_cast<void **>(&buf.d_pinned),
                             buf.pinnedHost, 0);
    cudaMalloc(&buf.d_baseF, pixelsPerBatch * sizeof(float));
    buf.capacity = pixelsPerBatch;
}

/**
 * @brief Frees pinned host memory and device buffers in HostDeviceBuffer.
 * @param buf Buffer struct to free.
 */
static void freeHostDeviceBuffer(HostDeviceBuffer &buf)
{
    if (buf.pinnedHost)
    {
        cudaFreeHost(buf.pinnedHost);
        buf.pinnedHost = nullptr;
        buf.d_pinned = nullptr;
    }
    if (buf.d_baseF)
    {
        cudaFree(buf.d_baseF);
        buf.d_baseF = nullptr;
    }
    buf.capacity = 0;
}

/**
 * @brief Computes pHashes of a list of images on disk, using m_batchSize images per batch.
 * @param imagePaths Paths to the images on disk.
 * @return A vector of pHash strings (binary "0"/"1" chars).
 */
std::vector<std::vector<uint32_t>> CudaPhash::phash(const std::vector<std::string> &imagePaths)
{
    size_t numImages = imagePaths.size();
    if (numImages == 0)
        return {};

    size_t pixelsPerImg = static_cast<size_t>(m_imgSize) * m_imgSize;

    HostDeviceBuffer buffers[2];
    allocateHostDeviceBuffer(buffers[0], pixelsPerImg * m_batchSize);
    allocateHostDeviceBuffer(buffers[1], pixelsPerImg * m_batchSize);

    cudaEvent_t gpuDoneEvent[2];
    cudaEventCreate(&gpuDoneEvent[0]);
    cudaEventCreate(&gpuDoneEvent[1]);

    std::vector<std::vector<uint32_t>> finalHashes(numImages);

    auto CpuPipeline = [&](size_t startIdx, size_t endIdx, HostDeviceBuffer &buf)
    {
        size_t count = endIdx - startIdx;
        std::vector<std::future<void>> futures;
        futures.reserve(count);

        for (size_t i = 0; i < count; ++i)
        {
            size_t globalIdx = startIdx + i;
            unsigned char *pinnedImgPtr = buf.pinnedHost + (i * pixelsPerImg);

            futures.emplace_back(std::async(std::launch::async, [&, globalIdx, pinnedImgPtr]()
                                            {
                cv::Mat dst(m_imgSize, m_imgSize, CV_8UC1, pinnedImgPtr);
                cv::Mat temp = cv::imread(imagePaths[globalIdx], cv::IMREAD_GRAYSCALE);
                if (temp.empty()) dst.setTo(0);
                else cv::resize(temp, dst, cv::Size(m_imgSize, m_imgSize), 0.0, 0.0, cv::INTER_LINEAR); }));
        }
        for (auto &f : futures)
            f.get();
    };

    auto GpuPipeline = [&](size_t batchOffset, size_t count, HostDeviceBuffer &buf, cudaEvent_t &doneEvent)
    {
        if (count == 0)
            return;

        dim3 block(512);
        dim3 grid((static_cast<int>((pixelsPerImg * count + 7) / 8) + block.x - 1) / block.x);
        host8uToFloatKernel<<<grid, block, 0, m_stream>>>(
            reinterpret_cast<const uint2 *>(buf.d_pinned),
            buf.d_baseF,
            m_imgSize,
            static_cast<int>(m_imgSize * count));

        std::vector<GpuImage> gpuImages(count);
        for (size_t i = 0; i < count; ++i)
        {
            gpuImages[i].data = buf.d_baseF + i * pixelsPerImg;
            gpuImages[i].pitch = m_imgSize * sizeof(float);
            gpuImages[i].width = m_imgSize;
            gpuImages[i].height = m_imgSize;
        }

        batchDct(gpuImages);

        for (auto &img : gpuImages)
        {
            img.width = m_hashSize;
            img.height = m_hashSize;
        }

        std::vector<unsigned int> hashes = computeMeanThresholdHash(gpuImages);

        int wordsPerImage = (m_hashSize * m_hashSize + 31) / 32;
        for (size_t i = 0; i < count; ++i)
        {
            std::vector<uint32_t> hashVec(wordsPerImage);
            for (int w = 0; w < wordsPerImage; ++w)
            {
                hashVec[w] = hashes[i * wordsPerImage + w];
            }
            finalHashes[batchOffset + i] = std::move(hashVec);
        }

        cudaEventRecord(doneEvent, m_stream);
    };

    size_t numBatches = (numImages + m_batchSize - 1) / m_batchSize;
    size_t currentStart = 0;
    size_t currentCount = std::min<size_t>(m_batchSize, numImages - currentStart);

    int currentBuf = 0, nextBuf = 1;
    CpuPipeline(currentStart, currentStart + currentCount, buffers[currentBuf]);

    for (size_t b = 0; b < numBatches; ++b)
    {
        size_t batchOffset = b * m_batchSize;
        size_t count = std::min<size_t>(m_batchSize, numImages - batchOffset);

        size_t nextBatchOffset = (b + 1) * m_batchSize;
        size_t nextCount = (b + 1 < numBatches) ? std::min<size_t>(m_batchSize, numImages - nextBatchOffset) : 0;

        cudaEventSynchronize(gpuDoneEvent[currentBuf]);

        GpuPipeline(batchOffset, count, buffers[currentBuf], gpuDoneEvent[currentBuf]);

        if (nextCount > 0)
        {
            std::future<void> f = std::async(std::launch::async, [&]()
                                             { CpuPipeline(nextBatchOffset, nextBatchOffset + nextCount, buffers[nextBuf]); });
            f.wait();
        }

        std::swap(currentBuf, nextBuf);
    }

    cudaEventSynchronize(gpuDoneEvent[currentBuf]);

    freeHostDeviceBuffer(buffers[0]);
    freeHostDeviceBuffer(buffers[1]);
    cudaEventDestroy(gpuDoneEvent[0]);
    cudaEventDestroy(gpuDoneEvent[1]);
    return finalHashes;
}