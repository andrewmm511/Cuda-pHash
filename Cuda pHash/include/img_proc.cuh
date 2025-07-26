//#pragma once
//#ifndef IMAGE_H
//#define IMAGE_H
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "cublas_v2.h"
//#include <vector>
//#include <string>
//#include <nvjpeg.h>
//#include <cstdint>
//#include <future>
//
//#include "memory_manager.cuh"
//
//#ifdef BUILDING_DLL
//#define API_EXPORT __declspec(dllexport)
//#else
//#define API_EXPORT __declspec(dllimport)
//#endif
//
//// Data structure
//struct Image {
//    std::shared_ptr<MemoryManager> memMgr;
//    std::string path;
//    int file_size = 0;
//    unsigned char* rawFileData = nullptr;    // pinned host memory
//    int originalWidth = 0;
//    int originalHeight = 0;
//    unsigned char* DecodedData = nullptr;    // device memory
//    int resizedWidth = 0;
//    int resizedHeight = 0;
//    float* resizedData = nullptr;    // device memory
//};
//
//struct CudaResources {
//    cudaStream_t stream;
//    cudaEvent_t event;
//
//    // For resize operation - persistent allocations
//    unsigned char** d_inputsArray;
//    float** d_outputsArray;
//    int* d_inWidths;
//    int* d_inHeights;
//    size_t currentBatchCapacity;
//
//    CudaResources(size_t initialBatchSize = 32)
//        : stream(nullptr), event(nullptr),
//        d_inputsArray(nullptr), d_outputsArray(nullptr),
//        d_inWidths(nullptr), d_inHeights(nullptr),
//        currentBatchCapacity(0) {
//        // Create stream and event
//        cudaStreamCreate(&stream);
//        cudaEventCreate(&event);
//
//        // Allocate arrays for resize operation
//        ensureCapacity(initialBatchSize);
//    }
//
//    ~CudaResources() {
//        // Clean up
//        if (stream) cudaStreamDestroy(stream);
//        if (event) cudaEventDestroy(event);
//
//        if (d_inputsArray) cudaFree(d_inputsArray);
//        if (d_outputsArray) cudaFree(d_outputsArray);
//        if (d_inWidths) cudaFree(d_inWidths);
//        if (d_inHeights) cudaFree(d_inHeights);
//    }
//
//    // Ensure arrays have enough capacity for a given batch size
//    void ensureCapacity(size_t batchSize) {
//        if (batchSize <= currentBatchCapacity) return;
//
//        // Free old arrays if they exist
//        if (d_inputsArray) cudaFree(d_inputsArray);
//        if (d_outputsArray) cudaFree(d_outputsArray);
//        if (d_inWidths) cudaFree(d_inWidths);
//        if (d_inHeights) cudaFree(d_inHeights);
//
//        // Allocate new arrays with the required capacity
//        cudaMalloc(&d_inputsArray, sizeof(unsigned char*) * batchSize);
//        cudaMalloc(&d_outputsArray, sizeof(float*) * batchSize);
//        cudaMalloc(&d_inWidths, sizeof(int) * batchSize);
//        cudaMalloc(&d_inHeights, sizeof(int) * batchSize);
//
//        currentBatchCapacity = batchSize;
//    }
//};
//
//
//__global__ void batchBilinearResizeKernel(const unsigned char* const* d_inputs,
//    const int* inWidths,
//    const int* inHeights,
//    float* const* d_outputs,
//    int outWidth,
//    int outHeight);
//
//void bilinearResize(const std::vector<unsigned char*>& d_inputVec,
//    const std::vector<int>& widths,
//    const std::vector<int>& heights,
//    const std::vector<float*>& d_outputVec,
//    int outWidth,
//    int outHeight,
//    CudaResources& resources);
//
//std::vector<Image> process_images(const std::vector<std::string>& imgPaths, int targetSize, bool verbose = false);
//
//#endif // IMAGE_H