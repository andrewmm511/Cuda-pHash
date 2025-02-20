#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "phash_base.h"
#include "phash_cuda.cuh"

class Phash {
public:
    Phash(int hash_size, int batch_size)
        : hash_size_(hash_size), batch_size_(batch_size)
    {
        if (is_cuda_available()) impl_ = std::make_unique<PhashGPU>(hash_size_, batch_size_);
        else impl_ = std::make_unique<PhashCPU>(hash_size_, batch_size_);
    }

    std::vector<std::string> phash(const std::vector<std::string>& images) {
        return impl_->run(images);
    }

private:
    // Helper to detect CUDA
    bool is_cuda_available() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess && device_count > 0);
    }

    int hash_size_;
    int batch_size_;

    // A pointer to either PhashCPU or PhashGPU
    std::unique_ptr<PhashBase> impl_;
};