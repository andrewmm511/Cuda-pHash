#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <nvjpeg.h>

#include <string>
#include <iostream>

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

/**
 * @struct FreeSegment
 * @brief Represents a free segment of memory in a variable-size pool.
 */
struct FreeSegment {
    size_t offset;
    size_t size;
};

/**
 * @struct MemoryRegion
 * @brief Encapsulates the data structures needed to manage a memory pool.
 */
struct MemoryRegion {
    std::string str;
    std::mutex mtx;
    std::condition_variable cv;
    void* basePtr = nullptr;
    size_t totalSize = 0;

    // For variable-sized allocations
    std::vector<FreeSegment> freeList;
    std::unordered_map<void*, size_t> allocations;

    void initFreeList();
};

class MemoryManager {
public:
    /**
     * @enum PoolType
     * @brief Identifies the various pools managed by MemoryManager.
     */
    enum PoolType {
        PINNED_POOL,   // Host-pinned memory pool
        DEVICE_POOL    // Device memory pool
    };

    /**
     * @brief Constructor. Must call InitializePools() to allocate memory.
     */
    MemoryManager();

    /**
     * @brief Destructor - frees all allocated resources.
     */
    ~MemoryManager();

    /**
	 * @brief Initializes the pools to 87% available memory.
     */
    void InitializePools();

    /**
     * @brief Allocate from a specified memory pool.
     *
     * @param poolType Which pool to allocate from
     * @param size     Number of elements (will be multiplied by sizeof(T))
     * @return Pointer to allocated memory
     */
    template<typename T>
    T* Allocate(PoolType poolType, size_t size = 1)
    {
        size_t sizeInBytes = size * sizeof(T);
        switch (poolType) {
        case PINNED_POOL:
            return reinterpret_cast<T*>(allocate(m_pinnedPool, sizeInBytes));
        case DEVICE_POOL:
            return reinterpret_cast<T*>(allocate(m_devicePool, sizeInBytes));
        default:
            throw std::runtime_error("Invalid pool type.");
        }
    }

    /**
     * @brief Free previously allocated memory back to the appropriate pool.
     *
     * @param poolType The pool the pointer was allocated from
     * @param ptr      Pointer to free
     */
    template<typename T>
    void Free(PoolType poolType, T* ptr)
    {
        if (!ptr) return;
        switch (poolType) {
        case PINNED_POOL:
            free(m_pinnedPool, reinterpret_cast<unsigned char*>(ptr));
            break;
        case DEVICE_POOL:
            free(m_devicePool, reinterpret_cast<unsigned char*>(ptr));
            break;
        default:
            throw std::runtime_error("Invalid pool type.");
        }
    }

    /**
     * @brief Getter for the nvJPEG device allocator.
     */
    nvjpegDevAllocatorV2_t* getNvjDevAllocator();

    /**
     * @brief Getter for the nvJPEG pinned allocator.
     */
    nvjpegPinnedAllocatorV2_t* getNvjPinnedAllocator();

private:
    // Allocation helper
    unsigned char* allocate(MemoryRegion& region, size_t size);

    // Free helper
    void free(MemoryRegion& region, unsigned char* ptr);

    // Actual backing pointers
    void* m_devPtr = nullptr;    // Single device allocation
    void* m_pinnedPtr = nullptr; // Single pinned allocation

    // Memory regions
    MemoryRegion m_devicePool;
    MemoryRegion m_pinnedPool;

    // nvJPEG allocators
    nvjpegDevAllocatorV2_t m_devAllocator;
    nvjpegPinnedAllocatorV2_t m_pinnedAllocator;

    // nvJPEG callback implementations
    static int nvjpegDevMalloc(void* ctx, void** ptr, size_t size, cudaStream_t stream);
    static int nvjpegDevFree(void* ctx, void* ptr, size_t size, cudaStream_t stream);
    static int nvjpegPinnedMalloc(void* ctx, void** ptr, size_t size, cudaStream_t stream);
    static int nvjpegPinnedFree(void* ctx, void* ptr, size_t size, cudaStream_t stream);
};