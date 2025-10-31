#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <random>
#include <sys/stat.h>
#include <windows.h>

#include "memory_manager.cuh"
#include "logger.hpp"

/**
 * @brief Aligns a size to a specified boundary, adjusting alignment based on size.
 *
 * @param size      Size to align
 * @param alignment Default alignment boundary (must be power of 2)
 * @return Aligned size
 */
inline size_t alignSize(size_t size, size_t alignment = 16) {
    // Adjust alignment based on allocation size for better CUDA performance
    if (size >= 4096)      alignment = 256;
    else if (size >= 1024) alignment = 128;
    else if (size >= 256)  alignment = 64;

    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Helper function to insert a new free segment into a free list and coalesce with neighbors.
 *
 * @param freeList The free list
 * @param newSeg   The segment to insert
 */
static void insertSegmentAndCoalesce(std::vector<FreeSegment>& freeList, const FreeSegment& newSeg) {
    if (freeList.empty()) {
        freeList.push_back(newSeg);
        return;
    }

    // Fast path for appending at the end
    if (newSeg.offset > freeList.back().offset) {
        if (freeList.back().offset + freeList.back().size == newSeg.offset) {
            freeList.back().size += newSeg.size;
        }
        else {
            freeList.push_back(newSeg);
        }
        return;
    }

    // Fast path for prepending at the beginning
    if (newSeg.offset + newSeg.size == freeList.front().offset) {
        freeList.front().offset = newSeg.offset;
        freeList.front().size += newSeg.size;
        return;
    }
    if (newSeg.offset < freeList.front().offset) {
        freeList.insert(freeList.begin(), newSeg);
        return;
    }

    // Binary search to find insertion point
    auto it = std::lower_bound(
        freeList.begin(),
        freeList.end(),
        newSeg,
        [](const FreeSegment& a, const FreeSegment& b) {
            return a.offset < b.offset;
        }
    );

    size_t insertIdx = static_cast<size_t>(std::distance(freeList.begin(), it));
    bool hasPrev = (insertIdx > 0);
    bool hasNext = (insertIdx < freeList.size());

    // Coalesce forward
    if (hasNext && (newSeg.offset + newSeg.size == freeList[insertIdx].offset)) {
        freeList[insertIdx].offset = newSeg.offset;
        freeList[insertIdx].size += newSeg.size;

        // Also coalesce backward if adjacent
        if (hasPrev &&
            (freeList[insertIdx - 1].offset + freeList[insertIdx - 1].size == freeList[insertIdx].offset))
        {
            freeList[insertIdx - 1].size += freeList[insertIdx].size;
            freeList.erase(freeList.begin() + insertIdx);
        }
        return;
    }

    // Coalesce backward
    if (hasPrev &&
        (freeList[insertIdx - 1].offset + freeList[insertIdx - 1].size == newSeg.offset))
    {
        freeList[insertIdx - 1].size += newSeg.size;
        return;
    }

    // No coalescing, just insert
    freeList.insert(it, newSeg);
}

// ----------------------------------------------------------------------------------
// MemoryRegion Member
// ----------------------------------------------------------------------------------

/**
 * @brief Initialize the free list to contain the entire region as free.
 */
void MemoryRegion::initFreeList() {
    freeList.clear();
    freeList.reserve(128);
    freeList.push_back({ 0, totalSize });
    allocations.reserve(128);
}

// ----------------------------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------------------------

MemoryManager::MemoryManager() {
    // Provide minimal initial allocations; real sizes come from InitializePools()
    size_t initialSize = 1024 * 1024 * 10; // 10MB
    CUDA_CHECK(cudaMalloc(&m_devPtr, initialSize));
    CUDA_CHECK(cudaHostAlloc(&m_pinnedPtr, initialSize, cudaHostAllocMapped));

    // Device pool
    m_devicePool.totalSize = initialSize;
    m_devicePool.basePtr = m_devPtr;
    m_devicePool.initFreeList();

    // Unified pinned pool
    m_pinnedPool.totalSize = initialSize;
    m_pinnedPool.basePtr = m_pinnedPtr;
    m_pinnedPool.initFreeList();

    // nvJPEG allocators
    m_devAllocator.dev_malloc = nvjpegDevMalloc;
    m_devAllocator.dev_free = nvjpegDevFree;
    m_devAllocator.dev_ctx = this;
    m_pinnedAllocator.pinned_malloc = nvjpegPinnedMalloc;
    m_pinnedAllocator.pinned_free = nvjpegPinnedFree;
    m_pinnedAllocator.pinned_ctx = this;
}

MemoryManager::~MemoryManager() {
    if (m_devPtr)    CUDA_CHECK_NOTHROW(cudaFree(m_devPtr));
    if (m_pinnedPtr) CUDA_CHECK_NOTHROW(cudaFreeHost(m_pinnedPtr));
}

// ----------------------------------------------------------------------------------
// Initial Pool Setup
// ----------------------------------------------------------------------------------

void MemoryManager::InitializePools()
{
    // Query current available GPU memory
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // Device pool = 87% of the *currently available* GPU memory
    size_t deviceSize = static_cast<size_t>(freeMem * 0.80);

    // Pinned pool = 800 MB
    size_t pinnedSize = 800ULL * 1024ULL * 1024ULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&m_devPtr, deviceSize));
    m_devicePool.basePtr = m_devPtr;
    m_devicePool.totalSize = deviceSize;
    m_devicePool.initFreeList();

    // Allocate pinned memory
    CUDA_CHECK(cudaHostAlloc(&m_pinnedPtr, pinnedSize, cudaHostAllocMapped));
    m_pinnedPool.basePtr = m_pinnedPtr;
    m_pinnedPool.totalSize = pinnedSize;
    m_pinnedPool.initFreeList();

    INFO(__func__, "Initialized Pools. Device=", deviceSize / (1024 * 1024), "MB, Pinned=", pinnedSize / (1024 * 1024), "MB");

    // Reassign nvJPEG allocators in case anything changed
    m_devAllocator.dev_malloc = nvjpegDevMalloc;
    m_devAllocator.dev_free = nvjpegDevFree;
    m_devAllocator.dev_ctx = this;

    m_pinnedAllocator.pinned_malloc = nvjpegPinnedMalloc;
    m_pinnedAllocator.pinned_free = nvjpegPinnedFree;
    m_pinnedAllocator.pinned_ctx = this;
}

// ----------------------------------------------------------------------------------
// nvJPEG Allocator Getters
// ----------------------------------------------------------------------------------

nvjpegDevAllocatorV2_t* MemoryManager::getNvjDevAllocator() {
    return &m_devAllocator;
}

nvjpegPinnedAllocatorV2_t* MemoryManager::getNvjPinnedAllocator() {
    return &m_pinnedAllocator;
}

// ----------------------------------------------------------------------------------
// Internal Allocation / Free Helpers
// ----------------------------------------------------------------------------------

/**
 * @brief Allocate a region of memory from the given pool via best-fit, blocking up to 500ms if no space is found.
 *
 * @param region Memory region to allocate from
 * @param size   Requested size in bytes
 * @return Pointer to allocated memory
 */
unsigned char* MemoryManager::allocate(MemoryRegion& region, size_t size) {
    size_t alignedSize = alignSize(size);
    auto startTime = std::chrono::steady_clock::now();

    std::unique_lock<std::mutex> lock(region.mtx);

    while (true)
    {
        auto bestIt = region.freeList.end();
        size_t bestSize = std::numeric_limits<size_t>::max();

        for (auto it = region.freeList.begin(); it != region.freeList.end(); ++it) {
            if (it->size >= alignedSize && it->size < bestSize) {
                bestIt = it;
                bestSize = it->size;
            }
        }

        if (bestIt != region.freeList.end()) {
            // Found a suitable segment
            size_t offset = bestIt->offset;

            // Take whole segment if size close to avoid too much fragmentation on small leftover
            if (bestIt->size <= alignedSize + 128) {
                region.freeList.erase(bestIt);
                unsigned char* userPtr = static_cast<unsigned char*>(region.basePtr) + offset;
                region.allocations[userPtr] = bestSize;
                return userPtr;
            }
            else {
                // Split segment
                bestIt->offset += alignedSize;
                bestIt->size -= alignedSize;

                unsigned char* userPtr = static_cast<unsigned char*>(region.basePtr) + offset;
                region.allocations[userPtr] = alignedSize;
                return userPtr;
            }
        }

		// No segment was large enough, check search duration
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
        if (elapsedMs >= 500) {
            std::cerr << "Out of memory in pool. Lower batch size or increase pool." << std::endl;
            throw std::runtime_error("MemoryManager: out of memory after 500ms wait.");
        }

        // Wait for something to be freed up to max timeout
		long remainingMs = std::max<long>(500 - static_cast<long>(elapsedMs), 1);
        region.cv.wait_for(lock, std::chrono::milliseconds(remainingMs));
    }
}

/**
 * @brief Free a pointer in a variable-size pool with coalescing, notifying waiting threads if needed.
 *
 * @param region Memory region to free into
 * @param ptr    Pointer to free
 */
void MemoryManager::free(MemoryRegion& region, unsigned char* ptr) {
    if (!ptr) return;
    std::unique_lock<std::mutex> lock(region.mtx);

    auto it = region.allocations.find(ptr);
    if (it == region.allocations.end()) return;

    size_t freedSize = it->second;
    region.allocations.erase(it);

    size_t offset = static_cast<size_t>(
        reinterpret_cast<unsigned char*>(ptr) -
        static_cast<unsigned char*>(region.basePtr)
        );

    insertSegmentAndCoalesce(region.freeList, { offset, freedSize });
    region.cv.notify_all();
}

// ----------------------------------------------------------------------------------
// nvJPEG Callbacks
// ----------------------------------------------------------------------------------

int MemoryManager::nvjpegDevMalloc(void* ctx, void** ptr, size_t size, cudaStream_t /*stream*/) {
    try {
        auto* mgr = static_cast<MemoryManager*>(ctx);
        // Round up the size for alignment
        *ptr = mgr->allocate(mgr->m_devicePool, (size + 255) & ~255);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "nvjpegDevMalloc error: " << e.what() << std::endl;
        *ptr = nullptr;
        return -1;
    }
}

int MemoryManager::nvjpegDevFree(void* ctx, void* ptr, size_t /*size*/, cudaStream_t /*stream*/) {
    try {
        if (!ptr) return 0;
        auto* mgr = static_cast<MemoryManager*>(ctx);
        mgr->free(mgr->m_devicePool, static_cast<unsigned char*>(ptr));
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "nvjpegDevFree error: " << e.what() << std::endl;
        return -1;
    }
}

int MemoryManager::nvjpegPinnedMalloc(void* ctx, void** ptr, size_t size, cudaStream_t /*stream*/) {
    try {
        auto* mgr = static_cast<MemoryManager*>(ctx);
        // Round up the size for alignment
        *ptr = mgr->allocate(mgr->m_pinnedPool, (size + 255) & ~255);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "nvjpegPinnedMalloc error: " << e.what() << std::endl;
        *ptr = nullptr;
        return -1;
    }
}

int MemoryManager::nvjpegPinnedFree(void* ctx, void* ptr, size_t /*size*/, cudaStream_t /*stream*/) {
    try {
        if (!ptr) return 0;
        auto* mgr = static_cast<MemoryManager*>(ctx);
        mgr->free(mgr->m_pinnedPool, static_cast<unsigned char*>(ptr));
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "nvjpegPinnedFree error: " << e.what() << std::endl;
        return -1;
    }
}