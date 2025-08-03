#include "pch.h"
#include "CppUnitTest.h"

#include "memory_manager.cuh"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace
{
    constexpr size_t KB = 1024;
    constexpr size_t MB = KB * KB;
}

/*
    Unit-tests for MemoryManager
        • Exercise both pools (device & pinned)
        • Verify best-fit allocation, splitting and coalescing
        • Check alignment guarantees
        • Detect leaks / corruption under contention
        • Validate 500 ms OOM path and the blocking / wake-up logic
        • Ensure nvJPEG callback wrappers behave
*/
namespace MemoryManagerTests
{
    TEST_CLASS(MemoryManagerTests)
    {
    public:

        TEST_METHOD(PinnedAllocateFreeReuse)
        {
            MemoryManager mgr;

            const size_t nInts = 256;
            int* p1 = mgr.Allocate<int>(MemoryManager::PINNED_POOL, nInts);
            Assert::IsNotNull(p1);

            mgr.Free(MemoryManager::PINNED_POOL, p1);

            int* p2 = mgr.Allocate<int>(MemoryManager::PINNED_POOL, nInts);
            Assert::IsNotNull(p2);
            Assert::AreEqual(reinterpret_cast<void*>(p1),
                reinterpret_cast<void*>(p2),
                L"Pointer should be reused after free");

            mgr.Free(MemoryManager::PINNED_POOL, p2);
        }

        TEST_METHOD(DevicePoolCoalescing)
        {
            MemoryManager mgr;

            const size_t block = 256 * KB;

            unsigned char* a = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, block);
            unsigned char* b = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, block);
            unsigned char* c = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, block);

            mgr.Free(MemoryManager::DEVICE_POOL, b);

            unsigned char* mid = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, block / 2);
            Assert::IsTrue(mid >= b && mid < b + block,
                L"Mid allocation did not originate from the freed segment");

            // Return everything and be sure that the three neighbouring free
            // segments are coalesced into a single big one.
            mgr.Free(MemoryManager::DEVICE_POOL, a);
            mgr.Free(MemoryManager::DEVICE_POOL, c);
            mgr.Free(MemoryManager::DEVICE_POOL, mid);

            unsigned char* big =
                mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, block * 3 + 128);

            Assert::IsNotNull(big, L"Unable to allocate big block after coalescing");

            mgr.Free(MemoryManager::DEVICE_POOL, big);
        }

        TEST_METHOD(OutOfMemoryThrows)
        {
            MemoryManager      mgr;
            std::vector<void*> ptrs;
            bool               threw = false;

            try
            {
                while (true)
                    ptrs.push_back(
                        mgr.Allocate<unsigned char>(MemoryManager::PINNED_POOL, 256 * KB));
            }
            catch (const std::runtime_error&)
            {
                threw = true;
            }

            for (auto* p : ptrs)
                mgr.Free(MemoryManager::PINNED_POOL, static_cast<unsigned char*>(p));

            Assert::IsTrue(threw, L"Expected out-of-memory exception was not thrown");
        }

        TEST_METHOD(BlockingWaitSucceeds)
        {
            MemoryManager mgr;

            // Grab almost the whole pool so that subsequent allocations block.
            unsigned char* big = mgr.Allocate<unsigned char>(MemoryManager::PINNED_POOL, 8 * MB);

            std::atomic<bool> workerDone{ false };

            std::thread t([&]
                {
                    unsigned char* small =
                        mgr.Allocate<unsigned char>(MemoryManager::PINNED_POOL, 1 * MB);
                    workerDone = true;
                    mgr.Free(MemoryManager::PINNED_POOL, small);
                });

            // Give the worker a moment to reach the blocking wait
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            mgr.Free(MemoryManager::PINNED_POOL, big);
            t.join();

            Assert::IsTrue(workerDone,
                L"Worker thread failed to obtain memory after space became available");
        }

        TEST_METHOD(MultiThreadStress)
        {
            MemoryManager mgr;

            constexpr size_t threadCount = 8;
            constexpr size_t iterations = 2000;

            std::atomic<bool> failed{ false };

            auto worker = [&](int seed)
                {
                    std::vector<void*> local;
                    std::mt19937       rng(seed);
                    std::uniform_int_distribution<size_t> dist(8, 512);

                    for (size_t i = 0; i < iterations && !failed; ++i)
                    {
                        bool doAlloc = local.empty() || (rng() % 2 == 0);

                        if (doAlloc)
                        {
                            size_t bytes = dist(rng);
                            try
                            {
                                local.push_back(
                                    mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, bytes));
                            }
                            catch (...)
                            {
                                failed = true;
                            }
                        }
                        else
                        {
                            size_t idx = rng() % local.size();
                            mgr.Free(MemoryManager::DEVICE_POOL,
                                static_cast<unsigned char*>(local[idx]));
                            local.erase(local.begin() + idx);
                        }
                    }

                    for (void* p : local)
                        mgr.Free(MemoryManager::DEVICE_POOL,
                            static_cast<unsigned char*>(p));
                };

            std::vector<std::thread> threads;
            for (int i = 0; i < threadCount; ++i)
                threads.emplace_back(worker, i + 1);

            for (auto& th : threads) th.join();

            // Pool should be completely free now.
            try
            {
                auto* p = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, 9 * MB);
                mgr.Free(MemoryManager::DEVICE_POOL, p);
            }
            catch (...)
            {
                Assert::Fail(L"Leak or corruption detected after multithreaded stress test");
            }

            Assert::IsFalse(failed, L"Failure detected inside stress threads");
        }

        TEST_METHOD(DoubleFreeIsNoOp)
        {
            MemoryManager mgr;
            auto* p = mgr.Allocate<int>(MemoryManager::PINNED_POOL, 64);

            mgr.Free(MemoryManager::PINNED_POOL, p);
            mgr.Free(MemoryManager::PINNED_POOL, p);   // second free must be harmless

            auto* q = mgr.Allocate<int>(MemoryManager::PINNED_POOL, 64);
            Assert::IsNotNull(q);
            mgr.Free(MemoryManager::PINNED_POOL, q);
        }

        TEST_METHOD(ZeroSizeAllocation)
        {
            MemoryManager     mgr;
            unsigned char* p = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, 0);

            Assert::IsNotNull(p);
            mgr.Free(MemoryManager::DEVICE_POOL, p);
        }

        TEST_METHOD(AlignmentGuarantee)
        {
            MemoryManager  mgr;

            unsigned char* p1 = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, 3);
            unsigned char* p2 = mgr.Allocate<unsigned char>(MemoryManager::DEVICE_POOL, 3);

            ptrdiff_t diff = p2 - p1;

            Assert::IsTrue(diff % 16 == 0,
                L"Consecutive small allocations are not 16-byte aligned");

            mgr.Free(MemoryManager::DEVICE_POOL, p1);
            mgr.Free(MemoryManager::DEVICE_POOL, p2);
        }

        TEST_METHOD(NvjpegAllocatorCallbacks)
        {
            MemoryManager                       mgr;
            nvjpegDevAllocatorV2_t* devAlloc = mgr.getNvjDevAllocator();
            void* ptr = nullptr;

            Assert::AreEqual(0, devAlloc->dev_malloc(devAlloc->dev_ctx, &ptr, 1024, nullptr));
            Assert::IsNotNull(ptr);
            Assert::AreEqual(0, devAlloc->dev_free(devAlloc->dev_ctx, ptr, 0, nullptr));
        }
    };
}