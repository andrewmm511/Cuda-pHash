# Repository Guidelines

This C++ and CUDA repository implements a GPU‑accelerated perceptual hash (pHash) pipeline using CUDA, cuBLAS, and nvJPEG.

## Project Structure & Module Organization
- CudapHash/: core library and GPU pipeline
  - CudapHash/include/: public headers (e.g., phash_cuda.cuh, memory_manager.cuh, img_proc.cuh)
  - CudapHash/src/: CUDA/C++ implementation
- App/: CLI entry point (src/main.cpp)
- Test/: Google Test suites
- third_party/: external dependencies
- build/: CMake artifacts

## Build and Development Commands
Once complete with all tasks, run these commands in order:
- Configure: `cmake --build build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build -j 8 -DCMAKE_BUILD_TYPE=Release`

Correct any errors or warnings related to your changes if applicable.

## Coding Style & Naming Conventions
- C++20 and CUDA, 4‑space indent, 120‑column soft limit.
- Prefer RAII, `std::unique_ptr`/`std::shared_ptr`; avoid raw new/delete.
- Use clang‑format if a project config is present.

## Core Components

1. **CudaPhash** (`CudapHash/include/phash_cuda.cuh`): Main class that orchestrates the GPU pipeline
   - Manages CUDA resources (streams, memory, handles)
   - Implements multi-stage pipeline: reading -> decoding -> resizing -> hashing

2. **MemoryManager** (`CudapHash/include/memory_manager.cuh`): Custom memory management for GPU operations

3. **CLI Application** (`App\main.cpp`): Command-line interface
   - Two main commands: `hash` (compute hashes) and `similar` (compute hashes then find duplicates based on hamming distances)
   - Options for auto-deletion or interactive review of similar images

## Key Design Patterns

- **Asynchronous Pipeline**: Uses CUDA streams to overlap CPU-GPU transfers with computation
- **Batch Processing**: Processes images in configurable batches (default 500) to maximize GPU utilization
- **Work Queue System**: Multi-threaded CPU workers for I/O operations
- **Double Buffering**: Hides PCIe transfer latency by overlapping transfers with computation

## Important Notes

- You are running in a Linux environment, so use Linux/Unix commands