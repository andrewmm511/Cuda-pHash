# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA pHash is a high-performance, GPU-accelerated tool for computing perceptual hashes (pHash) of images and identifying visually similar images in a dataset. It's performant by leveraging CUDA, cuBLAS, and nvJPEG for massive parallelization of pHashes on GPU.

## Build Commands

This is a Visual Studio solution with CUDA support:

- **Build Solution**: Open `CudapHash.sln` in Visual Studio and build (Ctrl+Shift+B)
- **Build Configuration**: Release x64 is the primary configuration
- **Output**: Executables are generated in `x64\Release\`

## Architecture

The solution is structured into three projects:
1. **CudapHash**: Core library implementing the GPU pipeline
   - Contains CUDA kernels, memory management, and image processing logic
   - Exposes a C++ API
2. **App**: Command-line application that uses the core library
3. **Tests**: Unit tests and integration tests for the core library

### Core Components

1. **CudaPhash** (`CudapHash\include\phash_cuda.cuh`): Main class that orchestrates the GPU pipeline
   - Manages CUDA resources (streams, memory, handles)
   - Implements multi-stage pipeline: reading → decoding → resizing → hashing
   - Uses cuBLAS for DCT computation

2. **MemoryManager** (`CudapHash\include\memory_manager.cuh`): Custom memory management for GPU operations
   - Handles allocation/deallocation of device memory
   - Manages pinned host memory for faster transfers

3. **Image Processing** (`CudapHash\include\img_proc.cuh`): GPU kernels for image operations
   - JPEG decoding using nvJPEG
   - Image resizing kernels
   - DCT computation using cuBLAS

4. **CLI Application** (`App\main.cpp`): Command-line interface
   - Two main commands: `hash` (compute hashes) and `similar` (find duplicates)
   - Supports batch processing, recursive directory scanning
   - Options for auto-deletion or interactive review of similar images

### Key Design Patterns

- **Asynchronous Pipeline**: Uses CUDA streams to overlap CPU-GPU transfers with computation
- **Batch Processing**: Processes images in configurable batches (default 500) to maximize GPU utilization
- **Work Queue System**: Multi-threaded CPU workers for I/O operations
- **Double Buffering**: Hides PCIe transfer latency by overlapping transfers with computation

## Running Tests

The project includes a Google Test based test suite:

```bash
# Run all tests (from x64\Release directory)
Test.dll

# Or use the test playlists
# Quick Test.playlist - subset of tests
# Full Test.playlist - comprehensive test suite
```

## Important Notes

- Primary development is on Windows with MSVC compiler