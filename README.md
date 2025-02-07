 CUDA-Accelerated Perceptual Hash (pHash)

## Overview

Cuda pHash is a high-performance, GPU-accelerated tool for computing the perceptual hash (pHash) of images. 

Cuda pHash outperforms other leading perceptual hash implementations by a wide margin, processing massive datasets via highly-optimized compute pipelines.

## Performance Benchmark 🚀

| Implementation                            | Wall clock Time* ⏱️ | Speedup vs. CUDA ⚡ |
|------------------------------------------|-----------------|----------------- |
| **⚡ Cuda pHash**            | **000.000**     | **Baseline** 🏆 |
| **OpenCV pHash**                     | 000.000         | **0.0× slower** 🐢 |
| **Python pHash**      | 000.000         | **0.0× slower** 🐢 |

> **\*** pHash calculation on **COCO 2017** (~163,000 images) using **13th Gen Intel i9 13900K** and **NVIDIA RTX 3080** over PCIe.

## Quick Start

Below is a simple example demonstrating how to instantiate a CUDA pHash object, compute hashes, and compare them using Hamming distance:

```cpp
#include "cuda_phash.cuh"
#include <iostream>
#include <vector>
#include <string>

// Computes Hamming distance between two bit-packed hash vectors
int hammingDistance(const std::vector<uint32_t>& hashA, const std::vector<uint32_t>& hashB) {
    int distance = 0;
    for (size_t i = 0; i < hashA.size(); ++i) {
        uint32_t diff = hashA[i] ^ hashB[i]; // XOR to find differing bits
        while (diff) {
            distance += (diff & 1);
            diff >>= 1;
        }
    }
    return distance;
}

int main() {
    // Initialize CUDA-based pHash with:
    // - hashSize = 8
    // - highFreqFactor = 4
    // - batchSize = 5000
    CudaPhash phasher(8, 4, 5000);

    // Image paths to process
    std::vector<std::string> imagePaths = {
        "image1.jpg", 
        "image2.jpg", 
        "image3.jpg"
    };

    // Compute perceptual hashes
    std::vector<std::vector<uint32_t>> results = phasher.phash(imagePaths);

    // Compute Hamming distance between image1 and image2
    int dist = hammingDistance(results[0], results[1]);
    std::cout << "Hamming distance between image1 and image2: " << dist << std::endl;

    return 0;
}
```

### Hash Output Format

The `phash()` function returns a `std::vector<std::vector<uint32_t>>`, where each inner vector represents the bit-packed hash of an image. If desired, these can be converted to binary or hexadecimal representations by iterating over the 32-bit words.

## How It Works

Perceptual hashing extracts a compact, **transformation-resistant signature** from an image. This implementation employs **discrete cosine transforms (DCT) and mean-thresholding**, optimized for **GPU acceleration** via CUDA.

### 1. **Discrete Cosine Transform (DCT)**  
The **2D DCT** of an image **f(x,y)** is computed as:

$$
DCT(u,v) = \alpha(u) \alpha(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x,y) 
\cos\left(\frac{\pi}{N} (x + 0.5) u \right) 
\cos\left(\frac{\pi}{N} (y + 0.5) v \right)
$$

where:

$$
\alpha(u) =
\begin{cases}
\frac{1}{\sqrt{N}}, & u = 0 \\
\sqrt{\frac{2}{N}}, & u > 0
\end{cases}
$$

By focusing on **low-frequency DCT coefficients** (top-left sub-block), we extract **global structural features** that remain stable under **rotation, scaling, and lighting variations**.

### 2. **Mean Thresholding & Bit Packing**  
- A subset of **n × n** DCT coefficients is selected.  
- These coefficients are **compared to their mean value**, generating a binary hash:  
  - **1** if the coefficient is above the mean  
  - **0** otherwise  
- The resulting bits are **packed into 32-bit words** for efficient storage and fast bitwise comparisons.

### 3. **GPU Acceleration with CUDA & cuBLAS**  
This implementation dramatically **reduces processing time** using **highly optimized GPU operations**:
- **Batched DCT Computation:** CUDA kernels leverage **cuBLAS** to process multiple images **in parallel**.
- **Pinned (Page-Locked) Host Memory:** Enables **faster CPU–GPU transfers** via direct memory mapping.
- **Asynchronous Pipelines:** Overlaps **image loading, resizing, and computation** to **eliminate bottlenecks**.

The result is a **robust, high-performance perceptual hashing** algorithm, capable of **handling large-scale datasets efficiently**.

## Installation

### 1. Clone or Download the Repository

```sh
git clone https://github.com/yourusername/cuda_phash.git
cd cuda_phash
```

### 2. Install Dependencies

Ensure the following are installed:

- **CUDA Toolkit** (tested with **CUDA 11+**)
- **cuBLAS** (included with CUDA)
- **OpenCV** (for image loading and preprocessing)
- **Modern C++ Compiler** (e.g., MSVC, GCC, or Clang)

### 3. Build from Source

Since no CMake configuration is provided, build the project using your preferred method:

#### **Windows (Visual Studio)**
1. Create a new **Visual C++ project**.
2. Add all `.cu` and `.cuh` files.
3. Configure project settings to enable CUDA:
   - Use `"CUDA Runtime API"` in **VS properties**.
4. Link against:
   - `cublas.lib`
   - `cudart.lib`
   - OpenCV libraries (e.g., `opencv_world455.lib`).

#### **Linux / macOS (Command Line)**
Compile using `nvcc` and link required libraries:

```sh
nvcc -o cuda_phash main.cu cuda_phash.cu -lcublas `pkg-config --cflags --libs opencv4`
```

> Adjust paths and library flags to match your system configuration.

### 4. Run the Program

Once compiled, include `cuda_phash.cuh` in your project and ensure **CUDA Toolkit** and **OpenCV** are correctly linked at runtime.
