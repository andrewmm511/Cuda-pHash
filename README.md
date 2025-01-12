# Cuda-pHash
Highly-optimized perceptual hash calculations on GPU

---

## High-Level Process

1. **Image Loading**  
   Each image is loaded from disk and converted to a 2D grayscale matrix, where each matrix element represents the intensity of one pixel.

2. **Downsampling (192×192)**  
   The loaded images are resized via bicubic interpolation.

3. **Convert to Float and Upload to GPU**  
   The grayscale data is stored in an OpenCV `GpuMat` of type `CV_32F`. Each pixel’s intensity is now stored as a single-precision float on the GPU.

4. **Discrete Cosine Transform (DCT)**  
   We apply a DCT (specifically the DCT-II) on each 192×192 float image.

5. **Crop to 16×16**  
   From the DCT result, we take the top-left N×N submatrix, where N is the desired hash length. This region concentrates the lowest frequencies, which are most significant for perceptual hashing.

6. **Thresholding and Bit-Packing**  
   - We sort each N×N block's coefficients to compute the median.  
   - Compare each DCT coefficient against the median: if the coefficient is larger, the corresponding bit is set to 1; otherwise, it is 0.  
   - Finally, we pack these N*N bits into N/2 × 32-bit unsigned integers.

7. **Return the Hash**  
   The N bits (in N/2 × 32-bit words) form the resulting pHash.  

This final hash string (or bit sequence) is the perceptual fingerprint of the image, and can be used for similarity search.

---
