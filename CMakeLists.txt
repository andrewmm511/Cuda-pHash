cmake_minimum_required(VERSION 3.18)
project(CudaPHash LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find cudadevrt library manually
find_library(CUDA_CUDADEVRT_LIBRARY cudadevrt 
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64)

# Set CUDA architectures (compute_86,sm_86 from vcxproj)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Global compile options for Release mode (matching Visual Studio settings)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    # MSVC specific flags from the vcxproj files
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /GL /Gy")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")  # EnableEnhancedInstructionSet
    
    # Enable whole program optimization
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    
    # CUDA specific flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/std:c++20\"")
    # Additional Windows-specific CUDA flags
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")
endif()

# Set output directories to match Visual Studio structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin/Release/x64)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin/Release/x64)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib/Release/x64)

# Define the CudapHash directory with proper path handling
set(CUDA_PHASH_DIR "${CMAKE_SOURCE_DIR}/CudapHash")
file(TO_CMAKE_PATH "${CUDA_PHASH_DIR}" CUDA_PHASH_DIR)

# CudapHash static library
# Use GLOB to handle paths with spaces more reliably
file(GLOB CUDA_PHASH_SOURCES "${CUDA_PHASH_DIR}/src/*.cu")
file(GLOB CUDA_PHASH_HEADERS 
    "${CUDA_PHASH_DIR}/include/*.hpp"
    "${CUDA_PHASH_DIR}/include/*.cuh"
    "${CUDA_PHASH_DIR}/include/*.h"
)

add_library(CudaPhashLib STATIC ${CUDA_PHASH_SOURCES} ${CUDA_PHASH_HEADERS})

# Set include directories for CudapHash library
target_include_directories(CudaPhashLib PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/third_party
    "${CUDA_PHASH_DIR}/include"
)

# Set compile definitions for CudapHash
target_compile_definitions(CudaPhashLib PRIVATE
    WIN32
    WIN64
    NDEBUG
    _CONSOLE
)

# CUDA compile options
set_target_properties(CudaPhashLib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set CUDA compilation flags per target
target_compile_options(CudaPhashLib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-code=arch=compute_86,code=sm_86
        --expt-extended-lambda
        -Xcompiler=/bigobj
    >
)

# Link CUDA libraries (though not needed for static lib, these were in the vcxproj)
target_link_libraries(CudaPhashLib PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvjpeg
    CUDA::cublas
)

# App executable
add_executable(App "App/src/main.cpp")

# Set include directories for App
target_include_directories(App PRIVATE
    "${CUDA_PHASH_DIR}/include"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/third_party
)

# Link App with CudapHash library and CUDA runtime
target_link_libraries(App PRIVATE
    CudaPhashLib
    CUDA::cudart
    ${CUDA_CUDADEVRT_LIBRARY}
    CUDA::nvjpeg
    CUDA::cublas
)

# Set compile definitions for App
target_compile_definitions(App PRIVATE
    NDEBUG
    _CONSOLE
)

# Test library (DLL) - using Google Test or MS Unit Test
# Note: The original uses MS Unit Test framework, but for CMake we'll use Google Test
# You may need to adjust this based on your testing framework preference

# Find or fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# OpenCV configuration (hardcoded path from vcxproj)
# You should make this configurable via CMake options
set(OPENCV_DIR "C:/opencv411/opencv-4.11.0/build/install" CACHE PATH "OpenCV installation directory")

# Test DLL
set(TEST_SOURCES
    "Test/src/pch.cpp"
    "Test/src/Integration.cpp"
    "Test/src/Unit.cpp"
)

set(TEST_HEADERS
    "Test/include/pch.h"
)

add_library(Test SHARED ${TEST_SOURCES} ${TEST_HEADERS})

# Configure precompiled header for Test
target_precompile_headers(Test PRIVATE "Test/include/pch.h")

# Set include directories for Test
target_include_directories(Test PRIVATE
    "${CUDA_PHASH_DIR}/include"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/Test/include
    ${OPENCV_DIR}/include
)

# Link Test with required libraries
target_link_libraries(Test PRIVATE
    CudaPhashLib
    gtest
    gtest_main
    CUDA::cudart
    ${CUDA_CUDADEVRT_LIBRARY}
    CUDA::nvjpeg
    CUDA::cublas
    ${OPENCV_DIR}/x64/vc17/lib/opencv_world4110.lib
    ${OPENCV_DIR}/x64/vc17/lib/opencv_img_hash4110.lib
)

# Set compile definitions for Test
target_compile_definitions(Test PRIVATE
    NDEBUG
)

# Enable testing
enable_testing()
add_test(NAME CudaPHashTests COMMAND Test)

# Add custom target to match VS solution structure
add_custom_target(Solution ALL
    DEPENDS CudaPhashLib App Test
)

# Print configuration summary
message(STATUS "=== CudapHash CMake Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV directory: ${OPENCV_DIR}")
message(STATUS "Output directories:")
message(STATUS "  Runtime: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  Library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "  Archive: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")