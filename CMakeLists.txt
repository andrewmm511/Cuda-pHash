cmake_minimum_required(VERSION 3.18)
project(CudaPHash LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find cudadevrt library manually
find_library(CUDA_CUDADEVRT_LIBRARY cudadevrt 
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86)

# Global compile options for Release mode
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

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin/Release/x64)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin/Release/x64)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib/Release/x64)

# Define the CudapHash directory with proper path handling
set(CUDA_PHASH_DIR "${CMAKE_SOURCE_DIR}/CudapHash")
file(TO_CMAKE_PATH "${CUDA_PHASH_DIR}" CUDA_PHASH_DIR)

# CudapHash static library
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

target_include_directories(CudaPhashLib PRIVATE
    "${CUDA_PHASH_DIR}/src"
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

# Link CUDA libraries
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

###########
#  TESTS  #
###########

add_subdirectory(third_party/googletest)

# Create a test executable that includes all test files
add_executable(Tests
    "Test/src/WorkQueueTests.cpp"
    "Test/src/Unit.cpp"
)

target_include_directories(Tests PRIVATE
    "${CUDA_PHASH_DIR}/include"
    "${CUDA_PHASH_DIR}/src"
    ${CMAKE_SOURCE_DIR}/Test/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(Tests PRIVATE
    CudaPhashLib
    gtest
    gtest_main
    Threads::Threads
    CUDA::cudart
    CUDA::nvjpeg
)

# Keep the WorkQueueTests target for backwards compatibility
add_executable(WorkQueueTests
    "Test/src/WorkQueueTests.cpp"
)

target_include_directories(WorkQueueTests PRIVATE
    "${CUDA_PHASH_DIR}/include"
    "${CUDA_PHASH_DIR}/src"
    ${CMAKE_SOURCE_DIR}/Test/include
)

target_link_libraries(WorkQueueTests PRIVATE
    gtest
    gtest_main
    Threads::Threads
)

enable_testing()

# For multi-config generators (e.g., Visual Studio), reference the target file so CTest
# can locate the correct config binary. Invoke ctest with -C <Debug|Release>.
add_test(NAME WorkQueueTests COMMAND $<TARGET_FILE:WorkQueueTests>)
add_test(NAME AllTests COMMAND $<TARGET_FILE:Tests>)

# Aggregate target
add_custom_target(Solution ALL
    DEPENDS CudaPhashLib App WorkQueueTests Tests
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
