# Cuda pHash

set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/memory_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/phash_cuda.cu
)

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/progress_tracker.cpp
)

set(PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/phash_cuda.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/progress_tracker.hpp
)

set(PRIVATE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logger.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/memory_manager.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/work_queue.hpp
)

add_library(CudaPhashLib STATIC)
add_library(CudaPHash::CudaPhashLib ALIAS CudaPhashLib)

target_sources(CudaPhashLib
    PRIVATE
        ${CUDA_SOURCES}
        ${CPP_SOURCES}
        ${PRIVATE_HEADERS}
    PUBLIC
        FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES ${PUBLIC_HEADERS}
)

target_include_directories(CudaPhashLib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link to common options interface (only during build, not for export)
target_link_libraries(CudaPhashLib
    PRIVATE
        $<BUILD_INTERFACE:CudaPHash::CommonOptions>
)

set_target_properties(CudaPhashLib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${CUDAPHASH_CUDA_ARCHITECTURES}"
)

if(CUDAPHASH_ENABLE_LTO AND CUDAPHASH_IPO_SUPPORTED)
    set_target_properties(CudaPhashLib PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE
    )
endif()

option(CUDAPHASH_STATIC_CUDART "Link CUDA runtime statically" ON)
set(CUDART_TARGET $<IF:$<BOOL:${CUDAPHASH_STATIC_CUDART}>,CUDA::cudart_static,CUDA::cudart>)

target_link_libraries(CudaPhashLib 
    PUBLIC
        ${CUDART_TARGET}
        CUDA::cuda_driver
        CUDA::nvjpeg
        CUDA::cublas
)

install(TARGETS CudaPhashLib
    EXPORT CudaPHashTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudaphash
)