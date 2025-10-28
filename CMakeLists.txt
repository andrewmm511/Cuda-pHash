cmake_minimum_required(VERSION 3.31)
project(CudaPHash VERSION 0.1.0 LANGUAGES CXX CUDA)

cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0077 NEW)

include(GNUInstallDirs)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDAPHASH_CUDA_ARCHITECTURES
    "86-real;89-real;90-real;90-virtual"
    CACHE STRING "CUDA architectures to build for" FORCE
)
set(CMAKE_CUDA_ARCHITECTURES ${CUDAPHASH_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

option(CUDAPHASH_STATIC_MSVC_RUNTIME "Link MSVC runtime statically" ON)
if(MSVC AND CUDAPHASH_STATIC_MSVC_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(CUDAPHASH_ENABLE_LTO "Enable Link Time Optimization" ON)
option(CUDAPHASH_LTO_EXECUTABLE_ONLY "Apply LTO only to executables, not static libraries" ON)
option(CUDAPHASH_ENABLE_NATIVE_ARCH "Build for native CPU architecture" OFF)
option(CUDAPHASH_ENABLE_FAST_MATH "Enable fast math optimizations" ON)
option(CUDAPHASH_ENABLE_PARALLEL_COMPILE "Enable parallel compilation" ON)
option(CUDA_VERBOSE_PTXAS "Show CUDA PTX assembly statistics for optimization" OFF)


if(CUDAPHASH_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CUDAPHASH_IPO_SUPPORTED OUTPUT CUDAPHASH_IPO_ERROR)
    if(CUDAPHASH_IPO_SUPPORTED)
        message(STATUS "Link Time Optimization (IPO/LTO) enabled")
        if(MSVC)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /CGTHREADS:8 /LTCG:INCREMENTAL")
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /HEAP:0x80000000 /OPT:REF /OPT:ICF=3")
            else()
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /HEAP:0x40000000")
            endif()
        endif()
    else()
        message(STATUS "Link Time Optimization not supported: ${CUDAPHASH_IPO_ERROR}")
        set(CUDAPHASH_ENABLE_LTO OFF)
    endif()
endif()

if(MSVC AND CUDAPHASH_ENABLE_PARALLEL_COMPILE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)
    set(CMAKE_VS_GLOBALS "UseMultiToolTask=true" "EnforceProcessCountAcrossBuilds=true")
endif()

include(CheckCXXCompilerFlag)
option(CUDAPHASH_ENABLE_AVX2 "Enable AVX2 instructions if available" ON)

if(CUDAPHASH_ENABLE_AVX2)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        set(AVX2_FLAG "/arch:AVX2")
    else()
        check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
        set(AVX2_FLAG "-mavx2 -mfma")
    endif()

    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2 support detected - enabling AVX2 optimizations")
        set(CUDAPHASH_AVX2_AVAILABLE ON)
    else()
        message(STATUS "AVX2 not supported by compiler - using default instruction set")
        set(CUDAPHASH_AVX2_AVAILABLE OFF)
    endif()
else()
    set(CUDAPHASH_AVX2_AVAILABLE OFF)
endif()

option(ENABLE_CCACHE "Enable ccache/sccache for build caching" ON)
if(ENABLE_CCACHE)
    find_program(CCACHE_PROGRAM NAMES sccache ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "Using ${CCACHE_PROGRAM} for build caching")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else()
        message(STATUS "Build caching requested but ccache/sccache not found")
    endif()
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(cudaphash_common_options INTERFACE)
add_library(CudaPHash::CommonOptions ALIAS cudaphash_common_options)

target_compile_options(cudaphash_common_options INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
        $<$<CONFIG:Release>:/O2 /Ob3 /Oi /Ot /Oy /GL /Gy /GF /Gw /fp:fast /Qpar /favor:blend /Qfast_transcendentals /Zc:inline
            $<$<BOOL:${CUDAPHASH_AVX2_AVAILABLE}>:${AVX2_FLAG}>>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Ob2 /Oi /Ot /fp:fast
            $<$<BOOL:${CUDAPHASH_AVX2_AVAILABLE}>:${AVX2_FLAG}>>
        $<$<CONFIG:MinSizeRel>:/O1 /Ob1 /Os /Oy /GL /Gy /GF /Gw>
        $<$<CONFIG:Debug>:/Od /RTC1 /JMC /sdl /GS>
        /W3 /permissive- /Zc:__cplusplus /Zc:inline /Zc:throwingNew
    >
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
        $<$<CONFIG:Release>:-O3 -ffast-math -funroll-loops -ftree-vectorize
            $<$<BOOL:${CUDAPHASH_ENABLE_NATIVE_ARCH}>:-march=native>
            $<$<BOOL:${CUDAPHASH_AVX2_AVAILABLE}>:${AVX2_FLAG}>>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g
            $<$<BOOL:${CUDAPHASH_AVX2_AVAILABLE}>:${AVX2_FLAG}>>
        $<$<CONFIG:MinSizeRel>:-Os>
        $<$<CONFIG:Debug>:-O0 -g3>
        -Wall -Wextra -Wpedantic
    >
)

target_compile_options(cudaphash_common_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
        --threads 0
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:
            --use_fast_math
            --ftz=true
            --prec-div=false
            --prec-sqrt=false
            --fmad=true
            -O3
            $<$<CONFIG:Release>:--extra-device-vectorization>
            $<$<CONFIG:Release>:--optimize-float-atomics>
            $<$<CONFIG:Release>:--split-compile 8>
        >
        $<$<CONFIG:RelWithDebInfo>:-lineinfo>
        $<$<CONFIG:Debug>:-G --debug -O0>
        $<$<BOOL:${CUDA_VERBOSE_PTXAS}>:--ptxas-options=-v>
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/bigobj -Xcompiler=/fp:fast>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=-ffast-math>
    >
)

target_compile_definitions(cudaphash_common_options INTERFACE
    $<$<PLATFORM_ID:Windows>:WIN32 _WIN64 NOMINMAX>
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
    $<$<CONFIG:MinSizeRel>:NDEBUG>
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# For multi-config generators (Visual Studio)
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/${CONFIG}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/${CONFIG}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/${CONFIG}/lib)
endforeach()

option(BUILD_TESTS "Build test suite" ON)
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Code coverage enabled")
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
elseif(ENABLE_COVERAGE AND MSVC)
    message(WARNING "Code coverage not supported for MSVC")
endif()

add_subdirectory(CudapHash)
add_subdirectory(App)

if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    add_subdirectory(Test)
endif()

add_custom_target(cudaphash_all ALL
    DEPENDS CudaPhashLib App
    $<$<BOOL:${BUILD_TESTS}>:Tests>
    COMMENT "Building all CudaPHash components"
)

install(EXPORT CudaPHashTargets
    FILE CudaPHashTargets.cmake
    NAMESPACE CudaPHash::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaPHash
    COMPONENT Development
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CudaPHashConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CudaPHashConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CudaPHashConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CudaPHashConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CudaPHashConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaPHash
    COMPONENT Development
)

# --- CPack Configuration ---

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"
   OR (CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE))
    set(CPACK_PACKAGE_NAME "CudaPHash")
    set(CPACK_PACKAGE_VENDOR "Andrew Martin")
    set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

    set(CPACK_PACKAGE_INSTALL_DIRECTORY "CudaPHash")
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")

    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_PACKAGE_NAME "CudaPHash ${PROJECT_VERSION}")
    set(CPACK_NSIS_DISPLAY_NAME "CudaPHash ${PROJECT_VERSION}")

    set(CPACK_COMPONENTS_ALL Runtime Development)
    set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "CudaPHash Application")
    set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "The CudaPHash executable and required runtime libraries")
    set(CPACK_COMPONENT_RUNTIME_REQUIRED ON)

    set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development Files")
    set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Headers and CMake files for using CudaPHash as a library")
    set(CPACK_COMPONENT_DEVELOPMENT_DISABLED ON)
    set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Runtime)

    if(WIN32)
        set(CPACK_GENERATOR "NSIS;ZIP")
    else()
        set(CPACK_GENERATOR "TGZ")
    endif()

    set(CPACK_NSIS_COMPONENT_INSTALL OFF)

    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-win64")

    include(CPack)
endif()