# Test

include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG        v1.17.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

set(APP_UNIT_TEST_SOURCES
    unit/App/arguments_test.cpp
    unit/App/helpers_test.cpp
)

set(CUDAPHASH_UNIT_TEST_SOURCES
    unit/CudapHash/kernels_test.cpp
    unit/CudapHash/memoryManager_test.cpp
    unit/CudapHash/workQueue_test.cpp
)

set(CUDA_TEST_SOURCES
    unit/CudapHash/kernels_test_wrapper.cu
)

set(APP_IMPLEMENTATION_SOURCES
    ../App/src/arguments.cpp
    ../App/src/helpers.cpp
    ../App/src/hash_command.cpp
    ../App/src/similar_command.cpp
    ../App/src/save_results.cpp
)

add_executable(Tests
    main.cpp
    ${APP_UNIT_TEST_SOURCES}
    ${CUDAPHASH_UNIT_TEST_SOURCES}
    ${CUDA_TEST_SOURCES}
    ${APP_IMPLEMENTATION_SOURCES}
)
set_target_properties(Tests PROPERTIES OUTPUT_NAME "cudaphash_tests")

target_include_directories(Tests PRIVATE
    $<TARGET_PROPERTY:CudaPHash::CudaPhashLib,INTERFACE_INCLUDE_DIRECTORIES>
    # Private headers (needed for testing internal components)
    ../CudapHash/src
    ../App/include
    include
    unit/CudapHash
)

target_link_libraries(Tests PRIVATE
    CudaPHash::CudaPhashLib
    $<BUILD_INTERFACE:CudaPHash::CommonOptions>
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main
    Threads::Threads
)

set_target_properties(Tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CUDAPHASH_CUDA_ARCHITECTURES}"
)

if(CUDAPHASH_ENABLE_LTO AND CUDAPHASH_IPO_SUPPORTED AND NOT CUDAPHASH_LTO_EXECUTABLE_ONLY)
    set_target_properties(Tests PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE
    )
endif()

target_compile_definitions(Tests
    PRIVATE
        _CONSOLE
        GTEST_LINKED_AS_SHARED_LIBRARY=0
)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Tests PRIVATE --coverage -O0 -g)
    target_link_options(Tests PRIVATE --coverage)
endif()

include(GoogleTest)

if(CMAKE_CONFIGURATION_TYPES)
    # For multi-config generators (Visual Studio), add test manually
    add_test(NAME AllTests 
        COMMAND Tests
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
    )
    set_tests_properties(AllTests PROPERTIES
        TIMEOUT 60
    )
else()
    # For single-config generators, use gtest_discover_tests
    gtest_discover_tests(Tests
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        PROPERTIES TIMEOUT 60
    )
endif()

# Custom target to run tests with output
add_custom_target(test-run
    COMMAND $<TARGET_FILE:Tests> --gtest_color=yes
    DEPENDS Tests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:Tests>
    COMMENT "Running unit tests..."
    USES_TERMINAL
)

# Custom target to run tests with detailed output
add_custom_target(test-verbose
    COMMAND $<TARGET_FILE:Tests> --gtest_color=yes --gtest_print_time=yes --gtest_output=xml:test_results.xml
    DEPENDS Tests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:Tests>
    COMMENT "Running unit tests with detailed output..."
    USES_TERMINAL
)

# Custom target using CTest
add_custom_target(test-check
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure --verbose
    DEPENDS Tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests via CTest..."
    USES_TERMINAL
)

# Set IDE folder for Visual Studio
set_target_properties(Tests PROPERTIES FOLDER "Tests")
set_target_properties(test-run test-verbose test-check PROPERTIES FOLDER "Tests/Commands")