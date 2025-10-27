# App

set(APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arguments.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hash_command.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/similar_command.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/save_results.cpp
)

add_executable(App ${APP_SOURCES})
set_target_properties(App PROPERTIES OUTPUT_NAME "cudaphash")

target_include_directories(App
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(App
    PRIVATE
        CudaPHash::CudaPhashLib
        $<BUILD_INTERFACE:CudaPHash::CommonOptions>
)

add_custom_command(TARGET App POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -D TARGET_FILE=$<TARGET_FILE:App>
          -D DEST_DIR=$<TARGET_FILE_DIR:App>
          -D EXTRA_DIRS=${CUDAToolkit_BIN_DIR}
          -P ${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake
)

target_compile_definitions(App
    PRIVATE
        _CONSOLE
)

if(CUDAPHASH_ENABLE_LTO AND CUDAPHASH_IPO_SUPPORTED)
    set_target_properties(App PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE
    )
endif()

install(TARGETS App
    EXPORT CudaPHashTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)

install(TARGETS App
    RUNTIME_DEPENDENCIES
	DIRECTORIES "${CUDAToolkit_BIN_DIR}"
    PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
    POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
    COMPONENT Runtime
)