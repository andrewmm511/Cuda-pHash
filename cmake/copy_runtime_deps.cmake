# cmake/copy_runtime_deps.cmake
# Arguments passed via -D:
#   TARGET_FILE: full path to the exe or dll to scan
#   DEST_DIR:    directory to copy resolved DLLs into
#   EXTRA_DIRS:  optional semicolon list of directories to search (e.g. CUDA bin)

if(NOT DEFINED TARGET_FILE OR NOT EXISTS "${TARGET_FILE}")
  message(FATAL_ERROR "TARGET_FILE not found: ${TARGET_FILE}")
endif()
if(NOT DEFINED DEST_DIR)
  message(FATAL_ERROR "DEST_DIR not set")
endif()

set(_dirs)
if(DEFINED EXTRA_DIRS AND NOT EXTRA_DIRS STREQUAL "")
  list(APPEND _dirs ${EXTRA_DIRS})
endif()

# Scan runtime deps of TARGET_FILE and resolve DLLs.
file(GET_RUNTIME_DEPENDENCIES
  RESOLVED_DEPENDENCIES_VAR _deps
  UNRESOLVED_DEPENDENCIES_VAR _unresolved
  EXECUTABLES "${TARGET_FILE}"
  DIRECTORIES ${_dirs}
  PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
  POST_EXCLUDE_REGEXES ".*[\\/]system32[\\/].*\\.dll"
)

if(_unresolved)
  message(WARNING "Unresolved runtime deps for ${TARGET_FILE}: ${_unresolved}")
endif()

if(NOT _deps)
  message(STATUS "No non-system runtime DLLs to copy for ${TARGET_FILE}")
  return()
endif()

foreach(dll IN LISTS _deps)
  file(COPY "${dll}" DESTINATION "${DEST_DIR}")
endforeach()

message(STATUS "Copied ${_deps}; destination: ${DEST_DIR}")
