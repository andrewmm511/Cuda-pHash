#pragma once

#include <filesystem>
#include <string>

namespace phash_app {

namespace defaults {
    constexpr const char* DEFAULT_EXTENSIONS = "jpg,jpeg";
    constexpr const char* DEFAULT_OUTPUT = "";

    constexpr int BATCH_SIZE = 500;
    constexpr int THREADS = -1;
    constexpr int PREFETCH_FACTOR = 8;
    constexpr int LOG_LEVEL = 4;
    constexpr int HASH_SIZE = 8;
    constexpr int FREQ_FACTOR = 4;
    constexpr bool RECURSIVE = false;

    // Similar-specific defaults
    constexpr int THRESHOLD = 5;
    constexpr int NUM_TABLES = 32;
    constexpr int BITS_PER_TABLE = 8;
    constexpr bool AUTO_DELETE = false;
    constexpr bool INTERACTIVE = false;
    constexpr bool PRINT_ONLY = false;
    constexpr bool DRY_RUN = false;
}

struct Arguments {
    std::filesystem::path directory;
    bool recursive;
    std::string extensions;
    int hashSize;
    int freqFactor;
    int batchSize;
    int threads;
    int prefetchFactor;
    int logLevel;
    std::string outputPath;

    // Similar-specific arguments
    int threshold;
    int numTables;
    int bitsPerTable;
    bool autoDelete;
    bool interactive;
    bool printOnly;
    bool dryRun;
};

} // namespace phash_app