#pragma once

#include <filesystem>
#include <string>
#include <vector>

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

    constexpr int THRESHOLD = 5;
    constexpr int NUM_TABLES = 32;
    constexpr int BITS_PER_TABLE = 8;
    constexpr bool AUTO_DELETE = false;
    constexpr bool INTERACTIVE = false;
    constexpr bool PRINT_ONLY = false;
    constexpr bool DRY_RUN = false;
}

struct RawArguments {
    std::filesystem::path directory;
    bool recursive = defaults::RECURSIVE;
    std::string extensions = defaults::DEFAULT_EXTENSIONS;
    int hashSize = defaults::HASH_SIZE;
    int freqFactor = defaults::FREQ_FACTOR;
    int batchSize = defaults::BATCH_SIZE;
    int threads = defaults::THREADS;
    int prefetchFactor = defaults::PREFETCH_FACTOR;
    int logLevel = defaults::LOG_LEVEL;
    std::string outputPath = defaults::DEFAULT_OUTPUT;

    int threshold = defaults::THRESHOLD;
    int numTables = defaults::NUM_TABLES;
    int bitsPerTable = defaults::BITS_PER_TABLE;
    bool autoDelete = defaults::AUTO_DELETE;
    bool interactive = defaults::INTERACTIVE;
    bool printOnly = defaults::PRINT_ONLY;
    bool dryRun = defaults::DRY_RUN;
};

class Arguments {
public:
    enum class Command { Hash, Similar };

    Arguments(const RawArguments& raw, Command command);

    const std::filesystem::path directory;
    const bool recursive;
    const std::vector<std::string> extensions;
    const int hashSize;
    const int freqFactor;
    const int batchSize;
    const int threads;
    const int prefetchFactor;
    const int logLevel;
    const std::string outputPath;

    const int threshold;
    const int numTables;
    const int bitsPerTable;
    const bool autoDelete;
    const bool interactive;
    const bool printOnly;
    const bool dryRun;

};

} // namespace phash_app
