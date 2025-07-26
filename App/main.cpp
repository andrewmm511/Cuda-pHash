//
//  main.cpp – CLI front-end for CUDA pHash library
//

#include "argparse.hpp"
#include "..\Cuda pHash\src\phash_cuda.cu"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

/* --- helpers --- */

static std::string toLower(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::unordered_set<std::string> parseExtensions(const std::string& csv)
{
    std::unordered_set<std::string> exts;
    size_t start = 0;
    while (start < csv.size()) {
        size_t pos = csv.find_first_of(",;", start);
        std::string token = csv.substr(start, pos == std::string::npos ? csv.npos : pos - start);
        token = toLower(token);
        if (!token.empty() && token.front() == '.') token.erase(0, 1);
        if (!token.empty()) exts.insert(token);
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return exts;
}

static std::vector<std::string> collectImagePaths(const fs::path& dir,
    const std::unordered_set<std::string>& allowed,
    bool recursive)
{
    std::vector<std::string> files;
    const auto opts = fs::directory_options::skip_permission_denied;

    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(dir, opts)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = toLower(entry.path().extension().string());
                if (!ext.empty() && ext.front() == '.') ext.erase(0, 1);
                if (allowed.empty() || allowed.contains(ext))
                    files.emplace_back(entry.path().string());
            }
        }
        else {
            for (const auto& entry : fs::directory_iterator(dir, opts)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = toLower(entry.path().extension().string());
                if (!ext.empty() && ext.front() == '.') ext.erase(0, 1);
                if (allowed.empty() || allowed.contains(ext))
                    files.emplace_back(entry.path().string());
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error scanning directory: " << e.what() << '\n';
    }

    std::sort(files.begin(), files.end());
    return files;
}

static bool queryYesNo(const std::string& prompt)
{
    std::cout << prompt << " [y/N] " << std::flush;
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return false;
    char c = static_cast<char>(std::tolower(line.front()));
    return c == 'y' || c == '1';
}

static std::string formatFileSize(std::uintmax_t bytes)
{
    const char* units[] = { "B", "KB", "MB", "GB", "TB" };
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

static std::string pHashToHex(const pHash& hash, int hashSize)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    int totalBits = hashSize * hashSize;
    if (totalBits <= 64) {
        // Only output the relevant hex digits
        int hexDigits = (totalBits + 3) / 4;  // Round up to nearest hex digit
        oss << std::setw(hexDigits) << (hash.words[0] >> (64 - totalBits));
    }
    else {
        // Use both words
        oss << std::setw(16) << hash.words[0];
        int remainingBits = totalBits - 64;
        int hexDigits = (remainingBits + 3) / 4;
        oss << std::setw(hexDigits) << (hash.words[1] >> (64 - remainingBits));
    }

    return oss.str();
}

static void saveHashesCSV(const std::string& path, const std::vector<std::string>& files,
    const std::vector<pHash>& hashes, int hashSize)
{
    std::ofstream csv(path);
    if (!csv) {
        std::cerr << "Warning: Cannot create output file '" << path << "'\n";
        return;
    }

    csv << "filepath,phash_hex\n";
    for (size_t i = 0; i < files.size() && i < hashes.size(); ++i) {
        csv << '"' << files[i] << "\"," << pHashToHex(hashes[i], hashSize) << '\n';
    }

    std::cout << "Hashes saved to " << path << '\n';
}

static void saveSimilarImagesCSV(const std::string& path, const std::vector<Image>& similar, int threshold)
{
    std::ofstream csv(path);
    if (!csv) {
        std::cerr << "Warning: Cannot create output file '" << path << "'\n";
        return;
    }

    csv << "image_file,similar_to,estimated_difference\n";
    for (const auto& img : similar) {
        // We don't have exact distance in the Image struct, but we know it's < threshold
        csv << '"' << img.path << "\",\"" << img.mostSimilarImage << "\"," << (threshold - 1) << '\n';
    }

    std::cout << "Similar images list saved to " << path << '\n';
}

static void printConfiguration(int hashSize, int freqFactor, int batchSize, int threads, int threshold = -1)
{
    std::cout << "Configuration:\n";
    std::cout << "  Hash size: " << hashSize << "x" << hashSize
        << " (" << (hashSize * hashSize) << " bits)\n";
    std::cout << "  Image size for analysis: " << (hashSize * freqFactor) << "x" << (hashSize * freqFactor)
        << " pixels\n";
    std::cout << "  GPU batch size: " << batchSize << " images\n";
    std::cout << "  CPU threads: " << (threads == -1 ? "auto-detect" : std::to_string(threads)) << "\n";
    if (threshold >= 0) {
        std::cout << "  Similarity threshold: " << threshold
            << " (lower = more similar, 0 = identical)\n";
    }
    std::cout << "\n";
}

/* --- command handlers --- */

static int handleHashCommand(argparse::ArgumentParser& program)
{
    // Extract options
    const fs::path directory = program.get<std::string>("--directory");
    const bool recursive = program.get<bool>("--recursive");
    const auto extensions = parseExtensions(program.get<std::string>("--extensions"));
    const int hashSize = program.get<int>("--hash-size");
    const int freqFactor = program.get<int>("--freq-factor");
    const int batchSize = program.get<int>("--batch-size");
    const int threads = program.get<int>("--threads");
    const int prefetchFactor = program.get<int>("--prefetch-factor");
    const int logLevel = program.get<int>("--log-level");
    const std::string outputPath = program.get<std::string>("--output");

    // Validate directory
    std::error_code ec;
    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        std::cerr << "Error: '" << directory.string() << "' is not a valid directory\n";
        return 1;
    }

    // Validate hash size
    if (hashSize < 5 || hashSize > 11) {
        std::cerr << "Error: Hash size must be between 5 and 11\n";
        return 1;
    }

    // Collect files
    std::cout << "Searching for images in " << directory.string();
    if (recursive) std::cout << " (recursive)";
    std::cout << "...\n";

    const auto filePaths = collectImagePaths(directory, extensions, recursive);
    if (filePaths.empty()) {
        std::cout << "No images found.\n";
        return 0;
    }

    std::cout << "Found " << filePaths.size() << " image(s)\n\n";

    // Compute hashes
    try {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Computing perceptual hashes...\n";
        printConfiguration(hashSize, freqFactor, batchSize, threads);

        CudaPhash phash(hashSize, freqFactor, batchSize, threads, prefetchFactor, logLevel);
        const auto hashes = phash.computeHashes(filePaths);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nCompleted in " << duration.count() / 1000.0 << " seconds\n";
        std::cout << "Successfully computed " << hashes.size() << " hashes\n";

        // Save to CSV if requested
        if (!outputPath.empty()) {
            saveHashesCSV(outputPath, filePaths, hashes, hashSize);
        }

        // Display first few hashes as examples
        std::cout << "\nHashes:\n";
        for (size_t i = 0; i < std::min<size_t>(5, hashes.size()); ++i) {
            std::cout << "  " << fs::path(filePaths[i]).filename().string()
                << ": " << pHashToHex(hashes[i], hashSize) << '\n';
        }
        if (hashes.size() > 5) {
            std::cout << "  ... and " << (hashes.size() - 5) << " more\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

static int handleSimilarCommand(argparse::ArgumentParser& program)
{
    // Extract options
    const fs::path directory = program.get<std::string>("--directory");
    const bool recursive = program.get<bool>("--recursive");
    const auto extensions = parseExtensions(program.get<std::string>("--extensions"));
    const int hashSize = program.get<int>("--hash-size");
    const int freqFactor = program.get<int>("--freq-factor");
    const int batchSize = program.get<int>("--batch-size");
    const int threads = program.get<int>("--threads");
    const int prefetchFactor = program.get<int>("--prefetch-factor");
    const int logLevel = program.get<int>("--log-level");
    const int threshold = program.get<int>("--threshold");
    const int numTables = program.get<int>("--num-tables");
    const int bitsPerTable = program.get<int>("--bits-per-table");
    const bool autoDelete = program.get<bool>("--auto-delete");
    const bool interactive = program.get<bool>("--interactive");
    const bool printOnly = program.get<bool>("--print-only");
    const bool dryRun = program.get<bool>("--dry-run");
    const std::string outputPath = program.get<std::string>("--output");

    // Validate directory
    std::error_code ec;
    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        std::cerr << "Error: '" << directory.string() << "' is not a valid directory\n";
        return 1;
    }

    // Validate hash size
    if (hashSize < 5 || hashSize > 11) {
        std::cerr << "Error: Hash size must be between 5 and 11\n";
        return 1;
    }

    // Check conflicting options
    if (autoDelete && interactive) {
        std::cerr << "Error: Cannot use both --auto-delete and --interactive\n";
        return 1;
    }

    // Collect files
    std::cout << "Searching for images in " << directory.string();
    if (recursive) std::cout << " (recursive)";
    std::cout << "...\n";

    const auto filePaths = collectImagePaths(directory, extensions, recursive);
    if (filePaths.empty()) {
        std::cout << "No images found.\n";
        return 0;
    }

    std::cout << "Found " << filePaths.size() << " image(s)\n\n";

    // Find similar images
    try {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Finding visually similar images...\n";
        printConfiguration(hashSize, freqFactor, batchSize, threads, threshold);

        if (dryRun) {
            std::cout << "(DRY RUN - no files will be deleted)\n\n";
        }

        CudaPhash phash(hashSize, freqFactor, batchSize, threads, prefetchFactor, logLevel);
        auto similar = phash.findDuplicatesGPU(filePaths, threshold, numTables, bitsPerTable);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nCompleted in " << duration.count() / 1000.0 << " seconds\n";
        std::cout << "Found " << similar.size() << " visually similar image(s)\n\n";

        if (similar.empty()) {
            std::cout << "No similar images found.\n";
            return 0;
        }

        // Save to CSV if requested
        if (!outputPath.empty()) {
            saveSimilarImagesCSV(outputPath, similar, threshold);
        }

        // Calculate potential space savings
        std::uintmax_t totalSize = 0;
        for (const auto& img : similar) {
            std::error_code ec;
            auto size = fs::file_size(img.path, ec);
            if (!ec) totalSize += size;
        }

        // Display similar images
        std::cout << "Visually similar images:\n";
        std::cout << std::string(80, '-') << '\n';

        size_t displayCount = printOnly ? similar.size() : std::min<size_t>(20, similar.size());
        for (size_t i = 0; i < displayCount; ++i) {
            std::cout << "Remove:     " << similar[i].path << '\n';
            std::cout << "Similar to: " << similar[i].mostSimilarImage << '\n';
            std::cout << '\n';
        }

        if (displayCount < similar.size()) {
            std::cout << "... and " << (similar.size() - displayCount) << " more similar images\n\n";
        }

        std::cout << "Total similar images that could be removed: " << similar.size() << '\n';
        std::cout << "Potential space savings: " << formatFileSize(totalSize) << '\n';

        // Handle deletion
        if (printOnly || (!autoDelete && !interactive)) {
            std::cout << "\nNo files deleted (use --auto-delete or --interactive to remove similar images)\n";
            return 0;
        }

        if (dryRun) {
            std::cout << "\nDRY RUN: Would delete " << similar.size() << " files\n";
            return 0;
        }

        // Auto-delete mode
        if (autoDelete) {
            if (!queryYesNo("\nDelete " + std::to_string(similar.size()) + " similar images?")) {
                std::cout << "Deletion cancelled.\n";
                return 0;
            }

            size_t deleted = 0;
            std::uintmax_t freedSpace = 0;

            for (const auto& img : similar) {
                std::error_code ec;
                auto size = fs::file_size(img.path, ec);
                if (!ec && fs::remove(img.path, ec)) {
                    deleted++;
                    freedSpace += size;
                }
                else if (ec) {
                    std::cerr << "Error deleting '" << img.path << "': " << ec.message() << '\n';
                }
            }

            std::cout << "\nDeleted " << deleted << " file(s)\n";
            std::cout << "Freed " << formatFileSize(freedSpace) << " of disk space\n";
        }
        // Interactive mode
        else if (interactive) {
            size_t deleted = 0;
            std::uintmax_t freedSpace = 0;

            std::cout << "\nInteractive deletion mode:\n";

            for (const auto& img : similar) {
                std::cout << "\nImage:      " << img.path << '\n';
                std::cout << "Similar to: " << img.mostSimilarImage << '\n';

                std::error_code ec;
                auto size = fs::file_size(img.path, ec);
                if (!ec) {
                    std::cout << "Size: " << formatFileSize(size) << '\n';
                }

                std::string response;
                std::cout << "Delete this file? ([y]es/[n]o/[q]uit): " << std::flush;
                std::getline(std::cin, response);

                if (response.empty()) response = "n";
                char c = static_cast<char>(std::tolower(response[0]));

                if (c == 'q') {
                    std::cout << "Quitting...\n";
                    break;
                }
                else if (c == 'y') {
                    if (fs::remove(img.path, ec)) {
                        deleted++;
                        if (size > 0) freedSpace += size;
                        std::cout << "Deleted.\n";
                    }
                    else {
                        std::cerr << "Error deleting file: " << ec.message() << '\n';
                    }
                }
                else {
                    std::cout << "Kept.\n";
                }
            }

            std::cout << "\nSummary:\n";
            std::cout << "Deleted " << deleted << " file(s)\n";
            std::cout << "Freed " << formatFileSize(freedSpace) << " of disk space\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

/* --- main --- */

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("phash", "1.0");
    program.add_description("CUDA-accelerated perceptual hash calculator for finding visually similar images");
    program.add_epilog("Examples:\n"
        "  phash hash -d ./photos -o hashes.csv\n"
        "  phash similar -d ./photos -t 3 --interactive");

    // Add subcommands
    argparse::ArgumentParser hash_command("hash");
    hash_command.add_description("Compute perceptual hashes for images");

    argparse::ArgumentParser similar_command("similar");
    similar_command.add_description("Find visually similar images");

    // Common arguments for both commands
    auto addCommonArgs = [](argparse::ArgumentParser& cmd) {
        cmd.add_argument("-d", "--directory")
            .required()
            .help("Directory containing images to process");

        cmd.add_argument("-e", "--extensions")
            .default_value(std::string("jpg,jpeg"))
            .help("Image file extensions to include (comma-separated)");

        cmd.add_argument("-r", "--recursive")
            .implicit_value(true)
            .default_value(false)
            .help("Search directories recursively for images");

        cmd.add_argument("-b", "--batch-size")
            .default_value(500)
            .scan<'i', int>()
            .help("Number of images to process in each GPU batch");

        cmd.add_argument("-T", "--threads")
            .default_value(-1)
            .scan<'i', int>()
            .help("Number of CPU threads for file I/O (-1 = auto-detect)");

        cmd.add_argument("--prefetch-factor")
            .default_value(8)
            .scan<'i', int>()
            .help("Multiplier for I/O queue size (higher = faster but more memory)");

        cmd.add_argument("-l", "--log-level")
            .default_value(4)
            .scan<'i', int>()
            .help("Internal logging verbosity (4=errors only, 3=info, 2=debug, 1=trace)");

        cmd.add_argument("-hs", "--hash-size")
            .default_value(8)
            .scan<'i', int>()
            .help("Hash dimensions in bits (5-11, default 8 = 64-bit hash)");

        cmd.add_argument("-f", "--freq-factor")
            .default_value(4)
            .scan<'i', int>()
            .help("Frequency oversampling factor (higher = more accurate but slower)");
        };

    // Add common arguments to both commands
    addCommonArgs(hash_command);
    addCommonArgs(similar_command);

    // Hash-specific arguments
    hash_command.add_argument("-o", "--output")
        .default_value(std::string(""))
        .help("Save computed hashes to CSV file");

    // Similar-specific arguments
    similar_command.add_argument("-t", "--threshold")
        .default_value(5)
        .scan<'i', int>()
        .help("Similarity threshold (0=exact duplicate, 10=entirely different)");

    similar_command.add_argument("--num-tables")
        .default_value(32)
        .scan<'i', int>()
        .help("Number of hash tables for similarity search (advanced)");

    similar_command.add_argument("--bits-per-table")
        .default_value(8)
        .scan<'i', int>()
        .help("Bits sampled per hash table (advanced)");

    similar_command.add_argument("-a", "--auto-delete")
        .implicit_value(true)
        .default_value(false)
        .help("Automatically delete similar images (with confirmation prompt)");

    similar_command.add_argument("-i", "--interactive")
        .implicit_value(true)
        .default_value(false)
        .help("Review and confirm each deletion individually");

    similar_command.add_argument("-p", "--print-only")
        .implicit_value(true)
        .default_value(false)
        .help("Only display similar images without deleting");

    similar_command.add_argument("--dry-run")
        .implicit_value(true)
        .default_value(false)
        .help("Show what would be deleted without actually deleting");

    similar_command.add_argument("-o", "--output")
        .default_value(std::string(""))
        .help("Save list of similar images to CSV file");

    // Add subparsers
    program.add_subparser(hash_command);
    program.add_subparser(similar_command);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        std::cerr << program;
        return 1;
    }

    // Handle subcommands
    if (program.is_subcommand_used("hash")) {
        return handleHashCommand(hash_command);
    }
    else if (program.is_subcommand_used("similar")) {
        return handleSimilarCommand(similar_command);
    }
    else {
        std::cerr << "No command specified. Use 'hash' or 'similar'\n";
        std::cerr << program;
        return 1;
    }
}