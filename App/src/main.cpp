//
//  CLI front-end for CUDA pHash library
//

#include "argparse.hpp"
#include "rang.hpp"
#include "indicators.hpp"
#include "phash_cuda.cuh"
#include "helpers.hpp"
#include "save_results.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;
using namespace rang;
using namespace phash_app;

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
        std::cerr << fg::red << "Error: '" << directory.string() << "' is not a valid directory\n" << fg::reset;
        return 1;
    }

    // Collect files
    const auto filePaths = collectImagePaths(directory, extensions, recursive);
    if (filePaths.empty()) {
        std::cerr << fg::red << "No images found.\n" << fg::reset;
        return 1;
    }

    // Compute hashes
    try {
        auto start = std::chrono::high_resolution_clock::now();

        printConfiguration(directory.string(), filePaths.size(), hashSize, freqFactor, batchSize, threads);

        hideCursor();

        std::array<indicators::ProgressBar, 5> bar_arr = {
            bar("Total  ", true, true),
            bar("Read   "),
            bar("Decode "),
            bar("Resize "),
            bar("Hash   ")
        };
        indicators::MultiProgress<indicators::ProgressBar, 5> bars(bar_arr[0], bar_arr[1], bar_arr[2], bar_arr[3], bar_arr[4]);

        // Create progress callback that updates the progress bar
        auto progressCallback = [&bars, &bar_arr](const ProgressInfo& info) {
            bars.set_progress<0>(info.percentComplete());
            bars.set_progress<1>(info.percentRead());
            bars.set_progress<2>(info.percentDecoded());
            bars.set_progress<3>(info.percentResized());
            bars.set_progress<4>(info.percentHashed());

            if (info.hasFailures()) {
                bar_arr[0].set_option(indicators::option::ForegroundColor{indicators::Color::yellow});
                bar_arr[0].set_option(indicators::option::PostfixText{ "(" + std::to_string(info.failedImages) + " failed image(s))" });
            }
        };

        CudaPhash phash(hashSize, freqFactor, batchSize, threads, prefetchFactor, logLevel, progressCallback);
        const auto hashes = phash.computeHashes(filePaths);

        showCursor();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << fg::green << "\nCompleted " << withCommas(hashes.size()) << " images in " << duration.count() / 1000.0 << " seconds\n\n" << fg::reset;

        // Report failures if any
        size_t failedCount = filePaths.size() - hashes.size();
        if (failedCount > 0) {
            std::cout << fg::yellow << "Warning: " << withCommas(failedCount) << " image(s) failed to process" << fg::reset << "\n";
        }

        // Save to CSV if requested
        if (!outputPath.empty()) {
            saveHashesCSV(outputPath, filePaths, hashes, hashSize);
        }

        // Display first few hashes as examples
        printHashResults(filePaths, hashes, outputPath, hashSize);

    }
    catch (const std::exception& e) {
        showCursor();  // Restore cursor on error
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
        printConfiguration(directory.string(), filePaths.size(), hashSize, freqFactor, batchSize, threads, threshold);

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
    program.add_description("CUDA-accelerated perceptual hash calculator");
    program.add_epilog("Examples:\n  phash hash -d ./photos -o hashes.csv\n  phash similar -d ./photos -t 3 --interactive\n\n"
                       "For detailed options: phash <command> --help");

    // Add subcommands
    argparse::ArgumentParser hash_command("hash");
    hash_command.add_description("Compute the perceptual hashes of images");

    argparse::ArgumentParser similar_command("similar");
    similar_command.add_description("Compute hashes and calculate visual similarity");

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

    // Validate arguments
    /*if (program.get<int>("--hash-size") < 5 || program.get<int>("--hash-size") > 11) {
        std::cerr << fg::red << "Error: Hash size must be between 5 and 11\n" << fg::reset;
        return 1;
    }*/

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