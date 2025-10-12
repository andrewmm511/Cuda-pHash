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

namespace fs = std::filesystem;
using namespace rang;
using namespace phash_app;

struct Arguments {
    fs::path directory;
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

/* --- command handlers --- */

static int handleHashCommand(Arguments args)
{
    const auto filePaths = collectImagePaths(args.directory, args.extensions, args.recursive);

    // Compute hashes
    try {
        auto start = std::chrono::high_resolution_clock::now();

        printConfiguration(args.directory.string(), filePaths.size(), args.hashSize, args.freqFactor, args.batchSize, args.threads);

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
            bars.set_progress<0>(info.percentComplete(PipelineStage::All));
            bars.set_progress<1>(info.percentComplete(PipelineStage::Read));
            bars.set_progress<2>(info.percentComplete(PipelineStage::Decode));
            bars.set_progress<3>(info.percentComplete(PipelineStage::Resize));
            bars.set_progress<4>(info.percentComplete(PipelineStage::Hash));

            if (info.failedImages > 0) {
                bar_arr[0].set_option(indicators::option::ForegroundColor{indicators::Color::yellow});
                bar_arr[0].set_option(indicators::option::PostfixText{ "(" + std::to_string(info.failedImages) + " failed image(s))" });
            }
        };

        CudaPhash phash(args.hashSize, args.freqFactor, args.batchSize, args.threads, args.prefetchFactor, args.logLevel, progressCallback);
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
        if (!args.outputPath.empty()) {
            saveHashesCSV(args.outputPath, filePaths, hashes, args.hashSize);
        }

        // Display first few hashes as examples
        printHashResults(filePaths, hashes, args.outputPath, args.hashSize);

    }
    catch (const std::exception& e) {
        showCursor();  // Restore cursor on error
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

static int handleSimilarCommand(Arguments args)
{
    // Check conflicting options
    if (args.autoDelete && args.interactive) {
        std::cerr << "Error: Cannot use both --auto-delete and --interactive\n";
        return 1;
    }

    // Collect files
    std::cout << "Searching for images in " << args.directory.string();
    if (args.recursive) std::cout << " (recursive)";
    std::cout << "...\n";

    const auto filePaths = collectImagePaths(args.directory, args.extensions, args.recursive);

    std::cout << "Found " << filePaths.size() << " image(s)\n\n";

    // Find similar images
    try {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Finding visually similar images...\n";
        printConfiguration(args.directory.string(), filePaths.size(), args.hashSize, args.freqFactor, args.batchSize, args.threads, args.threshold);

        if (args.dryRun) {
            std::cout << "(DRY RUN - no files will be deleted)\n\n";
        }

        CudaPhash phash(args.hashSize, args.freqFactor, args.batchSize, args.threads, args.prefetchFactor, args.logLevel);
        auto similar = phash.findDuplicatesGPU(filePaths, args.threshold, args.numTables, args.bitsPerTable);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nCompleted in " << duration.count() / 1000.0 << " seconds\n";
        std::cout << "Found " << similar.size() << " visually similar image(s)\n\n";

        if (similar.empty()) {
            std::cout << "No similar images found.\n";
            return 0;
        }

        // Save to CSV if requested
        if (!args.outputPath.empty()) {
            saveSimilarImagesCSV(args.outputPath, similar, args.threshold);
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

        size_t displayCount = args.printOnly ? similar.size() : std::min<size_t>(20, similar.size());
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
        if (args.printOnly || (!args.autoDelete && !args.interactive)) {
            std::cout << "\nNo files deleted (use --auto-delete or --interactive to remove similar images)\n";
            return 0;
        }

        if (args.dryRun) {
            std::cout << "\nDRY RUN: Would delete " << similar.size() << " files\n";
            return 0;
        }

        // Auto-delete mode
        if (args.autoDelete) {
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
        else if (args.interactive) {
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

    Arguments args;

    // Add subcommands
    argparse::ArgumentParser hash_command("hash");
    hash_command.add_description("Compute the perceptual hashes of images");

    argparse::ArgumentParser similar_command("similar");
    similar_command.add_description("Compute hashes and calculate visual similarity");

    // Common arguments for both commands
    auto addCommonArgs = [&](argparse::ArgumentParser& cmd) {
        cmd.add_argument("-d", "--directory")
            .required()
            .store_into(args.directory)
            .help("Directory containing images to process");

        cmd.add_argument("-e", "--extensions")
            .default_value(defaults::DEFAULT_EXTENSIONS)
			.store_into(args.extensions)
            .help("Image file extensions to include (comma-separated)");

        cmd.add_argument("-r", "--recursive")
            .implicit_value(true)
            .default_value(defaults::RECURSIVE)
			.store_into(args.recursive)
            .help("Search directories recursively for images");

        cmd.add_argument("-b", "--batch-size")
			.default_value(defaults::BATCH_SIZE)
            .scan<'i', int>()
            .store_into(args.batchSize)
            .help("Number of images to process in each GPU batch");

        cmd.add_argument("-T", "--threads")
			.default_value(defaults::THREADS)
            .scan<'i', int>()
			.store_into(args.threads)
            .help("Number of CPU threads for file I/O (-1 = auto-detect)");

        cmd.add_argument("--prefetch-factor")
			.default_value(defaults::PREFETCH_FACTOR)
            .scan<'i', int>()
			.store_into(args.prefetchFactor)
            .help("Multiplier for I/O queue size (higher = faster but more memory)");

        cmd.add_argument("-l", "--log-level")
            .default_value(defaults::LOG_LEVEL)
            .scan<'i', int>()
			.store_into(args.logLevel)
            .help("Internal logging verbosity (4=errors only, 3=info, 2=debug, 1=trace)");

        cmd.add_argument("-hs", "--hash-size")
			.default_value(defaults::HASH_SIZE)
            .scan<'i', int>()
			.store_into(args.hashSize)
            .help("Hash dimensions in bits (5-11, default 8 = 64-bit hash)");

        cmd.add_argument("-f", "--freq-factor")
			.default_value(defaults::FREQ_FACTOR)
            .scan<'i', int>()
			.store_into(args.freqFactor)
            .help("Frequency oversampling factor (higher = more accurate but slower)");
        };

    addCommonArgs(hash_command);
    addCommonArgs(similar_command);

    // Hash-specific arguments
    hash_command.add_argument("-o", "--output")
        .default_value(std::string(""))
		.store_into(args.outputPath)
        .help("Save computed hashes to CSV file");

    // Similar-specific arguments
    similar_command.add_argument("-t", "--threshold")
		.default_value(defaults::THRESHOLD)
        .scan<'i', int>()
		.store_into(args.threshold)
        .help("Similarity threshold (0=exact duplicate, 10=entirely different)");

    similar_command.add_argument("--num-tables")
		.default_value(defaults::NUM_TABLES)
        .scan<'i', int>()
		.store_into(args.numTables)
        .help("Number of hash tables for similarity search (advanced)");

    similar_command.add_argument("--bits-per-table")
		.default_value(defaults::BITS_PER_TABLE)
        .scan<'i', int>()
		.store_into(args.bitsPerTable)
        .help("Bits sampled per hash table (advanced)");

    similar_command.add_argument("-a", "--auto-delete")
        .implicit_value(true)
		.default_value(defaults::AUTO_DELETE)
		.store_into(args.autoDelete)
        .help("Automatically delete similar images (with confirmation prompt)");

    similar_command.add_argument("-i", "--interactive")
        .implicit_value(true)
		.default_value(defaults::INTERACTIVE)
		.store_into(args.interactive)
        .help("Review and confirm each deletion individually");

    similar_command.add_argument("-p", "--print-only")
        .implicit_value(true)
		.default_value(defaults::PRINT_ONLY)
		.store_into(args.printOnly)
        .help("Only display similar images without deleting");

    similar_command.add_argument("--dry-run")
        .implicit_value(true)
		.default_value(defaults::DRY_RUN)
		.store_into(args.dryRun)
        .help("Show what would be deleted without actually deleting");

    similar_command.add_argument("-o", "--output")
        .default_value(std::string(""))
		.store_into(args.outputPath)
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
        return handleHashCommand(args);
    }
    else if (program.is_subcommand_used("similar")) {
        return handleSimilarCommand(args);
    }
    else {
        std::cerr << "No command specified. Use 'hash' or 'similar'\n";
        std::cerr << program;
        return 1;
    }
}