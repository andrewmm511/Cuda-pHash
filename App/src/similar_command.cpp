#include "similar_command.hpp"
#include "rang.hpp"
#include "phash_cuda.cuh"
#include "helpers.hpp"
#include "save_results.hpp"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>
#include <cctype>

namespace fs = std::filesystem;
using namespace rang;
using namespace phash_app;

int phash_app::handleSimilarCommand(const Arguments& args)
{
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
