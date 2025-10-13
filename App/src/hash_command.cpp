#include "hash_command.hpp"
#include "rang.hpp"
#include "indicators.hpp"
#include "phash_cuda.cuh"
#include "helpers.hpp"
#include "save_results.hpp"

#include <iostream>
#include <chrono>
#include <array>

namespace fs = std::filesystem;
using namespace rang;
using namespace phash_app;

int phash_app::handleHashCommand(const Arguments& args)
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