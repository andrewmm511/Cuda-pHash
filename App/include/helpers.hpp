//
// helpers.hpp
// General utility and helper functions for the pHash CLI application
//

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include "indicators.hpp"
#include "phash_cuda.cuh"

namespace phash_app {

// Cursor visibility control
void hideCursor();
void showCursor();

// Progress bar creation
indicators::ProgressBar bar(std::string_view prefix, bool show_elapsed = false, bool show_remaining = false);

// Number formatting
std::string withCommas(auto number);

// String manipulation
std::string toLower(std::string s);

// File collection
std::vector<std::string> collectImagePaths(const std::filesystem::path& dir,
                                            const std::vector<std::string>& allowedExtensions,
                                            bool recursive);

// User interaction
bool queryYesNo(const std::string& prompt);

// File size formatting
std::string formatFileSize(std::uintmax_t bytes);

// Result display functions
void printConfiguration(std::string folder, int numImages, int hashSize,
                       int freqFactor, int batchSize, int threads,
                       int threshold = -1);

void printHashResults(std::vector<std::string> paths,
                      std::vector<pHash> hashes,
                      std::string outputPath,
                      int hashSize);

} // namespace phash_app
