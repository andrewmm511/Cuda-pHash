//
// save_results.hpp
// Functions for saving computation results to CSV files
//

#pragma once

#include <string>
#include <vector>
#include "phash_cuda.cuh"

namespace phash_app {

/**
 * Save computed hashes to a CSV file
 * @param path Output file path
 * @param files Vector of file paths
 * @param hashes Vector of computed perceptual hashes
 * @param hashSize Size of the hash dimension
 */
void saveHashesCSV(const std::string& path,
                   const std::vector<std::string>& files,
                   const std::vector<pHash>& hashes,
                   int hashSize);

/**
 * Save similar images list to a CSV file
 * @param path Output file path
 * @param similar Vector of similar images
 * @param threshold Similarity threshold used in detection
 */
void saveSimilarImagesCSV(const std::string& path,
                          const std::vector<Image>& similar,
                          int threshold);

} // namespace phash_app