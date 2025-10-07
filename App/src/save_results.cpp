//
// save_results.cpp
// Functions for saving computation results to CSV files
//

#include "save_results.hpp"
#include <fstream>
#include <iostream>

namespace phash_app {

void saveHashesCSV(const std::string& path,
                   const std::vector<std::string>& files,
                   const std::vector<pHash>& hashes,
                   int hashSize)
{
    std::ofstream csv(path);
    if (!csv) {
        std::cerr << "Warning: Cannot create output file '" << path << "'\n";
        return;
    }

    csv << "filepath,phash_hex\n";
    for (size_t i = 0; i < files.size() && i < hashes.size(); ++i) {
        csv << '"' << files[i] << "\"," << hashes[i].to_string(hashSize) << '\n';
    }
}

void saveSimilarImagesCSV(const std::string& path,
                          const std::vector<Image>& similar,
                          int threshold)
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

} // namespace phash_app