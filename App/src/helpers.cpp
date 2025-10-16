//
// helpers.cpp
// General utility and helper functions for the pHash CLI application
//

#include "helpers.hpp"
#include "rang.hpp"
#include "tabulate.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <locale>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace phash_app {

namespace fs = std::filesystem;
using namespace rang;
using namespace tabulate;

void printLogo()
{
    constexpr std::string_view asciiArt = R"(
  .d8888b. 88    88  888888b.      .d8               88    88                88    
 d88P  Y8b 88    88  88   Y8b     d888b              88    88                88    
 888       88    88  88    88    d8P888      8888b.  88888888   8888b. .d888 8888b. 
 888       88    88  88    88   d8P  888     88 "8b  88    88      "8b  "Y8. 88  8b
 Y88b  d8P Y8b  d8P  88   d8P  d888888888    88  88  88    88  888  88    88 88  88 
  "Y8888P"  "Y888P"  888888P  d8P      888   8888P"  88    88  "Y88888 888P' 88  88 
                                             88                                    
                                             88                                    

)";
    std::cout << asciiArt;
}

std::string centerText(std::string_view text, int width) {
    if (text.length() >= static_cast<size_t>(width)) return std::string(text);
    int leftPadding = (width - static_cast<int>(text.length())) / 2;
    return std::string(leftPadding, ' ') + std::string(text);
}

indicators::ProgressBar bar(std::string_view prefix, bool show_elapsed, bool show_remaining) {
    return indicators::ProgressBar{
        indicators::option::BarWidth{30},
        indicators::option::PrefixText{std::string(prefix)},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{show_elapsed},
        indicators::option::ShowRemainingTime{show_remaining},
        indicators::option::Stream{std::cout}
    };
}

std::string withCommas(std::integral auto number) {
    static const auto loc = std::locale("");
    return std::format(loc, "{:L}", number);
}

std::string toLower(std::string s) {
    for (char& c : s) c = std::tolower(static_cast<unsigned char>(c));
    return s;
}

std::vector<std::string> collectImagePaths(const fs::path& dir,
    const std::vector<std::string>& allowedExtensions,
    bool recursive)
{
    const std::unordered_set<std::string> exts(allowedExtensions.begin(), allowedExtensions.end());

    auto normalizeExt = [](const fs::path& p) {
        std::string ext = toLower(p.extension().string());
        return ext.starts_with('.') ? ext.substr(1) : ext;
        };

    auto shouldInclude = [&](const auto& entry) {
        return entry.is_regular_file() && (exts.empty() || exts.contains(normalizeExt(entry.path())));
        };

    std::vector<std::string> files;
    constexpr auto opts = fs::directory_options::skip_permission_denied;

    try {
        auto processDir = [&](auto&& iterator) {
            for (const auto& entry : iterator) {
                if (shouldInclude(entry)) files.emplace_back(entry.path().string());
            }
        };

        recursive ? processDir(fs::recursive_directory_iterator(dir, opts))
            : processDir(fs::directory_iterator(dir, opts));
    }
    catch (const fs::filesystem_error& e) {
        throw std::runtime_error(std::format("Error scanning directory: {}", e.what()));
    }

    std::ranges::sort(files);

    if (files.empty()) {
        throw std::runtime_error(std::format("No images found in directory '{}'", dir.string()));
    }

    return files;
}

bool queryYesNo(const std::string& prompt)
{
    std::cout << prompt << " [y/N] " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line) || line.empty()) return false;
    char c = std::tolower(static_cast<unsigned char>(line[0]));
    return c == 'y' || line[0] == '1';
}

std::string formatFileSize(std::uintmax_t bytes)
{
    const char* units[] = { "B", "KB", "MB", "GB", "TB" };
    if (bytes == 0) return "0.00 B";

    int unit = std::min(4, static_cast<int>(std::log(bytes) / std::log(1024)));
    double size = bytes / std::pow(1024.0, unit);

    return std::format("{:.2f} {}", size, units[unit]);
}

void printConfiguration(std::string folder, int numImages, int hashSize, int freqFactor, int batchSize, int threads, int threshold)
{
    printLogo();

    int colWidth = 16;
    int tableWidth = (threshold >= 0) ? (colWidth * 5) + 7 : (colWidth * 4) + 5;

    std::cout << style::italic << centerText(folder, tableWidth) << '\n' << centerText(withCommas(numImages) + " images", tableWidth) 
        << style::reset << '\n';

    std::string hashSizeStr = std::to_string(hashSize) + "x" + std::to_string(hashSize) + " (" + std::to_string(hashSize * hashSize) + " bits)";
    std::string imgSizeStr = std::to_string(hashSize * freqFactor) + "x" + std::to_string(hashSize * freqFactor) + " pixels";
    std::string batchSizeStr = withCommas(batchSize);
    std::string threadsStr = (threads == -1 ? "auto" : std::to_string(threads));

    Table::Row_t headerRow = { "Hash Size", "Img Size", "GPU Batch", "CPU Threads" };
    Table::Row_t dataRow = { hashSizeStr, imgSizeStr, batchSizeStr, threadsStr };

    if (threshold >= 0) {
        headerRow.push_back("Threshold");
        dataRow.push_back(std::to_string(threshold));
    }

    Table configurations;
    configurations.add_row(headerRow).add_row(dataRow);
    configurations.format().width(colWidth).font_align(FontAlign::center);
    configurations[0].format().font_style({ FontStyle::bold });

    std::cout << configurations << std::endl << std::endl;
}

void printHashResults(std::vector<std::string> paths, std::vector<pHash> hashes, std::string outputPath, int hashSize)
{
    constexpr int tableWidth = (4 * 16) + 2;
    constexpr int colWidth = tableWidth / 2;

    std::cout << style::italic << centerText("Results", tableWidth) << style::reset << '\n';

    Table results;
    results.add_row({ "File", "Hash (Hex)" });

    for (size_t i : std::views::iota(0u, std::min<size_t>(5, hashes.size()))) {
        results.add_row({ fs::path(paths[i]).filename().string(), hashes[i].to_string(hashSize) });
    }

    results.format().width(colWidth).font_align(FontAlign::center);
    results[0].format().font_style({ FontStyle::bold });

    std::cout << results << '\n';

    if (hashes.size() > 5 && !outputPath.empty()) {
        std::cout << style::italic << fg::green << centerText(std::format("{} hashes saved to {}",
                withCommas(hashes.size()), outputPath), tableWidth) << style::reset << fg::reset << '\n';
    }
}

} // namespace phash_app