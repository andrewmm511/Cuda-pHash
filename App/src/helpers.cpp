//
// helpers.cpp
// General utility and helper functions for the pHash CLI application
//

#include "helpers.hpp"
#include "rang.hpp"
#include "tabulate.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <format>
#include <locale>
#include <string_view>
#include <stdexcept>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#endif

namespace phash_app {

namespace fs = std::filesystem;
using namespace rang;
using namespace tabulate;

// Cursor visibility control
#ifdef _WIN32
void hideCursor() {
    HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO info;
    info.dwSize = 100;
    info.bVisible = FALSE;
    SetConsoleCursorInfo(consoleHandle, &info);
}

void showCursor() {
    HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO info;
    info.dwSize = 100;
    info.bVisible = TRUE;
    SetConsoleCursorInfo(consoleHandle, &info);
}
#else
void hideCursor() {
    std::cout << "\033[?25l" << std::flush;  // ANSI escape code to hide cursor
}

void showCursor() {
    std::cout << "\033[?25h" << std::flush;  // ANSI escape code to show cursor
}
#endif

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

std::string withCommas(auto number) {
    return std::format(std::locale("en_US.UTF-8"), "{:L}", number);
}

std::string toLower(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::string> collectImagePaths(const fs::path& dir,
    const std::vector<std::string>& allowedExtensions,
    bool recursive)
{
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        throw std::runtime_error("Error: '" + dir.string() + "' is not a valid directory");
    }

    std::vector<std::string> files;
    const auto opts = fs::directory_options::skip_permission_denied;

    std::unordered_set<std::string> extensions(allowedExtensions.begin(), allowedExtensions.end());
    const bool filterActive = !extensions.empty();

    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(dir, opts)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = toLower(entry.path().extension().string());
                if (!ext.empty() && ext.front() == '.') ext.erase(0, 1);
                if (!filterActive || extensions.contains(ext))
                    files.emplace_back(entry.path().string());
            }
        }
        else {
            for (const auto& entry : fs::directory_iterator(dir, opts)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = toLower(entry.path().extension().string());
                if (!ext.empty() && ext.front() == '.') ext.erase(0, 1);
                if (!filterActive || extensions.contains(ext))
                    files.emplace_back(entry.path().string());
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        throw std::runtime_error(std::string("Error scanning directory: ") + e.what());
    }

    std::sort(files.begin(), files.end());

    if (files.empty()) {
        throw std::runtime_error("No images found in directory '" + dir.string() + "'");
	}

    return files;
}

bool queryYesNo(const std::string& prompt)
{
    std::cout << prompt << " [y/N] " << std::flush;
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return false;
    char c = static_cast<char>(std::tolower(line.front()));
    return c == 'y' || c == '1';
}

std::string formatFileSize(std::uintmax_t bytes)
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

void printConfiguration(std::string folder, int numImages, int hashSize, int freqFactor, int batchSize, int threads, int threshold)
{
    std::cout << "\n  .d8888b. 88    88  888888b.      .d8               88    88                88    " << std::endl;
    std::cout << " d88P  Y8b 88    88  88   Y8b     d888b              88    88                88    " << std::endl;
    std::cout << " 888       88    88  88    88    d8P888      8888b.  88888888   8888b. .d888 8888b. " << std::endl;
    std::cout << " 888       88    88  88    88   d8P  888     88 \"8b  88    88      \"8b  \"Y8. 88  8b" << std::endl;
    std::cout << " Y88b  d8P Y8b  d8P  88   d8P  d888888888    88  88  88    88  888  88    88 88  88 " << std::endl;
    std::cout << "  \"Y8888P\"  \"Y888P\"  888888P  d8P      888   8888P\"  88    88  \"Y88888 888P' 88  88 " << std::endl;
    std::cout << "                                             88                                    " << std::endl;
    std::cout << "                                             88                                    \n" << std::endl;

    int colWidth = 16;
    int tableWidth = (threshold >= 0) ? (colWidth * 5) + 7 : (colWidth * 4) + 5;

    auto centerText = [tableWidth](const std::string& text) {
        int textLen = text.length();
        if (textLen >= tableWidth)  return text;
        int leftPadding = (tableWidth - textLen) / 2;
        return std::string(leftPadding, ' ') + text;
    };

    std::cout << style::italic << centerText(folder) << style::reset << std::endl;
    std::cout << style::italic << centerText(withCommas(numImages) + " images") << style::reset << std::endl;

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
    configurations.add_row(headerRow);
    configurations.add_row(dataRow);
    configurations.format().width(colWidth).font_align(FontAlign::center);
    configurations[0].format().font_style({ FontStyle::bold });

    std::cout << configurations << std::endl << std::endl;
}

void printHashResults(std::vector<std::string> paths, std::vector<pHash> hashes, std::string outputPath, int hashSize)
{
    int tableWidth = (4 * 16) + 2;
    int colWidth = tableWidth / 2;

    auto centerText = [tableWidth](const std::string& text) {
        int textLen = text.length();
        if (textLen >= tableWidth)  return text;
        int leftPadding = ((tableWidth + 4) - textLen) / 2;
        return std::string(leftPadding, ' ') + text;
    };

    std::cout << style::italic << centerText("Results") << style::reset << std::endl;

    Table::Row_t headerRow = { "File", "Hash (Hex)" };
    std::vector<Table::Row_t> dataRows;

    for (size_t i = 0; i < std::min<size_t>(5, hashes.size()); ++i) {
        dataRows.push_back({ fs::path(paths[i]).filename().string(), hashes[i].to_string(hashSize) });
    }

    Table results;
    results.add_row(headerRow);

    for (const auto& row : dataRows) {
        results.add_row(row);
    }

    results.format().width(colWidth).font_align(FontAlign::center);
    results[0].format().font_style({ FontStyle::bold });

    std::cout << results << std::endl;

    if (hashes.size() > 5 && outputPath != "") {
        std::cout << style::italic << fg::green << centerText(withCommas(hashes.size()) + " hashes saved to " + outputPath) << style::reset << fg::reset << std::endl;
    }
}

} // namespace phash_app
