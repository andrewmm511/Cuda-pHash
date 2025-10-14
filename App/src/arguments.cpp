#include "arguments.hpp"

#include <algorithm>
#include <cctype>
#include <concepts>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace phash_app {

namespace fs = std::filesystem;

namespace {

std::string trimCopy(std::string_view value)
{
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};

    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(start, end - start + 1));
}

std::string toLowerCopy(std::string value)
{
    std::ranges::transform(value,
                           value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> parseExtensionList(const std::string& exts)
{
    std::vector<std::string> extensions;
    std::unordered_set<std::string> seen;

    if (exts.empty()) return extensions;

    std::string_view sv = exts;
    size_t start = 0;

    while (start < sv.size()) {
        const auto pos = sv.find_first_of(",;", start);
        const auto token = sv.substr(start, pos == std::string_view::npos ? sv.size() - start : pos - start);
        auto cleaned = trimCopy(token);

        if (!cleaned.empty() && cleaned.front() == '.') {
            cleaned.erase(0, 1);
        }

        cleaned = toLowerCopy(std::move(cleaned));
        if (!cleaned.empty() && seen.insert(cleaned).second) {
            extensions.emplace_back(std::move(cleaned));
        }

        if (pos == std::string_view::npos) break;
        
        start = pos + 1;
    }

    return extensions;
}

template<std::integral T>
T validateInRange(std::string_view flag, T value, T min, T max)
{
    if (value < min || value > max) {
        throw std::invalid_argument(std::format("{} must be between {} and {}).", flag, min, max, value));
    }
    return value;
}

fs::path validateDirectory(const fs::path& input)
{
    if (input.empty()) {
        throw std::invalid_argument("A directory must be provided (--directory).");
    }

    std::error_code ec;
    if (!fs::exists(input, ec)) {
        throw std::invalid_argument("Directory '" + input.string() + "' does not exist.");
    }

    if (!fs::is_directory(input, ec)) {
        throw std::invalid_argument("Path '" + input.string() + "' is not a directory.");
    }

    const auto absolutePath = fs::weakly_canonical(input, ec);
    if (ec) {
        throw std::invalid_argument("Unable to resolve directory '" + input.string() + "': " + ec.message());
    }

    return absolutePath;
}

int validatePositive(std::string_view flag, int value)
{
    if (value <= 0) {
        throw std::invalid_argument(
            std::format("{} must be greater than zero (received {}).", flag, value));
    }
    return value;
}

int validateThreads(int value)
{
    if (value == -1 || value > 0) return value;

    throw std::invalid_argument(
        std::format("--threads must be -1 (auto) or greater than zero (received {}).", value));
}

void validateSimilarOptions(bool autoDelete, bool interactive)
{
    if (autoDelete && interactive) {
        throw std::invalid_argument("Cannot combine --auto-delete with --interactive.");
    }
}

} // namespace

Arguments::Arguments(const RawArguments& raw, Command command)
    : directory(validateDirectory(raw.directory)),
      recursive(raw.recursive),
      extensions(parseExtensionList(raw.extensions)),
      hashSize(validateInRange("--hash-size", raw.hashSize, 5, 11)),
      freqFactor(validatePositive("--freq-factor", raw.freqFactor)),
      batchSize(validatePositive("--batch-size", raw.batchSize)),
      threads(validateThreads(raw.threads)),
      prefetchFactor(validatePositive("--prefetch-factor", raw.prefetchFactor)),
      logLevel(validateInRange("--log-level", raw.logLevel, 1, 4)),
      outputPath(raw.outputPath),
      threshold(command == Command::Similar ? validateInRange("--threshold", raw.threshold, 0, 64) : raw.threshold),
      numTables(validatePositive("--num-tables", raw.numTables)),
      bitsPerTable(validatePositive("--bits-per-table", raw.bitsPerTable)),
      autoDelete(raw.autoDelete),
      interactive(raw.interactive),
      printOnly(raw.printOnly),
      dryRun(raw.dryRun)
{
    if (command == Command::Similar) {
        validateSimilarOptions(autoDelete, interactive);
    }
}

} // namespace phash_app
