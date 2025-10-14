#include "arguments.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace phash_app {

namespace fs = std::filesystem;

namespace {

std::string trimCopy(std::string_view value)
{
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(start, end - start + 1));
}

std::string toLowerCopy(std::string value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> parseExtensionList(const std::string& csv)
{
    std::vector<std::string> extensions;
    std::unordered_set<std::string> seen;

    if (csv.empty()) {
        return extensions;
    }

    size_t start = 0;
    while (start <= csv.size()) {
        const auto pos = csv.find_first_of(",;", start);
        const auto token = csv.substr(start, pos == std::string::npos ? pos : pos - start);
        auto cleaned = trimCopy(token);

        if (!cleaned.empty() && cleaned.front() == '.') {
            cleaned.erase(0, 1);
        }

        cleaned = toLowerCopy(cleaned);
        if (!cleaned.empty() && seen.insert(cleaned).second) {
            extensions.emplace_back(std::move(cleaned));
        }

        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }

    return extensions;
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

int validateHashSize(int value)
{
    if (value < 5 || value > 11) {
        throw std::invalid_argument("--hash-size must be between 5 and 11.");
    }
    return value;
}

int validatePositive(std::string_view flag, int value)
{
    if (value <= 0) {
        std::ostringstream oss;
        oss << flag << " must be greater than zero (received " << value << ").";
        throw std::invalid_argument(oss.str());
    }
    return value;
}

int validateThreads(int value)
{
    if (value == -1 || value > 0) {
        return value;
    }

    std::ostringstream oss;
    oss << "--threads must be -1 (auto) or greater than zero (received " << value << ").";
    throw std::invalid_argument(oss.str());
}

int validateLogLevel(int value)
{
    if (value < 1 || value > 4) {
        std::ostringstream oss;
        oss << "--log-level must be between 1 and 4 (received " << value << ").";
        throw std::invalid_argument(oss.str());
    }
    return value;
}

int validateThreshold(int value)
{
    if (value < 0 || value > 64) {
        std::ostringstream oss;
        oss << "--threshold must be between 0 and 64 (received " << value << ").";
        throw std::invalid_argument(oss.str());
    }
    return value;
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
      hashSize(validateHashSize(raw.hashSize)),
      freqFactor(validatePositive("--freq-factor", raw.freqFactor)),
      batchSize(validatePositive("--batch-size", raw.batchSize)),
      threads(validateThreads(raw.threads)),
      prefetchFactor(validatePositive("--prefetch-factor", raw.prefetchFactor)),
      logLevel(validateLogLevel(raw.logLevel)),
      outputPath(raw.outputPath),
      threshold(command == Command::Similar ? validateThreshold(raw.threshold) : raw.threshold),
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
