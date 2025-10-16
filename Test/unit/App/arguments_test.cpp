#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <Windows.h>

#include "arguments.hpp"

namespace fs = std::filesystem;
using namespace phash_app;

class ArgumentsTest : public ::testing::Test {
protected:
    fs::path tempDir;

    void SetUp() override {
        auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string dirName = "phash_test_" + std::to_string(::GetCurrentProcessId()); + "_" + std::to_string(timestamp);
        tempDir = fs::temp_directory_path() / dirName;

        std::error_code ec;
        fs::create_directories(tempDir, ec);
        ASSERT_FALSE(ec) << "Failed to create temp directory: " << ec.message();
    }

    void TearDown() override {
        if (!tempDir.empty() && fs::exists(tempDir)) {
            std::error_code ec;
            fs::remove_all(tempDir, ec);
        }
    }

    RawArguments createValidRawArgs() {
        RawArguments raw;
        raw.directory = tempDir;
        return raw;
    }
};

// Tests for valid construction
TEST_F(ArgumentsTest, ConstructorWithValidHashCommand) {
    RawArguments raw = createValidRawArgs();

    Arguments args(raw, Arguments::Command::Hash);

    EXPECT_EQ(args.directory, fs::weakly_canonical(tempDir));
    EXPECT_EQ(args.recursive, defaults::RECURSIVE);
    EXPECT_EQ(args.hashSize, defaults::HASH_SIZE);
    EXPECT_EQ(args.freqFactor, defaults::FREQ_FACTOR);
    EXPECT_EQ(args.batchSize, defaults::BATCH_SIZE);
    EXPECT_EQ(args.threads, defaults::THREADS);
    EXPECT_EQ(args.prefetchFactor, defaults::PREFETCH_FACTOR);
    EXPECT_EQ(args.logLevel, defaults::LOG_LEVEL);
    EXPECT_EQ(args.outputPath, defaults::DEFAULT_OUTPUT);
    EXPECT_EQ(args.threshold, defaults::THRESHOLD);
    EXPECT_EQ(args.numTables, defaults::NUM_TABLES);
    EXPECT_EQ(args.bitsPerTable, defaults::BITS_PER_TABLE);
    EXPECT_EQ(args.autoDelete, defaults::AUTO_DELETE);
    EXPECT_EQ(args.interactive, defaults::INTERACTIVE);
    EXPECT_EQ(args.printOnly, defaults::PRINT_ONLY);
    EXPECT_EQ(args.dryRun, defaults::DRY_RUN);
}

TEST_F(ArgumentsTest, ConstructorWithValidSimilarCommand) {
    RawArguments raw = createValidRawArgs();
    raw.threshold = 10;

    Arguments args(raw, Arguments::Command::Similar);

    EXPECT_EQ(args.threshold, 10);
}

TEST_F(ArgumentsTest, ConstructorWithCustomValues) {
    RawArguments raw = createValidRawArgs();
    raw.recursive = true;
    raw.hashSize = 9;
    raw.freqFactor = 8;
    raw.batchSize = 1000;
    raw.threads = 4;
    raw.prefetchFactor = 16;
    raw.logLevel = 2;
    raw.outputPath = "output.txt";
    raw.threshold = 15;
    raw.numTables = 64;
    raw.bitsPerTable = 16;
    raw.printOnly = true;
    raw.dryRun = true;

    Arguments args(raw, Arguments::Command::Hash);

    EXPECT_TRUE(args.recursive);
    EXPECT_EQ(args.hashSize, 9);
    EXPECT_EQ(args.freqFactor, 8);
    EXPECT_EQ(args.batchSize, 1000);
    EXPECT_EQ(args.threads, 4);
    EXPECT_EQ(args.prefetchFactor, 16);
    EXPECT_EQ(args.logLevel, 2);
    EXPECT_EQ(args.outputPath, "output.txt");
    EXPECT_EQ(args.numTables, 64);
    EXPECT_EQ(args.bitsPerTable, 16);
    EXPECT_TRUE(args.printOnly);
    EXPECT_TRUE(args.dryRun);
}

// Directory validation tests
TEST_F(ArgumentsTest, DirectoryValidation_EmptyPath) {
    RawArguments raw = createValidRawArgs();
    raw.directory = "";

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, DirectoryValidation_NonExistentDirectory) {
    RawArguments raw = createValidRawArgs();
    raw.directory = tempDir / "non_existent_dir";

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, DirectoryValidation_FileInsteadOfDirectory) {
    fs::path filePath = tempDir / "test_file.txt";
    std::ofstream(filePath) << "test";

    RawArguments raw = createValidRawArgs();
    raw.directory = filePath;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, DirectoryValidation_RelativePathConverted) {
    fs::path subDir = tempDir / "subdir";
    fs::create_directories(subDir);

    fs::path originalDir = fs::current_path();
    fs::current_path(tempDir);

    RawArguments raw;
    raw.directory = "subdir";

    try {
        Arguments args(raw, Arguments::Command::Hash);
        EXPECT_EQ(args.directory, fs::weakly_canonical(subDir));
    } catch(...) {
        fs::current_path(originalDir);
        throw;
    }

    fs::current_path(originalDir);
}

// Extension parsing tests
TEST_F(ArgumentsTest, ExtensionParsing_DefaultExtensions) {
    RawArguments raw = createValidRawArgs();

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 2u);
    EXPECT_EQ(args.extensions[0], "jpg");
    EXPECT_EQ(args.extensions[1], "jpeg");
}

TEST_F(ArgumentsTest, ExtensionParsing_EmptyString) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "";

    Arguments args(raw, Arguments::Command::Hash);

    EXPECT_TRUE(args.extensions.empty());
}

TEST_F(ArgumentsTest, ExtensionParsing_SingleExtension) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "png";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 1u);
    EXPECT_EQ(args.extensions[0], "png");
}

TEST_F(ArgumentsTest, ExtensionParsing_MultipleExtensions) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "png,jpg,gif,bmp";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 4u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
    EXPECT_EQ(args.extensions[3], "bmp");
}

TEST_F(ArgumentsTest, ExtensionParsing_WithDots) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = ".png,.jpg,.gif";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

TEST_F(ArgumentsTest, ExtensionParsing_WithWhitespace) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "  png  ,  jpg  ,  gif  ";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

TEST_F(ArgumentsTest, ExtensionParsing_UppercaseConverted) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "PNG,JPG,GIF";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

TEST_F(ArgumentsTest, ExtensionParsing_MixedCase) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "PnG,JpG,GiF";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

TEST_F(ArgumentsTest, ExtensionParsing_DuplicatesRemoved) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "png,jpg,png,gif,jpg";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

TEST_F(ArgumentsTest, ExtensionParsing_EmptyTokensIgnored) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "png,,jpg,,,gif";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

TEST_F(ArgumentsTest, ExtensionParsing_TrailingDelimiters) {
    RawArguments raw = createValidRawArgs();
    raw.extensions = "png,jpg,gif,";

    Arguments args(raw, Arguments::Command::Hash);

    ASSERT_EQ(args.extensions.size(), 3u);
    EXPECT_EQ(args.extensions[0], "png");
    EXPECT_EQ(args.extensions[1], "jpg");
    EXPECT_EQ(args.extensions[2], "gif");
}

// Hash size validation tests
TEST_F(ArgumentsTest, HashSizeValidation_MinimumValid) {
    RawArguments raw = createValidRawArgs();
    raw.hashSize = 5;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.hashSize, 5);
}

TEST_F(ArgumentsTest, HashSizeValidation_MaximumValid) {
    RawArguments raw = createValidRawArgs();
    raw.hashSize = 11;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.hashSize, 11);
}

TEST_F(ArgumentsTest, HashSizeValidation_TooSmall) {
    RawArguments raw = createValidRawArgs();
    raw.hashSize = 4;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, HashSizeValidation_TooLarge) {
    RawArguments raw = createValidRawArgs();
    raw.hashSize = 12;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

// Positive value validation tests
TEST_F(ArgumentsTest, FreqFactorValidation_Positive) {
    RawArguments raw = createValidRawArgs();
    raw.freqFactor = 10;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.freqFactor, 10);
}

TEST_F(ArgumentsTest, FreqFactorValidation_Zero) {
    RawArguments raw = createValidRawArgs();
    raw.freqFactor = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, FreqFactorValidation_Negative) {
    RawArguments raw = createValidRawArgs();
    raw.freqFactor = -1;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, BatchSizeValidation_Positive) {
    RawArguments raw = createValidRawArgs();
    raw.batchSize = 1000;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.batchSize, 1000);
}

TEST_F(ArgumentsTest, BatchSizeValidation_Zero) {
    RawArguments raw = createValidRawArgs();
    raw.batchSize = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, PrefetchFactorValidation_Positive) {
    RawArguments raw = createValidRawArgs();
    raw.prefetchFactor = 32;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.prefetchFactor, 32);
}

TEST_F(ArgumentsTest, PrefetchFactorValidation_Zero) {
    RawArguments raw = createValidRawArgs();
    raw.prefetchFactor = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, NumTablesValidation_Positive) {
    RawArguments raw = createValidRawArgs();
    raw.numTables = 64;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.numTables, 64);
}

TEST_F(ArgumentsTest, NumTablesValidation_Zero) {
    RawArguments raw = createValidRawArgs();
    raw.numTables = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, BitsPerTableValidation_Positive) {
    RawArguments raw = createValidRawArgs();
    raw.bitsPerTable = 16;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.bitsPerTable, 16);
}

TEST_F(ArgumentsTest, BitsPerTableValidation_Zero) {
    RawArguments raw = createValidRawArgs();
    raw.bitsPerTable = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

// Thread validation tests
TEST_F(ArgumentsTest, ThreadsValidation_Auto) {
    RawArguments raw = createValidRawArgs();
    raw.threads = -1;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.threads, -1);
}

TEST_F(ArgumentsTest, ThreadsValidation_Positive) {
    RawArguments raw = createValidRawArgs();
    raw.threads = 8;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.threads, 8);
}

TEST_F(ArgumentsTest, ThreadsValidation_Zero) {
    RawArguments raw = createValidRawArgs();
    raw.threads = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, ThreadsValidation_NegativeOtherThanMinusOne) {
    RawArguments raw = createValidRawArgs();
    raw.threads = -2;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

// Log level validation tests
TEST_F(ArgumentsTest, LogLevelValidation_MinimumValid) {
    RawArguments raw = createValidRawArgs();
    raw.logLevel = 1;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.logLevel, 1);
}

TEST_F(ArgumentsTest, LogLevelValidation_MaximumValid) {
    RawArguments raw = createValidRawArgs();
    raw.logLevel = 4;

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.logLevel, 4);
}

TEST_F(ArgumentsTest, LogLevelValidation_TooSmall) {
    RawArguments raw = createValidRawArgs();
    raw.logLevel = 0;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

TEST_F(ArgumentsTest, LogLevelValidation_TooLarge) {
    RawArguments raw = createValidRawArgs();
    raw.logLevel = 5;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Hash), std::invalid_argument);
}

// Threshold validation tests (for Similar command)
TEST_F(ArgumentsTest, ThresholdValidation_MinimumValid) {
    RawArguments raw = createValidRawArgs();
    raw.threshold = 0;

    Arguments args(raw, Arguments::Command::Similar);
    EXPECT_EQ(args.threshold, 0);
}

TEST_F(ArgumentsTest, ThresholdValidation_MaximumValid) {
    RawArguments raw = createValidRawArgs();
    raw.threshold = 64;

    Arguments args(raw, Arguments::Command::Similar);
    EXPECT_EQ(args.threshold, 64);
}

TEST_F(ArgumentsTest, ThresholdValidation_Negative) {
    RawArguments raw = createValidRawArgs();
    raw.threshold = -1;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Similar), std::invalid_argument);
}

TEST_F(ArgumentsTest, ThresholdValidation_TooLarge) {
    RawArguments raw = createValidRawArgs();
    raw.threshold = 65;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Similar), std::invalid_argument);
}

TEST_F(ArgumentsTest, ThresholdValidation_NotValidatedForHashCommand) {
    RawArguments raw = createValidRawArgs();
    raw.threshold = 100; // Invalid value but shouldn't be validated for Hash command

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.threshold, 100); // Should pass through without validation
}

// Similar command specific validations
TEST_F(ArgumentsTest, SimilarCommand_AutoDeleteAlone) {
    RawArguments raw = createValidRawArgs();
    raw.autoDelete = true;
    raw.interactive = false;

    Arguments args(raw, Arguments::Command::Similar);
    EXPECT_TRUE(args.autoDelete);
    EXPECT_FALSE(args.interactive);
}

TEST_F(ArgumentsTest, SimilarCommand_InteractiveAlone) {
    RawArguments raw = createValidRawArgs();
    raw.autoDelete = false;
    raw.interactive = true;

    Arguments args(raw, Arguments::Command::Similar);
    EXPECT_FALSE(args.autoDelete);
    EXPECT_TRUE(args.interactive);
}

TEST_F(ArgumentsTest, SimilarCommand_CannotCombineAutoDeleteAndInteractive) {
    RawArguments raw = createValidRawArgs();
    raw.autoDelete = true;
    raw.interactive = true;

    EXPECT_THROW(Arguments(raw, Arguments::Command::Similar), std::invalid_argument);
}

TEST_F(ArgumentsTest, HashCommand_CanHaveBothAutoDeleteAndInteractive) {
    RawArguments raw = createValidRawArgs();
    raw.autoDelete = true;
    raw.interactive = true;

    // Should not throw for Hash command
    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_TRUE(args.autoDelete);
    EXPECT_TRUE(args.interactive);
}

// Boolean flag tests
TEST_F(ArgumentsTest, BooleanFlags_AllFalse) {
    RawArguments raw = createValidRawArgs();
    raw.recursive = false;
    raw.autoDelete = false;
    raw.interactive = false;
    raw.printOnly = false;
    raw.dryRun = false;

    Arguments args(raw, Arguments::Command::Hash);

    EXPECT_FALSE(args.recursive);
    EXPECT_FALSE(args.autoDelete);
    EXPECT_FALSE(args.interactive);
    EXPECT_FALSE(args.printOnly);
    EXPECT_FALSE(args.dryRun);
}

TEST_F(ArgumentsTest, BooleanFlags_AllTrue) {
    RawArguments raw = createValidRawArgs();
    raw.recursive = true;
    raw.autoDelete = true;
    raw.interactive = true;
    raw.printOnly = true;
    raw.dryRun = true;

    Arguments args(raw, Arguments::Command::Hash);

    EXPECT_TRUE(args.recursive);
    EXPECT_TRUE(args.autoDelete);
    EXPECT_TRUE(args.interactive);
    EXPECT_TRUE(args.printOnly);
    EXPECT_TRUE(args.dryRun);
}

// Output path tests
TEST_F(ArgumentsTest, OutputPath_Empty) {
    RawArguments raw = createValidRawArgs();
    raw.outputPath = "";

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.outputPath, "");
}

TEST_F(ArgumentsTest, OutputPath_CustomPath) {
    RawArguments raw = createValidRawArgs();
    raw.outputPath = "/path/to/output.txt";

    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.outputPath, "/path/to/output.txt");
}