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

// Start with just the basic test that worked in diagnostic
TEST(ArgumentsMinimal, BasicDefaults) {
    EXPECT_EQ(defaults::HASH_SIZE, 8);
    EXPECT_EQ(defaults::BATCH_SIZE, 500);
}

// Now add the test fixture class
class ArgumentsTestMinimal : public ::testing::Test {
protected:
    fs::path tempDir;

    void SetUp() override {
        auto pid = std::to_string(::GetCurrentProcessId());
        auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string dirName = "phash_test_" + pid + "_" + std::to_string(timestamp);
        tempDir = fs::temp_directory_path() / dirName;

        std::error_code ec;
        fs::create_directories(tempDir, ec);
        // Comment out GTEST_SKIP for now - it might be the issue
        // if (ec) {
        //     GTEST_SKIP() << "Failed to create temp directory: " << ec.message();
        // }
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

// Add one simple test using the fixture
TEST_F(ArgumentsTestMinimal, SimpleConstructor) {
    RawArguments raw = createValidRawArgs();
    Arguments args(raw, Arguments::Command::Hash);
    EXPECT_EQ(args.hashSize, defaults::HASH_SIZE);
}

// Add one test with EXPECT_THROW
TEST_F(ArgumentsTestMinimal, EmptyDirectoryThrows) {
    RawArguments raw = createValidRawArgs();
    raw.directory = "";

    EXPECT_THROW({
        Arguments args(raw, Arguments::Command::Hash);
    }, std::invalid_argument);
}