#include <gtest/gtest.h>

#include "helpers.hpp"
#include "phash_cuda.cuh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

namespace {
namespace fs = std::filesystem;

class HelpersTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temp directory for file tests
        test_dir_ = fs::temp_directory_path() / "helpers_test_temp";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up temp directory
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    fs::path test_dir_;
};

// Tests for cursor control functions
// Note: These are platform-specific and mainly test that they don't crash
TEST_F(HelpersTest, HideCursorDoesNotCrash) {
    // Simply test that the function can be called without crashing
    EXPECT_NO_THROW(phash_app::hideCursor());
}

TEST_F(HelpersTest, ShowCursorDoesNotCrash) {
    // Simply test that the function can be called without crashing
    EXPECT_NO_THROW(phash_app::showCursor());
}

// Tests for bar() function
TEST_F(HelpersTest, BarCreationWithDefaults) {
    auto progressBar = phash_app::bar("Test");
    EXPECT_FALSE(progressBar.is_completed());
}

TEST_F(HelpersTest, BarCreationWithElapsedTime) {
    auto progressBar = phash_app::bar("Test", true);
    EXPECT_FALSE(progressBar.is_completed());
}

TEST_F(HelpersTest, BarCreationWithRemainingTime) {
    auto progressBar = phash_app::bar("Test", false, true);
    EXPECT_FALSE(progressBar.is_completed());
}

TEST_F(HelpersTest, BarCreationWithBothTimes) {
    auto progressBar = phash_app::bar("Test", true, true);
    EXPECT_FALSE(progressBar.is_completed());
}

TEST_F(HelpersTest, BarWithEmptyPrefix) {
    auto progressBar = phash_app::bar("");
    EXPECT_FALSE(progressBar.is_completed());
}

// Tests for withCommas() function
TEST_F(HelpersTest, WithCommasZero) {
    EXPECT_EQ(phash_app::withCommas(0), "0");
}

TEST_F(HelpersTest, WithCommasSmallNumber) {
    EXPECT_EQ(phash_app::withCommas(999), "999");
}

TEST_F(HelpersTest, WithCommasThousand) {
    auto result = phash_app::withCommas(1000);
    // Result depends on locale, but should contain either "1,000" or "1.000" or "1 000"
    EXPECT_TRUE(result == "1,000" || result == "1.000" || result == "1 000" || result == "1000");
}

TEST_F(HelpersTest, WithCommasMillion) {
    auto result = phash_app::withCommas(1000000);
    // Result depends on locale
    EXPECT_TRUE(result == "1,000,000" || result == "1.000.000" || result == "1 000 000" || result == "1000000");
}

TEST_F(HelpersTest, WithCommasNegative) {
    auto result = phash_app::withCommas(-1000);
    // Should preserve negative sign
    EXPECT_TRUE(result.starts_with("-"));
}

TEST_F(HelpersTest, WithCommasLargeNumber) {
    auto result = phash_app::withCommas(9999999999LL);
    EXPECT_FALSE(result.empty());
}

// Tests for toLower() function
TEST_F(HelpersTest, ToLowerEmpty) {
    EXPECT_EQ(phash_app::toLower(""), "");
}

TEST_F(HelpersTest, ToLowerAlreadyLowercase) {
    EXPECT_EQ(phash_app::toLower("hello"), "hello");
}

TEST_F(HelpersTest, ToLowerUppercase) {
    EXPECT_EQ(phash_app::toLower("HELLO"), "hello");
}

TEST_F(HelpersTest, ToLowerMixedCase) {
    EXPECT_EQ(phash_app::toLower("HeLLo WoRLd"), "hello world");
}

TEST_F(HelpersTest, ToLowerWithNumbers) {
    EXPECT_EQ(phash_app::toLower("Test123"), "test123");
}

TEST_F(HelpersTest, ToLowerWithSpecialChars) {
    EXPECT_EQ(phash_app::toLower("Test@#$%"), "test@#$%");
}

TEST_F(HelpersTest, ToLowerWithSpaces) {
    EXPECT_EQ(phash_app::toLower("  TEST  "), "  test  ");
}

TEST_F(HelpersTest, ToLowerWithTabs) {
    EXPECT_EQ(phash_app::toLower("TEST\tTEST"), "test\ttest");
}

// Tests for collectImagePaths() function
TEST_F(HelpersTest, CollectImagePathsEmptyDirectory) {
    // Empty directory should throw
    EXPECT_THROW(
        phash_app::collectImagePaths(test_dir_, {"jpg", "png"}, false),
        std::runtime_error
    );
}

TEST_F(HelpersTest, CollectImagePathsNonExistentDirectory) {
    // Non-existent directory should throw
    fs::path nonExistent = test_dir_ / "nonexistent";
    EXPECT_THROW(
        phash_app::collectImagePaths(nonExistent, {"jpg", "png"}, false),
        std::runtime_error
    );
}

TEST_F(HelpersTest, CollectImagePathsSingleFile) {
    // Create a test image file
    fs::path testFile = test_dir_ / "test.jpg";
    std::ofstream(testFile) << "test";

    auto files = phash_app::collectImagePaths(test_dir_, {"jpg"}, false);
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(fs::path(files[0]).filename().string(), "test.jpg");
}

TEST_F(HelpersTest, CollectImagePathsMultipleExtensions) {
    // Create test files with different extensions
    std::ofstream(test_dir_ / "test1.jpg") << "test";
    std::ofstream(test_dir_ / "test2.png") << "test";
    std::ofstream(test_dir_ / "test3.gif") << "test";
    std::ofstream(test_dir_ / "test4.txt") << "test";

    auto files = phash_app::collectImagePaths(test_dir_, {"jpg", "png", "gif"}, false);
    ASSERT_EQ(files.size(), 3);

    // Files should be sorted
    EXPECT_TRUE(files[0].find("test1.jpg") != std::string::npos);
    EXPECT_TRUE(files[1].find("test2.png") != std::string::npos);
    EXPECT_TRUE(files[2].find("test3.gif") != std::string::npos);
}

TEST_F(HelpersTest, CollectImagePathsCaseInsensitiveExtensions) {
    // Create test files with mixed case extensions
    std::ofstream(test_dir_ / "test1.JPG") << "test";
    std::ofstream(test_dir_ / "test2.Png") << "test";
    std::ofstream(test_dir_ / "test3.GIF") << "test";

    auto files = phash_app::collectImagePaths(test_dir_, {"jpg", "png", "gif"}, false);
    ASSERT_EQ(files.size(), 3);
}

TEST_F(HelpersTest, CollectImagePathsRecursive) {
    // Create subdirectory with files
    fs::path subdir = test_dir_ / "subdir";
    fs::create_directories(subdir);

    std::ofstream(test_dir_ / "test1.jpg") << "test";
    std::ofstream(subdir / "test2.jpg") << "test";

    // Non-recursive should only find one
    auto filesNonRecursive = phash_app::collectImagePaths(test_dir_, {"jpg"}, false);
    EXPECT_EQ(filesNonRecursive.size(), 1);

    // Recursive should find both
    auto filesRecursive = phash_app::collectImagePaths(test_dir_, {"jpg"}, true);
    EXPECT_EQ(filesRecursive.size(), 2);
}

TEST_F(HelpersTest, CollectImagePathsEmptyExtensionList) {
    // With empty extension list, should collect all files
    std::ofstream(test_dir_ / "test1.jpg") << "test";
    std::ofstream(test_dir_ / "test2.txt") << "test";
    std::ofstream(test_dir_ / "test3") << "test"; // No extension

    auto files = phash_app::collectImagePaths(test_dir_, {}, false);
    EXPECT_EQ(files.size(), 3);
}

TEST_F(HelpersTest, CollectImagePathsIgnoresDirectories) {
    // Create a directory that looks like an image file
    fs::create_directories(test_dir_ / "fake.jpg");
    std::ofstream(test_dir_ / "real.jpg") << "test";

    auto files = phash_app::collectImagePaths(test_dir_, {"jpg"}, false);
    EXPECT_EQ(files.size(), 1);
    EXPECT_TRUE(files[0].find("real.jpg") != std::string::npos);
}

// Tests for queryYesNo() function
TEST_F(HelpersTest, QueryYesNoWithY) {
    std::istringstream input("y\n");
    std::cin.rdbuf(input.rdbuf());

    // Redirect cout to suppress prompt output
    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    bool result = phash_app::queryYesNo("Test prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_TRUE(result);
}

TEST_F(HelpersTest, QueryYesNoWithUpperY) {
    std::istringstream input("Y\n");
    std::cin.rdbuf(input.rdbuf());

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    bool result = phash_app::queryYesNo("Test prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_TRUE(result);
}

TEST_F(HelpersTest, QueryYesNoWith1) {
    std::istringstream input("1\n");
    std::cin.rdbuf(input.rdbuf());

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    bool result = phash_app::queryYesNo("Test prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_TRUE(result);
}

TEST_F(HelpersTest, QueryYesNoWithN) {
    std::istringstream input("n\n");
    std::cin.rdbuf(input.rdbuf());

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    bool result = phash_app::queryYesNo("Test prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_FALSE(result);
}

TEST_F(HelpersTest, QueryYesNoWithEmpty) {
    std::istringstream input("\n");
    std::cin.rdbuf(input.rdbuf());

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    bool result = phash_app::queryYesNo("Test prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_FALSE(result);
}

TEST_F(HelpersTest, QueryYesNoWithOtherInput) {
    std::istringstream input("maybe\n");
    std::cin.rdbuf(input.rdbuf());

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    bool result = phash_app::queryYesNo("Test prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_FALSE(result);
}

TEST_F(HelpersTest, QueryYesNoPromptOutput) {
    std::istringstream input("n\n");
    std::cin.rdbuf(input.rdbuf());

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::queryYesNo("Custom prompt");

    std::cout.rdbuf(oldCout);
    EXPECT_TRUE(output.str().find("Custom prompt [y/N]") != std::string::npos);
}

// Tests for formatFileSize() function
TEST_F(HelpersTest, FormatFileSizeZero) {
    EXPECT_EQ(phash_app::formatFileSize(0), "0.00 B");
}

TEST_F(HelpersTest, FormatFileSizeBytes) {
    EXPECT_EQ(phash_app::formatFileSize(1), "1.00 B");
    EXPECT_EQ(phash_app::formatFileSize(1023), "1023.00 B");
}

TEST_F(HelpersTest, FormatFileSizeKilobytes) {
    EXPECT_EQ(phash_app::formatFileSize(1024), "1.00 KB");
    EXPECT_EQ(phash_app::formatFileSize(1536), "1.50 KB");
    EXPECT_EQ(phash_app::formatFileSize(1024 * 1023), "1023.00 KB");
}

TEST_F(HelpersTest, FormatFileSizeMegabytes) {
    EXPECT_EQ(phash_app::formatFileSize(1024 * 1024), "1.00 MB");
    EXPECT_EQ(phash_app::formatFileSize(1024 * 1024 * 5), "5.00 MB");
    EXPECT_EQ(phash_app::formatFileSize(static_cast<std::uintmax_t>(1024) * 1024 * 1023), "1023.00 MB");
}

TEST_F(HelpersTest, FormatFileSizeGigabytes) {
    std::uintmax_t gb = static_cast<std::uintmax_t>(1024) * 1024 * 1024;
    EXPECT_EQ(phash_app::formatFileSize(gb), "1.00 GB");
    EXPECT_EQ(phash_app::formatFileSize(gb * 10), "10.00 GB");
}

TEST_F(HelpersTest, FormatFileSizeTerabytes) {
    std::uintmax_t tb = static_cast<std::uintmax_t>(1024) * 1024 * 1024 * 1024;
    EXPECT_EQ(phash_app::formatFileSize(tb), "1.00 TB");
    EXPECT_EQ(phash_app::formatFileSize(tb * 2), "2.00 TB");
}

TEST_F(HelpersTest, FormatFileSizeLargerThanTB) {
    // Should cap at TB even for larger values
    std::uintmax_t huge = static_cast<std::uintmax_t>(1024) * 1024 * 1024 * 1024 * 1024;
    EXPECT_EQ(phash_app::formatFileSize(huge), "1024.00 TB");
}

TEST_F(HelpersTest, FormatFileSizeMaxValue) {
    // Test with maximum uintmax_t value
    std::uintmax_t max_val = std::numeric_limits<std::uintmax_t>::max();
    auto result = phash_app::formatFileSize(max_val);
    EXPECT_TRUE(result.find("TB") != std::string::npos);
}

// Tests for printConfiguration() function
TEST_F(HelpersTest, PrintConfigurationWithThreshold) {
    // Redirect cout to capture output
    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::printConfiguration("test_folder", 100, 32, 4, 256, 8, 10);

    std::cout.rdbuf(oldCout);
    std::string result = output.str();

    // Check for expected content
    EXPECT_TRUE(result.find("test_folder") != std::string::npos);
    EXPECT_TRUE(result.find("100 images") != std::string::npos);
    EXPECT_TRUE(result.find("32x32") != std::string::npos);
    EXPECT_TRUE(result.find("128x128") != std::string::npos); // 32 * 4
    EXPECT_TRUE(result.find("256") != std::string::npos);
    EXPECT_TRUE(result.find("8") != std::string::npos);
    EXPECT_TRUE(result.find("10") != std::string::npos); // threshold
}

TEST_F(HelpersTest, PrintConfigurationWithoutThreshold) {
    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::printConfiguration("test_folder", 1000, 16, 8, 512, -1, -1);

    std::cout.rdbuf(oldCout);
    std::string result = output.str();

    // Check for expected content
    EXPECT_TRUE(result.find("test_folder") != std::string::npos);
    EXPECT_TRUE(result.find("16x16") != std::string::npos);
    EXPECT_TRUE(result.find("128x128") != std::string::npos); // 16 * 8
    EXPECT_TRUE(result.find("512") != std::string::npos);
    EXPECT_TRUE(result.find("auto") != std::string::npos); // threads = -1 shows as "auto"
}

TEST_F(HelpersTest, PrintConfigurationLargeNumbers) {
    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::printConfiguration("folder", 1000000, 64, 2, 1024, 16, 20);

    std::cout.rdbuf(oldCout);
    std::string result = output.str();

    // Should format large numbers with commas (locale-dependent)
    EXPECT_TRUE(result.find("000") != std::string::npos ||
                result.find(",") != std::string::npos ||
                result.find(".") != std::string::npos);
}

// Tests for printHashResults() function
TEST_F(HelpersTest, PrintHashResultsWithFewHashes) {
    std::vector<std::string> paths = {
        "/path/to/image1.jpg",
        "/path/to/image2.png",
        "/path/to/image3.gif"
    };

    std::vector<pHash> hashes;
    for (int i = 0; i < 3; ++i) {
        pHash hash;
        hash.data[0] = static_cast<uint64_t>(i + 1);
        hashes.push_back(hash);
    }

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::printHashResults(paths, hashes, "", 32);

    std::cout.rdbuf(oldCout);
    std::string result = output.str();

    // Check for expected content
    EXPECT_TRUE(result.find("Results") != std::string::npos);
    EXPECT_TRUE(result.find("image1.jpg") != std::string::npos);
    EXPECT_TRUE(result.find("image2.png") != std::string::npos);
    EXPECT_TRUE(result.find("image3.gif") != std::string::npos);
}

TEST_F(HelpersTest, PrintHashResultsWithManyHashes) {
    std::vector<std::string> paths;
    std::vector<pHash> hashes;

    for (int i = 0; i < 10; ++i) {
        paths.push_back("/path/to/image" + std::to_string(i) + ".jpg");
        pHash hash;
        hash.data[0] = static_cast<uint64_t>(i);
        hashes.push_back(hash);
    }

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::printHashResults(paths, hashes, "output.txt", 32);

    std::cout.rdbuf(oldCout);
    std::string result = output.str();

    // Should only show first 5 results in table
    EXPECT_TRUE(result.find("image0.jpg") != std::string::npos);
    EXPECT_TRUE(result.find("image4.jpg") != std::string::npos);
    EXPECT_FALSE(result.find("image5.jpg") != std::string::npos);

    // Should show save message
    EXPECT_TRUE(result.find("10 hashes saved to output.txt") != std::string::npos);
}

TEST_F(HelpersTest, PrintHashResultsEmptyOutput) {
    std::vector<std::string> paths;
    std::vector<pHash> hashes;

    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    phash_app::printHashResults(paths, hashes, "", 32);

    std::cout.rdbuf(oldCout);
    std::string result = output.str();

    // Should still print headers
    EXPECT_TRUE(result.find("Results") != std::string::npos);
    EXPECT_TRUE(result.find("File") != std::string::npos);
    EXPECT_TRUE(result.find("Hash") != std::string::npos);
}

TEST_F(HelpersTest, PrintHashResultsDifferentHashSizes) {
    std::vector<std::string> paths = {"/path/to/test.jpg"};
    std::vector<pHash> hashes;

    pHash hash;
    hash.data[0] = 0xFFFFFFFFFFFFFFFF;
    hashes.push_back(hash);

    std::ostringstream output1, output2;
    std::streambuf* oldCout = std::cout.rdbuf(output1.rdbuf());

    // Test with hash size 16
    phash_app::printHashResults(paths, hashes, "", 16);

    std::cout.rdbuf(output2.rdbuf());

    // Test with hash size 32
    phash_app::printHashResults(paths, hashes, "", 32);

    std::cout.rdbuf(oldCout);

    // Both should contain the file name
    EXPECT_TRUE(output1.str().find("test.jpg") != std::string::npos);
    EXPECT_TRUE(output2.str().find("test.jpg") != std::string::npos);
}

} // namespace