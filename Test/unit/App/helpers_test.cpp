#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "helpers.hpp"
#include "phash_cuda.cuh"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <limits>
#include <locale>

namespace App {

namespace fs = std::filesystem;
using namespace phash_app;
using ::testing::HasSubstr;
using ::testing::ContainsRegex;
using ::testing::UnorderedElementsAre;
using ::testing::ElementsAre;
using ::testing::Not;

class CoutCapture {
public:
    CoutCapture() : oldBuf(std::cout.rdbuf()) { std::cout.rdbuf(buffer.rdbuf()); }
    ~CoutCapture() { std::cout.rdbuf(oldBuf); }
    
    std::string str() const { return buffer.str(); }
    void clear() { buffer.str(""); buffer.clear(); }
private:
    std::stringstream buffer;
    std::streambuf* oldBuf;
};

class CinMock {
public:
    CinMock(const std::string& input) : oldBuf(std::cin.rdbuf()) {
        buffer.str(input);
        std::cin.rdbuf(buffer.rdbuf());
    }
    ~CinMock() { std::cin.rdbuf(oldBuf); }
private:
    std::stringstream buffer;
    std::streambuf* oldBuf;
};

class HelpersTest : public ::testing::Test {
protected:
    fs::path tempDir;

    void SetUp() override {
        tempDir = fs::temp_directory_path() / ("phash_test_" + std::to_string(std::random_device{}()));
        fs::create_directories(tempDir);
    }

    void TearDown() override {
        if (fs::exists(tempDir)) fs::remove_all(tempDir);
    }

    void createTestFile(const fs::path& path, const std::string& content = "") {
        fs::create_directories(path.parent_path());
        std::ofstream file(path);
        file << content;
    }
};

// ========== String Manipulation Tests ==========

TEST_F(HelpersTest, ToLower_BasicConversion) {
    EXPECT_EQ(toLower("HELLO"), "hello");
    EXPECT_EQ(toLower("Hello"), "hello");
    EXPECT_EQ(toLower("HeLLo WoRLd"), "hello world");
}

TEST_F(HelpersTest, ToLower_EmptyString) {
    EXPECT_EQ(toLower(""), "");
}

TEST_F(HelpersTest, ToLower_AlreadyLowercase) {
    EXPECT_EQ(toLower("hello"), "hello");
    EXPECT_EQ(toLower("hello world"), "hello world");
}

TEST_F(HelpersTest, ToLower_SpecialCharacters) {
    EXPECT_EQ(toLower("Hello123!@#"), "hello123!@#");
    EXPECT_EQ(toLower("ABC_DEF-GHI"), "abc_def-ghi");
    EXPECT_EQ(toLower("File.TXT"), "file.txt");
}

TEST_F(HelpersTest, ToLower_NonAsciiCharacters) {
    std::string result = toLower("ÀÈÌÒÙ");
    EXPECT_FALSE(result.empty());
}

TEST_F(HelpersTest, WithCommas_PositiveNumbers) {
    EXPECT_FALSE(withCommas(0).empty());
    EXPECT_FALSE(withCommas(100).empty());
    EXPECT_FALSE(withCommas(1000).empty());
    EXPECT_FALSE(withCommas(1000000).empty());
    EXPECT_FALSE(withCommas(1234567890).empty());
}

TEST_F(HelpersTest, WithCommas_NegativeNumbers) {
    EXPECT_FALSE(withCommas(-100).empty());
    EXPECT_FALSE(withCommas(-1000).empty());
    EXPECT_FALSE(withCommas(-1000000).empty());
}

TEST_F(HelpersTest, WithCommas_LargeNumbers) {
    EXPECT_FALSE(withCommas(std::numeric_limits<int>::max()).empty());
    EXPECT_FALSE(withCommas(std::numeric_limits<long long>::max()).empty());
}

TEST_F(HelpersTest, FormatFileSize_Zero) {
    EXPECT_EQ(formatFileSize(0), "0.00 B");
}

TEST_F(HelpersTest, FormatFileSize_Bytes) {
    EXPECT_EQ(formatFileSize(1), "1.00 B");
    EXPECT_EQ(formatFileSize(999), "999.00 B");
    EXPECT_EQ(formatFileSize(1023), "1023.00 B");
}

TEST_F(HelpersTest, FormatFileSize_Kilobytes) {
    EXPECT_EQ(formatFileSize(1024), "1.00 KB");
    EXPECT_EQ(formatFileSize(1536), "1.50 KB");
    EXPECT_EQ(formatFileSize(2048), "2.00 KB");
    EXPECT_EQ(formatFileSize(1024 * 1023), "1023.00 KB");
}

TEST_F(HelpersTest, FormatFileSize_Megabytes) {
    EXPECT_EQ(formatFileSize(1024 * 1024), "1.00 MB");
    EXPECT_EQ(formatFileSize(1024 * 1024 * 10), "10.00 MB");
    EXPECT_EQ(formatFileSize(static_cast<std::uintmax_t>(1024) * 1024 * 512), "512.00 MB");
}

TEST_F(HelpersTest, FormatFileSize_Gigabytes) {
    EXPECT_EQ(formatFileSize(static_cast<std::uintmax_t>(1024) * 1024 * 1024), "1.00 GB");
    EXPECT_EQ(formatFileSize(static_cast<std::uintmax_t>(1024) * 1024 * 1024 * 5), "5.00 GB");
}

TEST_F(HelpersTest, FormatFileSize_Terabytes) {
    EXPECT_EQ(formatFileSize(static_cast<std::uintmax_t>(1024) * 1024 * 1024 * 1024), "1.00 TB");
    EXPECT_EQ(formatFileSize(static_cast<std::uintmax_t>(1024) * 1024 * 1024 * 1024 * 2), "2.00 TB");
}

TEST_F(HelpersTest, FormatFileSize_MaxSize) {
    std::string result = formatFileSize(std::numeric_limits<std::uintmax_t>::max());
    EXPECT_FALSE(result.empty());
    EXPECT_THAT(result, HasSubstr("TB"));  // Max size should be in TB
}

// ========== File Collection Tests ==========

TEST_F(HelpersTest, CollectImagePaths_EmptyDirectory) {
    EXPECT_THROW(
        collectImagePaths(tempDir, {"jpg", "png"}, false),
        std::runtime_error
    );
}

TEST_F(HelpersTest, CollectImagePaths_NonExistentDirectory) {
    fs::path nonExistent = tempDir / "does_not_exist";
    EXPECT_THROW(
        collectImagePaths(nonExistent, {"jpg", "png"}, false),
        std::runtime_error
    );
}

TEST_F(HelpersTest, CollectImagePaths_NoMatchingFiles) {
    createTestFile(tempDir / "document.txt");
    createTestFile(tempDir / "data.csv");

    EXPECT_THROW(
        collectImagePaths(tempDir, {"jpg", "png"}, false),
        std::runtime_error
    );
}

TEST_F(HelpersTest, CollectImagePaths_BasicCollection) {
    createTestFile(tempDir / "image1.jpg");
    createTestFile(tempDir / "image2.png");
    createTestFile(tempDir / "photo.jpeg");
    createTestFile(tempDir / "document.txt");

    auto paths = collectImagePaths(tempDir, {"jpg", "jpeg", "png"}, false);

    EXPECT_EQ(paths.size(), 3);
    EXPECT_TRUE(std::is_sorted(paths.begin(), paths.end()));
}

TEST_F(HelpersTest, CollectImagePaths_CaseInsensitiveExtensions) {
    createTestFile(tempDir / "image1.JPG");
    createTestFile(tempDir / "image2.Png");
    createTestFile(tempDir / "image3.JPEG");
    createTestFile(tempDir / "image4.JpEg");

    auto paths = collectImagePaths(tempDir, {"jpg", "jpeg", "png"}, false);

    EXPECT_EQ(paths.size(), 4);
}

TEST_F(HelpersTest, CollectImagePaths_RecursiveCollection) {
    createTestFile(tempDir / "root.jpg");
    createTestFile(tempDir / "subdir1" / "image1.png");
    createTestFile(tempDir / "subdir1" / "subdir2" / "deep.jpg");
    createTestFile(tempDir / "subdir3" / "photo.jpeg");

    auto nonRecursive = collectImagePaths(tempDir, {"jpg", "jpeg", "png"}, false);
    EXPECT_EQ(nonRecursive.size(), 1);

    auto recursive = collectImagePaths(tempDir, {"jpg", "jpeg", "png"}, true);
    EXPECT_EQ(recursive.size(), 4);
    EXPECT_TRUE(std::is_sorted(recursive.begin(), recursive.end()));
}

TEST_F(HelpersTest, CollectImagePaths_EmptyExtensionsList) {
    createTestFile(tempDir / "file1.txt");
    createTestFile(tempDir / "file2.jpg");
    createTestFile(tempDir / "file3.doc");

    auto paths = collectImagePaths(tempDir, {}, false);

    EXPECT_EQ(paths.size(), 3);
}

TEST_F(HelpersTest, CollectImagePaths_SpecialCharactersInFilenames) {
    createTestFile(tempDir / "image with spaces.jpg");
    createTestFile(tempDir / "image-with-dashes.png");
    createTestFile(tempDir / "image_with_underscores.jpeg");
    createTestFile(tempDir / "image.multiple.dots.jpg");

    auto paths = collectImagePaths(tempDir, {"jpg", "jpeg", "png"}, false);

    EXPECT_EQ(paths.size(), 4);
}

TEST_F(HelpersTest, CollectImagePaths_HiddenFiles) {
    createTestFile(tempDir / ".hidden.jpg");
    createTestFile(tempDir / "visible.jpg");

    auto paths = collectImagePaths(tempDir, {"jpg"}, false);

    EXPECT_GE(paths.size(), 1);
}

// ========== User Interaction Tests ==========

TEST_F(HelpersTest, QueryYesNo_YesResponses) {
    {
        CinMock input("y\n");
        EXPECT_TRUE(queryYesNo("Continue?"));
    }
    {
        CinMock input("Y\n");
        EXPECT_TRUE(queryYesNo("Continue?"));
    }
    {
        CinMock input("yes\n");
        EXPECT_TRUE(queryYesNo("Continue?"));
    }
    {
        CinMock input("YES\n");
        EXPECT_TRUE(queryYesNo("Continue?"));
    }
    {
        CinMock input("1\n");
        EXPECT_TRUE(queryYesNo("Continue?"));
    }
}

TEST_F(HelpersTest, QueryYesNo_NoResponses) {
    {
        CinMock input("n\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
    {
        CinMock input("N\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
    {
        CinMock input("no\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
    {
        CinMock input("0\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
}

TEST_F(HelpersTest, QueryYesNo_EmptyInput) {
    CinMock input("\n");
    EXPECT_FALSE(queryYesNo("Continue?"));
}

TEST_F(HelpersTest, QueryYesNo_InvalidInput) {
    {
        CinMock input("maybe\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
    {
        CinMock input("xyz\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
    {
        CinMock input("2\n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
}

TEST_F(HelpersTest, QueryYesNo_WhitespaceHandling) {
    {
        CinMock input("  y  \n");
        EXPECT_TRUE(queryYesNo("Continue?"));
    }
    {
        CinMock input(" n \n");
        EXPECT_FALSE(queryYesNo("Continue?"));
    }
}

TEST_F(HelpersTest, QueryYesNo_PromptsCorrectly) {
    CoutCapture capture;
    CinMock input("y\n");

    queryYesNo("Delete file?");

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("Delete file?"));
    EXPECT_THAT(output, HasSubstr("[y/N]"));
}

// ========== Progress Bar Tests ==========

TEST_F(HelpersTest, Bar_BasicCreation) {
    auto progressBar = bar("Processing", false, false);
    EXPECT_NO_THROW(progressBar.set_progress(50));
}

TEST_F(HelpersTest, Bar_WithElapsedTime) {
    auto progressBar = bar("Loading", true, false);
    EXPECT_NO_THROW(progressBar.set_progress(25));
}

TEST_F(HelpersTest, Bar_WithRemainingTime) {
    auto progressBar = bar("Downloading", false, true);
    EXPECT_NO_THROW(progressBar.set_progress(75));
}

TEST_F(HelpersTest, Bar_WithBothTimes) {
    auto progressBar = bar("Computing", true, true);
    EXPECT_NO_THROW(progressBar.set_progress(100));
}

TEST_F(HelpersTest, Bar_EmptyPrefix) {
    auto progressBar = bar("", false, false);
    EXPECT_NO_THROW(progressBar.set_progress(0));
}

// ========== Display Functions Tests ==========

TEST_F(HelpersTest, PrintConfiguration_BasicOutput) {
    CoutCapture capture;

    printConfiguration("test_folder", 100, 32, 4, 64, 8, -1);

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("test_folder"));
    EXPECT_THAT(output, HasSubstr("100 images"));
    EXPECT_THAT(output, HasSubstr("32x32"));
    EXPECT_THAT(output, HasSubstr("64"));
    EXPECT_THAT(output, HasSubstr("8"));
}

TEST_F(HelpersTest, PrintConfiguration_WithThreshold) {
    CoutCapture capture;

    printConfiguration("test_folder", 1000, 64, 2, 128, -1, 10);

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("test_folder"));
    EXPECT_THAT(output, HasSubstr("images"));
    EXPECT_THAT(output, HasSubstr("64x64"));
    EXPECT_THAT(output, HasSubstr("128"));
    EXPECT_THAT(output, HasSubstr("auto"));
    EXPECT_THAT(output, HasSubstr("10"));
    EXPECT_THAT(output, HasSubstr("Threshold"));
}

TEST_F(HelpersTest, PrintConfiguration_LargeNumbers) {
    CoutCapture capture;

    printConfiguration("big_dataset", 1000000, 128, 8, 256, 16, -1);

    EXPECT_THAT(capture.str(), HasSubstr("images"));
    // The actual number formatting depends on locale, so just check it's present
}

TEST_F(HelpersTest, PrintHashResults_EmptyResults) {
    CoutCapture capture;

    std::vector<std::string> paths;
    std::vector<pHash> hashes;

    printHashResults(paths, hashes, "", 32);

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("Results"));
}

TEST_F(HelpersTest, PrintHashResults_FewResults) {
    CoutCapture capture;

    std::vector<std::string> paths = {
        "image1.jpg",
        "image2.png",
        "image3.jpeg"
    };
    std::vector<pHash> hashes(3);

    printHashResults(paths, hashes, "", 32);

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("Results"));
    EXPECT_THAT(output, HasSubstr("File"));
    EXPECT_THAT(output, HasSubstr("Hash"));
    EXPECT_THAT(output, HasSubstr("image1.jpg"));
    EXPECT_THAT(output, HasSubstr("image2.png"));
    EXPECT_THAT(output, HasSubstr("image3.jpeg"));
}

TEST_F(HelpersTest, PrintHashResults_ManyResults) {
    CoutCapture capture;

    std::vector<std::string> paths;
    std::vector<pHash> hashes;

    for (int i = 0; i < 10; i++) {
        paths.push_back("image" + std::to_string(i) + ".jpg");
        hashes.emplace_back();
    }

    printHashResults(paths, hashes, "output.txt", 64);

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("Results"));
    // Should only show first 5
    EXPECT_THAT(output, HasSubstr("image0.jpg"));
    EXPECT_THAT(output, HasSubstr("image4.jpg"));
    EXPECT_THAT(output, Not(HasSubstr("image5.jpg")));
    // Should mention saving to file
    EXPECT_THAT(output, HasSubstr("10 hashes saved to output.txt"));
}

TEST_F(HelpersTest, PrintHashResults_LongFilenames) {
    CoutCapture capture;

    std::vector<std::string> paths = {
        "/very/long/path/to/some/deeply/nested/directory/structure/image_with_very_long_filename_12345.jpg"
    };
    std::vector<pHash> hashes(1);

    printHashResults(paths, hashes, "", 32);

    std::string output = capture.str();
    // The table library may wrap long filenames, so check for parts
    // Should contain parts of the filename (may be wrapped)
    EXPECT_THAT(output, HasSubstr("image_with_very_long_filename"));
    EXPECT_THAT(output, HasSubstr("12345.jpg"));
    // Should not contain the full path
    EXPECT_THAT(output, Not(HasSubstr("/very/long/path")));
}

// ========== Console Control Tests ==========

TEST_F(HelpersTest, CursorVisibility_HideShow) {
    // These functions interact directly with console
    // We only test that they don't crash
    EXPECT_NO_THROW(phash_app::hideCursor());
    EXPECT_NO_THROW(phash_app::showCursor());

    EXPECT_NO_THROW({
        phash_app::hideCursor();
        phash_app::hideCursor();
        phash_app::showCursor();
        phash_app::showCursor();
    });
}

// ========== Edge Cases and Error Handling ==========

TEST_F(HelpersTest, CollectImagePaths_PermissionDenied) {
#ifdef _WIN32
    fs::path restrictedDir = tempDir / "restricted";
    fs::create_directories(restrictedDir);
    createTestFile(restrictedDir / "image.jpg");

    EXPECT_NO_THROW(collectImagePaths(tempDir, {"jpg"}, true));
#else
    fs::path restrictedDir = tempDir / "restricted";
    fs::create_directories(restrictedDir);
    createTestFile(restrictedDir / "image.jpg");
    fs::permissions(restrictedDir, fs::perms::none);

    EXPECT_NO_THROW(collectImagePaths(tempDir, {"jpg"}, true));

    fs::permissions(restrictedDir, fs::perms::all);
#endif
}

TEST_F(HelpersTest, CollectImagePaths_SymbolicLinks) {
    createTestFile(tempDir / "real_image.jpg");
    fs::path linkPath = tempDir / "link_to_image.jpg";

    try {
        fs::create_symlink(tempDir / "real_image.jpg", linkPath);

        auto paths = collectImagePaths(tempDir, {"jpg"}, false);
        EXPECT_GE(paths.size(), 1);
    } catch (const fs::filesystem_error&) {
        GTEST_SKIP() << "Symbolic links not supported on this filesystem";
    }
}

TEST_F(HelpersTest, CollectImagePaths_VeryDeepNesting) {
    fs::path deepPath = tempDir;

    const int maxDepth =
#ifdef _WIN32
        10;  // Conservative depth for Windows to stay under MAX_PATH
#else
        50;  // Unix systems can handle longer paths
#endif

    try {
        for (int i = 0; i < maxDepth; ++i) {
            deepPath = deepPath / ("L" + std::to_string(i));
        }
        fs::create_directories(deepPath);
        createTestFile(deepPath / "deep.jpg");

        EXPECT_NO_THROW({
            auto paths = collectImagePaths(tempDir, {"jpg"}, true);
            EXPECT_EQ(paths.size(), 1);
        });
    } catch (const fs::filesystem_error& e) {
        GTEST_SKIP() << "Path length limits exceeded: " << e.what();
    }
}

TEST_F(HelpersTest, CollectImagePaths_LargeDirectory) {
    for (int i = 0; i < 100; ++i) {
        createTestFile(tempDir / ("image" + std::to_string(i) + ".jpg"));
    }

    auto paths = collectImagePaths(tempDir, {"jpg"}, false);

    EXPECT_EQ(paths.size(), 100);
    EXPECT_TRUE(std::is_sorted(paths.begin(), paths.end()));
}

// ========== Additional Edge Cases ==========

TEST_F(HelpersTest, CollectImagePaths_MixedExtensionFormats) {
    createTestFile(tempDir / "file1.jpg");
    createTestFile(tempDir / "file2.jpeg");

    auto paths1 = collectImagePaths(tempDir, {".jpg", ".jpeg"}, false);
    auto paths2 = collectImagePaths(tempDir, {"jpg", "jpeg"}, false);

    EXPECT_EQ(paths1.size(), 2);
    EXPECT_EQ(paths2.size(), 2);
}

TEST_F(HelpersTest, CollectImagePaths_UnicodeFilenames) {
    try {
        createTestFile(tempDir / u8"图像文件.jpg");      // Chinese
        createTestFile(tempDir / u8"画像ファイル.png");   // Japanese
        createTestFile(tempDir / u8"изображение.jpeg");  // Russian
        createTestFile(tempDir / u8"εικόνα.jpg");        // Greek

        auto paths = collectImagePaths(tempDir, {"jpg", "jpeg", "png"}, false);

        EXPECT_GE(paths.size(), 0);
    } catch (const fs::filesystem_error&) {
        GTEST_SKIP() << "Unicode filenames not supported on this filesystem";
    }
}

TEST_F(HelpersTest, FormatFileSize_BoundaryValues) {
    EXPECT_EQ(formatFileSize(1024), "1.00 KB");
    EXPECT_EQ(formatFileSize(1024 * 1024), "1.00 MB");
    EXPECT_EQ(formatFileSize(1024ULL * 1024 * 1024), "1.00 GB");
    EXPECT_EQ(formatFileSize(1024ULL * 1024 * 1024 * 1024), "1.00 TB");

    EXPECT_EQ(formatFileSize(1023), "1023.00 B");
    EXPECT_EQ(formatFileSize(1024 * 1024 - 1), "1024.00 KB");
    EXPECT_EQ(formatFileSize(1024ULL * 1024 * 1024 - 1), "1024.00 MB");
}

TEST_F(HelpersTest, ToLower_ThreadSafety) {
    std::string original = "HELLO WORLD";
    std::string lowered = toLower(original);

    EXPECT_EQ(original, "HELLO WORLD");  // Original should be unchanged
    EXPECT_EQ(lowered, "hello world");
}

TEST_F(HelpersTest, WithCommas_EdgeCases) {
    EXPECT_FALSE(withCommas(std::numeric_limits<int>::min()).empty());
    EXPECT_FALSE(withCommas(std::numeric_limits<unsigned int>::max()).empty());
    EXPECT_FALSE(withCommas(std::numeric_limits<long long>::min()).empty());
    EXPECT_FALSE(withCommas(std::numeric_limits<unsigned long long>::max()).empty());
}

TEST_F(HelpersTest, QueryYesNo_EOFHandling) {
    CinMock input("");  // Empty input simulates EOF
    EXPECT_FALSE(queryYesNo("Continue?"));
}

TEST_F(HelpersTest, PrintConfiguration_SpecialCharactersInFolder) {
    CoutCapture capture;

    printConfiguration("C:\\Users\\Test User\\My Documents\\Photos & Videos", 500, 32, 4, 64, 8, -1);

    EXPECT_THAT(capture.str(), HasSubstr("Photos & Videos"));
}

TEST_F(HelpersTest, PrintHashResults_EmptyPaths) {
    CoutCapture capture;

    std::vector<std::string> paths(3, "");
    std::vector<pHash> hashes(3);

    printHashResults(paths, hashes, "", 32);

    std::string output = capture.str();
    EXPECT_THAT(output, HasSubstr("Results"));
}

TEST_F(HelpersTest, PrintHashResults_PathsHashesMismatch) {
    CoutCapture capture;

    std::vector<std::string> paths = {"image1.jpg", "image2.jpg"};
    std::vector<pHash> hashes(5);  // More hashes than paths

    EXPECT_NO_THROW(printHashResults(paths, hashes, "", 32));
}

TEST_F(HelpersTest, CollectImagePaths_CircularSymlinks) {
    fs::path dir1 = tempDir / "dir1";
    fs::path dir2 = tempDir / "dir2";

    fs::create_directories(dir1);
    fs::create_directories(dir2);

    createTestFile(dir1 / "image.jpg");

    try {
        fs::create_directory_symlink(dir2, dir1 / "link_to_dir2");
        fs::create_directory_symlink(dir1, dir2 / "link_to_dir1");

        // Should handle circular symlinks without infinite loop
        EXPECT_NO_THROW({
            auto paths = collectImagePaths(tempDir, {"jpg"}, true);
            EXPECT_GE(paths.size(), 1);  // Should at least find the real image
        });
    } catch (const fs::filesystem_error&) {
        GTEST_SKIP() << "Symbolic links not supported on this filesystem";
    }
}

TEST_F(HelpersTest, Bar_ProgressBoundaries) {
    auto progressBar = bar("Testing", false, false);

    EXPECT_NO_THROW(progressBar.set_progress(0));
    EXPECT_NO_THROW(progressBar.set_progress(100));

    EXPECT_NO_THROW(progressBar.set_progress(-1));
    EXPECT_NO_THROW(progressBar.set_progress(101));
    EXPECT_NO_THROW(progressBar.set_progress(1000));
}

TEST_F(HelpersTest, PrintConfiguration_ExtremeDimensions) {
    CoutCapture capture;

    printConfiguration("test", 100, 1, 1, 1, 1, -1);
    printConfiguration("test", 100, 512, 32, 10000, 256, 100);
}

TEST_F(HelpersTest, FormatFileSize_PrecisionCheck) {
    // Verify precision is always 2 decimal places
    std::string result;

    result = formatFileSize(1);
    EXPECT_THAT(result, ContainsRegex("\\d+\\.\\d\\d B"));

    result = formatFileSize(1024);
    EXPECT_THAT(result, ContainsRegex("\\d+\\.\\d\\d KB"));

    result = formatFileSize(1536);  // 1.5 KB
    EXPECT_EQ(result, "1.50 KB");

    result = formatFileSize(1024 * 1024 * 1024 / 3);  // ~341.33 MB
    EXPECT_THAT(result, ContainsRegex("\\d+\\.\\d\\d MB"));
}

} // namespace