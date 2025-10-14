#include <gtest/gtest.h>

// First, let's test if this basic file compiles
TEST(ArgumentsDiagnostic, BasicCompiles) {
    EXPECT_EQ(1, 1);
}

// Now try including just the filesystem header
#include <filesystem>

TEST(ArgumentsDiagnostic, FilesystemWorks) {
    std::filesystem::path p = "test";
    EXPECT_FALSE(p.empty());
}

// Now try including arguments.hpp
// If tests disappear after uncommenting this, we know the header is the problem
#include "arguments.hpp"

TEST(ArgumentsDiagnostic, ArgumentsHeaderFound) {
    // Just test that we can reference the namespace
    using namespace phash_app;
    EXPECT_EQ(phash_app::defaults::HASH_SIZE, 8);
}

// Try creating a RawArguments
TEST(ArgumentsDiagnostic, CanCreateRawArguments) {
    phash_app::RawArguments raw;
    raw.hashSize = 8;
    EXPECT_EQ(raw.hashSize, 8);
}

// Try creating an Arguments object (this requires linking)
TEST(ArgumentsDiagnostic, CanCreateArguments) {
    phash_app::RawArguments raw;
    raw.directory = std::filesystem::current_path();

    // This will test if we can actually link with the Arguments constructor
    phash_app::Arguments args(raw, phash_app::Arguments::Command::Hash);
    EXPECT_EQ(args.hashSize, 8);
}