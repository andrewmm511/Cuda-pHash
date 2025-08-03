#include "pch.h"
#include "CppUnitTest.h"

#include "phash_cuda.cuh"

#include <intrin.h>
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
namespace fs = std::filesystem;


// Helpers
namespace TestHelpers
{
    // Convert UTF-8 std::string to std::wstring
    inline std::wstring s2ws(const std::string& s)
    {
        return std::wstring(s.begin(), s.end());
    }

    // Convert wide string to UTF-8 std::string
    inline std::string ws2s(const std::wstring& w)
    {
        if (w.empty()) return {};
        int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0,
            w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
        std::string ret(sizeNeeded, 0);
        WideCharToMultiByte(CP_UTF8, 0,
            w.data(), static_cast<int>(w.size()),
            ret.data(), sizeNeeded, nullptr, nullptr);
        return ret;
    }

    // RAII helper that mimics a streaming logger
    struct LogStream
    {
        std::ostringstream os;
        ~LogStream()
        {
            Logger::WriteMessage(os.str().c_str());
        }

        template<typename T>
        LogStream& operator<<(const T& v)
        {
            os << v;
            return *this;
        }
    };
} // namespace TestHelpers

#define LOG_INFO() TestHelpers::LogStream()


// Hashing utilities
namespace
{
    // CUDA pHash structure to binary string
    std::string hashToString(const pHash& h)
    {
        std::string out;
        out.reserve(128);
        for (int w = 0; w < 2; ++w)
            for (int b = 0; b < 64; ++b)
                out.push_back((h.words[w] & (1ULL << b)) ? '1' : '0');
        return out;
    }

    std::string formatHashString(const std::string& bin, int group = 8)
    {
        std::string ret;
        for (size_t i = 0; i < bin.size(); i += group)
        {
            if (i) ret += ' ';
            ret.append(bin.substr(i, group));
        }
        return ret;
    }

    int hammingDistance(const pHash& a, const pHash& b)
    {
        uint64_t x0 = a.words[0] ^ b.words[0];
        uint64_t x1 = a.words[1] ^ b.words[1];
        return static_cast<int>(__popcnt64(x0) + __popcnt64(x1));
    }

    std::string getFileName(const std::string& path)
    {
        return fs::path(path).filename().string();
    }

    std::string formatDuration(double seconds)
    {
        std::ostringstream os;
        if (seconds < 1.0)
            os << std::fixed << std::setprecision(2) << seconds * 1000 << " ms";
        else if (seconds < 60.0)
            os << std::fixed << std::setprecision(2) << seconds << " s";
        else
        {
            int m = static_cast<int>(seconds / 60);
            os << m << " m " << std::fixed << std::setprecision(1)
                << seconds - m * 60 << " s";
        }
        return os.str();
    }

    // Scan directory for JPEGs (handles wide-char file names on NTFS)
    std::vector<std::string> collectImagePaths(const std::string& dir,
        bool truncate = false,
        size_t maxCount = 50'000)
    {
        std::vector<std::string> out;
        WIN32_FIND_DATAW fd{};
        std::wstring search = TestHelpers::s2ws(dir) + L"*.jp*";

        HANDLE h = FindFirstFileW(search.c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE)
            return out;

        do
        {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                out.emplace_back(dir + TestHelpers::ws2s(fd.cFileName));
                if (truncate && out.size() >= maxCount) break;
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
        return out;
    }
} // anonymous namespace


// BAPPS helper structs
struct ImageTriplet
{
    std::string image_id;
    std::string ref_path, p0_path, p1_path;
    std::string human_choice;
    std::string human_choice_path;
};

struct ComparisonResult
{
    std::string image_id;
    std::string phash_choice;
    std::string human_choice;
    bool        agreement{ false };
    int         dist_ref_p0{ 0 };
    int         dist_ref_p1{ 0 };
};

class BAPPSTestHelper
{
public:
    static std::vector<ImageTriplet> readCsv(const std::string& csv)
    {
        std::ifstream in(csv);
        if (!in.is_open())
            throw std::runtime_error("Cannot open CSV: " + csv);

        std::vector<ImageTriplet> out;
        std::string line;
        bool header = true;
        while (std::getline(in, line))
        {
            if (header) { header = false; continue; }
            std::string_view sv(line);
            ImageTriplet t;
            size_t pos = 0, prev = 0;
            int fld = 0;
            while ((pos = sv.find(',', prev)) != std::string::npos)
            {
                switch (fld)
                {
                case 0: t.image_id = std::string(sv.substr(prev, pos - prev)); break;
                case 1: t.ref_path = std::string(sv.substr(prev, pos - prev)); break;
                case 2: t.p0_path = std::string(sv.substr(prev, pos - prev)); break;
                case 3: t.p1_path = std::string(sv.substr(prev, pos - prev)); break;
                case 4: t.human_choice = std::string(sv.substr(prev, pos - prev)); break;
                }
                prev = pos + 1;
                ++fld;
            }
            t.human_choice_path = std::string(sv.substr(prev));
            out.emplace_back(std::move(t));
        }
        return out;
    }

    static std::vector<std::string> getAllImagePaths(const std::vector<ImageTriplet>& t)
    {
        std::vector<std::string> paths;
        paths.reserve(t.size() * 3);
        for (const auto& it : t)
        {
            paths.push_back(it.ref_path);
            paths.push_back(it.p0_path);
            paths.push_back(it.p1_path);
        }
        std::sort(paths.begin(), paths.end());
        paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
        return paths;
    }

    static std::vector<ComparisonResult> compareHashes(
        const std::vector<ImageTriplet>& triplets,
        const std::unordered_map<std::string, pHash>& map)
    {
        std::vector<ComparisonResult> res;
        res.reserve(triplets.size());

        for (const auto& tr : triplets)
        {
            if (!map.count(tr.ref_path) || !map.count(tr.p0_path) || !map.count(tr.p1_path))
                continue;

            const auto& ref = map.at(tr.ref_path);
            const auto& p0 = map.at(tr.p0_path);
            const auto& p1 = map.at(tr.p1_path);

            int d0 = ::hammingDistance(ref, p0);
            int d1 = ::hammingDistance(ref, p1);

            ComparisonResult r;
            r.image_id = tr.image_id;
            r.phash_choice = (d0 <= d1 ? "p0" : "p1");
            r.human_choice = tr.human_choice;
            r.agreement = (r.phash_choice == tr.human_choice);
            r.dist_ref_p0 = d0;
            r.dist_ref_p1 = d1;
            res.emplace_back(std::move(r));
        }
        return res;
    }
};

// Main test class
namespace IntegrationTests
{
    TEST_CLASS(Hashing)
    {
        TEST_METHOD(HashSmallDataset)
        {
            Logger::WriteMessage("Tests ability to hash 5 images\n\n");

            auto images = collectImagePaths("C:/dedup_test3/", true, 5);
            if (images.empty()) Assert::Fail(L"No images found in test dataset");

            try
            {
                CudaPhash hasher(8, 4, 20);

                auto hashes = hasher.computeHashes(images);

                Assert::AreEqual<size_t>(images.size(), hashes.size(), L"One hash should be produced per image.");

                LOG_INFO() << "Computed " << hashes.size() << "/5 hashes\n";
                LOG_INFO() << "Checking for non-zero hashes...\n";

                for (size_t i = 0; i < hashes.size(); ++i)
                {
                    LOG_INFO() << getFileName(images[i]) << " = " << formatHashString(hashToString(hashes[i]).substr(0, 64)) << "\n";

                    Assert::IsTrue(hashes[i].words[0] != 0 || hashes[i].words[1] != 0,
                        (L"Hash for " + TestHelpers::s2ws(getFileName(images[i])) + L" is zero").c_str());
                }
            }
            catch (const std::exception& e)
            {
                Assert::Fail(TestHelpers::s2ws(e.what()).c_str());
            }
        }

        TEST_METHOD(HashMediumDataset)
        {
            Logger::WriteMessage("Tests ability to hash 2,000 images\n\n");

            auto images = collectImagePaths("C:/coco2017_2k/", true, 2000);
            if (images.size() < 2000)
                Assert::Fail(L"Not enough images for batch test");

            try
            {
                CudaPhash hasher(8, 4, 20);

                auto hashes = hasher.computeHashes(images);

                Assert::AreEqual<size_t>(images.size(), hashes.size(), L"One hash should be produced per image.");

                LOG_INFO() << "Computed " << hashes.size() << "/2000 hashes\n";
                LOG_INFO() << "Checking for non-zero hashes...\n";

                for (size_t i = 0; i < hashes.size(); ++i)
                {
                    Assert::IsTrue(hashes[i].words[0] != 0 || hashes[i].words[1] != 0,
                        (L"Hash for " + TestHelpers::s2ws(getFileName(images[i])) + L" is zero").c_str());
                }
                LOG_INFO() << "No non-zero hashes\n";
            }
            catch (const std::exception& e)
            {
                Assert::Fail(TestHelpers::s2ws(e.what()).c_str());
            }
        }

        TEST_METHOD(HashLargeDataset)
        {
            const std::string coco = "C:/coco2017/";
            if (!fs::exists(coco))
                Assert::Fail(L"COCO dataset not available");

            auto images = collectImagePaths(coco);
            if (images.empty())
                Assert::Fail(L"No images in COCO folder");

            const int perfTargetSec = 30;
            LOG_INFO() << "COCO PERFORMANCE TEST – " << images.size() << " files";

            try
            {
                CudaPhash ph(8, 4, 1200);

                auto t0 = std::chrono::high_resolution_clock::now();
                ph.findDuplicatesGPU(images, 3);
                double secs = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                LOG_INFO() << "Processing finished in " << formatDuration(secs);

                if (secs > perfTargetSec)
                {
                    std::ostringstream os;
                    os << "Performance target (" << perfTargetSec << " s) exceeded by "
                        << (secs - perfTargetSec) << " s";
                    Assert::Fail(TestHelpers::s2ws(os.str()).c_str());
                }
            }
            catch (const std::exception& e)
            {
                Assert::Fail(TestHelpers::s2ws(e.what()).c_str());
            }
        }

        TEST_METHOD(RelativeSimilarity)
        {
            Logger::WriteMessage("Tests distances of 3 image pairs. Expects Pair 1 < Pair 2 < Pair 3\n\n");

            const std::vector<std::pair<std::string, std::string>> pairs = {
                {"C:/dedup_test3/16.jpg",   "C:/dedup_test3/16-dup.jpg"}, // Very similar
                {"C:/dedup_test3/31.jpg",   "C:/dedup_test3/31-dup.jpg"}, // Similar
                {"C:/dedup_test3/12.jpg",   "C:/dedup_test3/13.jpg"} };   // Different

            bool missing = false;
            for (auto& p : pairs)
                if (!fs::exists(p.first) || !fs::exists(p.second))
                    missing = true;

            if (missing)
                Assert::Fail(L"Required JPEG files for this test are not present");

            CudaPhash hasher(8, 4, 20, -1, 8, 4);
            std::vector<int> distances;

            for (size_t i = 0; i < pairs.size(); ++i)
            {
                const auto& A = pairs[i].first;
                const auto& B = pairs[i].second;

                std::vector<pHash> hashes = hasher.computeHashes({ A,B });
                int dist = hammingDistance(hashes[0], hashes[1]);

                LOG_INFO() << "[Pair " << i + 1 << "] Distance = " << dist << " (" << getFileName(A) << " and " << getFileName(B) << ")\n";
                distances.push_back(dist);
            }

            if (distances.size() != 3 || distances[0] >= distances[1] || distances[1] >= distances[2])
            {
                std::ostringstream os;
                os << "Relative distances are not correct. Expected Pair 1 < Pair 2 < Pair 3\n";
                Assert::Fail(TestHelpers::s2ws(os.str()).c_str());
            }
        }

        TEST_METHOD(MultipleRuns)
        {
            Logger::WriteMessage("Tests consistency of pHash object across multiple runs\n");
            Logger::WriteMessage("Expects: Multiple runs of `computeHashes` results in same hash\n\n");

            const std::string img = "C:/dedup_test3/16.jpg";
            if (!fs::exists(img))
                Assert::Fail(L"Reference image not present");

            CudaPhash hasher(8, 4, 20);
            std::vector<pHash> runs;

            for (int i = 0; i < 5; ++i)
            {
                auto res = hasher.computeHashes({ img });
                runs.push_back(res[0]);
                LOG_INFO() << "Run " << i + 1 << " hash: " << formatHashString(hashToString(res[0]).substr(0, 32)) << "...\n";
            }

            for (size_t i = 1; i < runs.size(); ++i)
                if (runs[i].words[0] != runs[0].words[0] || runs[i].words[1] != runs[0].words[1])
                    Assert::Fail(L"Inconsistent hash values across runs");
        }
    };

    TEST_CLASS(Duplicates)
    {
        TEST_METHOD(DuplicateDetection)
        {
            Logger::WriteMessage("Tests ability to detect 2 duplicates in a small dataset\n\n");

            auto images = collectImagePaths("C:/dedup_test3/");
            if (images.empty()) Assert::Fail(L"No images found in test dataset");

            auto make_sorted_pair = [](const std::string& a, const std::string& b) {
                return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
                };

            const std::set<std::pair<std::string, std::string>> expected{
                make_sorted_pair("16.jpg", "16-dup.jpg"),
                make_sorted_pair("31.jpg", "31-dup.jpg") };

            try
            {
                CudaPhash hasher(8, 4, 100);

                std::vector<Image> dups = hasher.findDuplicatesGPU(images, 9);

                std::set<std::pair<std::string, std::string>> found;
                for (const auto& im : dups)
                {
                    auto a = getFileName(im.path);
                    auto b = getFileName(im.mostSimilarImage);
                    found.emplace(make_sorted_pair(a, b));
                }

                // Verification
                bool countsOk = (dups.size() == expected.size());
                bool allFound = std::all_of(expected.begin(), expected.end(),
                    [&found](auto& p) { return found.count(p); });
                bool noExtra = std::all_of(found.begin(), found.end(),
                    [&expected](auto& p) { return expected.count(p); });

                if (!countsOk || !allFound || !noExtra)
                {
                    std::ostringstream os;
                    os << "Duplicate detection failed.\nExpected (" << expected.size() << "): ";
                    for (auto& p : expected) os << '[' << p.first << "<->" << p.second << "] ";
                    os << "\nFound (" << found.size() << "): ";
                    for (auto& p : found)    os << '[' << p.first << "<->" << p.second << "] ";
                    Assert::Fail(TestHelpers::s2ws(os.str()).c_str());
                }
            }
            catch (const std::exception& e)
            {
                Assert::Fail(TestHelpers::s2ws(e.what()).c_str());
            }
        }
        
        TEST_METHOD(BAPPSDatasetEvaluation)
        {
            const std::string csv = "C:/bapps/clean/bapps_clean.csv";
            if (!fs::exists(csv))
                Assert::Fail(L"BAPPS CSV not present");

            LOG_INFO() << "BAPPS DATASET EVALUATION";

            try
            {
                auto triplets = BAPPSTestHelper::readCsv(csv);
                Assert::IsTrue(!triplets.empty(), L"No triplets parsed from CSV");

                auto allImgs = BAPPSTestHelper::getAllImagePaths(triplets);

                CudaPhash ph(8, 8, 1500);
                auto t0 = std::chrono::high_resolution_clock::now();
                auto allHashes = ph.computeHashes(allImgs);
                double secs = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                LOG_INFO() << "Hashes for " << allImgs.size() << " pictures computed in "
                    << formatDuration(secs);

                std::unordered_map<std::string, pHash> map;
                for (size_t i = 0; i < allImgs.size(); ++i) map[allImgs[i]] = allHashes[i];

                auto cmp = BAPPSTestHelper::compareHashes(triplets, map);

                size_t correct = std::count_if(cmp.begin(), cmp.end(),
                    [](const ComparisonResult& r) { return r.agreement; });
                double acc = cmp.empty() ? 0.0 : 100.0 * correct / cmp.size();

                LOG_INFO() << "Agreement with human choice: " << std::fixed
                    << std::setprecision(2) << acc << '%';

                if (acc < 50.0)
                {
                    std::ostringstream os;
                    os << "Agreement below threshold: " << acc << '%';
                    Assert::Fail(TestHelpers::s2ws(os.str()).c_str());
                }
            }
            catch (const std::exception& ex)
            {
                Assert::Fail(TestHelpers::s2ws(ex.what()).c_str());
            }
        }
    };
} // namespace IntegrationTests