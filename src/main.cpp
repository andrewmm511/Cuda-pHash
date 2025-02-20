#include <iostream>
#include "third_party/argparse.hpp"

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("phash");

    program.add_argument("--hash_size", "-h")
        .default_value(8)
        .scan<'i', int>()
        .help("Size of the hash");

    program.add_argument("--freq_factor", "-f")
        .default_value(6)
        .scan<'i', int>()
        .help("High frequency factor");

    program.add_argument("--batch_size", "-b")
        .default_value(100)
        .scan<'i', int>()
        .help("Number of images to process in a batch");

    program.add_argument("--cpu-only", "-c")
        .default_value(false)
        .implicit_value(true)
        .help("Use CPU instead of GPU (automatic if no GPU found or dataset too small)");

    program.add_argument("--directory", "-d")
        .required()
        .help("Path to the directory containing images");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
    }

    int hash_size = program.get<int>("--hash_size");
    int batch_size = program.get<int>("--batch_size");
    std::string directory = program.get<std::string>("--directory");

    std::cout << "Hash size: " << hash_size << "\nBatch size: " << batch_size << "\nDirectory: " << directory << "\n";

    return 0;
}