//
//  CLI front-end for CUDA pHash library
//

#include "argparse.hpp"
#include "arguments.hpp"
#include "hash_command.hpp"
#include "similar_command.hpp"

#include <iostream>

using namespace phash_app;

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("phash", "1.0");
    program.add_description("CUDA-accelerated perceptual hash calculator");
    program.add_epilog("Examples:\n  phash hash -d ./photos -o hashes.csv\n  phash similar -d ./photos -t 3 --interactive\n\n"
                       "For detailed options: phash <command> --help");

    Arguments args;

    // Add subcommands
    argparse::ArgumentParser hash_command("hash");
    hash_command.add_description("Compute the perceptual hashes of images");

    argparse::ArgumentParser similar_command("similar");
    similar_command.add_description("Compute hashes and calculate visual similarity");

    // Common arguments for both commands
    auto addCommonArgs = [&](argparse::ArgumentParser& cmd) {
        cmd.add_argument("-d", "--directory")
            .required()
            .store_into(args.directory)
            .help("Directory containing images to process");

        cmd.add_argument("-e", "--extensions")
            .default_value(defaults::DEFAULT_EXTENSIONS)
			.store_into(args.extensions)
            .help("Image file extensions to include (comma-separated)");

        cmd.add_argument("-r", "--recursive")
            .implicit_value(true)
            .default_value(defaults::RECURSIVE)
			.store_into(args.recursive)
            .help("Search directories recursively for images");

        cmd.add_argument("-b", "--batch-size")
			.default_value(defaults::BATCH_SIZE)
            .scan<'i', int>()
            .store_into(args.batchSize)
            .help("Number of images to process in each GPU batch");

        cmd.add_argument("-T", "--threads")
			.default_value(defaults::THREADS)
            .scan<'i', int>()
			.store_into(args.threads)
            .help("Number of CPU threads for file I/O (-1 = auto-detect)");

        cmd.add_argument("--prefetch-factor")
			.default_value(defaults::PREFETCH_FACTOR)
            .scan<'i', int>()
			.store_into(args.prefetchFactor)
            .help("Multiplier for I/O queue size (higher = faster but more memory)");

        cmd.add_argument("-l", "--log-level")
            .default_value(defaults::LOG_LEVEL)
            .scan<'i', int>()
			.store_into(args.logLevel)
            .help("Internal logging verbosity (4=errors only, 3=info, 2=debug, 1=trace)");

        cmd.add_argument("-hs", "--hash-size")
			.default_value(defaults::HASH_SIZE)
            .scan<'i', int>()
			.store_into(args.hashSize)
            .help("Hash dimensions in bits (5-11, default 8 = 64-bit hash)");

        cmd.add_argument("-f", "--freq-factor")
			.default_value(defaults::FREQ_FACTOR)
            .scan<'i', int>()
			.store_into(args.freqFactor)
            .help("Frequency oversampling factor (higher = more accurate but slower)");
        };

    addCommonArgs(hash_command);
    addCommonArgs(similar_command);

    // Hash-specific arguments
    hash_command.add_argument("-o", "--output")
        .default_value(std::string(""))
		.store_into(args.outputPath)
        .help("Save computed hashes to CSV file");

    // Similar-specific arguments
    similar_command.add_argument("-t", "--threshold")
		.default_value(defaults::THRESHOLD)
        .scan<'i', int>()
		.store_into(args.threshold)
        .help("Similarity threshold (0=exact duplicate, 10=entirely different)");

    similar_command.add_argument("--num-tables")
		.default_value(defaults::NUM_TABLES)
        .scan<'i', int>()
		.store_into(args.numTables)
        .help("Number of hash tables for similarity search (advanced)");

    similar_command.add_argument("--bits-per-table")
		.default_value(defaults::BITS_PER_TABLE)
        .scan<'i', int>()
		.store_into(args.bitsPerTable)
        .help("Bits sampled per hash table (advanced)");

    similar_command.add_argument("-a", "--auto-delete")
        .implicit_value(true)
		.default_value(defaults::AUTO_DELETE)
		.store_into(args.autoDelete)
        .help("Automatically delete similar images (with confirmation prompt)");

    similar_command.add_argument("-i", "--interactive")
        .implicit_value(true)
		.default_value(defaults::INTERACTIVE)
		.store_into(args.interactive)
        .help("Review and confirm each deletion individually");

    similar_command.add_argument("-p", "--print-only")
        .implicit_value(true)
		.default_value(defaults::PRINT_ONLY)
		.store_into(args.printOnly)
        .help("Only display similar images without deleting");

    similar_command.add_argument("--dry-run")
        .implicit_value(true)
		.default_value(defaults::DRY_RUN)
		.store_into(args.dryRun)
        .help("Show what would be deleted without actually deleting");

    similar_command.add_argument("-o", "--output")
        .default_value(std::string(""))
		.store_into(args.outputPath)
        .help("Save list of similar images to CSV file");

    // Add subparsers
    program.add_subparser(hash_command);
    program.add_subparser(similar_command);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        std::cerr << program;
        return 1;
    }

    // Handle subcommands
    if (program.is_subcommand_used("hash")) {
        return handleHashCommand(args);
    }
    else if (program.is_subcommand_used("similar")) {
        return handleSimilarCommand(args);
    }
    else {
        std::cerr << "No command specified. Use 'hash' or 'similar'\n";
        std::cerr << program;
        return 1;
    }
}