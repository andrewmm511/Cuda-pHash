#pragma once

#include "arguments.hpp"

namespace phash_app {

/**
 * Handles the 'hash' command for computing perceptual hashes of images
 * @param args Command arguments containing directory, output options, and processing parameters
 * @return 0 on success, 1 on error
 */
int handleHashCommand(const Arguments& args);

} // namespace phash_app