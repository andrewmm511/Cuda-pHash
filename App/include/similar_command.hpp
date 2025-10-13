#pragma once

#include "arguments.hpp"

namespace phash_app {

/**
 * Handles the 'similar' command for finding visually similar images
 * @param args Command arguments containing directory, threshold, and deletion options
 * @return 0 on success, 1 on error
 */
int handleSimilarCommand(const Arguments& args);

} // namespace phash_app