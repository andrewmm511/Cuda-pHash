// phash_base.h
#pragma once
#include <vector>
#include <string>

class PhashBase {
public:
    virtual ~PhashBase() = default;
    virtual std::vector<std::vector<uint32_t>> phash(const std::vector<std::string>& imagePaths) = 0;
};