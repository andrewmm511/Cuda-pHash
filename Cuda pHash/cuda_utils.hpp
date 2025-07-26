#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <tuple>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                                 " - " + cudaGetErrorString(error)); \
    } \
} while(0)

// Constants
constexpr int DEFAULT_RNG_SEED = 12345;
constexpr size_t MAX_EDGES_FACTOR = 200;  // max edges = 2 * n * MAX_EDGES_FACTOR
constexpr float PI = 3.14159265358979323846f;

// Helper functions
inline std::vector<int> coverByHighestDegree(const std::vector<std::pair<int, int>>& edges, int n) {
    if (edges.empty()) { return {}; }

    std::unordered_map<int, std::vector<int>> adj;
    std::unordered_map<int, int> degree;

    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
        degree[e.first]++;
        degree[e.second]++;
    }

    int remainingEdges = static_cast<int>(edges.size());
    std::vector<int> toDelete;
    toDelete.reserve(degree.size());

    using PII = std::pair<int, int>;
    std::priority_queue<PII> pq;

    for (const auto& [node, deg] : degree) {
        if (deg > 0) { pq.push({ deg, node }); }
    }

    std::unordered_set<int> deleted;

    while (remainingEdges > 0 && !pq.empty()) {
        auto [deg, maxNode] = pq.top();
        pq.pop();

        if (deleted.count(maxNode) || degree[maxNode] != deg) { continue; }

        deleted.insert(maxNode);
        toDelete.push_back(maxNode);

        for (int neighbor : adj[maxNode]) {
            if (!deleted.count(neighbor) && degree[neighbor] > 0) {
                degree[neighbor]--;
                remainingEdges--;
                if (degree[neighbor] > 0) {
                    pq.push({ degree[neighbor], neighbor });
                }
            }
        }
    }

    return toDelete;
}