#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>

// Structure containing progress information for all pipeline stages
struct ProgressInfo {
    size_t totalImages = 0;
    size_t readCompleted = 0;
    size_t decodedCompleted = 0;
    size_t resizedCompleted = 0;
    size_t hashedCompleted = 0;
    size_t failedImages = 0;  // Track failed images

    // Queue depths for bottleneck analysis
    size_t decodeQueueDepth = 0;
    size_t resizeQueueDepth = 0;
    size_t hashQueueDepth = 0;

    // Calculate overall progress as a percentage (0.0 to 1.0)
    float overallProgress() const {
        if (totalImages == 0) return 0.0f;

        // Each stage contributes 25% to overall progress
        float stageContribution = 0.25f;
        float progress = 0.0f;

        progress += stageContribution * (static_cast<float>(readCompleted) / totalImages);
        progress += stageContribution * (static_cast<float>(decodedCompleted) / totalImages);
        progress += stageContribution * (static_cast<float>(resizedCompleted) / totalImages);
        progress += stageContribution * (static_cast<float>(hashedCompleted) / totalImages);

        return progress;
    }

    size_t percentComplete() const {
        return static_cast<size_t>(overallProgress() * 100);
    }

    size_t percentRead() const {
        return (totalImages == 0) ? 0 : static_cast<size_t>((static_cast<float>(readCompleted) / totalImages) * 100);
    }

    size_t percentDecoded() const {
        return (totalImages == 0) ? 0 : static_cast<size_t>((static_cast<float>(decodedCompleted) / totalImages) * 100);
	}

    size_t percentResized() const {
		return (totalImages == 0) ? 0 : static_cast<size_t>((static_cast<float>(resizedCompleted) / totalImages) * 100);
	}

    size_t percentHashed() const {
		return (totalImages == 0) ? 0 : static_cast<size_t>((static_cast<float>(hashedCompleted) / totalImages) * 100);
	}

    // Check if any failures occurred
    bool hasFailures() const {
        return failedImages > 0;
    }
};

// Thread-safe progress tracker with rate-limited callbacks
class ProgressTracker {
public:
    using ProgressCallback = std::function<void(const ProgressInfo&)>;

    // Constructor
    ProgressTracker(size_t totalImages, ProgressCallback callback = nullptr);

    // Update methods for each stage (thread-safe)
    void updateRead(size_t count);
    void updateDecoded(size_t count);
    void updateResized(size_t count);
    void updateHashed(size_t count);
    void updateFailed(size_t count);  // Update failed image count

    // Update queue depths
    void updateQueueDepths(size_t decodeDepth, size_t resizeDepth, size_t hashDepth);

    // Force an immediate callback invocation (useful for final update)
    void forceUpdate();

    // Get current progress info
    ProgressInfo getProgress() const;

private:
    // Atomic counters for lock-free updates
    std::atomic<size_t> m_totalImages;
    std::atomic<size_t> m_readCompleted;
    std::atomic<size_t> m_decodedCompleted;
    std::atomic<size_t> m_resizedCompleted;
    std::atomic<size_t> m_hashedCompleted;
    std::atomic<size_t> m_failedImages;

    // Queue depths (less frequent updates, so using mutex)
    mutable std::mutex m_queueMutex;
    size_t m_decodeQueueDepth;
    size_t m_resizeQueueDepth;
    size_t m_hashQueueDepth;

    // Callback and rate limiting
    ProgressCallback m_callback;
    mutable std::mutex m_callbackMutex;
    std::chrono::steady_clock::time_point m_lastCallbackTime;
    const std::chrono::milliseconds m_minCallbackInterval;

    // Internal method to check if callback should be invoked
    void tryInvokeCallback();
};