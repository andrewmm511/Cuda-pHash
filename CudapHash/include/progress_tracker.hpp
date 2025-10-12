#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>

enum class PipelineStage {
    All,
    Read,
    Decode,
    Resize,
    Hash
};

struct ProgressInfo {
    size_t totalImages = 0;
    size_t readCompleted = 0;
    size_t decodedCompleted = 0;
    size_t resizedCompleted = 0;
    size_t hashedCompleted = 0;
    size_t failedImages = 0;

    float overallProgress() const {
        if (totalImages == 0) return 0.0f;

        float stageContribution = 0.25f;
        float progress = 0.0f;

        progress += stageContribution * (static_cast<float>(readCompleted) / totalImages);
        progress += stageContribution * (static_cast<float>(decodedCompleted) / totalImages);
        progress += stageContribution * (static_cast<float>(resizedCompleted) / totalImages);
        progress += stageContribution * (static_cast<float>(hashedCompleted) / totalImages);

        return progress;
    }

    size_t percentComplete(PipelineStage stage) const {
		if (totalImages == 0) return 0;

        switch (stage) {
            case PipelineStage::Read:
                return static_cast<size_t>((static_cast<float>(readCompleted) / totalImages) * 100);
            case PipelineStage::Decode:
                return static_cast<size_t>((static_cast<float>(decodedCompleted) / totalImages) * 100);
            case PipelineStage::Resize:
                return static_cast<size_t>((static_cast<float>(resizedCompleted) / totalImages) * 100);
            case PipelineStage::Hash:
                return static_cast<size_t>((static_cast<float>(hashedCompleted) / totalImages) * 100);
            default:
                return static_cast<size_t>(overallProgress() * 100);
        }
    }
};

// Thread-safe progress tracker with rate-limited callbacks
class ProgressTracker {
public:
    using ProgressCallback = std::function<void(const ProgressInfo&)>;

    ProgressTracker(size_t totalImages, ProgressCallback callback = nullptr);

    void update(PipelineStage stage, size_t count);
    void updateFailed(size_t count);
    void forceUpdate();

    ProgressInfo getProgress() const;

private:
    std::atomic<size_t> m_totalImages;
    std::atomic<size_t> m_readCompleted;
    std::atomic<size_t> m_decodedCompleted;
    std::atomic<size_t> m_resizedCompleted;
    std::atomic<size_t> m_hashedCompleted;
    std::atomic<size_t> m_failedImages;

    ProgressCallback m_callback;
    mutable std::mutex m_callbackMutex;
    std::chrono::steady_clock::time_point m_lastCallbackTime;
    const std::chrono::milliseconds m_minCallbackInterval;

    void tryInvokeCallback();
};