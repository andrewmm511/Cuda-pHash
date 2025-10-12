#include "../include/progress_tracker.hpp"

ProgressTracker::ProgressTracker(size_t totalImages, ProgressCallback callback)
    : m_totalImages(totalImages),
      m_readCompleted(0),
      m_decodedCompleted(0),
      m_resizedCompleted(0),
      m_hashedCompleted(0),
      m_failedImages(0),
      m_callback(callback),
      m_lastCallbackTime(std::chrono::steady_clock::now()),
      m_minCallbackInterval(500)  // 500ms = 2 updates per second
{
}

void ProgressTracker::update(PipelineStage stage, size_t count) {
    switch (stage) {
        case PipelineStage::Read:
            m_readCompleted.fetch_add(count, std::memory_order_relaxed);
            break;
        case PipelineStage::Decode:
            m_decodedCompleted.fetch_add(count, std::memory_order_relaxed);
            break;
        case PipelineStage::Resize:
            m_resizedCompleted.fetch_add(count, std::memory_order_relaxed);
            break;
        case PipelineStage::Hash:
            m_hashedCompleted.fetch_add(count, std::memory_order_relaxed);
            break;
        default:
            // Invalid stage for single update
            break;
	}
    
    tryInvokeCallback();
}

void ProgressTracker::updateFailed(size_t count) {
    m_failedImages.fetch_add(count, std::memory_order_relaxed);
    tryInvokeCallback();
}

void ProgressTracker::forceUpdate() {
    if (!m_callback) return;

    std::lock_guard<std::mutex> lock(m_callbackMutex);

    ProgressInfo info = getProgress();
    m_callback(info);
    m_lastCallbackTime = std::chrono::steady_clock::now();
}

ProgressInfo ProgressTracker::getProgress() const {
    ProgressInfo info;

    info.totalImages = m_totalImages.load(std::memory_order_relaxed);
    info.readCompleted = m_readCompleted.load(std::memory_order_relaxed);
    info.decodedCompleted = m_decodedCompleted.load(std::memory_order_relaxed);
    info.resizedCompleted = m_resizedCompleted.load(std::memory_order_relaxed);
    info.hashedCompleted = m_hashedCompleted.load(std::memory_order_relaxed);
    info.failedImages = m_failedImages.load(std::memory_order_relaxed);

    return info;
}

void ProgressTracker::tryInvokeCallback() {
    if (!m_callback) return;

    std::unique_lock<std::mutex> lock(m_callbackMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastCallbackTime);
    if (elapsed < m_minCallbackInterval) return;

    // Invoke callback
    ProgressInfo info = getProgress();
    m_callback(info);
    m_lastCallbackTime = now;
}