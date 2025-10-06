#include "../include/progress_tracker.hpp"

ProgressTracker::ProgressTracker(size_t totalImages, ProgressCallback callback)
    : m_totalImages(totalImages),
      m_readCompleted(0),
      m_decodedCompleted(0),
      m_resizedCompleted(0),
      m_hashedCompleted(0),
      m_failedImages(0),
      m_decodeQueueDepth(0),
      m_resizeQueueDepth(0),
      m_hashQueueDepth(0),
      m_callback(callback),
      m_lastCallbackTime(std::chrono::steady_clock::now()),
      m_minCallbackInterval(500)  // 500ms = 2 updates per second
{
}

void ProgressTracker::updateRead(size_t count) {
    m_readCompleted.fetch_add(count, std::memory_order_relaxed);
    tryInvokeCallback();
}

void ProgressTracker::updateDecoded(size_t count) {
    m_decodedCompleted.fetch_add(count, std::memory_order_relaxed);
    tryInvokeCallback();
}

void ProgressTracker::updateResized(size_t count) {
    m_resizedCompleted.fetch_add(count, std::memory_order_relaxed);
    tryInvokeCallback();
}

void ProgressTracker::updateHashed(size_t count) {
    m_hashedCompleted.fetch_add(count, std::memory_order_relaxed);
    tryInvokeCallback();
}

void ProgressTracker::updateFailed(size_t count) {
    m_failedImages.fetch_add(count, std::memory_order_relaxed);
    tryInvokeCallback();
}

void ProgressTracker::updateQueueDepths(size_t decodeDepth, size_t resizeDepth, size_t hashDepth) {
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_decodeQueueDepth = decodeDepth;
        m_resizeQueueDepth = resizeDepth;
        m_hashQueueDepth = hashDepth;
    }
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

    // Read atomic values
    info.totalImages = m_totalImages.load(std::memory_order_relaxed);
    info.readCompleted = m_readCompleted.load(std::memory_order_relaxed);
    info.decodedCompleted = m_decodedCompleted.load(std::memory_order_relaxed);
    info.resizedCompleted = m_resizedCompleted.load(std::memory_order_relaxed);
    info.hashedCompleted = m_hashedCompleted.load(std::memory_order_relaxed);
    info.failedImages = m_failedImages.load(std::memory_order_relaxed);

    // Read queue depths under lock
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        info.decodeQueueDepth = m_decodeQueueDepth;
        info.resizeQueueDepth = m_resizeQueueDepth;
        info.hashQueueDepth = m_hashQueueDepth;
    }

    return info;
}

void ProgressTracker::tryInvokeCallback() {
    if (!m_callback) return;

    // Check if enough time has passed since last callback
    auto now = std::chrono::steady_clock::now();

    // Try to acquire lock without blocking
    std::unique_lock<std::mutex> lock(m_callbackMutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        // Another thread is already invoking callback, skip this update
        return;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastCallbackTime);
    if (elapsed < m_minCallbackInterval) {
        // Not enough time has passed, skip this update
        return;
    }

    // Invoke callback
    ProgressInfo info = getProgress();
    m_callback(info);
    m_lastCallbackTime = now;
}