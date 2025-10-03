#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>

struct Image;

class WorkQueue {
private:
    std::vector<Image*> m_items;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond;
    std::atomic<bool> m_sentinel{ false };
    size_t m_maxCapacity;
    std::string m_name;

public:
    WorkQueue(size_t maxCapacity, std::string name)
        : m_maxCapacity(maxCapacity), m_name(std::move(name)) {
    }

    void push(Image* item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return m_items.size() < m_maxCapacity; });
        m_items.emplace_back(item);
        m_cond.notify_one();
    }

    void pushMany(const std::vector<Image*>& items) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this, &items] { return m_items.size() + items.size() <= m_maxCapacity; });
        m_items.insert(m_items.end(), items.begin(), items.end());
        m_cond.notify_all();
    }

    Image* pop() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return !m_items.empty() || m_sentinel.load(std::memory_order_acquire); });
        if (m_items.empty()) { return nullptr; }
        Image* result = m_items.back();
        m_items.pop_back();
        m_cond.notify_one();
        return result;
    }

    std::vector<Image*> popMax(size_t maxItems) {
        std::vector<Image*> items;
        items.reserve(maxItems);
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this, maxItems]() { return m_sentinel.load(std::memory_order_acquire) || m_items.size() >= maxItems; });
        size_t count = std::min(m_items.size(), maxItems);
        for (size_t i = 0; i < count; ++i) {
            items.push_back(std::move(m_items.back()));
            m_items.pop_back();
        }
        m_cond.notify_all();
        return items;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_items.empty();
    }

    bool isSentinel() const {
        return m_sentinel.load(std::memory_order_acquire);
    }

    void setSentinel() {
        m_sentinel.store(true, std::memory_order_release);
        m_cond.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_items.size();
    }
};
