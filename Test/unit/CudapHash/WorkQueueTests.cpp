// WorkQueue unit tests using Google Test
// These tests exercise blocking semantics, sentinel behavior, capacity bounds,
// and basic concurrency for the WorkQueue without depending on the full Image definition.

#include "pch.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "work_queue.hpp"

using namespace std::chrono_literals;

// Helper to create a distinct, real pointer without depending on Image definition.
// We allocate raw storage and cast to Image*; we never dereference, so layout isn't needed.
static Image* make_fake_item() {
    return static_cast<Image*>(::operator new(1));
}

static void free_fake_item(Image* p) {
    ::operator delete(static_cast<void*>(p));
}

TEST(WorkQueueTests, PopBlocksUntilPush) {
    WorkQueue q(2, "pop-blocks");

    std::promise<Image*> popped_prom;
    auto popped_fut = popped_prom.get_future();

    std::thread consumer([&] {
        popped_prom.set_value(q.pop());
    });

    // Ensure pop() is blocking (future should time out briefly)
    EXPECT_EQ(std::future_status::timeout, popped_fut.wait_for(50ms));

    Image* item = make_fake_item();
    q.push(item);

    ASSERT_EQ(std::future_status::ready, popped_fut.wait_for(500ms));
    Image* got = popped_fut.get();
    EXPECT_EQ(got, item);

    consumer.join();
    free_fake_item(item);
}

TEST(WorkQueueTests, PopReturnsNullAfterSentinelWhenEmpty) {
    WorkQueue q(1, "sentinel-empty");
    q.setSentinel();
    Image* got = q.pop();
    EXPECT_EQ(got, nullptr);
}

TEST(WorkQueueTests, PopMaxBlocksUntilEnoughOrSentinel) {
    WorkQueue q(10, "popmax");

    std::promise<std::vector<Image*>> prom;
    auto fut = prom.get_future();

    std::thread consumer([&] {
        prom.set_value(q.popMax(3));
    });

    // Not enough items yet, should still be waiting
    EXPECT_EQ(std::future_status::timeout, fut.wait_for(50ms));

    Image* a = make_fake_item();
    Image* b = make_fake_item();
    q.push(a);
    q.push(b);

    // Still waiting for 3 items (or sentinel)
    EXPECT_EQ(std::future_status::timeout, fut.wait_for(50ms));

    // Unblock via sentinel; should return the two available items
    q.setSentinel();

    ASSERT_EQ(std::future_status::ready, fut.wait_for(500ms));
    std::vector<Image*> got = fut.get();
    ASSERT_EQ(got.size(), 2u);

    // WorkQueue is LIFO: last pushed is first popped
    EXPECT_EQ(got[0], b);
    EXPECT_EQ(got[1], a);

    consumer.join();
    free_fake_item(a);
    free_fake_item(b);
}

TEST(WorkQueueTests, PushBlocksWhenFullUntilPop) {
    WorkQueue q(2, "cap");

    Image* a = make_fake_item();
    Image* b = make_fake_item();
    Image* c = make_fake_item();

    q.push(a);
    q.push(b);
    ASSERT_EQ(q.size(), 2u);

    std::promise<void> pushed_prom;
    auto pushed_fut = pushed_prom.get_future();

    std::thread producer([&] {
        q.push(c); // should block until one item is popped
        pushed_prom.set_value();
    });

    // Confirm producer is blocked
    EXPECT_EQ(std::future_status::timeout, pushed_fut.wait_for(50ms));

    // Make space
    Image* got = q.pop();
    (void)got; // not used further

    // Producer should complete promptly
    ASSERT_EQ(std::future_status::ready, pushed_fut.wait_for(500ms));
    producer.join();

    EXPECT_EQ(q.size(), 2u);

    free_fake_item(a);
    free_fake_item(b);
    free_fake_item(c);
}

TEST(WorkQueueTests, PushManyWaitsForSufficientSpace) {
    WorkQueue q(3, "pushmany");

    Image* a = make_fake_item();
    Image* b = make_fake_item();
    Image* c = make_fake_item();
    Image* d = make_fake_item();

    q.push(a); // occupancy 1/3

    std::vector<Image*> batch{ b, c, d }; // needs space for 3 more

    std::promise<void> done_prom;
    auto done_fut = done_prom.get_future();

    std::thread producer([&] {
        q.pushMany(batch); // should block until queue is empty enough
        done_prom.set_value();
    });

    // Not enough space yet
    EXPECT_EQ(std::future_status::timeout, done_fut.wait_for(50ms));

    // Free space by popping the single item
    (void)q.pop();

    ASSERT_EQ(std::future_status::ready, done_fut.wait_for(500ms));
    producer.join();

    EXPECT_EQ(q.size(), 3u);

    // Drain to avoid leaks in test allocation
    for (size_t i = 0; i < 3; ++i) (void)q.pop();

    free_fake_item(a);
    free_fake_item(b);
    free_fake_item(c);
    free_fake_item(d);
}

TEST(WorkQueueTests, MultiProducerMultiConsumerWithSentinel) {
    constexpr int producers = 3;
    constexpr int itemsPerProducer = 50;
    const int totalItems = producers * itemsPerProducer;

    WorkQueue q(128, "mpmc");

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    // Producers
    std::vector<std::thread> prodThreads;
    prodThreads.reserve(producers);
    for (int p = 0; p < producers; ++p) {
        prodThreads.emplace_back([&] {
            for (int i = 0; i < itemsPerProducer; ++i) {
                Image* it = make_fake_item();
                q.push(it);
                produced.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Consumers
    constexpr int consumers = 4;
    std::vector<std::thread> consThreads;
    consThreads.reserve(consumers);
    std::atomic<bool> running{true};

    for (int cidx = 0; cidx < consumers; ++cidx) {
        consThreads.emplace_back([&] {
            for (;;) {
                Image* it = q.pop();
                if (!it) break; // sentinel observed and queue empty
                // release storage for the fake item
                free_fake_item(it);
                consumed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Join producers then signal completion
    for (auto& t : prodThreads) t.join();
    q.setSentinel();

    for (auto& t : consThreads) t.join();

    EXPECT_EQ(produced.load(), totalItems);
    EXPECT_EQ(consumed.load(), totalItems);
    EXPECT_TRUE(q.isSentinel());
    EXPECT_TRUE(q.empty());
}

