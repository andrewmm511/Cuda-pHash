#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>   // For sprintf
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <charconv>

#define TRACE(...) logger::trace(__VA_ARGS__)
#define DEBUG(...) logger::debug(__VA_ARGS__)
#define INFO(...) logger::info(__VA_ARGS__)
#define WARN(...) logger::warn(__VA_ARGS__)
#define ERROR_L(...) logger::error(__VA_ARGS__)

/**
 * -------------
 * USAGE OVERVIEW
 * -------------
 *
 * // Initialize once, specifying log level and ring size (power-of-two):
 * logger::init(logger::Level::INFO, 4096);
 *
 * // Log from multiple threads with:
 * logger::log(logger::Level::DEBUG, "identifier", arg1, argN);
 *
 * // On shutdown:
 * logger::shutdown();
*/

namespace logger
{
    enum Level : uint8_t
    {
        TRACE = 0,
        DEBUG,
        INFO,
        WARN,
        ERROR_L
    };

    constexpr const char* levelToStr(Level lv) noexcept
    {
        switch (lv)
        {
        case TRACE: return "TRACE";
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO";
        case WARN:  return "WARN";
        case ERROR_L: return "ERROR";
        }
        return "UNKNOWN";
    }

    //-----------------------------------
    // Internal Helper to Build Strings
    //-----------------------------------
    namespace detail
    {
        // Append C-string
        inline void appendOne(std::string& dest, const char* str)
        {
            if (str) {
                dest += str;
            }
        }

        // Append std::string
        inline void appendOne(std::string& dest, const std::string& s)
        {
            dest += s;
        }

        // Append arithmetic (int, float, etc.) via std::to_chars
        template <typename T,
            typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
        inline void appendOne(std::string& dest, T val)
        {
            char buf[64];
            auto end = std::to_chars(buf, buf + sizeof(buf), val).ptr;
            dest.append(buf, static_cast<size_t>(end - buf));
        }

        // Fallback for any other type that implements operator<<
        template <typename T,
            typename std::enable_if_t<!std::is_arithmetic_v<std::decay_t<T>> &&
            !std::is_same_v<std::decay_t<T>, std::string> &&
            !std::is_same_v<std::decay_t<T>, const char*>, int> = 0>
        inline void appendOne(std::string& dest, const T& val)
        {
            thread_local std::ostringstream oss;
            oss.str(std::string{});
            oss.clear();
            oss << val;
            dest += oss.str();
        }

        // Variadic build
        template <typename... Ts>
        inline void buildString(std::string& dest, Ts&&... args)
        {
            (appendOne(dest, std::forward<Ts>(args)), ...);
        }
    } // namespace detail

    //-----------------------------------
    // Internal Ring Buffer
    //-----------------------------------
    struct LogMessage
    {
        Level level;
		char padding[7]; // Padding for 8-byte alignment
        int64_t microsSinceEpoch;
        std::string id;
        std::string text;
    };

    class LogRing
    {
    public:
        LogRing(size_t size) : size_(size), mask_(size - 1), buffer_(new LogMessage[size]), head_(0), tail_(0) {}

        ~LogRing() { delete[] buffer_; }

        // Non-blocking multi-producer push. Returns true if stored, false if the ring was full.
        bool tryPush(LogMessage&& msg)
        {
            const auto pos = tail_.fetch_add(1, std::memory_order_acq_rel);
            const auto idx = pos & mask_;

            if (pos - head_.load(std::memory_order_acquire) >= size_) {
                tail_.fetch_sub(1, std::memory_order_release);
                return false;
            }

            buffer_[idx] = std::move(msg);
            return true;
        }

        // Single-consumer pop
        bool tryPop(LogMessage& out)
        {
            const auto currentHead = head_.load(std::memory_order_relaxed);
            if (currentHead >= tail_.load(std::memory_order_acquire)) return false;

            out = std::move(buffer_[currentHead & mask_]);
            head_.store(currentHead + 1, std::memory_order_release);
            return true;
        }

    private:
        // Prevent false sharing by separating atomic variables
        size_t size_;
        size_t mask_;
        LogMessage* buffer_;
        alignas(64) std::atomic<uint64_t> head_;  // Align to cache line boundary
        char padding[64 - sizeof(std::atomic<uint64_t>)]; // Padding
        alignas(64) std::atomic<uint64_t> tail_;  // Align to cache line boundary
    };

    //-----------------------------------
    // The Logger Singleton
    //-----------------------------------
    class Logger
    {
    public:
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        static Logger& instance()
        {
            static Logger s;
            return s;
        }

        // Called once at startup
        void init(Level level, size_t ringSize)
        {
            if (!ring_)
            {
                ring_ = new LogRing(ringSize);
                currentLevel_ = level;
                running_.store(true, std::memory_order_release);
                consumerThread_ = std::thread(&Logger::consumerLoop, this);
            }
        }

        // Non-blocking log function
        template<typename IdType, typename... Args>
        void log(Level lv, IdType&& id, Args&&... args)
        {
            if (lv < currentLevel_) return;

            // Build the user message
            std::string text;
            text.reserve(128); // Avoid repeated allocations
            detail::buildString(text, std::forward<Args>(args)...);

            // Create the log message struct
            LogMessage msg;
            msg.level = lv;
            msg.microsSinceEpoch = nowMicrosSinceEpoch();
            msg.id = std::forward<IdType>(id);
            msg.text = std::move(text);

            ring_->tryPush(std::move(msg));
        }

        void shutdown()
        {
            if (running_.exchange(false, std::memory_order_acq_rel))
            {
                if (consumerThread_.joinable()) {
                    consumerThread_.join();
                }
                
                delete ring_;
                ring_ = nullptr;
            }
        }

    private:
        Logger() : ring_(nullptr), consumerThread_() {}
        ~Logger() { shutdown(); }

        static int64_t nowMicrosSinceEpoch()
        {
            using namespace std::chrono;
            return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
        }

        void consumerLoop()
        {
            // Continuously pop from ring and print
            while (running_.load(std::memory_order_acquire))
            {
                LogMessage msg;
                while (ring_->tryPop(msg)) { printMessage(msg); }

                // Sleep a bit to reduce CPU usage
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // Drain any remaining
            LogMessage leftover;
            while (ring_->tryPop(leftover)) { printMessage(leftover); }
        }

        static void printMessage(const LogMessage& lm)
        {
            using namespace std::chrono;
            auto micros = microseconds(lm.microsSinceEpoch);
            auto tp = system_clock::time_point(micros);

            std::time_t t = system_clock::to_time_t(tp);
            std::tm tmBuf{};
#ifdef _WIN32
            gmtime_s(&tmBuf, &t);
#else
            gmtime_r(&t, &tmBuf);
#endif

            // Compute mm part
            auto dur = tp.time_since_epoch();
            auto msPart = duration_cast<milliseconds>(dur).count() % 1000;

            char timeBuf[32];
            std::snprintf(timeBuf, sizeof(timeBuf),
                "%02d:%02d:%02d.%03d",
                tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec,
                static_cast<int>(msPart));

            std::cout << "[" << timeBuf << "] "
                << "[" << levelToStr(lm.level) << "] ";
            if (!lm.id.empty()) {
                std::cout << "[" << lm.id << "] ";
            }
            std::cout << lm.text << "\n";
        }

        // Fields
        LogRing* ring_;
        Level currentLevel_;
        std::atomic<bool> running_;
        std::thread consumerThread_;
    };

    //-----------------------------------
    // 4) Public API
    //-----------------------------------
    inline void init(Level lv, size_t ringSize = 4096)
    {
        Logger::instance().init(lv, ringSize);
    }

    inline void shutdown()
    {
        Logger::instance().shutdown();
    }

    template<typename IdType, typename... Args>
    inline void log(Level lv, IdType&& id, Args&&... args)
    {
        Logger::instance().log(lv, std::forward<IdType>(id), std::forward<Args>(args)...);
    }

    template<typename IdType, typename... Args>
    inline void trace(IdType&& id, Args&&... args) { log(TRACE, std::forward<IdType>(id), std::forward<Args>(args)...); }

    template<typename IdType, typename... Args>
    inline void debug(IdType&& id, Args&&... args) { log(DEBUG, std::forward<IdType>(id), std::forward<Args>(args)...); }

    template<typename IdType, typename... Args>
    inline void info(IdType&& id, Args&&... args) { log(INFO, std::forward<IdType>(id), std::forward<Args>(args)...); }

    template<typename IdType, typename... Args>
    inline void warn(IdType&& id, Args&&... args) { log(WARN, std::forward<IdType>(id), std::forward<Args>(args)...); }

    template<typename IdType, typename... Args>
    inline void error(IdType&& id, Args&&... args) { log(ERROR_L, std::forward<IdType>(id), std::forward<Args>(args)...); }


} // namespace logger
