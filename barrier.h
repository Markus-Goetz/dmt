#ifndef DMT_BARRIER_H
#define DMT_BARRIER_H

#include <atomic>
#include <condition_variable>
#include <mutex>

class Barrier {
protected:
    std::mutex mutex_;
    std::condition_variable condition_;

    std::atomic<size_t> count_;
    size_t generation_;
    size_t threshold_;

public:
    explicit Barrier(std::size_t count) : count_(count), generation_(0), threshold_(count) {}

    void wait() {
        std::unique_lock<std::mutex> lock{ this->mutex_ };
        size_t last_generation = this->generation_;
        if (! --this->count_) {
            ++this->generation_;
            this->count_.store(this->threshold_);
            this->condition_.notify_all();
        } else {
            this->condition_.wait(lock, [this, last_generation] { return last_generation != this->generation_; });
        }
    }
};

#endif //DMT_BARRIER_H
