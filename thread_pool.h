#ifndef CONCURRENT_THREADPOOL_H
#define CONCURRENT_THREADPOOL_H

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
#include <thread>

/**
 *  Simple ThreadPool that creates `ThreadCount` threads upon its creation,
 *  and pulls from a queue to get new jobs. The default is 10 threads.
 *
 *  This class requires a number of c++11 features be present in your compiler.
 */
class ThreadPool {
    size_t                               thread_count;
    std::vector<std::thread>             threads;
    std::list<std::function<void(void)>> queue;

    std::atomic_int         jobs_left;
    std::atomic_bool        bailout;
    std::atomic_bool        finished;
    std::condition_variable job_available_var;
    std::condition_variable wait_var;
    std::mutex              wait_mutex;
    std::mutex              queue_mutex;

    /**
     *  Take the next job in the queue and run it.
     *  Notify the main thread that a job has completed.
     */
    void task() {
        while(!this->bailout) {
            this->next_job()();
            --this->jobs_left;
            this->wait_var.notify_one();
        }
    }

    /**
     *  Get the next job; pop the first item in the queue, 
     *  otherwise wait for a signal from the main thread.
     */
    std::function<void(void)> next_job() {
        std::function<void(void)> res;
        std::unique_lock<std::mutex> job_lock(queue_mutex);

        // Wait for a job if we don't have any.
        this->job_available_var.wait(job_lock, [this]() -> bool { return this->queue.size() || this->bailout; });
        
        // Get job from the queue
        if (!this->bailout) {
            res = this->queue.front();
            this->queue.pop_front();
        // If we're bailing out, 'inject' a job into the queue to keep jobs_left accurate.
        } else {
            res = []{};
            ++this->jobs_left;
        }
        return res;
    }

public:
    ThreadPool(size_t _thread_count) : thread_count(_thread_count), jobs_left(0), bailout(false), finished(false) {
        this->threads.reserve(_thread_count);
        for (size_t i = 0; i < _thread_count; ++i) {
            this->threads.push_back(std::thread([this] { this->task(); }));
        }
    }

    /**
     *  JoinAll on deconstruction
     */
    ~ThreadPool() {
        this->join_all();
    }

    /**
     *  Get the number of threads in this pool
     */
    inline size_t size() const {
        return this->thread_count;
    }

    /**
     *  Get the number of jobs left in the queue.
     */
    inline unsigned jobs_remaining() {
        std::lock_guard<std::mutex> guard(queue_mutex);
        return this->queue.size();
    }

    /**
     *  Add a new job to the pool. If there are no jobs in the queue,
     *  a thread is woken up to take the job. If all threads are busy,
     *  the job is added to the end of the queue.
     */
    void add_job(std::function<void(void)> job) {
        std::lock_guard<std::mutex> guard(queue_mutex);
        this->queue.emplace_back(job);
        ++this->jobs_left;
        this->job_available_var.notify_one();
    }

    /**
     *  Join with all threads. Block until all threads have completed.
     *  Params: WaitForAll: If true, will wait for the queue to empty 
     *          before joining with threads. If false, will complete
     *          current jobs, then inform the threads to exit.
     *  The queue will be empty after this call, and the threads will
     *  be done. After invoking `ThreadPool::JoinAll`, the pool can no
     *  longer be used. If you need the pool to exist past completion
     *  of jobs, look to use `ThreadPool::WaitAll`.
     */
    void join_all(bool wait_for_all=true) {
        if (!this->finished ) {
            if (wait_for_all) {
                this->wait_all();
            }

            // note that we're done, and wake up any thread that's
            // waiting for a new job
            this->bailout = true;
            this->job_available_var.notify_all();

            for (auto &x : threads) {
                if (x.joinable()) {
                    x.join();
                }
            }
            this->finished = true;
        }
    }

    /**
     *  Wait for the pool to empty before continuing. 
     *  This does not call `std::thread::join`, it only waits until
     *  all jobs have finished executing.
     */
    void wait_all() {
        if (this->jobs_left > 0) {
            std::unique_lock<std::mutex> lk(wait_mutex);
            this->wait_var.wait(lk, [this] { return this->jobs_left == 0; });
            lk.unlock();
        }
    }
};

#endif //CONCURRENT_THREADPOOL_H
