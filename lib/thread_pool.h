#include <condition_variable>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

using namespace std;

static constexpr uint NUM_THREADS = 4;

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads)
      : is_waiting_(false) {
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(thread([this]() { Loop(); }));
    }
  }

  void Submit(function<void(void)> task) {
    unique_lock<mutex> lock(mu_);
    tasks_.push(task);
    cond_.notify_one();
  }

  void Wait() {
    {
      unique_lock<mutex> lock(mu_);
      is_waiting_ = true;
      cond_.notify_all();
    }

    for (thread& th : threads_) {
      th.join();
    }
  }

 private:
  void Loop() {
    while (true) {
      function<void(void)> task;
      {
        unique_lock<mutex> lock(mu_);
        if (tasks_.empty()) {
          if (is_waiting_)
            return;
          cond_.wait(lock);
        }

        if (tasks_.empty())
          continue;

        task = tasks_.top();
        tasks_.pop();
      }
      task();
    }
  }

  vector<thread> threads_;
  mutex mu_;
  condition_variable cond_;
  stack<function<void(void)>> tasks_;
  bool is_waiting_;
};
