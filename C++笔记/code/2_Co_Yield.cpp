#include <concepts>
#include <coroutine>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

struct User {
  std::string name;
  int id;
};

template <typename T>
struct Generator {
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type  // required
  {
    T value_;
    std::exception_ptr exception_;

    Generator get_return_object() {
      return Generator(handle_type::from_promise(*this));
    }
    std::suspend_always initial_suspend() {
      return {};
    }
    std::suspend_always final_suspend() noexcept {
      return {};
    }
    void unhandled_exception() {
      exception_ = std::current_exception();
    }  // saving
       // exception
    template <typename From>
    std::suspend_always yield_value(From &&from) {
      // value_ = std::forward<From>(from); // caching the result in promise
      value_ = User{from.name, from.id + 1};
      return {};
    }
    void return_void() {}
  };

  handle_type h_;

  Generator(handle_type h) : h_(h) {}
  ~Generator() {
    h_.destroy();
  }
  Generator(const Generator &) = delete;
  Generator &operator=(const Generator &) = delete;
  explicit operator bool() {
    fill();  // The only way to reliably find out whether or not we finished coroutine,
             // whether or not there is going to be a next value generated (co_yield)
             // in coroutine via C++ getter (operator () below) is to execute/resume
             // coroutine until the next co_yield point (or let it fall off end).
             // Then we store/cache result in promise to allow getter (operator() below
             // to grab it without executing coroutine).
    return !h_.done();
  }
  T operator()() {
    fill();
    full_ = false;  // we are going to move out previously cached
                    // result to make promise empty again
    return std::move(h_.promise().value_);
  }

 private:
  bool full_ = false;

  void fill() {
    if (!full_) {
      h_();
      if (h_.promise().exception_) std::rethrow_exception(h_.promise().exception_);
      full_ = true;
    }
  }
};

Generator<User> gen_user(unsigned n) {
  for (unsigned i = 0; i < n; ++i) {
    std::string name = std::to_string(i);
    User u{name, i};
    co_yield u;
  }
}

int main() {
  try {
    auto gen = gen_user(5);
    while (gen) {
      auto v = gen();
      std::cout << "user:" << v.name << "," << v.id << "\n";
    }
  }
  catch (const std::exception &ex) {
    std::cerr << "Exception: " << ex.what() << '\n';
  }
  catch (...) {
    std::cerr << "Unknown exception.\n";
  }
}