#pragma once
// Single-threaded shim for wasm32-wasi: a std::thread that runs its callable
// immediately on construction. join()/detach() are no-ops. Deterministic by
// construction, which is what --threads 1 gives on native builds.
#include <__config>
#include <utility>
#include <tuple>
#include <chrono>
namespace std {
class thread {
  bool joinable_ = false;
public:
  struct id { unsigned v=0; friend bool operator==(id a,id b){return a.v==b.v;}
              friend bool operator!=(id a,id b){return a.v!=b.v;} };
  using native_handle_type = int;
  thread() noexcept = default;
  template<class F, class... A> explicit thread(F&& f, A&&... a) : joinable_(true) {
    // run now, in-line
    ::std::forward<F>(f)(::std::forward<A>(a)...);
  }
  thread(const thread&) = delete; thread& operator=(const thread&) = delete;
  thread(thread&& o) noexcept : joinable_(o.joinable_) { o.joinable_=false; }
  thread& operator=(thread&& o) noexcept { joinable_=o.joinable_; o.joinable_=false; return *this; }
  ~thread() = default;
  bool joinable() const noexcept { return joinable_; }
  void join() { joinable_=false; }
  void detach() { joinable_=false; }
  id get_id() const noexcept { return id{joinable_?1u:0u}; }
  native_handle_type native_handle() { return 0; }
  static unsigned hardware_concurrency() noexcept { return 1u; }
  void swap(thread& o) noexcept { bool t=joinable_; joinable_=o.joinable_; o.joinable_=t; }
};
namespace this_thread {
  inline void yield() noexcept {}
  inline thread::id get_id() noexcept { return thread::id{1u}; }
  template<class R,class P> void sleep_for(const chrono::duration<R,P>&) {}
  template<class C,class D> void sleep_until(const chrono::time_point<C,D>&) {}
}
}
