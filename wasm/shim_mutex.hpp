#pragma once
// Single-threaded shim for wasm32-wasi (libc++ built without threads).
#include <__config>
namespace std {
struct mutex { void lock(){} void unlock(){} bool try_lock(){return true;} };
struct recursive_mutex { void lock(){} void unlock(){} bool try_lock(){return true;} };
struct defer_lock_t{}; struct try_to_lock_t{}; struct adopt_lock_t{};
inline constexpr defer_lock_t defer_lock{}; inline constexpr try_to_lock_t try_to_lock{};
inline constexpr adopt_lock_t adopt_lock{};
template<class M> struct lock_guard {
  using mutex_type=M; explicit lock_guard(M&){} lock_guard(M&, adopt_lock_t){} ~lock_guard(){}
  lock_guard(const lock_guard&)=delete; lock_guard& operator=(const lock_guard&)=delete; };
template<class M> struct unique_lock {
  using mutex_type=M; M* m_=nullptr; bool owns_=false;
  unique_lock()=default; explicit unique_lock(M& m):m_(&m),owns_(true){}
  unique_lock(M& m, defer_lock_t):m_(&m),owns_(false){}
  unique_lock(M& m, try_to_lock_t):m_(&m),owns_(true){}
  unique_lock(M& m, adopt_lock_t):m_(&m),owns_(true){}
  ~unique_lock(){} void lock(){owns_=true;} void unlock(){owns_=false;}
  bool try_lock(){owns_=true;return true;} bool owns_lock() const {return owns_;}
  explicit operator bool() const {return owns_;} M* mutex() const {return m_;}
  unique_lock(const unique_lock&)=delete; unique_lock& operator=(const unique_lock&)=delete;
  unique_lock(unique_lock&& o):m_(o.m_),owns_(o.owns_){o.m_=nullptr;o.owns_=false;} };
template<class... M> struct scoped_lock { explicit scoped_lock(M&...){} ~scoped_lock(){}
  scoped_lock(const scoped_lock&)=delete; scoped_lock& operator=(const scoped_lock&)=delete; };
}
