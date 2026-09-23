#pragma once
#include <__config>
#include <mutex>
#include <chrono>
namespace std {
enum class cv_status { no_timeout, timeout };
class condition_variable {
public:
  void notify_one() noexcept {} void notify_all() noexcept {}
  template<class L> void wait(L&) {}
  template<class L, class P> void wait(L&, P p) { while(!p()) break; }
  template<class L,class R,class Pd> cv_status wait_for(L&, const chrono::duration<R,Pd>&){ return cv_status::timeout; }
  template<class L,class R,class Pd,class P> bool wait_for(L&, const chrono::duration<R,Pd>&, P p){ return p(); }
};
using condition_variable_any = condition_variable;
}
