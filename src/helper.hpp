#include <chrono>

// A simple stopwatch.
class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;
  using msec_t = std::chrono::duration<double, std::milli>;
  inline void tic() {
    start = Clock::now();
  }
  inline void toc() {
    accumulated += std::chrono::duration_cast<msec_t>(Clock::now() - start);
    count += 1;
  }
  inline void reset() {
    accumulated = std::chrono::duration<double>(0);
    count = 0;
  }
  inline double total() const {
    return accumulated.count();
  }
  inline double avg() const {
    return accumulated.count() / count;
  }
private:
  Clock::time_point start;
  msec_t accumulated;
  size_t count = 0;
};

// Get cpu info from system.
#ifdef _MSC_VER
#include <intrin.h>
std::string cpu_info()
{
  int id[4] = { -1 };
  char info[32] = { 0 };
  std::string cpu_info;
  for (uint i = 0; i < 3; i++) {
    __cpuid(id, 0x80000002 + i);
    memcpy(info, id, sizeof(id));
    cpu_info += info;
  }
  return cpu_info;
}
#else
#include <fstream>
std::string cpu_info()
{
  std::ifstream file("/proc/cpuinfo");
  std::string line;
  for (int i = 0; i < 5; i++)
    std::getline(file, line);
  return line.substr(13);
}
#endif