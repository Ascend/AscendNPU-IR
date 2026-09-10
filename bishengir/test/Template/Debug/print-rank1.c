// RUN: %host_cxx -std=c++17 -x c++ -Wno-attributes \
// RUN:   -Wno-format-security -Wno-unknown-attributes \
// RUN:   -Wno-compound-token-split-by-macro -I%bishengir_src_root \
// RUN:   -I%bishengir_src_root/lib/Template/include %s -o %t
// RUN: %t | FileCheck %s --match-full-lines
//
// CHECK: rank1:
// CHECK-NEXT: [1,2,3]

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#define HIVM_MLIR_TEMPLATE_DEBUG_UTILS_H
#define CCE_PRINT_SIMT 1
#define CCE_PRINT_CC
// Turn [aicore] into an ignored C++ attribute for host compilation.
#define aicore [npuir_test::aicore]
#define __gm__
#define __ubuf__ volatile
#define pipe_barrier(...)
#define trap() __builtin_trap()
#define __cce_simt_get_BLOCKID() 0
#define REGISTER_PRINT_SCALAR(type, mem)
#define REGISTER_PRINT_1TO8D_TENSOR(type, mem)
#define REGISTER_ASSERT_SCALAR(mem)
#define REGISTER_ASSERT_1TO8D_TENSOR(mem)

struct half {
  unsigned short bits;
  operator float() const { return 0.0f; }
};
using bfloat16_t = unsigned short;

template <typename T, std::size_t Dim>
struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

namespace cce {
template <typename... Args>
void printf(const char *format, Args... args) {
  std::printf(format, args...);
}
} // namespace cce

#include "lib/Template/lib/Debug/Debug.cpp"

int main() {
  int32_t data[] = {1, 2, 3};
  memref_t<int32_t, 1> arg{data, data, 0, {3}, {1}};
  char prefix[] = "rank1:";
  print_nd_core<int32_t, int32_t, 1>(prefix, 6, &arg, 0);
}
