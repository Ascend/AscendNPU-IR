/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HIVM_MLIR_TEMPLATE_UTILS_H
#define HIVM_MLIR_TEMPLATE_UTILS_H

#include <CPUDebug/CPUDebugUtils.h>

#if ENABLE_CPU_TRACE_INTRINSIC
#else
#if ENABLE_SYCL
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
using namespace cl::sycl;
#define __aiv__ SYCL_EXTERNAL __aivector__[aicore]
#else
#define __aiv__ [aicore]
#endif

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

#define CEIL_DIV(x, y) (((x) + ((y)-1)) / (y))
// Compute the smallest value that is >= x and can divide y
#define CEIL_FACTOR(x, y) (((x) + ((y)-1)) / (y) * (y))
#define FLOOR_FACTOR(x, y) ((x / y) * (y))
// meta op library use synchronization id among 6 and 7 to avoid conflict with
// auto sychronization of compiler which use id among 0 and 5.
#define MIN(x, y) (x < y ? x : y)
#define MAX(x, y) (x > y ? x : y)
#define NAN (__builtin_nanf(""))

// Reserved eventIds for template library use, currently LIB_EVENT_ID1 is not
// being use, if needed or any other eventId, please update sync related passes
// (inject-sync::reservedEventIdNum) to prevent any conflicts between library
// instructions and AscendNPU-IR inserted instructions.
#define LIB_EVENT_ID0 EVENT_ID7
#define LIB_EVENT_ID1 EVENT_ID6

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//
constexpr int32_t BYTES_B8 = 1;
constexpr int32_t BYTES_B16 = 2;
constexpr int32_t BYTES_B32 = 4;
constexpr int32_t BYTES_B64 = 8;
constexpr int32_t UB_ALIGN_BYTES = 32;
constexpr int32_t BITS_B8 = 8;
constexpr int32_t FRACTAL_BLOCK_NUM = 16;
constexpr int32_t L0AB_BUFFER_BYTES = 65536;
constexpr int32_t L1_ALIGN_BYTES = 32;
//===----------------------------------------------------------------------===//
// Typedefs
//===----------------------------------------------------------------------===//
template <typename T, size_t Dim>
struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

// Calculate the bit width of the current type.
template <typename T>
__aiv__ __attribute__((always_inline)) constexpr int bitwidthOf() {
  if constexpr (std::is_same<T, bool>::value) {
    return 1;
  } else {
    return sizeof(T) * 8;
  }
}

// Determine whether the size is block aligned.
template <typename T>
__aiv__ __attribute__((always_inline)) bool isSizeAlignedToBlock(int size) {
  constexpr int num_per_block = UB_ALIGN_BYTES / sizeof(T);
  return size % num_per_block == 0;
}

// Determine whether the starting address is 32byte aligned.
template <typename T>
__aiv__ __attribute__((always_inline)) bool isAddress32ByteAligned(__ubuf__ T* ptr) {
  auto address = reinterpret_cast<uintptr_t>(ptr);
  return (address & 0x1F) == 0;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool is32ByteAligned(int64_t value) {
  return (value * sizeof(T)) % UB_ALIGN_BYTES == 0;
}
#endif // HIVM_MLIR_TEMPLATE_UTILS_H