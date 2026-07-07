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

// Avoid including <cmath> when building ARM target
#define INFINITY (__builtin_inff())
// pi
#define M_PI 3.14159265358979323846

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
constexpr int32_t INTR_BITS_PER_BLOCK = 256;
constexpr int32_t REG_REGISTER_SIZE = 256;
constexpr uint8_t BIT_32_LEN = 32;
constexpr uint8_t BIT_64_LEN = 64;
constexpr uint64_t UINT_64_MAX = 0xFFFFFFFFFFFFFFFF;
constexpr uint64_t UINT_64_HIGHEST_BIT_MASK = 0x8000000000000000;
constexpr uint64_t INT_64_MAX = 0x7FFFFFFFFFFFFFFF;
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

template <typename IntType> struct FpTraits;
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
__aiv__ __attribute__((always_inline)) bool
isAddress32ByteAligned(__ubuf__ T *ptr) {
  auto address = reinterpret_cast<uintptr_t>(ptr);
  return (address & 0x1F) == 0;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool is32ByteAligned(int64_t value) {
  return (value * sizeof(T)) % UB_ALIGN_BYTES == 0;
}

template <typename T, int Dim>
__aiv__ __attribute__((always_inline))
bool is_memref_single_element(memref_t<__ubuf__ T, Dim> *buf) {
  for (int i = 0; i < Dim; i++) {
    if (buf->sizes[i] != 1) {
      return false;
    }
  }
  return true;
}

template <typename T, int Dim>
__aiv__ __attribute__((always_inline)) bool is_memref_aligned(memref_t<__ubuf__ T, Dim> *buf) {
  // Check if buf pointer is null
  if (buf == nullptr) {
    return false;
  }

  // Calculate alignment factor: number of elements per alignment unit
  // UB_ALIGN_BYTES * BITS_B8 converts bytes to bits, then divide by element bit width
  constexpr int align_factor = (UB_ALIGN_BYTES * BITS_B8) / bitwidthOf<T>();

  // Check if offset is aligned to the alignment factor
  if ((buf->offset % align_factor) != 0) {
    return false;
  }

  // If the offset is aligned and the memref has only one element, take the vector path.
  if (is_memref_single_element<T, Dim>(buf)) {
    return true;
  }

  // Check if last dimension is contiguous in memory (stride must be 1)
  if (buf->strides[Dim - 1] != 1) {
    return false;
  }

  // For 2D and higher dimensions, check alignment of second-to-last dimension stride
  if constexpr (Dim >= 2) {
    return ((buf->strides[Dim - 2] % align_factor) == 0);
  }
  return true;
}

#if defined(__DAV_C310__)
template <typename T>
__simt_callee__ __aiv__ __attribute__((always_inline)) T
UintDivImpl(T dividend, T magic, T shift) {
  static_assert(std::is_same<T, uint32_t>::value ||
                    std::is_same<T, uint64_t>::value,
                "Input type T only supports uint32_t, uint64_t.");

  T q = 0;
  if constexpr (std::is_same<T, uint32_t>::value) {
    q = __umulhi(dividend, magic);
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    q = __umul64hi(dividend, magic);
  }

  T sum = dividend + q;
  return sum >> shift;
}

template <int countValue>
__aiv__ __attribute__((always_inline)) static int64_t
ScalarGetCountOfValueImpl(uint64_t valueIn) {
  if constexpr (countValue == 1) {
    // This instruction counts the number of 1 in the number in Xn, and then
    // writes the result to Xd.
    return bcnt1(valueIn);
  } else if constexpr (countValue == 0) {
    // This instruction counts the number of 0 in the number in Xn, and then
    // writes the result to Xd.
    return bcnt0(valueIn);
  } else {
    static_assert(((countValue == 0) || (countValue == 1)) &&
                  "countValue must be 1 or 0");
    return 0;
  }
}

__aiv__ __attribute__((always_inline)) static int64_t
ScalarCountLeadingZeroImpl(uint64_t valueIn) {
  // This instruction counts leading zero bits of number in Xn.
  return clz(valueIn);
}

__aiv__ __attribute__((always_inline)) static uint64_t
GetUintDivMagic(uint64_t dividend, uint64_t divisor) {
  uint64_t quotient = 0;
  uint64_t remainder = dividend;
  uint64_t borrow = 0;
  // handle low 64 bit
  for (int i = 0; i < BIT_64_LEN; i++) {
    quotient <<= 1;
    borrow = (remainder & UINT_64_HIGHEST_BIT_MASK) > 0 ? 1 : 0;
    remainder = (remainder << 1);

    if (borrow == 1) {
      remainder = UINT_64_MAX - divisor + 1 + remainder;
      quotient |= 1;
    } else if (remainder >= divisor) {
      remainder -= divisor;
      quotient |= 1;
    }
  }
  return quotient + 1;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
GetUintDivMagicAndShiftImpl(T &magic, T &shift, T divisor) {
  static_assert(std::is_same<T, uint32_t>::value ||
                    std::is_same<T, uint64_t>::value,
                "Input type T only supports uint32_t, uint64_t.");
  int64_t pos = BIT_64_LEN - ScalarCountLeadingZeroImpl(divisor);
  int64_t cnt1 = ScalarGetCountOfValueImpl<1>(divisor);
  shift = cnt1 == 1 ? pos - 1 : pos;
  if constexpr (std::is_same<T, uint32_t>::value) {
    magic = (1l << BIT_32_LEN) * ((1l << shift) - divisor) / divisor + 1;
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    uint64_t dividend = 0;
    if (shift < BIT_64_LEN) {
      dividend = (1l << shift) - divisor;
    } else if (shift == BIT_64_LEN) {
      // divisor must be greater than 0, so will not overflow
      dividend = UINT_64_MAX - divisor + 1;
    } else {
      return;
    }
    magic = GetUintDivMagic(dividend, divisor);
  }
}

template <typename DTYPE>
__simt_callee__ __aiv__ __attribute__((always_inline))
constexpr DTYPE MakeSentinelNegOne() {
  if constexpr (std::is_same<DTYPE, half2>::value) {
    return half2{static_cast<half>(-1), static_cast<half>(-1)};
  } else if constexpr (std::is_same<DTYPE, bfloat16x2_t>::value) {
    return bfloat16x2_t{static_cast<bfloat16_t>(-1),
                        static_cast<bfloat16_t>(-1)};
  } else {
    return static_cast<DTYPE>(-1);
  }
}
#endif
#endif // HIVM_MLIR_TEMPLATE_UTILS_H
