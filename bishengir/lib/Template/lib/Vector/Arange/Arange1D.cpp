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

#include "Utils.h"
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/VecUtils.h"
#include <cstdint>
#include <type_traits>

/// expand the sequance [0, sub_n) to [0, sub_n + expand_n)
/// constraits:
/// 1. both dst_ptr and dst_ptr + sub_n are aligned to UB_ALIGN_BYTES
/// 2. expand_n <= sub_n
template <typename T>
__aiv__ __attribute__((always_inline)) void
expand_arange(__ubuf__ T *dst_ptr, int64_t sub_n, int64_t expand_n,
              int64_t stride) {
  // use vadds to implement it, where src0 is the data from [dst_ptr,
  // dst_ptr+expand_n), src1 is sub_n

  // use counter mode
  INTRINSIC(set_vector_mask, 0, expand_n);
  T scalar;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  if constexpr (std::is_same<T, half>::value) {
    scalar = {static_cast<unsigned short>(sub_n * stride)};
  } else {
    scalar = static_cast<T>(sub_n * stride);
  }
#else
  scalar = static_cast<T>(sub_n * stride);
#endif

  if constexpr (std::is_same<T, int64_t>::value) {
    // PIPE_S waits PIPE_V before writing
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < expand_n; i++) {
      dst_ptr[sub_n + i] = dst_ptr[i] + scalar;
    }
    // PIPE_V waits PIPE_S either before return or before expansion
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  } else {
    vector_eltwise_vs_intrin<VectorOpTy::VADDS, T>(
        intrin_args<1, T>{dst_ptr + sub_n, // dst_ptr
                          {dst_ptr},       // src_ptr
                          scalar,          // scalar
                          1,               // repeat num
                          1,               // dst block stride
                          {1},             // src block stride
                          8,               // dst repeat stride
                          {8}}             // src repeat stride
    );
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_1d_core(__ubuf__ T *dst_ptr, int64_t n, int64_t offset, int64_t stride) {
  static_assert((sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
                "Arange_1d_core do not support this data type");
  size_t sub_n = UB_ALIGN_BYTES / sizeof(T);
  if (sub_n > n)
    sub_n = n;
  // PIPE_S waits PIPE_V before writing
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < sub_n; ++i) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    if constexpr (std::is_same<T, half>::value) {
      dst_ptr[i] = {static_cast<unsigned short>(offset + stride * i)};
    } else {
      dst_ptr[i] = offset + stride * i;
    }
#else
    dst_ptr[i] = offset + stride * i;
#endif
  }

  // PIPE_V waits PIPE_S either before return or before expansion
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  if (n <= sub_n)
    return;

  // expand arange by vadds, using counter mode
  INTRINSIC_NO_ARGS(set_mask_count);
  while (sub_n * 2 < n) {
    expand_arange(dst_ptr, sub_n, sub_n, stride);
    sub_n *= 2;
    INTRINSIC(pipe_barrier, PIPE_V);
  }
  // tail
  expand_arange(dst_ptr, sub_n, n - sub_n, stride);

  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_1d_uncontinue_core(memref_t<__ubuf__ T, 1> *dst, int64_t offset,
                          int64_t stride) {
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  auto dst_ptr = dst->aligned + dst->offset;
  int64_t dst_stride = dst->strides[0];
  for (int i = 0; i < dst->sizes[0]; ++i) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    if constexpr (std::is_same<T, half>::value) {
      dst_ptr[i * dst_stride] = {
          static_cast<unsigned short>(offset * dst_stride + stride * i)};
    } else {
      dst_ptr[i * dst_stride] = offset * dst_stride + stride * i;
    }
#else
    dst_ptr[i * dst_stride] = offset * dst_stride + stride * i;
#endif
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_arange_1d(memref_t<__ubuf__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
#endif
}

/// Generate the a sequance of number, range from [offset, offset + n *stride)
/// to dst, where n is the dst_size
///
/// Constraints:
/// 1. dim of dst must be 1 (Arange - 1D)
template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_1d(memref_t<__ubuf__ T, 1> *dst, int64_t offset, int64_t stride) {
  static_assert((sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
                "arange_1d unsupport this type");
  // Input parameter constraints assert.
  check_inputs_of_arange_1d(dst);

  if (dst->strides[0] != 1) {
    arange_1d_uncontinue_core<T>(dst, offset, stride);
    return;
  }
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  arange_1d_core<T>(dst_ptr, dst->sizes[0], offset, stride);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Vector arange 1d
//===-------------------------------------------------------------------===//
REGISTE_ARANGE_1D(int16_t)
REGISTE_ARANGE_1D(int32_t)
REGISTE_ARANGE_1D(half)
REGISTE_ARANGE_1D(float)
REGISTE_ARANGE_1D(int64_t)
}
