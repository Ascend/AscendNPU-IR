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
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) uint16_t merge_b8_to_u16(T src_val) {
  static_assert((sizeof(T) == 1) &&
                "Brc_scalar_core do not support this data type");
  uint16_t val_low = ((uint16_t)((uint16_t)src_val << 8ULL)) >> 8ULL;
  uint16_t dst_val = ((uint16_t)(val_low << 8ULL)) + val_low;

  return dst_val;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_scalar_core_1d(T src_val, __ubuf__ T *dst_ptr, int64_t count) {
  static_assert((sizeof(T) == 2 || sizeof(T) == 4) &&
                "Brc_scalar_core_1d do not support this data type");
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0x0, count); // in counter mode
  INTRINSIC(vector_dup,
            dst_ptr, // dst
            src_val, // src
            1,       // repeat
            1,       // dst blk stride
            0,       // src blk stride
            8,       // dst rep stride
            0);      // src rep stride
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// Broadcast scalar to vector, broadcast scalar to dst (m, n)[s, 1]
///
/// Constraints:
/// 1. dim of dst must be 2
/// 2. s and n of dst must be aligned to ub_block_unit
/// 3. src must be scalar
template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_scalar_core_2d(T src_val, __ubuf__ T *dst_ptr, int64_t row_num,
                   int64_t col_num, int64_t stride) {
  static_assert((sizeof(T) == 2 || sizeof(T) == 4) &&
                "Brc_scalar_core_2d do not support this data type");
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  if (col_num <= num_per_repeat) {
    // Repeat over all the rows
    int64_t repeat_times = row_num;
    int64_t loop_num = repeat_times / INTR_MAX_REPEAT_CNTS;
    int64_t dst_repeat_stride = stride * sizeof(T) / UB_ALIGN_BYTES;
    for (int64_t i = 0; i < loop_num; ++i) {
      INTRINSIC_NO_ARGS(set_mask_norm);
      SetMaskValueByCount(col_num);
      INTRINSIC(vector_dup,
                dst_ptr + i * stride * INTR_MAX_REPEAT_CNTS, // dst
                src_val,                                     // src
                INTR_MAX_REPEAT_CNTS,                        // repeat
                1,                                           // dst blk stride
                0,                                           // src blk stride
                dst_repeat_stride,                           // dst rep stride
                0);                                          // src rep stride
    }
    if (repeat_times % INTR_MAX_REPEAT_CNTS != 0) {
      int64_t remain_repeat_times = repeat_times % INTR_MAX_REPEAT_CNTS;
      INTRINSIC_NO_ARGS(set_mask_norm);
      SetMaskValueByCount(col_num);
      INTRINSIC(vector_dup,
                dst_ptr + loop_num * stride * INTR_MAX_REPEAT_CNTS, // dst
                src_val,                                            // src
                remain_repeat_times,                                // repeat
                1,                 // dst blk stride
                0,                 // src blk stride
                dst_repeat_stride, // dst rep stride
                0);                // src rep stride
    }
  } else {
    // For loop over all the rows
    for (int64_t i = 0; i < row_num; ++i) {
      brc_scalar_core_1d<T>(src_val, dst_ptr + i * stride, col_num);
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_scalar_core_with_mask_mode(T src_val, __ubuf__ T *dst_ptr, int64_t repeat,
                               uint64_t mask_value) {
  static_assert(sizeof(T) == 4 &&
                "Brc_scalar_core_normal do not support this data type");
  INTRINSIC_NO_ARGS(set_mask_norm);
  INTRINSIC(set_vector_mask, 0x0, mask_value);
  INTRINSIC(vector_dup,
            dst_ptr, // dst
            src_val, // src
            repeat,  // repeat
            1,       // dst blk stride
            0,       // src blk stride
            8,       // dst rep stride
            0);      // src rep stride
}

// Select the interval_index-th data from interval_num to brc, and the interval
// is equal to interval_num.
__aiv__ __attribute__((always_inline)) uint64_t
get_interval_mask(uint64_t interval_num, uint64_t interval_index,
                  uint64_t mask_length) {
  // Since it is an equally spaced jump, the relationship between the value of
  // the mask and the number of terms should satisfy the geometric sequence.
  uint64_t first_num = pow((uint64_t)2, interval_num - interval_index - 1);
  uint64_t common_radio = pow((uint64_t)2, interval_num);
  uint64_t num = mask_length / interval_num;
  uint64_t mask = (pow(common_radio, num) - 1) / (common_radio - 1) * first_num;
  return mask;
}

// broadcast scalar to dst (b, ), dst is not stored continuously and is
// determined based on mask_value.
template <typename T, typename = typename std::enable_if<sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar_interval(T scalar, __ubuf__ T *dst_ptr, int64_t brc_num,
                          uint64_t interval_num, uint64_t interval_index) {
  uint64_t num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(int32_t);
  // part 1: main block
  auto main_size = brc_num / num_per_repeat;
  auto main_loop_num = main_size / INTR_MAX_REPEAT_CNTS;
  uint64_t main_mask_value =
      get_interval_mask(interval_num, interval_index, num_per_repeat);
  for (int64_t i = 0; i < main_loop_num; ++i) {
    brc_scalar_core_with_mask_mode<int32_t>(
        scalar,
        (__ubuf__ int32_t *)dst_ptr + i * INTR_MAX_REPEAT_CNTS * num_per_repeat,
        INTR_MAX_REPEAT_CNTS, main_mask_value);
  }
  if (main_size % INTR_MAX_REPEAT_CNTS != 0) {
    brc_scalar_core_with_mask_mode<int32_t>(
        scalar,
        (__ubuf__ int32_t *)dst_ptr +
            main_loop_num * INTR_MAX_REPEAT_CNTS * num_per_repeat,
        main_size % INTR_MAX_REPEAT_CNTS, main_mask_value);
  }
  // part 2: tail block
  auto tail_size = brc_num % num_per_repeat;
  uint64_t tail_mask_value =
      get_interval_mask(interval_num, interval_index, tail_size);
  if (tail_size != 0) {
    brc_scalar_core_with_mask_mode<int32_t>(
        scalar, (__ubuf__ int32_t *)dst_ptr + main_size * num_per_repeat, 1,
        tail_mask_value);
  }
}

template <typename T, size_t Dim>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_broadcast_scalar(T scalar, memref_t<__ubuf__ T, Dim> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(dst->strides[Dim - 1] == 1 && "The dst last dim must be continuous.");
  if constexpr (Dim == 2) {
    assert(isSizeAlignedToBlock<T>(dst->strides[0]) &&
           "The dst strides[0] must be aligned to block.");
  }
#endif
}

/// Broadcast scalar to vector, broadcast scalar to dst (b, ), here b is
/// broadcast axis, customized for b64.
///
/// Constraints:
/// 1. dim of dst must be 1, dst must be continuous vector
/// 2. src must be scalar
template <typename T>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar_b64(T scalar, memref_t<__ubuf__ T, 1> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_broadcast_scalar(scalar, dst);

  static_assert(sizeof(T) == 8 && "broadcast_scalar_b64 only support b64.");
  // Step1: Split b64 scalar into high 32bits and low 32bits.
  // get the lower 32 bits value.
  int32_t low_bits_value = (int32_t)(scalar & 0xffffffff);
  // get the high 32 bits value.
  int32_t high_bits_value =
      (int32_t)(((uint64_t)(scalar & 0xffffffff00000000)) >> 32);

  // Step2: Perform brc on the high 32bits and low 32bits respectively, and
  // store the results at intervals.
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  // View (a,) b64 as (2*a,) b32.
  int64_t num = dst->sizes[0] * 2;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(int32_t);
  auto tail_size = num % num_per_repeat;
  // The high 32bits value is stored in intervals of 10101010...
  broadcast_scalar_interval(high_bits_value, (__ubuf__ int32_t *)dst_ptr, num,
                            2, 0);
  // The low 32bits value is stored in intervals of 01010101...
  broadcast_scalar_interval(low_bits_value, (__ubuf__ int32_t *)dst_ptr, num, 2,
                            1);
}

/// Broadcast scalar to vector, broadcast scalar to dst (b, )
///
/// Constraints:
/// 1. dim of dst must be 1, and last dim stride must be 1.
/// 2. src must be scalar
template <typename T>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar_1d(T scalar, memref_t<__ubuf__ T, 1> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_broadcast_scalar(scalar, dst);
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  if constexpr (sizeof(T) == 1) {
    bool is_size_odd = (dst->sizes[0] & 1) == 1;
    if (is_size_odd) {
      // when dst size is odd, the last element will be reg_mov.
      *(dst_ptr + dst->sizes[0] - 1) = scalar;
      INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    }
    uint16_t src_u16 = merge_b8_to_u16<T>(scalar);
    brc_scalar_core_1d<uint16_t>(src_u16, (__ubuf__ uint16_t *)dst_ptr,
                                 dst->sizes[0] / 2);
    if (is_size_odd) {
      INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    }
  } else if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    brc_scalar_core_1d<T>(scalar, dst_ptr, dst->sizes[0]);
  } else if constexpr (sizeof(T) == 8) {
    broadcast_scalar_b64<T>(scalar, dst);
  }
}

/// Broadcast scalar to vector, broadcast scalar to dst (m, n)[s, 1]
///
/// Constraints:
/// 1. dim of dst must be 2
/// 3. src must be scalar
template <typename T>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar_2d(T scalar, memref_t<__ubuf__ T, 2> *dst) {
  if constexpr (sizeof(T) == 1) {
    memref_t<__ubuf__ T, 1> dst_1d;
    dst_1d.allocated = dst->allocated;
    dst_1d.aligned = dst->aligned;
    dst_1d.offset = dst->offset;
    dst_1d.sizes[0] = dst->sizes[0] * dst->strides[0];
    dst_1d.strides[0] = 1;
    broadcast_scalar_1d<T>(scalar, &dst_1d);
  }
  if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    brc_scalar_core_2d<T>(scalar, dst_ptr, dst->sizes[0], dst->sizes[1],
                          dst->strides[0]);
  }
  if constexpr (sizeof(T) == 8) {
    for (int64_t i = 0; i < dst->sizes[0]; ++i) {
      memref_t<__ubuf__ T, 1> dst_1d;
      dst_1d.allocated = dst->allocated;
      dst_1d.aligned = dst->aligned;
      dst_1d.offset = dst->offset + i * dst->strides[0];
      dst_1d.sizes[0] = dst->sizes[1];
      dst_1d.strides[0] = 1;
      broadcast_scalar_b64<T>(scalar, &dst_1d);
    }
  }
}

/// Broadcast scalar to vector, broadcast scalar to dst (b, ) [1] or (m,n) [s,1]
///
/// Constraints:
/// 1. dim of dst must be 1 or 2
/// 2. src must be scalar
template <typename T, size_t Dim>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar(T scalar, memref_t<__ubuf__ T, Dim> *dst) {
  static_assert(
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
      (Dim == 1 || Dim == 2) &&
      "Brcoadcast_scalar do not support this data type or dimension");

  if constexpr (Dim == 1) {
    broadcast_scalar_1d<T>(scalar, dst);
  } else if constexpr (Dim == 2) {
    broadcast_scalar_2d<T>(scalar, dst);
  }
}

// partial template specialization for bool.
//
// Constraints:
// 1. If the tail axis of dst is not aligned with 16, then the size of the dst
// converted to float type will be automatically aligned upward by 16, and there
// is a risk of memory trampling.
template <size_t Dim,
          typename = typename std::enable_if<Dim == 1 || Dim == 2>::type>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar(bool scalar, memref_t<__ubuf__ bool, Dim> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_broadcast_scalar(scalar, dst);

  memref_t<__ubuf__ uint16_t, Dim> dst_b16;
  view_as<bool, uint16_t, Dim>(dst, &dst_b16);
  __ubuf__ uint16_t *dst_b16_ptr = dst_b16.aligned + dst_b16.offset;
  int64_t loop_num = (Dim == 1) ? 1 : dst->sizes[0];
  for (int64_t i = 0; i < loop_num; ++i) {
    brc_scalar_core_1d<uint16_t>(scalar ? MAX_UINT16 : (uint16_t)0,
                                 dst_b16_ptr + i * dst_b16.strides[0],
                                 dst_b16.sizes[Dim - 1]);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_broadcast_scalar(memref_t<__ubuf__ T, 1> *src,
                                 memref_t<__ubuf__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(dst->strides[0] == 1 && "The dst must be continuous vector.");
#endif
}

/// Broadcast src (1, ) to dst (b, ), here b is broadcast axis.
///
/// Constraints:
/// 1. dim of src/dst must be 1, src/dst must be continuous vector, and stride
/// must be 1.
/// 2. src/dst must be continuous vector
template <typename T, typename = typename std::enable_if<
                          sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                          sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_broadcast_scalar(src, dst);

  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  T scalar = src_ptr[0];
  if constexpr (sizeof(T) == 1) {
    if (dst->sizes[0] % 2 != 0) {
      // when dst size is odd, the last element will be reg_mov.
      *(dst_ptr + dst->sizes[0] - 1) = scalar;
    }
    uint16_t src_u16 = merge_b8_to_u16<T>(scalar);
    brc_scalar_core_1d<uint16_t>(src_u16, (__ubuf__ uint16_t *)dst_ptr,
                                 FLOOR_FACTOR(dst->sizes[0], 2));
  } else if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    brc_scalar_core_1d<T>(scalar, dst_ptr, dst->sizes[0]);
  } else if constexpr (sizeof(T) == 8) {
    broadcast_scalar_b64<T>(scalar, dst);
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

// full template specialization for bool.
//
// Constraints:
// 1. If the tail axis of dst is not aligned with 16, then the size of the dst
// converted to float type will be automatically aligned upward by 16, and there
// is a risk of memory trampling.
template <>
__aiv__ __attribute__((always_inline)) void
broadcast_scalar(memref_t<__ubuf__ bool, 1> *src,
                 memref_t<__ubuf__ bool, 1> *dst) {
  __ubuf__ bool *src_ptr = src->aligned + src->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  bool scalar = src_ptr[0];
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  broadcast_scalar(scalar, dst);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Registe brc_scalar 1d interface
//===-------------------------------------------------------------------===//
REGISTE_BRC_SCALAR(1, bool);
REGISTE_BRC_SCALAR(1, int8_t);
REGISTE_BRC_SCALAR(1, int16_t);
REGISTE_BRC_SCALAR(1, int32_t);
REGISTE_BRC_SCALAR(1, int64_t);

//===-------------------------------------------------------------------===//
// Registe brc_scalar 2d interface
//===-------------------------------------------------------------------===//
REGISTE_BRC_SCALAR(2, bool);
REGISTE_BRC_SCALAR(2, int8_t);
REGISTE_BRC_SCALAR(2, int16_t);
REGISTE_BRC_SCALAR(2, int32_t);
REGISTE_BRC_SCALAR(2, int64_t);

//===-------------------------------------------------------------------===//
// Registe brc_scalar func
//===-------------------------------------------------------------------===//
REGISTE_BRC_SCALAR_CORE_1D(int16_t);
REGISTE_BRC_SCALAR_CORE_1D(uint16_t);
REGISTE_BRC_SCALAR_CORE_1D(half);
REGISTE_BRC_SCALAR_CORE_1D(int32_t);
REGISTE_BRC_SCALAR_CORE_1D(uint32_t);
REGISTE_BRC_SCALAR_CORE_1D(float);

//===-------------------------------------------------------------------===//
// Vector broadcast scalar to 1d
//===-------------------------------------------------------------------===//
REGISTE_BROADCAST_SCALAR(1, int8_t)
REGISTE_BROADCAST_SCALAR(1, uint8_t)
REGISTE_BROADCAST_SCALAR(1, bfloat16_t)
REGISTE_BROADCAST_SCALAR(1, int16_t)
REGISTE_BROADCAST_SCALAR(1, uint16_t)
REGISTE_BROADCAST_SCALAR(1, half)
REGISTE_BROADCAST_SCALAR(1, int32_t)
REGISTE_BROADCAST_SCALAR(1, uint32_t)
REGISTE_BROADCAST_SCALAR(1, float)
REGISTE_BROADCAST_SCALAR(1, int64_t)
REGISTE_BROADCAST_SCALAR(1, uint64_t)
REGISTE_BROADCAST_SCALAR(1, bool)

//===-------------------------------------------------------------------===//
// Vector broadcast scalar to 2d
//===-------------------------------------------------------------------===//
REGISTE_BROADCAST_SCALAR(2, int8_t)
REGISTE_BROADCAST_SCALAR(2, uint8_t)
REGISTE_BROADCAST_SCALAR(2, bfloat16_t)
REGISTE_BROADCAST_SCALAR(2, int16_t)
REGISTE_BROADCAST_SCALAR(2, uint16_t)
REGISTE_BROADCAST_SCALAR(2, half)
REGISTE_BROADCAST_SCALAR(2, int32_t)
REGISTE_BROADCAST_SCALAR(2, uint32_t)
REGISTE_BROADCAST_SCALAR(2, float)
REGISTE_BROADCAST_SCALAR(2, int64_t)
REGISTE_BROADCAST_SCALAR(2, uint64_t)
REGISTE_BROADCAST_SCALAR(2, bool)

//===-------------------------------------------------------------------===//
// Vector broadcast 1d
//===-------------------------------------------------------------------===//
REGISTE_BROADCAST_1D(int8_t)
REGISTE_BROADCAST_1D(uint8_t)
REGISTE_BROADCAST_1D(bfloat16_t)
REGISTE_BROADCAST_1D(int16_t)
REGISTE_BROADCAST_1D(uint16_t)
REGISTE_BROADCAST_1D(half)
REGISTE_BROADCAST_1D(int32_t)
REGISTE_BROADCAST_1D(uint32_t)
REGISTE_BROADCAST_1D(float)
REGISTE_BROADCAST_1D(int64_t)
REGISTE_BROADCAST_1D(uint64_t)
REGISTE_BROADCAST_1D(bool)

//===-------------------------------------------------------------------===//
// Vector broadcast scalar to 1d for b64
//===-------------------------------------------------------------------===//
REGISTE_BROADCAST_SCALAR_B64_1D(int64_t);
REGISTE_BROADCAST_SCALAR_B64_1D(uint64_t);
}
