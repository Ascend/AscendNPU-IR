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

#include "Vector/Cast/CastUtils.h"
#include "Vector/CmpSel/CmpUtils.h"
#include "Vector/VecUtils.h"

// Compile-time dispatched scalar comparison. Only the branch matching OP is
// instantiated, so there is zero runtime overhead from unused branches.
// Supports both vector-vs-vector (VCMP_*) and vector-vs-scalar (VCMPS_*) op
// codes.
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) bool scalar_compare_op_native(T a, T b) {
  constexpr bool is_valid_op =
      OP == VectorOpTy::VCMP_EQ || OP == VectorOpTy::VCMPS_EQ ||
      OP == VectorOpTy::VCMP_NE || OP == VectorOpTy::VCMPS_NE ||
      OP == VectorOpTy::VCMP_LT || OP == VectorOpTy::VCMPS_LT ||
      OP == VectorOpTy::VCMP_GT || OP == VectorOpTy::VCMPS_GT ||
      OP == VectorOpTy::VCMP_GE || OP == VectorOpTy::VCMPS_GE ||
      OP == VectorOpTy::VCMP_LE || OP == VectorOpTy::VCMPS_LE;
  static_assert(is_valid_op, "Cmp: Invalid comparison operator");

  if constexpr (OP == VectorOpTy::VCMP_EQ || OP == VectorOpTy::VCMPS_EQ) {
    return a == b;
  } else if constexpr (OP == VectorOpTy::VCMP_NE ||
                       OP == VectorOpTy::VCMPS_NE) {
    return a != b;
  } else if constexpr (OP == VectorOpTy::VCMP_LT ||
                       OP == VectorOpTy::VCMPS_LT) {
    return a < b;
  } else if constexpr (OP == VectorOpTy::VCMP_GT ||
                       OP == VectorOpTy::VCMPS_GT) {
    return a > b;
  } else if constexpr (OP == VectorOpTy::VCMP_GE ||
                       OP == VectorOpTy::VCMPS_GE) {
    return a >= b;
  } else if constexpr (OP == VectorOpTy::VCMP_LE ||
                       OP == VectorOpTy::VCMPS_LE) {
    return a <= b;
  }
  return false;
}

// Convert an IEEE 754 half-precision (fp16, 16-bit) bit pattern to an IEEE 754
// single-precision (fp32, 32-bit) float.
//
// Implementation principle:
//   Extract sign (1 bit), exponent (5 bits), mantissa (10 bits) from the 16-bit
//   pattern, handle zero / subnormal / infinity / NaN as special cases, then
//   reassemble into a 32-bit float pattern. The float value is obtained via
//   reinterpret_cast to avoid any floating-point arithmetic.
//
//   IEEE 754 half-precision (16 bits):        IEEE 754 single-precision (32
//   bits):
//   +-------+----------+----------+           +-------+----------+----------+
//   | Sign  | Exponent | Mantissa |           | Sign  | Exponent | Mantissa |
//   | 1 bit |  5 bits  | 10 bits  |           | 1 bit |  8 bits  | 23 bits  |
//   +-------+----------+----------+           +-------+----------+----------+
//
//   Exponent bias: half = 15, float = 127.  Bias difference = 127 - 15 = 112.
//   Mantissa shift: 10 bits -> 23 bits = left shift by 13.
__aiv__ __attribute__((always_inline)) float half_bits_to_float(uint16_t h) {
  // Extract sign (bit 15), exponent (bits 10-14), mantissa (bits 0-9)
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exponent = (h & 0x7C00u) >> 10;
  uint32_t mantissa = h & 0x03FFu;
  uint32_t f_bits;

  if (exponent == 0) {
    // Case 1: exponent all zeros -> zero (mantissa == 0) or subnormal (mantissa
    // != 0). Subnormal value = (-1)^sign * 2^(-14) * 0.mantissa (no implicit
    // leading 1). To convert, normalize by shifting mantissa left until the
    // implicit leading 1 appears at bit 10, then adjust exponent accordingly.
    if (mantissa != 0) {
      // Shift left until bit 10 (0x0400) is set, decrementing exponent per
      // shift to preserve the numerical value.
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      // Remove the now-visible implicit leading 1 (bit 10) from mantissa
      exponent++;
      mantissa &= 0x03FFu;
    }
    // Zero: sign | 0 | 0 = correctly signed zero in float.
    // Subnormal after normalization: adjusted exponent + 112 maps to float
    // range.
    f_bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  } else if (exponent == 31) {
    // Case 2: exponent all ones (0x1F).
    // mantissa == 0: infinity (+/- per sign bit).
    // mantissa != 0: NaN (quiet or signaling).
    // In float: exponent = 0xFF (all ones), mantissa shifted left by 13 bits.
    f_bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    // Case 3: normal number.
    // value = (-1)^sign * 2^(exponent - 15) * 1.mantissa
    // Convert bias: float_exponent = exponent - 15 + 127 = exponent + 112
    f_bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  return *reinterpret_cast<float *>(&f_bits);
}

// Compare two half-precision (fp16) values by converting both to float (fp32)
// first. Converting to float delegates to native float comparison, which
// correctly handles all IEEE 754 semantics including signed zero (0.0 == -0.0)
// and NaN.
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) bool compare_half(uint16_t val0,
                                                         uint16_t val1) {
  float fa = half_bits_to_float(val0);
  float fb = half_bits_to_float(val1);
  return scalar_compare_op_native<OP, float>(fa, fb);
}

// Store a boolean comparison result as a single bit in a packed byte buffer.
// The destination uses a compact bitmask representation (1 bit per element)
// instead of a boolean array (8 bits per element) to save memory.
//
// Bit indexing: bit 0 = LSB of byte 0, bit 7 = MSB of byte 0,
//               bit 8 = LSB of byte 1, etc.
__aiv__ __attribute__((always_inline)) void
set_result_bit(__ubuf__ uint8_t *dst_byte_ptr, int64_t abs_bit_pos,
               bool result) {
  int64_t byte_pos = abs_bit_pos / BITS_PER_BYTE;
  int64_t bit_in_byte = abs_bit_pos % BITS_PER_BYTE;
  if (result) {
    dst_byte_ptr[byte_pos] |= (1 << bit_in_byte);
  } else {
    dst_byte_ptr[byte_pos] &= ~(1 << bit_in_byte);
  }
}

// Scalar fallback for element-wise comparison of two 1D memrefs (src0[i] OP
// src1[i]). Used when input memory is not aligned for vectorized vcmpv
// instructions.
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                     memref_t<__ubuf__ T, 1> *src1,
                     memref_t<__ubuf__ bool, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("compare vv 1d unaligned");
#endif
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  __ubuf__ uint8_t *dst_byte_ptr = (__ubuf__ uint8_t *)dst->aligned;
  const int64_t n = src0->sizes[0];
  for (int64_t i = 0; i < n; ++i) {
    int64_t abs_bit_pos = dst->offset + i * dst->strides[0];
    bool result;
    if constexpr (std::is_same<T, half>::value) {
      // Half precision: reinterpret as uint16_t and convert to float for
      // comparison
      __ubuf__ uint16_t *src0_u16 =
          reinterpret_cast<__ubuf__ uint16_t *>(src0->aligned + src0->offset);
      __ubuf__ uint16_t *src1_u16 =
          reinterpret_cast<__ubuf__ uint16_t *>(src1->aligned + src1->offset);
      uint16_t val0 = src0_u16[i * src0->strides[0]];
      uint16_t val1 = src1_u16[i * src1->strides[0]];
      result = compare_half<OP, T>(val0, val1);
    } else {
      // Other types: direct native comparison
      __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
      __ubuf__ T *src1_ptr = src1->aligned + src1->offset;
      T val0 = src0_ptr[i * src0->strides[0]];
      T val1 = src1_ptr[i * src1->strides[0]];
      result = scalar_compare_op_native<OP, T>(val0, val1);
    }
    set_result_bit(dst_byte_ptr, abs_bit_pos, result);
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

// Scalar fallback for element-wise comparison of a 1D memref against a scalar
// (src0[i] OP scalar). Same synchronization and half-handling pattern as
// scalar_compare_vv_1d.
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                     memref_t<__ubuf__ bool, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("compare vs 1d unaligned");
#endif
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  __ubuf__ uint8_t *dst_byte_ptr = (__ubuf__ uint8_t *)dst->aligned;
  const int64_t n = src0->sizes[0];
  uint16_t uscalar = *reinterpret_cast<uint16_t *>(&scalar);

  for (int64_t i = 0; i < n; ++i) {
    int64_t abs_bit_pos = dst->offset + i * dst->strides[0];
    bool result;
    if constexpr (std::is_same<T, half>::value) {
      // Half precision: reinterpret as uint16_t and convert to float for
      // comparison
      __ubuf__ uint16_t *src0_u16 =
          reinterpret_cast<__ubuf__ uint16_t *>(src0->aligned + src0->offset);
      uint16_t val0 = src0_u16[i * src0->strides[0]];
      result = compare_half<OP, T>(val0, uscalar);
    } else {
      // Other types: direct native comparison
      __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
      T val0 = src0_ptr[i * src0->strides[0]];
      result = scalar_compare_op_native<OP, T>(val0, scalar);
    }
    set_result_bit(dst_byte_ptr, abs_bit_pos, result);
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

// Check alignment for vector-vector compare: src0, src1, and dst must be
// aligned.
template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_memref_aligned_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                                memref_t<__ubuf__ T, 1> *src1,
                                memref_t<__ubuf__ bool, 1> *dst) {
  return (is_memref_aligned<T, 1>(src0) && is_memref_aligned<T, 1>(src1) &&
          is_memref_aligned<bool, 1>(dst));
}

// Check alignment for vector-scalar compare: src0, and dst must be aligned.
template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_memref_aligned_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0,
                                memref_t<__ubuf__ bool, 1> *dst) {
  return (is_memref_aligned<T, 1>(src0) && is_memref_aligned<bool, 1>(dst));
}

// Check ub size remained for vector-vector compare: src0, src1 vector access
// span (repeat-aligned) must not exceed UB limit.
template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_ub_repeat_space_enough_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                                        memref_t<__ubuf__ T, 1> *src1) {
  return (is_ub_repeat_space_enough<T, 1>(src0) &&
          is_ub_repeat_space_enough<T, 1>(src1));
}

// Check ub size remained for vector-scalar compare: src0 vector access span
// (repeat-aligned) must not exceed UB limit.
template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_ub_repeat_space_enough_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0) {
  return (is_ub_repeat_space_enough<T, 1>(src0));
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                     memref_t<__ubuf__ T, 1> *src1,
                     memref_t<__ubuf__ bool, 1> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vv_1d<T, bool>(src0, src1, dst);
  const int64_t src_total_size = src0->sizes[0];

  // view dst as uint8 because dst type of vcmp instr is uint8
  memref_t<__ubuf__ uint8_t, 1> dst_as_uint8;
  view_as<bool, uint8_t, 1>(dst, &dst_as_uint8);

  __ubuf__ uint8_t *dst_ptr = dst_as_uint8.aligned + dst_as_uint8.offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;

  constexpr int src_bytes = sizeof(T);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / src_bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int max_repeat_times = get_vcmp_max_repeat_cnts<T>();

  uint64_t repeat_times = CEIL_DIV(src_total_size, num_per_repeat);
  uint16_t src_repeat_stride = num_per_repeat / num_per_block;

  if (repeat_times >= max_repeat_times) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    assert(repeat_times / max_repeat_times < 2 && "ub overflow");
#endif

    vector_eltwise_vv_intrin<OP, T>(
        intrin_args<2, T, uint8_t>{dst_ptr,
                                   {src0_ptr, src1_ptr},
                                   0,
                                   max_repeat_times,
                                   1,      // dst_block_stride
                                   {1, 1}, // src_block_stride
                                   0,      // dst_repeat_stride
                                   {src_repeat_stride, src_repeat_stride}});
  }

  if (repeat_times % max_repeat_times != 0) {
    uint64_t main_repeat_times = FLOOR_FACTOR(repeat_times, max_repeat_times);
    uint64_t remain_repeat_times = repeat_times % max_repeat_times;
    uint64_t dst_offset = num_per_repeat / BITS_PER_BYTE * main_repeat_times;
    uint64_t src_offset = num_per_repeat * main_repeat_times;

    vector_eltwise_vv_intrin<OP, T>(intrin_args<2, T, uint8_t>{
        dst_ptr + dst_offset,
        {src0_ptr + src_offset, src1_ptr + src_offset},
        0,
        remain_repeat_times,
        1,      // dst_block_stride
        {1, 1}, // src_block_stride
        0,      // dst_repeat_stride
        {src_repeat_stride, src_repeat_stride}});
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_scalar_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                            memref_t<__ubuf__ T, 1> *src1,
                            memref_t<__ubuf__ bool, 1> *dst) {
  const int64_t src_total_size = src0->sizes[0];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  int64_t main_elements = FLOOR_FACTOR(src_total_size, num_per_repeat);

  if (main_elements > 0) {
    memref_t<__ubuf__ T, 1> main_src0 = *src0;
    memref_t<__ubuf__ T, 1> main_src1 = *src1;
    memref_t<__ubuf__ bool, 1> main_dst = *dst;
    main_src0.sizes[0] = main_elements;
    main_src1.sizes[0] = main_elements;
    main_dst.sizes[0] = main_elements;
    vector_compare_vv_1d<OP, T>(&main_src0, &main_src1, &main_dst);
    INTRINSIC(pipe_barrier, PIPE_V);
  }

  int64_t tail_elements = src_total_size - main_elements;
  if (tail_elements > 0) {
    memref_t<__ubuf__ T, 1> tail_src0 = *src0;
    memref_t<__ubuf__ T, 1> tail_src1 = *src1;
    memref_t<__ubuf__ bool, 1> tail_dst = *dst;
    tail_src0.offset = src0->offset + main_elements;
    tail_src1.offset = src1->offset + main_elements;
    tail_dst.offset = dst->offset + main_elements;
    tail_src0.sizes[0] = tail_elements;
    tail_src1.sizes[0] = tail_elements;
    tail_dst.sizes[0] = tail_elements;
    scalar_compare_vv_1d<OP, T>(&tail_src0, &tail_src1, &tail_dst);
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                     memref_t<__ubuf__ bool, 1> *dst) {
  check_inputs_of_vector_eltwise_v_1d<T, bool>(src0, dst);
  const int64_t src_total_size = src0->sizes[0];

  // view dst as uint8 because dst type of vcmp instr is uint8
  memref_t<__ubuf__ uint8_t, 1> dst_as_uint8;
  view_as<bool, uint8_t, 1>(dst, &dst_as_uint8);

  __ubuf__ uint8_t *dst_ptr = dst_as_uint8.aligned + dst_as_uint8.offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;

  constexpr int src_bytes = sizeof(T);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / src_bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / src_bytes;
  constexpr int max_repeat_times = get_vcmp_max_repeat_cnts<T>();

  uint64_t repeat_times = CEIL_DIV(src_total_size, num_per_repeat);
  uint16_t src_repeat_stride = num_per_repeat / num_per_block;

  if (repeat_times >= max_repeat_times) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    assert(repeat_times / max_repeat_times < 2 && "ub overflow");
#endif

    vector_eltwise_vs_intrin<OP, T>(
        intrin_args<1, T, uint8_t>{dst_ptr,
                                   {src0_ptr},
                                   scalar,
                                   max_repeat_times,
                                   1,   // dst_block_stride
                                   {1}, // src_block_stride
                                   0,   // dst_repeat_stride
                                   {src_repeat_stride}});
  }

  if (repeat_times % max_repeat_times != 0) {
    uint64_t main_repeat_times = FLOOR_FACTOR(repeat_times, max_repeat_times);
    uint64_t remain_repeat_times = repeat_times % max_repeat_times;
    uint64_t dst_offset = num_per_repeat / BITS_PER_BYTE * main_repeat_times;
    uint64_t src_offset = num_per_repeat * main_repeat_times;

    vector_eltwise_vs_intrin<OP, T>(
        intrin_args<1, T, uint8_t>{dst_ptr + dst_offset,
                                   {src0_ptr + src_offset},
                                   scalar,
                                   remain_repeat_times,
                                   1,   // dst_block_stride
                                   {1}, // src_block_stride
                                   0,   // dst_repeat_stride
                                   {src_repeat_stride}});
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_scalar_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                            memref_t<__ubuf__ bool, 1> *dst) {
  const int64_t src_total_size = src0->sizes[0];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  int64_t main_elements = FLOOR_FACTOR(src_total_size, num_per_repeat);

  if (main_elements > 0) {
    memref_t<__ubuf__ T, 1> main_src0 = *src0;
    memref_t<__ubuf__ bool, 1> main_dst = *dst;
    main_src0.sizes[0] = main_elements;
    main_dst.sizes[0] = main_elements;
    vector_compare_vs_1d<OP, T>(&main_src0, scalar, &main_dst);
    INTRINSIC(pipe_barrier, PIPE_V);
  }

  int64_t tail_elements = src_total_size - main_elements;
  if (tail_elements > 0) {
    memref_t<__ubuf__ T, 1> tail_src0 = *src0;
    memref_t<__ubuf__ bool, 1> tail_dst = *dst;
    tail_src0.offset = src0->offset + main_elements;
    tail_dst.offset = dst->offset + main_elements;
    tail_src0.sizes[0] = tail_elements;
    tail_dst.sizes[0] = tail_elements;
    scalar_compare_vs_1d<OP, T>(&tail_src0, scalar, &tail_dst);
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_compare_vv_1d<VectorOpTy::VCMP_NE, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ bool, 1> *dst) {
  /// because vcmpv instruction not support ne int32,
  /// step 1: dst = vcmpv_eq(src0 , src1)
  /// step 2: dst = vnot(dst)
  vector_compare_vv_1d<VectorOpTy::VCMP_EQ, int32_t>(src0, src1, dst);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_1d<VectorOpTy::VNOT, bool>(dst, dst);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_compare_vs_1d<VectorOpTy::VCMPS_NE, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, int32_t scalar,
    memref_t<__ubuf__ bool, 1> *dst) {
  /// because vcmpvs instruction not support ne int32,
  /// step 1: dst = vcmpvs_eq(src0 , src1)
  /// step 2: dst = vnot(dst)
  vector_compare_vs_1d<VectorOpTy::VCMPS_EQ, int32_t>(src0, scalar, dst);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_1d<VectorOpTy::VNOT, bool>(dst, dst);
}

extern "C" {
//===-------------------------------------------------------------------===//
// vcmpv, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, half)
REGISTE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, float)
REGISTE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, int32_t)

REGISTE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, half)
REGISTE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, float)
REGISTE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, int32_t)

REGISTE_CMP_VV(lt, VectorOpTy::VCMP_LT, 1, half)
REGISTE_CMP_VV(lt, VectorOpTy::VCMP_LT, 1, float)

REGISTE_CMP_VV(gt, VectorOpTy::VCMP_GT, 1, half)
REGISTE_CMP_VV(gt, VectorOpTy::VCMP_GT, 1, float)

REGISTE_CMP_VV(ge, VectorOpTy::VCMP_GE, 1, half)
REGISTE_CMP_VV(ge, VectorOpTy::VCMP_GE, 1, float)

REGISTE_CMP_VV(le, VectorOpTy::VCMP_LE, 1, half)
REGISTE_CMP_VV(le, VectorOpTy::VCMP_LE, 1, float)

//===-------------------------------------------------------------------===//
// vcmpvs, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, half)
REGISTE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, float)
REGISTE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, int32_t)

REGISTE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, half)
REGISTE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, float)
REGISTE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, int32_t)

REGISTE_CMP_VS(lt, VectorOpTy::VCMPS_LT, 1, half)
REGISTE_CMP_VS(lt, VectorOpTy::VCMPS_LT, 1, float)

REGISTE_CMP_VS(gt, VectorOpTy::VCMPS_GT, 1, half)
REGISTE_CMP_VS(gt, VectorOpTy::VCMPS_GT, 1, float)

REGISTE_CMP_VS(ge, VectorOpTy::VCMPS_GE, 1, half)
REGISTE_CMP_VS(ge, VectorOpTy::VCMPS_GE, 1, float)

REGISTE_CMP_VS(le, VectorOpTy::VCMPS_LE, 1, half)
REGISTE_CMP_VS(le, VectorOpTy::VCMPS_LE, 1, float)
}