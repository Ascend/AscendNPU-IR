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
#include "Vector/Cast/CastUtils.h"
#include "Vector/MulExtended/MulExtendedUtils.h"
#include "Vector/VecUtils.h"

/// Scalar implementation for unaligned mul_extended 1D
template <typename T>
__aiv__ __attribute__((always_inline)) void vector_mul_extended_1d_scalar_impl(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ int16_t, 1> *dst0, memref_t<__ubuf__ int16_t, 1> *dst1) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("mul_extended_1d");
#endif
  const int64_t size = dst0->sizes[0];
  int64_t src0_stride = src0->sizes[0] == 1 ? 0 : src0->strides[0];
  int64_t src1_stride = src1->sizes[0] == 1 ? 0 : src1->strides[0];
  int64_t dst0_stride = dst0->strides[0];
  int64_t dst1_stride = dst1->strides[0];

  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;
  __ubuf__ int16_t *dst0_ptr = dst0->aligned + dst0->offset;
  __ubuf__ int16_t *dst1_ptr = dst1->aligned + dst1->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  for (int64_t i = 0; i < size; ++i) {
    int32_t result =
        (int32_t)src0_ptr[i * src0_stride] * (int32_t)src1_ptr[i * src1_stride];
    dst0_ptr[i * dst0_stride] = (int16_t)(result >> 16);
    dst1_ptr[i * dst1_stride] = (int16_t)(result & 0xFFFF);
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

/// Check if mul_extended 1D is unaligned
template <typename T>
__aiv__ __attribute__((always_inline)) bool is_unaligned_mul_extended_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ int16_t, 1> *dst0, memref_t<__ubuf__ int16_t, 1> *dst1) {
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;
  __ubuf__ int16_t *dst0_ptr = dst0->aligned + dst0->offset;
  __ubuf__ int16_t *dst1_ptr = dst1->aligned + dst1->offset;

  bool is_addr_aligned =
      isAddress32ByteAligned(src0_ptr) && isAddress32ByteAligned(src1_ptr) &&
      isAddress32ByteAligned(dst0_ptr) && isAddress32ByteAligned(dst1_ptr);

  bool is_stride_continuous =
      (src0->strides[0] == 1) && (src1->strides[0] == 1) &&
      (dst0->strides[0] == 1) && (dst1->strides[0] == 1);

  return !is_addr_aligned || !is_stride_continuous;
}

/// mul_extended op description:
/// perform multiplication on src0(N-bits) and src1(N-bits),
/// return dst0(N-bits) and dst1(N-bits),
/// dst0 is the high N-bits of the produce(2*N-bits),
/// and dst1 is the low N-bits of the product(2*N-bits),
/// tmp buffer uses f32 / i32 type
///
/// \param src0 (type: memref<a x T>)
/// \param src1 (type: memref<a x T>)
/// \param dst0 (type: memref<a x int16_t>)
/// \param dst1 (type: memref<a x int16_t>)
/// \param tmp_buf (type: memref<3 x a x int32_t>)
///
/// Constraints:
/// 1. src0 and src1 should be int16
/// 2. unable to cast uint16 to float, only int16 is supported right now
/// 3. tmp_buf size should be at least aligned(3 x a x int32_t, num_per_block)

template <typename T>
__aiv__ __attribute__((always_inline)) void vector_mul_extended_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ int16_t, 1> *dst0, memref_t<__ubuf__ int16_t, 1> *dst1,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf) {

  // check type T, need to be int16_t
  static_assert(std::is_same<T, int16_t>::value,
                "mul extended op only support int16_t operands right now!");

  // Check if unaligned, use scalar implementation
  bool is_unalign = is_unaligned_mul_extended_1d<T>(src0, src1, dst0, dst1);
  if (is_unalign) [[unlikely]] {
    vector_mul_extended_1d_scalar_impl<T>(src0, src1, dst0, dst1);
    return;
  }

  const int64_t size0 = src0->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;

  // step1. cast src0(N-bits) and src1(N-bits) to 2*N-bits
  // cast src0: T -> f32 -> i32
  auto cast_to_f32_size = CEIL_FACTOR(size0, num_per_block);
  memref_t<__ubuf__ float, 1> cast_f32_tmp{(__ubuf__ float *)tmp_buf->allocated,
                                           (__ubuf__ float *)tmp_buf->aligned,
                                           tmp_buf->offset,
                                           {cast_to_f32_size},
                                           {1}};

  memref_t<__ubuf__ int32_t, 1> cast_i32_tmp0{tmp_buf->allocated,
                                              tmp_buf->aligned,
                                              cast_f32_tmp.offset +
                                                  cast_f32_tmp.sizes[0],
                                              {cast_to_f32_size},
                                              {1}};

  vector_cast_1d_with_mode<T, float>(src0, &cast_f32_tmp, CastMode::R);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_cast_1d_with_mode<float, int32_t>(&cast_f32_tmp, &cast_i32_tmp0,
                                           CastMode::R);

  // cast src1: T -> f32 -> i32
  memref_t<__ubuf__ int32_t, 1> cast_i32_tmp1{tmp_buf->allocated,
                                              tmp_buf->aligned,
                                              cast_i32_tmp0.offset +
                                                  cast_i32_tmp0.sizes[0],
                                              {cast_to_f32_size},
                                              {1}};
  vector_cast_1d_with_mode<T, float>(src1, &cast_f32_tmp, CastMode::R);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_cast_1d_with_mode<float, int32_t>(&cast_f32_tmp, &cast_i32_tmp1,
                                           CastMode::R);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step2. do vmul on src0_i32 and src1_i32
  memref_t<__ubuf__ int32_t, 1> dst_as_i32;
  view_as<float, int32_t, 1>(&cast_f32_tmp, &dst_as_i32);
  vector_eltwise_vv_1d<VectorOpTy::VMUL, int32_t>(&cast_i32_tmp0,
                                                  &cast_i32_tmp1, &dst_as_i32);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step3. cast vmul result i32 ptr to i16 ptr
  memref_t<__ubuf__ int16_t, 1> dst_as_i16;
  view_as<int32_t, int16_t, 1>(&dst_as_i32, &dst_as_i16);

  // step4. use vreducev2 to get high 16-bits and low 16-bits
  vreducev2_1d_with_pattern_mode<int16_t, PatternMode::INDEX_1_FROM_2_ELEMENTS>(
      &dst_as_i16, dst0);
  vreducev2_1d_with_pattern_mode<int16_t, PatternMode::INDEX_0_FROM_2_ELEMENTS>(
      &dst_as_i16, dst1);
}

extern "C" {
//===-------------------------------------------------------------------===//
// mul_extended op, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_MULEXTENDED(1, int16_t)
}