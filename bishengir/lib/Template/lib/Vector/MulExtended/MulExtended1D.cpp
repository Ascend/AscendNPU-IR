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