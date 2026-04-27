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
#include "Vector/Cast/CastUtils.h"
#include "Vector/CmpSel/CmpUtils.h"
#include "Vector/CmpSel/SelUtils.h"
#include "Vector/Pow/PowUtils.h"
#include "Vector/Reduction/ReductionUtils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
pow_int_min_exp_init_1d(memref_t<__ubuf__ T, 1> *exp,
                        memref_t<__ubuf__ T, 1> *tmp_buf) {
  // step1: find every exponent equals to INT32_MIN
  const int64_t size = exp->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int bits_int = bytes * BITS_PER_BYTE;
  constexpr int bits_bool = bitwidthOf<bool>();
  constexpr int num_bool_per_block =
      BITS_PER_BYTE * INTR_BYTES_PER_BLOCK / bits_bool;
  const int64_t tmp_bool_size = CEIL_FACTOR(size, num_bool_per_block);
  constexpr int int_bool_bits_factor = bits_int / bits_bool;
  memref_t<__ubuf__ bool, 1> tmp_condition{(__ubuf__ bool *)tmp_buf->allocated,
                                           (__ubuf__ bool *)tmp_buf->aligned,
                                           tmp_buf->offset *
                                               int_bool_bits_factor,
                                           {size},
                                           {1}};
  vector_compare_vs_1d<VectorOpTy::VCMPS_EQ, T>(exp, INT32_MIN, &tmp_condition);

  // step2: replace INT32_MIN with -2 in exponents
  // exp = (exp == INT_MIN) ? -2 : exp
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  const int64_t tmp_i32_size = CEIL_FACTOR(size, num_per_block);
  const int64_t tmp_bool_size_as_i32 = tmp_bool_size / int_bool_bits_factor;
  memref_t<__ubuf__ T, 1> tmp_brc_int{tmp_buf->allocated,
                                      tmp_buf->aligned,
                                      tmp_buf->offset + tmp_bool_size_as_i32,
                                      {size},
                                      {1}};
  __ubuf__ T *tmp_brc_int_ptr = tmp_brc_int.aligned + tmp_brc_int.offset;
  brc_scalar_core_1d<T>(-2, tmp_brc_int_ptr, tmp_brc_int.sizes[0]);
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ T, 1> tmp_condition_addr_buf{tmp_buf->allocated,
                                                 tmp_buf->aligned,
                                                 tmp_brc_int.offset +
                                                     tmp_i32_size,
                                                 {num_per_block},
                                                 {1}};
  vector_select_vv_1d<T>(&tmp_condition, &tmp_brc_int, exp, exp,
                         &tmp_condition_addr_buf);
}

template <typename T>
__aiv__ __attribute__((always_inline)) memref_t<__ubuf__ bool, 1>
pow_neg_exp_init_1d(memref_t<__ubuf__ T, 1> *base, memref_t<__ubuf__ T, 1> *exp,
                    memref_t<__ubuf__ T, 1> *tmp_buf) {
  // step1: store all negaive exponents
  const int64_t size = exp->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int bits_int = bytes * BITS_PER_BYTE;
  constexpr int bits_bool = bitwidthOf<bool>();
  constexpr int num_bool_per_block =
      BITS_PER_BYTE * INTR_BYTES_PER_BLOCK / bits_bool;
  const int64_t tmp_bool_size = CEIL_FACTOR(size, num_bool_per_block);
  constexpr int int_bool_bits_factor = bits_int / bits_bool;
  memref_t<__ubuf__ bool, 1> tmp_neg_exp_mask{
      (__ubuf__ bool *)tmp_buf->allocated,
      (__ubuf__ bool *)tmp_buf->aligned,
      tmp_buf->offset * int_bool_bits_factor,
      {size},
      {1}};
  // change the offset of temp buffer to avoid overwiting
  // the tmp_neg_exp_mask memory
  tmp_buf->offset += tmp_bool_size / int_bool_bits_factor;

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  const int64_t tmp_i32_size = CEIL_FACTOR(size, num_per_block);
  memref_t<__ubuf__ float, 1> tmp_cast_f32{(__ubuf__ float *)tmp_buf->allocated,
                                           (__ubuf__ float *)tmp_buf->aligned,
                                           tmp_buf->offset,
                                           {size},
                                           {1}};
  vector_cast_1d_with_mode<T, float>(exp, &tmp_cast_f32, CastMode::R);
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ T, 1> tmp_condition_addr_buf{tmp_buf->allocated,
                                                 tmp_buf->aligned,
                                                 tmp_cast_f32.offset +
                                                     tmp_i32_size,
                                                 {num_per_block},
                                                 {1}};
  vector_compare_vs_1d<VectorOpTy::VCMPS_LT, float>(&tmp_cast_f32, 0.0f,
                                                    &tmp_neg_exp_mask);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step2: change negative exponents to positive by multiply -1
  // is_neg_exp = exp < 0
  // exp = is_neg_exp ? -1 * exp : exp
  memref_t<__ubuf__ T, 1> tmp_brc_int;
  view_as<float, T, 1>(&tmp_cast_f32, &tmp_brc_int);
  __ubuf__ T *tmp_brc_int_ptr = tmp_brc_int.aligned + tmp_brc_int.offset;
  brc_scalar_core_1d<T>(-1, tmp_brc_int_ptr, tmp_brc_int.sizes[0]);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_1d<VectorOpTy::VMUL, T>(exp, &tmp_brc_int, &tmp_brc_int);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_select_vv_1d<T>(&tmp_neg_exp_mask, &tmp_brc_int, exp, exp,
                         &tmp_condition_addr_buf);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step3: tmp_neg_exp_mask = is_neg_exp && (base != 0)
  memref_t<__ubuf__ bool, 1> tmp_zero_base_mask{
      (__ubuf__ bool *)tmp_condition_addr_buf.allocated,
      (__ubuf__ bool *)tmp_condition_addr_buf.aligned,
      tmp_condition_addr_buf.offset * int_bool_bits_factor +
          tmp_condition_addr_buf.sizes[0] * int_bool_bits_factor,
      {size},
      {1}};
  // base != 0
  vector_compare_vs_1d<VectorOpTy::VCMPS_NE, T>(base, 0, &tmp_zero_base_mask);
  INTRINSIC(pipe_barrier, PIPE_V);

  vector_eltwise_vv_1d<VectorOpTy::VAND, bool>(
      &tmp_zero_base_mask, &tmp_neg_exp_mask, &tmp_neg_exp_mask);
  INTRINSIC(pipe_barrier, PIPE_V);
  return tmp_neg_exp_mask;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
pow_do_calculate_1d(memref_t<__ubuf__ T, 1> *base, memref_t<__ubuf__ T, 1> *exp,
                    memref_t<__ubuf__ T, 1> *res,
                    memref_t<__ubuf__ T, 1> *tmp_buf) {
  // step1: initialization for Binary Exponentiation pow calculation loop
  // vbrc(1, tmp)
  const int64_t size = exp->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  const int64_t tmp_i32_size = CEIL_FACTOR(size, num_per_block);
  memref_t<__ubuf__ T, 1> tmp_brc_int{tmp_buf->allocated,
                                      tmp_buf->aligned,
                                      tmp_buf->offset,
                                      {size},
                                      {1}};
  __ubuf__ T *tmp_brc_int_ptr = tmp_brc_int.aligned + tmp_brc_int.offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / bytes;
  brc_scalar_core_1d<T>(1, tmp_brc_int_ptr, tmp_brc_int.sizes[0]);
  INTRINSIC(pipe_barrier, PIPE_V);

  // store temp results between calculations
  memref_t<__ubuf__ T, 1> tmp_res{tmp_buf->allocated,
                                  tmp_buf->aligned,
                                  tmp_brc_int.offset + tmp_i32_size,
                                  {size},
                                  {1}};

  // temp buffer for reduce_sum op
  auto tmp_reduce_size = num_per_repeat + num_per_block;
  if (size <= num_per_repeat) {
    tmp_reduce_size = CEIL_FACTOR(3 * size / 2, num_per_block);
  }
  memref_t<__ubuf__ T, 1> tmp_reduce_buf{tmp_buf->allocated,
                                         tmp_buf->aligned,
                                         tmp_res.offset + tmp_i32_size,
                                         {tmp_reduce_size},
                                         {1}};

  // store temp results of vcmp op as operands for vsel op
  constexpr int bits_int = bytes * BITS_PER_BYTE;
  constexpr int bits_bool = bitwidthOf<bool>();
  constexpr int num_bool_per_block =
      BITS_PER_BYTE * INTR_BYTES_PER_BLOCK / bits_bool;
  const int64_t tmp_bool_size = CEIL_FACTOR(size, num_bool_per_block);
  constexpr int int_bool_bits_factor = bits_int / bits_bool;
  memref_t<__ubuf__ T, 1> tmp_condition_addr_buf{tmp_buf->allocated,
                                                 tmp_buf->aligned,
                                                 tmp_reduce_buf.offset +
                                                     tmp_reduce_buf.sizes[0],
                                                 {num_per_block},
                                                 {1}};
  memref_t<__ubuf__ bool, 1> tmp_condition{
      (__ubuf__ bool *)tmp_buf->allocated,
      (__ubuf__ bool *)tmp_buf->aligned,
      (tmp_condition_addr_buf.offset + tmp_condition_addr_buf.sizes[0]) *
          int_bool_bits_factor,
      {size},
      {1}};

  // step2: Binary Exponentiation pow calculation loop body
  // while (exp > 0)
  while (true) {
    // res = (exp & 1) ? res * base : res
    vector_eltwise_vv_1d<VectorOpTy::VAND, T>(exp, &tmp_brc_int, &tmp_res);
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_compare_vs_1d<VectorOpTy::VCMPS_EQ, T>(&tmp_res, 0, &tmp_condition);
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vv_1d<VectorOpTy::VMUL, T>(res, base, &tmp_res);
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_select_vv_1d<T>(&tmp_condition, res, &tmp_res, res,
                           &tmp_condition_addr_buf);

    // exp >> 1
    // if (exp == 0) -> return
    vector_eltwise_vs_1d<VectorOpTy::VSHR, T>(exp, 1, exp);
    INTRINSIC(pipe_barrier, PIPE_V);
    // reuse tmp_condition_addr_buf as result of reduce_sum
    reduce_r<ReduceOpTy::REDUCE_SUM, int32_t>(exp, &tmp_condition_addr_buf,
                                              &tmp_reduce_buf, 0);
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    T tmp_reduce_sum =
        *(tmp_condition_addr_buf.aligned + tmp_condition_addr_buf.offset);
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    if (tmp_reduce_sum == 0) {
      return;
    }

    // base *= base
    vector_eltwise_vv_1d<VectorOpTy::VMUL, T>(base, base, base);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
pow_replace_neg_exp_res_1d(memref_t<__ubuf__ T, 1> *res,
                           memref_t<__ubuf__ bool, 1> *tmp_neg_exp_mask,
                           memref_t<__ubuf__ T, 1> *tmp_buf) {
  // reciprocal = 1 / res
  const int64_t size = res->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  const int64_t tmp_i32_size = CEIL_FACTOR(size, num_per_block);
  memref_t<__ubuf__ float, 1> tmp_cast_f32{(__ubuf__ float *)tmp_buf->allocated,
                                           (__ubuf__ float *)tmp_buf->aligned,
                                           tmp_buf->offset,
                                           {size},
                                           {1}};
  vector_cast_1d_with_mode<T, float>(res, &tmp_cast_f32, CastMode::R);
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ float, 1> tmp_brc_float{
      (__ubuf__ float *)tmp_buf->allocated,
      (__ubuf__ float *)tmp_buf->aligned,
      tmp_cast_f32.offset + tmp_i32_size,
      {size},
      {1}};

  // TODO: need vdiv to suport vector-scalar op to replace the vbrc
  __ubuf__ float *tmp_brc_float_ptr =
      tmp_brc_float.aligned + tmp_brc_float.offset;
  brc_scalar_core_1d<float>(1.0f, tmp_brc_float_ptr, tmp_brc_float.sizes[0]);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_1d<VectorOpTy::VDIV, float>(&tmp_brc_float, &tmp_cast_f32,
                                                &tmp_cast_f32);
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ T, 1> tmp_res;
  view_as<float, T, 1>(&tmp_brc_float, &tmp_res);
  vector_cast_1d_with_mode<float, int32_t>(&tmp_cast_f32, &tmp_res,
                                           CastMode::Z);
  INTRINSIC(pipe_barrier, PIPE_V);

  // res = (exp < 0) ? 1 / res : res
  memref_t<__ubuf__ T, 1> tmp_condition_addr_buf{tmp_buf->allocated,
                                                 tmp_buf->aligned,
                                                 tmp_brc_float.offset +
                                                     tmp_i32_size,
                                                 {num_per_block},
                                                 {1}};
  vector_select_vv_1d<T>(tmp_neg_exp_mask, &tmp_res, res, res,
                         &tmp_condition_addr_buf);
}

/// pow op description:
/// take the power of each element in base(a) with exp(a),
/// and return the result res(a)
///
/// \param base (type: memref<a x T>)
/// \param exp (type: memref<a x T>)
/// \param res (type: memref<a x T>)
/// \param tmp_buf (type: memref<c x T>)
///
/// Constraints:
/// 1. only int32_t type of pow op needs to be implemented here,
/// i8/i16 cast to f16/f32
/// 2. tmp_buf size should be at least 2 x aligned(a x T, num_per_block) +
/// 2 x aligned(a x T, num_bool_per_block) / (bitwidth(int32) / bitwidth(bool))
/// + 1 x num_per_block + 1 x aligned(tmp_buf_reducer, num_per_block)
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_pow_1d(memref_t<__ubuf__ T, 1> *base, memref_t<__ubuf__ T, 1> *exp,
              memref_t<__ubuf__ T, 1> *res, memref_t<__ubuf__ T, 1> *tmp_buf) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_v_1d(base, exp);
  // check type T, need to be int32_t
  static_assert(std::is_same<T, int32_t>::value,
                "pow op only support int32_t operands in template!");
  //  exp = (exp == INT_MIN) ? -2 : exp;
  //  exp = (exp < 0) ? -exp : exp;
  //  res = 1;
  //  while (exp > 0) {
  //    if (exp & 1) {
  //      res *= base
  //    }
  //    exp >> 1
  //    base *= base
  //  }
  //  res = (exp < 0 && base != 0) ? 1 / res : res;

  // step1: initialize result
  // 1. set every element in res to 1
  //    vbrc(1, res)
  __ubuf__ T *res_ptr = res->aligned + res->offset;
  brc_scalar_core_1d<T>(1, res_ptr, res->sizes[0]);
  INTRINSIC(pipe_barrier, PIPE_V);

  // 2. handle INT_MIN in exponents
  //    exp = (exp == INT_MIN) ? -2 : exp
  pow_int_min_exp_init_1d<T>(exp, tmp_buf);

  // 3. handle negative exponents by multiplying -1
  //    tmp_neg_exp_mask = (exp < 0) && (base != 0)
  //    exp = (exp < 0) ? -exp : exp
  memref_t<__ubuf__ bool, 1> tmp_neg_exp_mask =
      pow_neg_exp_init_1d<T>(base, exp, tmp_buf);

  // step2: calculate pow results
  // while (exp > 0) {
  //   if (exp & 1) {
  //     res *= base
  //   }
  //   exp >> 1
  //   base *= base
  // }
  pow_do_calculate_1d<T>(base, exp, res, tmp_buf);

  // step3: replace results with reciprocal for negative exponents
  // res = (exp < 0) ? 1 / res : res
  pow_replace_neg_exp_res_1d<T>(res, &tmp_neg_exp_mask, tmp_buf);
}

extern "C" {
//===-------------------------------------------------------------------===//
// pow op, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_POW(1, int32_t)
}