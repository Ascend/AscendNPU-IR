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

#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/CmpSel/CmpUtils.h"
#include "Vector/CmpSel/SelUtils.h"
#include "Vector/VecUtils.h"
#include <cstdint>

/// General Constraints:
/// 1. src0 and src1 should be float or half -> hence all other types are
///    casted into float or half.
/// 2. Temporary buffer needed for scalar inputs which shared with the buffer
///    for boolean condition.
/// 3. The condition_addr_buf which acts as a temporary buffer should have at
///    least ub_block_unit to accomodate the minimum of 8 values and this
///    increases for each scalar value introduced by the same amount of
///    ub_block_unit. For 16bit inputs 16 values can placed.
/// 4. temporary buffer needs 3 * ub_block_unit memory when data type is int64
/// 5. Temp buffer needed for int64 VV case : aligned(0.5*size, ub_block_unit) +
///    aligned(0.25*size, ub_block_unit)

template <typename T>
__aiv__ __attribute__((always_inline)) void vector_select_1d_intrin(
    intrin_args<2, T> args, memref_t<__ubuf__ bool, 1> *condition,
    memref_t<__ubuf__ T, 1> *condition_addr_buf, int64_t total_size) {
  memref_t<__ubuf__ T, 1> condition_as_template;
  view_as<bool, T, 1>(condition, &condition_as_template);
  __ubuf__ T *condition_addr_ubuf_ptr =
      condition_addr_buf->aligned + condition_addr_buf->offset;
  __ubuf__ T *condition_ptr =
      condition_as_template.aligned + condition_as_template.offset;

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  (*(__ubuf__ uint64_t *)((__ubuf__ T *)condition_addr_ubuf_ptr)) =
      (uint64_t)((__ubuf__ T *)condition_ptr);
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  INTRINSIC(set_cmpmask, (__ubuf__ uint64_t *)condition_addr_ubuf_ptr);
  INTRINSIC(set_vector_mask, 0, total_size);
  INTRINSIC(pipe_barrier, PIPE_V);
  constexpr uint8_t mode = static_cast<uint8_t>(SelectMode::MODE2);

  INTRINSIC(vsel, args.dst, args.src[0], args.src[1], args.repeat,
            args.dst_block_stride, args.src_block_stride[0],
            args.src_block_stride[1], args.dst_repeat_stride,
            args.src_repeat_stride[0], args.src_repeat_stride[1], mode);

  INTRINSIC_NO_ARGS(set_mask_norm);
}

//===--------------------------------------------------------------------===//
// vsel, 1 dim => for Vector-Vector Inputs
//
// Constraints:
// 1. src0 and src1 should be float or half -> hence all other types
// are casted into float or half.
// 2. src0 and src1 are vector inputs.
// 3. The condition_addr_buf which acts as a temporary buffer should have at
// least ub_block_unit to accomodate the minimum of 8 values.
// 4. Temp buffer needed for int64 VV case : aligned(0.5*size, ub_block_unit) +
//    aligned(0.25*size, ub_block_unit)
//===--------------------------------------------------------------------===//
template <typename T, typename COND_T>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d(
    memref_t<__ubuf__ COND_T, 1> *condition, memref_t<__ubuf__ T, 1> *src0,
    memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  vector_select_vv_1d<T>(condition, src0, src1, dst, condition_addr_buf);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_vv_1d(memref_t<__ubuf__ bool, 1> *condition,
                    memref_t<__ubuf__ T, 1> *src0,
                    memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  vector_select_1d_intrin(intrin_args<2, T>{dst->aligned + dst->offset,
                                            {src0->aligned + src0->offset,
                                             src1->aligned + src1->offset},
                                            0,
                                            0,
                                            1,       // dst_block_stride
                                            {1, 1},  // src_block_stride
                                            8,       // dst_repeat_stride
                                            {8, 8}}, // src_repeat_stride
                          condition, condition_addr_buf, dst->sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d<int16_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ int16_t, 1> *src0,
    memref_t<__ubuf__ int16_t, 1> *src1, memref_t<__ubuf__ int16_t, 1> *dst,
    memref_t<__ubuf__ int16_t, 1> *condition_addr_buf) {
  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;

  view_as<int16_t, half, 1>(src0, &src0_as_half);
  view_as<int16_t, half, 1>(src1, &src1_as_half);
  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  vector_select_vv_1d<half>(condition, &src0_as_half, &src1_as_half,
                            &dst_as_half, &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d<uint16_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ uint16_t, 1> *src0,
    memref_t<__ubuf__ uint16_t, 1> *src1, memref_t<__ubuf__ uint16_t, 1> *dst,
    memref_t<__ubuf__ uint16_t, 1> *condition_addr_buf) {

  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;

  view_as<uint16_t, half, 1>(src0, &src0_as_half);
  view_as<uint16_t, half, 1>(src1, &src1_as_half);
  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  vector_select_vv_1d<half>(condition, &src0_as_half, &src1_as_half,
                            &dst_as_half, &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d<bfloat16_t>(
    memref_t<__ubuf__ bool, 1> *condition,
    memref_t<__ubuf__ bfloat16_t, 1> *src0,
    memref_t<__ubuf__ bfloat16_t, 1> *src1,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *condition_addr_buf) {

  // convert bfloat16 memref to half memref
  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;

  view_as<bfloat16_t, half, 1>(src0, &src0_as_half);
  view_as<bfloat16_t, half, 1>(src1, &src1_as_half);
  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  vector_select_vv_1d<half>(condition, &src0_as_half, &src1_as_half,
                            &dst_as_half, &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d<int32_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ int32_t, 1> *src0,
    memref_t<__ubuf__ int32_t, 1> *src1, memref_t<__ubuf__ int32_t, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *condition_addr_buf) {

  // convert int32 memref to float memref
  memref_t<__ubuf__ float, 1> src0_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;

  view_as<int32_t, float, 1>(src0, &src0_as_float);
  view_as<int32_t, float, 1>(src1, &src1_as_float);
  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);

  vector_select_vv_1d<float>(condition, &src0_as_float, &src1_as_float,
                             &dst_as_float, &cond_addr_buf_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d<uint32_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ uint32_t, 1> *src0,
    memref_t<__ubuf__ uint32_t, 1> *src1, memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *condition_addr_buf) {

  // convert uint32 memref to float memref
  memref_t<__ubuf__ float, 1> src0_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;

  view_as<uint32_t, float, 1>(src0, &src0_as_float);
  view_as<uint32_t, float, 1>(src1, &src1_as_float);
  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);

  vector_select_vv_1d<float>(condition, &src0_as_float, &src1_as_float,
                             &dst_as_float, &cond_addr_buf_as_float);
}

//===--------------------------------------------------------------------===//
// vsel, 1 dim => for Scalar-Scalar Inputs
//
// Constraints:
// 1. src0 and src1 should be float or half -> hence all other types
//    are casted into float or half.
// 2. src0 and src1 are scalar inputs.
// 3. The condition_addr_buf which acts as a temporary buffer should have at
//    least (3 * ub_block_unit) to accomodate the minimum of 8 values
//    for each of the 3 i.e., src0 src1 and condition variables.
//===--------------------------------------------------------------------===//
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_ss_1d(memref_t<__ubuf__ bool, 1> *condition, T src0, T src1,
                    memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  __ubuf__ T *src0_brc_ptr = (condition_addr_buf->aligned +
                              condition_addr_buf->offset + num_per_block);
  __ubuf__ T *src1_brc_ptr = (condition_addr_buf->aligned +
                              condition_addr_buf->offset + (2 * num_per_block));

  brc_scalar_core_1d(src0, src0_brc_ptr, (int64_t)(num_per_block));
  brc_scalar_core_1d(src1, src1_brc_ptr, (int64_t)(num_per_block));

  vector_select_1d_intrin(intrin_args<2, T>{dst->aligned + dst->offset,
                                            {src0_brc_ptr, src1_brc_ptr},
                                            0,
                                            0,
                                            1,       // dst_block_stride
                                            {0, 0},  // src_block_stride
                                            8,       // dst_repeat_stride
                                            {0, 0}}, // src_repeat_stride
                          condition, condition_addr_buf, dst->sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_ss_1d<int16_t>(
    memref_t<__ubuf__ bool, 1> *condition, int16_t src0, int16_t src1,
    memref_t<__ubuf__ int16_t, 1> *dst,
    memref_t<__ubuf__ int16_t, 1> *condition_addr_buf) {
  auto *src0_half = reinterpret_cast<half *>(&src0);
  auto *src1_half = reinterpret_cast<half *>(&src1);

  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;

  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  vector_select_ss_1d(condition, *src0_half, *src1_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_ss_1d<uint16_t>(
    memref_t<__ubuf__ bool, 1> *condition, uint16_t src0, uint16_t src1,
    memref_t<__ubuf__ uint16_t, 1> *dst,
    memref_t<__ubuf__ uint16_t, 1> *condition_addr_buf) {
  auto *src0_half = reinterpret_cast<half *>(&src0);
  auto *src1_half = reinterpret_cast<half *>(&src1);

  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;

  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  vector_select_ss_1d(condition, *src0_half, *src1_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_ss_1d<bfloat16_t>(
    memref_t<__ubuf__ bool, 1> *condition, bfloat16_t src0, bfloat16_t src1,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *condition_addr_buf) {
  auto *src0_half = reinterpret_cast<half *>(&src0);
  auto *src1_half = reinterpret_cast<half *>(&src1);

  // convert bfloat16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;

  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  vector_select_ss_1d(condition, *src0_half, *src1_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_ss_1d<int32_t>(
    memref_t<__ubuf__ bool, 1> *condition, int32_t src0, int32_t src1,
    memref_t<__ubuf__ int32_t, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *condition_addr_buf) {
  auto *src0_float = reinterpret_cast<float *>(&src0);
  auto *src1_float = reinterpret_cast<float *>(&src1);

  // convert int32_t memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;

  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);

  vector_select_ss_1d(condition, *src0_float, *src1_float, &dst_as_float,
                      &cond_addr_buf_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_ss_1d<uint32_t>(
    memref_t<__ubuf__ bool, 1> *condition, uint32_t src0, uint32_t src1,
    memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *condition_addr_buf) {
  auto *src0_float = reinterpret_cast<float *>(&src0);
  auto *src1_float = reinterpret_cast<float *>(&src1);

  // convert uint32_t memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;

  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);

  vector_select_ss_1d(condition, *src0_float, *src1_float, &dst_as_float,
                      &cond_addr_buf_as_float);
}

//===--------------------------------------------------------------------===//
// vsel, 1 dim => for Vector-Scalar Inputs
//
// Constraints:
// 1. src0 and src1 should be float or half -> hence all other types
//    are casted into float or half.
// 2. src0 is vector and src1 is scalar input.
// 3. The condition_addr_buf which acts as a temporary buffer
//    should have at least (2 * ub_block_unit) to accomodate the
//    minimum of 8 values for each of the 2 i.e., src1 and
//    condition variables.
//===--------------------------------------------------------------------===//
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_vs_1d(memref_t<__ubuf__ bool, 1> *condition,
                    memref_t<__ubuf__ T, 1> *src0, T src1,
                    memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  __ubuf__ T *src1_brc_ptr = (condition_addr_buf->aligned +
                              condition_addr_buf->offset + num_per_block);
  brc_scalar_core_1d(src1, src1_brc_ptr, (int64_t)(num_per_block));
  vector_select_1d_intrin(
      intrin_args<2, T>{dst->aligned + dst->offset,
                        {src0->aligned + src0->offset, src1_brc_ptr},
                        0,
                        0,
                        1,       // dst_block_stride
                        {1, 0},  // src_block_stride
                        8,       // dst_repeat_stride
                        {8, 0}}, // src_repeat_stride
      condition, condition_addr_buf, dst->sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vs_1d<int16_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ int16_t, 1> *src0,
    int16_t src1, memref_t<__ubuf__ int16_t, 1> *dst,
    memref_t<__ubuf__ int16_t, 1> *condition_addr_buf) {
  auto *src1_half = reinterpret_cast<half *>(&src1);

  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;
  memref_t<__ubuf__ half, 1> src0_as_half;

  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);
  view_as<int16_t, half, 1>(src0, &src0_as_half);

  vector_select_vs_1d(condition, &src0_as_half, *src1_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vs_1d<uint16_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ uint16_t, 1> *src0,
    uint16_t src1, memref_t<__ubuf__ uint16_t, 1> *dst,
    memref_t<__ubuf__ uint16_t, 1> *condition_addr_buf) {
  auto *src1_half = reinterpret_cast<half *>(&src1);

  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;
  memref_t<__ubuf__ half, 1> src0_as_half;

  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);
  view_as<uint16_t, half, 1>(src0, &src0_as_half);

  vector_select_vs_1d(condition, &src0_as_half, *src1_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vs_1d<bfloat16_t>(
    memref_t<__ubuf__ bool, 1> *condition,
    memref_t<__ubuf__ bfloat16_t, 1> *src0, bfloat16_t src1,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *condition_addr_buf) {
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> src0_as_half;
  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(src0, &src0_as_half);

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(bfloat16_t);
  __ubuf__ bfloat16_t *src1_brc_ptr =
      condition_addr_buf->aligned + condition_addr_buf->offset + num_per_block;
  brc_scalar_core_1d(src1, src1_brc_ptr, (int64_t)num_per_block);
  memref_t<__ubuf__ bfloat16_t, 1> src1_brc_buf{condition_addr_buf->allocated,
                                                condition_addr_buf->aligned,
                                                condition_addr_buf->offset +
                                                    num_per_block,
                                                {num_per_block},
                                                {1}};

  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;
  view_as<bfloat16_t, half, 1>(&src1_brc_buf, &src1_as_half);
  view_as<bfloat16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  intrin_args<2, half> args{/* dst */ dst_as_half.aligned + dst_as_half.offset,
                            /* src */
                            {src0_as_half.aligned + src0_as_half.offset,
                             src1_as_half.aligned + src1_as_half.offset},
                            /* scalar */ 0,
                            /* repeat */ 0,
                            /* dst_block_stride */ 1,
                            /* src_block_stride */ {1, 0},
                            /* dst_repeat_stride */ 8,
                            /* src_repeat_stride */ {8, 0}};

  vector_select_1d_intrin(args, condition, &cond_addr_buf_as_half,
                          dst_as_half.sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vs_1d<int32_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ int32_t, 1> *src0,
    int32_t src1, memref_t<__ubuf__ int32_t, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *condition_addr_buf) {
  auto *src1_float = reinterpret_cast<float *>(&src1);

  // convert int32_t memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;
  memref_t<__ubuf__ float, 1> src0_as_float;

  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);
  view_as<int32_t, float, 1>(src0, &src0_as_float);

  vector_select_vs_1d(condition, &src0_as_float, *src1_float, &dst_as_float,
                      &cond_addr_buf_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vs_1d<uint32_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ uint32_t, 1> *src0,
    uint32_t src1, memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *condition_addr_buf) {
  auto *src1_float = reinterpret_cast<float *>(&src1);

  // convert uint32_t memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;
  memref_t<__ubuf__ float, 1> src0_as_float;

  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);
  view_as<uint32_t, float, 1>(src0, &src0_as_float);

  vector_select_vs_1d(condition, &src0_as_float, *src1_float, &dst_as_float,
                      &cond_addr_buf_as_float);
}

//===--------------------------------------------------------------------===//
// vsel, 1 dim => for Scalar-Vector Inputs
//
// Constraints:
// 1. src0 and src1 should be float or half -> hence all other
//    types are casted into float or half.
// 2. src0 is scalar and src1 is vector input.
// 3. The condition_addr_buf which acts as a temporary buffer
//    should have at least (2 * ub_block_unit) to accomodate
//    the minimum of 8 values for each of the 2 i.e., src0 and
//    condition variables.
//===--------------------------------------------------------------------===//
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_sv_1d(memref_t<__ubuf__ bool, 1> *condition, T src0,
                    memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  __ubuf__ T *src0_brc_ptr = (condition_addr_buf->aligned +
                              condition_addr_buf->offset + num_per_block);
  brc_scalar_core_1d(src0, src0_brc_ptr, (int64_t)(num_per_block));
  vector_select_1d_intrin(
      intrin_args<2, T>{dst->aligned + dst->offset,
                        {src0_brc_ptr, src1->aligned + src1->offset},
                        0,
                        0,
                        1,       // dst_block_stride
                        {0, 1},  // src_block_stride
                        8,       // dst_repeat_stride
                        {0, 8}}, // src_repeat_stride
      condition, condition_addr_buf, dst->sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_sv_1d<int16_t>(
    memref_t<__ubuf__ bool, 1> *condition, int16_t src0,
    memref_t<__ubuf__ int16_t, 1> *src1, memref_t<__ubuf__ int16_t, 1> *dst,
    memref_t<__ubuf__ int16_t, 1> *condition_addr_buf) {
  auto *src0_half = reinterpret_cast<half *>(&src0);

  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;

  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);
  view_as<int16_t, half, 1>(src1, &src1_as_half);

  vector_select_sv_1d(condition, *src0_half, &src1_as_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_sv_1d<uint16_t>(
    memref_t<__ubuf__ bool, 1> *condition, uint16_t src0,
    memref_t<__ubuf__ uint16_t, 1> *src1, memref_t<__ubuf__ uint16_t, 1> *dst,
    memref_t<__ubuf__ uint16_t, 1> *condition_addr_buf) {
  auto *src0_half = reinterpret_cast<half *>(&src0);

  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;

  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);
  view_as<uint16_t, half, 1>(src1, &src1_as_half);

  vector_select_sv_1d(condition, *src0_half, &src1_as_half, &dst_as_half,
                      &cond_addr_buf_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_sv_1d<bfloat16_t>(
    memref_t<__ubuf__ bool, 1> *condition, bfloat16_t src0,
    memref_t<__ubuf__ bfloat16_t, 1> *src1,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *condition_addr_buf) {
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(src1, &src1_as_half);

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(bfloat16_t);
  __ubuf__ bfloat16_t *src0_brc_ptr =
      condition_addr_buf->aligned + condition_addr_buf->offset + num_per_block;
  brc_scalar_core_1d(src0, src0_brc_ptr, (int64_t)num_per_block);
  memref_t<__ubuf__ bfloat16_t, 1> src0_brc_buf{condition_addr_buf->allocated,
                                                condition_addr_buf->aligned,
                                                condition_addr_buf->offset +
                                                    num_per_block,
                                                {num_per_block},
                                                {1}};

  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> cond_addr_buf_as_half;
  view_as<bfloat16_t, half, 1>(&src0_brc_buf, &src0_as_half);
  view_as<bfloat16_t, half, 1>(condition_addr_buf, &cond_addr_buf_as_half);

  intrin_args<2, half> args{dst_as_half.aligned + dst_as_half.offset,
                            {src0_as_half.aligned + src0_as_half.offset,
                             src1_as_half.aligned + src1_as_half.offset},
                            /* scalar */ 0,
                            /* repeat */ 0,
                            /* dst_block_stride */ 1,
                            /* src_block_stride */ {0, 1},
                            /* dst_repeat_stride */ 8,
                            /* src_repeat_stride */ {0, 8}};

  vector_select_1d_intrin(args, condition, &cond_addr_buf_as_half,
                          dst_as_half.sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_sv_1d<int32_t>(
    memref_t<__ubuf__ bool, 1> *condition, int32_t src0,
    memref_t<__ubuf__ int32_t, 1> *src1, memref_t<__ubuf__ int32_t, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *condition_addr_buf) {
  auto *src0_float = reinterpret_cast<float *>(&src0);

  // convert int32_t memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;

  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);
  view_as<int32_t, float, 1>(src1, &src1_as_float);

  vector_select_sv_1d(condition, *src0_float, &src1_as_float, &dst_as_float,
                      &cond_addr_buf_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_sv_1d<uint32_t>(
    memref_t<__ubuf__ bool, 1> *condition, uint32_t src0,
    memref_t<__ubuf__ uint32_t, 1> *src1, memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *condition_addr_buf) {
  auto *src0_float = reinterpret_cast<float *>(&src0);

  // convert uint32_t memref to float memref
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;

  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(condition_addr_buf, &cond_addr_buf_as_float);
  view_as<uint32_t, float, 1>(src1, &src1_as_float);

  vector_select_sv_1d(condition, *src0_float, &src1_as_float, &dst_as_float,
                      &cond_addr_buf_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d<int64_t, bool>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ int64_t, 1> *src0,
    memref_t<__ubuf__ int64_t, 1> *src1, memref_t<__ubuf__ int64_t, 1> *dst,
    memref_t<__ubuf__ int64_t, 1> *condition_addr_buf) {
  __ubuf__ int64_t *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ int64_t *src1_ptr = src1->aligned + src1->offset;
  __ubuf__ int64_t *dst_ptr = dst->aligned + dst->offset;

  memref_t<__ubuf__ uint16_t, 1> cond_b16;
  view_as<bool, uint16_t, 1>(condition, &cond_b16);
  __ubuf__ uint16_t *condition_ptr = cond_b16.aligned + cond_b16.offset;

  // TODO: change to use simd instr to impl
  // use select scalar instr
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t size0 = src0->sizes[0];
  for (int i = 0; i < size0; ++i) {
    int64_t shift_bits = i % bitwidthOf<uint16_t>();
    int64_t b16_idx = i / bitwidthOf<uint16_t>();
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    uint16_t cond = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits;
    dst_ptr[i] = cond == 1 ? src0_ptr[i] : src1_ptr[i];
#else
    /// get conditon of select by this logic because of Little-Endian enabled
    /// here. e.g. a = 0b0010010000000100, in order to get the third digit, we
    /// need to use following formula,
    /// b = 0b0000000000000001 << 2 =  0b0000000000000100,
    /// cond = (a & b) >> 2 = 1
    dst_ptr[i] = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits == 1
                     ? src0_ptr[i]
                     : src1_ptr[i];
#endif
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void vector_select_vv_1d(
    memref_t<__ubuf__ int8_t, 1> *condition, memref_t<__ubuf__ T, 1> *src0,
    memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *condition_addr_buf) {
  memref_t<__ubuf__ bool, 1> cond_as_bool;
  bitwise_view_as<int8_t, bool, 1>(condition, &cond_as_bool);
  vector_select_vv_1d<T>(&cond_as_bool, src0, src1, dst, condition_addr_buf);
}

template <typename T, typename COND_T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_select_1d_with_temp(memref_t<__ubuf__ COND_T, 1> *cond,
                                           memref_t<__ubuf__ T, 1> *src0,
                                           memref_t<__ubuf__ T, 1> *src1,
                                           memref_t<__ubuf__ T, 1> *dst,
                                           memref_t<__ubuf__ T, 1> *temp) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  auto src1_ptr = src1->aligned + src1->offset;
  auto temp_ptr = temp->aligned + temp->offset;
  auto cond_ptr = cond->aligned + cond->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src0 must be 32byte aligned.");
  assert(isAddress32ByteAligned(src1_ptr) &&
         "The starting address of src1 must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(temp_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(isAddress32ByteAligned(cond_ptr) &&
         "The starting address of cond must be 32byte aligned.");
  assert((src0->strides[0] == 1 && src1->strides[0] == 1 &&
          dst->strides[0] == 1 && temp->strides[0] == 1 && cond->strides[0]) &&
         "The src/dst must be continuous vector.");
#endif
}

/// Vselect for Vector-Vector Inputs condition, src0 and src1 (m,) with stride 1
/// Stores output in dst (m,), uses condition_add_buf (k,) as temp buffer
/// \param condition (type: memref<m x int8_t>)
/// \param src0 (type: memref<m x int64_t>)
/// \param src1 (type: memref<m x int64_t>)
/// \param dst (type: memref<m x int64_t>)
/// \param condition_addr_buf (type: memref<k x int64_t>)
/// Constraints:
/// 1. src0 and src1 should be i64, condition should be i8 type
/// 2. src0 and src1 are vector inputs.
/// 4. The condition_addr_buf size minimum :
/// k = aligned(0.5*m, ub_block_unit) + aligned(0.25*m, ub_block_unit)
/// Example:
/// for condition : 010110100 => new_condition : 001100111100110000
/// vsel_int64(condition,src0,src1,dst) =>
/// vsel_float(new_condition,src0_as_float,src1_as_float,dst_as_float)
template <>
__aiv__ __attribute__((always_inline)) void
vector_select_vv_1d<int64_t, int8_t>(
    memref_t<__ubuf__ int8_t, 1> *condition,
    memref_t<__ubuf__ int64_t, 1> *src0, memref_t<__ubuf__ int64_t, 1> *src1,
    memref_t<__ubuf__ int64_t, 1> *dst,
    memref_t<__ubuf__ int64_t, 1> *condition_addr_buf) {

  // Input parameter constraints assert.
  check_inputs_of_vector_select_1d_with_temp<int64_t, int8_t>(
      condition, src0, src1, dst, condition_addr_buf);

  const int64_t size = condition->sizes[0];
  constexpr int bits_int64 = bitwidthOf<int64_t>();
  constexpr int bits_float = bitwidthOf<float>();
  constexpr int bits_half = bitwidthOf<half>();
  constexpr int bits_int32 = bitwidthOf<int32_t>();
  constexpr int bits_bool = bitwidthOf<bool>();
  constexpr int num_i32_per_block = // 8 * 32 / 32 = 8
      BITS_PER_BYTE * INTR_BYTES_PER_BLOCK / bits_int32;
  constexpr int num_half_per_block = // 8 * 32 / 16 = 16
      BITS_PER_BYTE * INTR_BYTES_PER_BLOCK / bits_half;
  constexpr int num_bool_per_block = // 8 * 32 / 1 = 256
      BITS_PER_BYTE * INTR_BYTES_PER_BLOCK / bits_bool;
  const int64_t tmp_i32_size = CEIL_FACTOR(size, num_i32_per_block);
  const int64_t tmp_half_size = CEIL_FACTOR(size, num_half_per_block);
  const int64_t tmp_bool_size = CEIL_FACTOR(size * 2, num_bool_per_block);

  memref_t<__ubuf__ int32_t, 1> cast_dst{
      (__ubuf__ int32_t *)condition_addr_buf->allocated,
      (__ubuf__ int32_t *)condition_addr_buf->aligned,
      condition_addr_buf->offset * bits_int64 / bits_int32,
      {tmp_i32_size},
      {1}};
  memref_t<__ubuf__ half, 1> cast_temp{
      (__ubuf__ half *)condition_addr_buf->allocated,
      (__ubuf__ half *)condition_addr_buf->aligned,
      (cast_dst.offset + cast_dst.sizes[0]) * bits_int32 / bits_half,
      {tmp_half_size},
      {1}};
  // The idea is to convert condition from 0b10101010 to 0b1100110011001100
  // step1: cast condition int8 to int32
  vector_cast_1d_with_mode<int8_t, half>(condition, &cast_temp, CastMode::R);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_cast_1d_with_mode<half, int32_t>(&cast_temp, &cast_dst, CastMode::R);

  // step2: multiply all elements by 65537
  //         65537 =        0b00000000000000010000000000000001
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vs_1d<VectorOpTy::VMULS, int32_t>(&cast_dst, 65537, &cast_dst);
  memref_t<__ubuf__ half, 1> vmul_as_half;
  // step3: int32 tensor view as half to double each condition element
  //  Split into halves:  0b0000000000000001 | 0b0000000000000001
  //                      (upper 16 bits)    | (lower 16 bits)
  view_as<int32_t, half, 1>(&cast_dst, &vmul_as_half);
  memref_t<__ubuf__ bool, 1> new_cond{
      (__ubuf__ bool *)condition_addr_buf->allocated,
      (__ubuf__ bool *)condition_addr_buf->aligned,
      (cast_temp.offset) * bits_half / bits_bool,
      {tmp_bool_size},
      {1}};
  INTRINSIC(pipe_barrier, PIPE_V);
  // step4: get new condition as bool with vcmpvs_ne
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  vector_compare_vs_1d<VectorOpTy::VCMPS_NE, half>(&vmul_as_half, {0},
                                                   &new_cond);
#else
  vector_compare_vs_1d<VectorOpTy::VCMPS_NE, half>(&vmul_as_half, 0, &new_cond);
#endif

  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> cond_addr_buf_as_float{
      (__ubuf__ float *)condition_addr_buf->allocated,
      (__ubuf__ float *)condition_addr_buf->aligned,
      condition_addr_buf->offset * bits_int64 / bits_float,
      {num_i32_per_block},
      {1}};

  memref_t<__ubuf__ float, 1> src0_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;

  view_as<int64_t, float, 1>(dst, &dst_as_float);
  view_as<int64_t, float, 1>(src0, &src0_as_float);
  view_as<int64_t, float, 1>(src1, &src1_as_float);
  // step5: call float vsel on source, double elements int64 view as float32
  vector_select_vv_1d<float>(&new_cond, &src0_as_float, &src1_as_float,
                             &dst_as_float, &cond_addr_buf_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_sv_1d<int64_t>(
    memref_t<__ubuf__ bool, 1> *condition, int64_t src0,
    memref_t<__ubuf__ int64_t, 1> *src1, memref_t<__ubuf__ int64_t, 1> *dst,
    memref_t<__ubuf__ int64_t, 1> *condition_addr_buf) {
  __ubuf__ int64_t *src1_ptr = src1->aligned + src1->offset;
  __ubuf__ int64_t *dst_ptr = dst->aligned + dst->offset;

  memref_t<__ubuf__ uint16_t, 1> cond_b16;
  view_as<bool, uint16_t, 1>(condition, &cond_b16);
  __ubuf__ uint16_t *condition_ptr = cond_b16.aligned + cond_b16.offset;

  // TODO: change to use simd instr to impl
  // use select scalar instr
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t size0 = src1->sizes[0];
  for (int i = 0; i < size0; ++i) {
    int64_t shift_bits = i % bitwidthOf<uint16_t>();
    int64_t b16_idx = i / bitwidthOf<uint16_t>();
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    uint16_t cond = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits;
    dst_ptr[i] = cond == 1 ? src0 : src1_ptr[i];
#else
    /// get conditon of select by this logic because of Little-Endian enabled
    /// here. e.g. a = 0b0010010000000100, in order to get the third digit, we
    /// need to use following formula,
    /// b = 0b0000000000000001 << 2 =  0b0000000000000100,
    /// cond = (a & b) >> 2 = 1
    dst_ptr[i] = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits == 1
                     ? src0
                     : src1_ptr[i];
#endif
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_vs_1d<int64_t>(
    memref_t<__ubuf__ bool, 1> *condition, memref_t<__ubuf__ int64_t, 1> *src0,
    int64_t src1, memref_t<__ubuf__ int64_t, 1> *dst,
    memref_t<__ubuf__ int64_t, 1> *condition_addr_buf) {
  __ubuf__ int64_t *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ int64_t *dst_ptr = dst->aligned + dst->offset;

  memref_t<__ubuf__ uint16_t, 1> cond_b16;
  view_as<bool, uint16_t, 1>(condition, &cond_b16);
  __ubuf__ uint16_t *condition_ptr = cond_b16.aligned + cond_b16.offset;

  // TODO: change to use simd instr to impl
  // use select scalar instr
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t size0 = src0->sizes[0];
  for (int i = 0; i < size0; ++i) {
    int64_t shift_bits = i % bitwidthOf<uint16_t>();
    int64_t b16_idx = i / bitwidthOf<uint16_t>();
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    uint16_t cond = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits;
    dst_ptr[i] = cond == 1 ? src0_ptr[i] : src1;
#else
    /// get conditon of select by this logic because of Little-Endian enabled
    /// here. e.g. a = 0b0010010000000100, in order to get the third digit, we
    /// need to use following formula,
    /// b = 0b0000000000000001 << 2 =  0b0000000000000100,
    /// cond = (a & b) >> 2 = 1
    dst_ptr[i] = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits == 1
                     ? src0_ptr[i]
                     : src1;
#endif
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_select_ss_1d<int64_t>(
    memref_t<__ubuf__ bool, 1> *condition, int64_t src0, int64_t src1,
    memref_t<__ubuf__ int64_t, 1> *dst,
    memref_t<__ubuf__ int64_t, 1> *condition_addr_buf) {
  __ubuf__ int64_t *dst_ptr = dst->aligned + dst->offset;

  memref_t<__ubuf__ uint16_t, 1> cond_b16;
  view_as<bool, uint16_t, 1>(condition, &cond_b16);
  __ubuf__ uint16_t *condition_ptr = cond_b16.aligned + cond_b16.offset;

  // TODO: change to use simd instr to impl
  // use select scalar instr
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t size0 = dst->sizes[0];
  for (int i = 0; i < size0; ++i) {
    int64_t shift_bits = i % bitwidthOf<uint16_t>();
    int64_t b16_idx = i / bitwidthOf<uint16_t>();
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    uint16_t cond = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits;
    dst_ptr[i] = cond == 1 ? src0 : src1;
#else
    /// get conditon of select by this logic because of Little-Endian enabled
    /// here. e.g. a = 0b0010010000000100, in order to get the third digit, we
    /// need to use following formula,
    /// b = 0b0000000000000001 << 2 =  0b0000000000000100,
    /// cond = (a & b) >> 2 = 1
    dst_ptr[i] = (condition_ptr[b16_idx] & (1 << shift_bits)) >> shift_bits == 1
                     ? src0
                     : src1;
#endif
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

extern "C" {
//===-------------------------------------------------------------------===//
// vsel, 1 dim => for Vector-Vector Inputs
//===-------------------------------------------------------------------===//
REGISTE_VSEL_VV(1, bool, half)
REGISTE_VSEL_VV(1, bool, float)
REGISTE_VSEL_VV(1, bool, int16_t)
REGISTE_VSEL_VV(1, bool, int32_t)
REGISTE_VSEL_VV(1, bool, uint16_t)
REGISTE_VSEL_VV(1, bool, uint32_t)
REGISTE_VSEL_VV(1, bool, bfloat16_t)
REGISTE_VSEL_VV(1, bool, int64_t);
REGISTE_VSEL_VV(1, int8_t, half)
REGISTE_VSEL_VV(1, int8_t, float)
REGISTE_VSEL_VV(1, int8_t, int16_t)
REGISTE_VSEL_VV(1, int8_t, int32_t)
REGISTE_VSEL_VV(1, int8_t, uint16_t)
REGISTE_VSEL_VV(1, int8_t, uint32_t)
REGISTE_VSEL_VV(1, int8_t, bfloat16_t)
REGISTE_VSEL_VV(1, int8_t, int64_t);

//===-------------------------------------------------------------------===//
// vsel, 1 dim => for Scalar-Scalar Inputs
//===-------------------------------------------------------------------===//
REGISTE_VSEL_SS(1, half)
REGISTE_VSEL_SS(1, float)
REGISTE_VSEL_SS(1, int16_t)
REGISTE_VSEL_SS(1, int32_t)
REGISTE_VSEL_SS(1, uint16_t)
REGISTE_VSEL_SS(1, uint32_t)
REGISTE_VSEL_SS(1, bfloat16_t)
REGISTE_VSEL_SS(1, int64_t)

//===-------------------------------------------------------------------===//
// vsel, 1 dim => for Vector-Scalar Inputs
//===-------------------------------------------------------------------===//
REGISTE_VSEL_VS(1, half)
REGISTE_VSEL_VS(1, float)
REGISTE_VSEL_VS(1, int16_t)
REGISTE_VSEL_VS(1, int32_t)
REGISTE_VSEL_VS(1, uint16_t)
REGISTE_VSEL_VS(1, uint32_t)
REGISTE_VSEL_VS(1, bfloat16_t)
REGISTE_VSEL_VS(1, int64_t)

//===-------------------------------------------------------------------===//
// vsel, 1 dim => for Scalar-Vector Inputs
//===-------------------------------------------------------------------===//
REGISTE_VSEL_SV(1, half)
REGISTE_VSEL_SV(1, float)
REGISTE_VSEL_SV(1, int16_t)
REGISTE_VSEL_SV(1, int32_t)
REGISTE_VSEL_SV(1, uint16_t)
REGISTE_VSEL_SV(1, uint32_t)
REGISTE_VSEL_SV(1, bfloat16_t)
REGISTE_VSEL_SV(1, int64_t)
}
