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
#include "Vector/Interleave/InterleaveUtils.h"
#include "Vector/VecUtils.h"

/// interleave op description:
/// interleave src0 (a,) and src1 (a,) into dst (2*a),
/// src0 in odd position, src1 in even position,
/// 'a' is size of src0 and src1, and size of dst is always 2*'a', tmp buffer
/// with size 'c'
///
/// \param src0 (type: memref<a x T>)
/// \param src1 (type: memref<a x T>)
/// \param dst (type: memref<2*a x T>)
/// \param tmp (type: memref<c x T>)
/// (A_align*num_per_block+A*2, A is size of src, and A_align is A align to
/// multiples of 8)
///
/// Constraints:
/// 1. 'c' is aligned(a, src_num_per_vbrcb_repeat)*num_per_block+a*2,
/// here src_num_per_vbrcb_repeat means the number of src that vbrcb intr handle
/// per repeat.
/// 2. because vsel instruction only support half and float,
/// int16/uint16 view as half, and int32/uint32 view as float in interleave op

template <typename T>
__aiv__ __attribute__((always_inline)) void vector_interleave_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *temp) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vv_1d(src0, src1, dst);

  const int64_t size0 = src0->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / bytes;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;

  // step 1: brc src0 (a, 1) to brc_src0 (a, 2)
  // view src0 (a) as brc_src0 (a,1)
  memref_t<__ubuf__ T, 2> brc_src0{
      src0->allocated, src0->aligned, src0->offset, {size0, 1}, {1, 1}};
  // additional tmp buffer for brc operation
  const int64_t src_num_per_vbrcb_repeat =
      CEIL_FACTOR(size0, kSrcNumPerRepeatOfVBRCB);
  const int64_t brc_tmp_buf_size = src_num_per_vbrcb_repeat * num_per_block;
  memref_t<__ubuf__ T, 1> brc_tmp_buf{
      temp->allocated, temp->aligned, temp->offset, {brc_tmp_buf_size}, {1}};
  // brc_dst0 is stored on tmp buffer
  memref_t<__ubuf__ T, 2> brc_dst0{temp->allocated,
                                   temp->aligned,
                                   temp->offset + brc_tmp_buf_size,
                                   {size0, 2},
                                   {2, 1}};
  vector_broadcast_last_axis_unalign_2d<T>(&brc_src0, &brc_dst0, &brc_tmp_buf);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step 2: brc src1 (a, 1) to brc_src1 (a, 2)
  // view src1 (a) as brc_src1 (a,1)
  memref_t<__ubuf__ T, 2> brc_src1{
      src1->allocated, src1->aligned, src1->offset, {size0, 1}, {1, 1}};
  // brc_dst1 is reused by interleave dst
  memref_t<__ubuf__ T, 2> brc_dst1{dst->allocated,
                                   dst->aligned,
                                   dst->offset,
                                   {dst->sizes[0] / 2, 2},
                                   {2, 1}};
  vector_broadcast_last_axis_unalign_2d<T>(&brc_src1, &brc_dst1, &brc_tmp_buf);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step 3: interleave select between brc_dst0 and brc_dst1
  memref_t<__ubuf__ T, 1> brc_dst0_as_1d{temp->allocated,
                                         temp->aligned,
                                         temp->offset + brc_tmp_buf_size,
                                         {size0 * 2},
                                         {1}};
  memref_t<__ubuf__ T, 1> brc_dst1_as_1d{
      dst->allocated, dst->aligned, dst->offset, {dst->sizes[0]}, {1}};
  interleave_vsel_1d<T>(&brc_dst0_as_1d, &brc_dst1_as_1d, dst, &brc_tmp_buf);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_interleave_1d<bfloat16_t>(memref_t<__ubuf__ bfloat16_t, 1> *src0,
                                 memref_t<__ubuf__ bfloat16_t, 1> *src1,
                                 memref_t<__ubuf__ bfloat16_t, 1> *dst,
                                 memref_t<__ubuf__ bfloat16_t, 1> *temp) {
  // convert bfloat16_t memref to half memref
  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;
  view_as<bfloat16_t, half, 1>(src0, &src0_as_half);
  view_as<bfloat16_t, half, 1>(src1, &src1_as_half);
  view_as<bfloat16_t, half, 1>(dst, &dst_as_half);
  view_as<bfloat16_t, half, 1>(temp, &temp_as_half);
  vector_interleave_1d<half>(&src0_as_half, &src1_as_half, &dst_as_half,
                             &temp_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_interleave_1d<int16_t>(
    memref_t<__ubuf__ int16_t, 1> *src0, memref_t<__ubuf__ int16_t, 1> *src1,
    memref_t<__ubuf__ int16_t, 1> *dst, memref_t<__ubuf__ int16_t, 1> *temp) {
  // convert int16 memref to half memref
  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;
  view_as<int16_t, half, 1>(src0, &src0_as_half);
  view_as<int16_t, half, 1>(src1, &src1_as_half);
  view_as<int16_t, half, 1>(dst, &dst_as_half);
  view_as<int16_t, half, 1>(temp, &temp_as_half);
  vector_interleave_1d<half>(&src0_as_half, &src1_as_half, &dst_as_half,
                             &temp_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_interleave_1d<uint16_t>(
    memref_t<__ubuf__ uint16_t, 1> *src0, memref_t<__ubuf__ uint16_t, 1> *src1,
    memref_t<__ubuf__ uint16_t, 1> *dst, memref_t<__ubuf__ uint16_t, 1> *temp) {
  // convert uint16 memref to half memref
  memref_t<__ubuf__ half, 1> src0_as_half;
  memref_t<__ubuf__ half, 1> src1_as_half;
  memref_t<__ubuf__ half, 1> dst_as_half;
  memref_t<__ubuf__ half, 1> temp_as_half;
  view_as<uint16_t, half, 1>(src0, &src0_as_half);
  view_as<uint16_t, half, 1>(src1, &src1_as_half);
  view_as<uint16_t, half, 1>(dst, &dst_as_half);
  view_as<uint16_t, half, 1>(temp, &temp_as_half);
  vector_interleave_1d<half>(&src0_as_half, &src1_as_half, &dst_as_half,
                             &temp_as_half);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_interleave_1d<int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ int32_t, 1> *dst, memref_t<__ubuf__ int32_t, 1> *temp) {
  // convert int32 memref to float32 memref
  memref_t<__ubuf__ float, 1> src0_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> temp_as_float;
  view_as<int32_t, float, 1>(src0, &src0_as_float);
  view_as<int32_t, float, 1>(src1, &src1_as_float);
  view_as<int32_t, float, 1>(dst, &dst_as_float);
  view_as<int32_t, float, 1>(temp, &temp_as_float);
  vector_interleave_1d<float>(&src0_as_float, &src1_as_float, &dst_as_float,
                              &temp_as_float);
}

template <>
__aiv__ __attribute__((always_inline)) void vector_interleave_1d<uint32_t>(
    memref_t<__ubuf__ uint32_t, 1> *src0, memref_t<__ubuf__ uint32_t, 1> *src1,
    memref_t<__ubuf__ uint32_t, 1> *dst, memref_t<__ubuf__ uint32_t, 1> *temp) {
  // convert uint32_t memref to float32 memref
  memref_t<__ubuf__ float, 1> src0_as_float;
  memref_t<__ubuf__ float, 1> src1_as_float;
  memref_t<__ubuf__ float, 1> dst_as_float;
  memref_t<__ubuf__ float, 1> temp_as_float;
  view_as<uint32_t, float, 1>(src0, &src0_as_float);
  view_as<uint32_t, float, 1>(src1, &src1_as_float);
  view_as<uint32_t, float, 1>(dst, &dst_as_float);
  view_as<uint32_t, float, 1>(temp, &temp_as_float);
  vector_interleave_1d<float>(&src0_as_float, &src1_as_float, &dst_as_float,
                              &temp_as_float);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
interleave_vsel_1d(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
                   memref_t<__ubuf__ T, 1> *dst,
                   memref_t<__ubuf__ T, 1> *tmp_buf) {
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;

  const int64_t total_size = dst->sizes[0];
  constexpr int bytes = sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  // step 1: generate interleave select condition 0b0101010101010101
  constexpr uint16_t scalar = 0b0101010101010101;
  // brc scalar to vector which size is CEIL_DIV(total_size, HALF_BITS)
  const int64_t total_size_align = CEIL_DIV(total_size, HALF_BITS);
  __ubuf__ T *tmp_ptr = tmp_buf->aligned + tmp_buf->offset + num_per_block;
  brc_scalar_core_1d<uint16_t>(scalar, (__ubuf__ uint16_t *)tmp_ptr,
                               total_size_align);
  INTRINSIC(pipe_barrier, PIPE_V);

  const int64_t factor = bytes / sizeof(uint16_t);
  memref_t<__ubuf__ T, 1> condition{tmp_buf->allocated,
                                    tmp_buf->aligned,
                                    tmp_buf->offset + num_per_block,
                                    {total_size_align / factor},
                                    {1}};
  memref_t<__ubuf__ T, 1> condition_addr_buf{tmp_buf->allocated,
                                             tmp_buf->aligned,
                                             tmp_buf->offset,
                                             {num_per_block},
                                             {1}};

  // step 2: vector_select(condition, src0, src1)
  vector_select_1d<T>(&condition, src0, src1, dst, &condition_addr_buf);
}

extern "C" {
//===-------------------------------------------------------------------===//
// interleave op, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_INTERLEAVE(1, half)
REGISTE_INTERLEAVE(1, bfloat16_t)
REGISTE_INTERLEAVE(1, float)
REGISTE_INTERLEAVE(1, int16_t)
REGISTE_INTERLEAVE(1, int32_t)
REGISTE_INTERLEAVE(1, uint16_t)
REGISTE_INTERLEAVE(1, uint32_t)
}