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

#include "CPUDebug/CPUDebugUtils.h"
#include "Utils.h"
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/Reduction/ReductionUtils.h"
#include "Vector/Reduction/ReduceRAImpl.h"
#include "Vector/VecUtils.h"
#include "DMA/DMAUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_ara(memref_t<__ubuf__ T, 3> *src0,
                               memref_t<__ubuf__ T, 3> *dst, T initvalue) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src0_ptr = src0->aligned + src0->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src0->strides[0]) &&
          isSizeAlignedToBlock<T>(src0->strides[1]) &&
          isSizeAlignedToBlock<T>(dst->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[1])) &&
         "The src/dst strides[0]/strides[1] must be aligned to block.");
  assert((src0->strides[2] == 1 && dst->strides[2] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

/// Scalar fallback for reduce_ara: reduce src [s0, s1, s2] along dim1
/// using scalar operations.
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_reduce_ara(memref_t<__ubuf__ T, 3> *src0,
                      memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("reduceARA");
#endif
  const int64_t s0 = src0->sizes[0];
  const int64_t s1 = src0->sizes[1];
  const int64_t s2 = src0->sizes[2];
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t dst_stride0 = dst->strides[0];

  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  for (int64_t i = 0; i < s0; ++i) {
    for (int64_t k = 0; k < s2; ++k) {
      T val = *(src0_ptr + i * src0_stride0 + k);
      for (int64_t j = 1; j < s1; ++j) {
        val = reduction_scalar_operation<OP, T>(
            val, *(src0_ptr + i * src0_stride0 + j * src0_stride1 + k));
      }
      *(dst_ptr + i * dst_stride0 + k) = val;
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

/// Core dichotomy + repeat implementation for reduce_ara.
/// Reduces src [s0, s1, s2] along dim1 to dst [s0, 1, s2].
///
/// Uses dichotomy (log2(s1) layers) with hardware repeat to process all s0
/// slices simultaneously. Each fold pairs rows from the first and second half,
/// and each row pair uses a [s0, s2] 2D view with repeat=s0.
///
/// Tail rows (beyond the largest power of 2 <= s1) are merged into fold-1
/// intermediate results rather than accumulated at the end, distributing
/// tail contributions across the dichotomy tree for better precision.
///
/// tmp buffer layout: [main_size/2 rows, each row = s0*s2 contiguous elements]
/// Row j offset = j * s0 * s2, viewed as [s0, s2] with stride [s2, 1].
/// Total tmp size = (main_size/2) * s0 * s2 elements.
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ara_core(memref_t<__ubuf__ T, 3> *src0,
                              memref_t<__ubuf__ T, 3> *dst,
                              memref_t<__ubuf__ T, 1> *tmp_buf,
                              T initvalue) {
  // initvalue is only consumed above for XOR tmp offset calculation.
  // The main dichotomy path combines row pairs directly (e.g., a+b, max(a,b)),
  // so no identity value is needed. Tail rows are accumulated into the
  // dichotomy result, which already contains the correct partial reduction.
  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();
  const int64_t s0 = src0->sizes[0];
  const int64_t s1 = src0->sizes[1];
  const int64_t s2 = src0->sizes[2];
  const int64_t stride0 = src0->strides[0];
  const int64_t stride1 = src0->strides[1];
  const int64_t dst_stride0 = dst->strides[0];

  const int64_t dichotomy_num = Log2(s1);
  const int64_t main_size = 1LL << dichotomy_num;
  const int64_t tail_size = s1 - main_size;

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t aligned_s2 = CEIL_FACTOR(s2, num_per_block);
  auto tmp_offset = tmp_buf->offset;

  if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
    tmp_offset =
        tmp_offset +
        CEIL_FACTOR((main_size / 2) * s0 * aligned_s2, num_per_block);
  }

  // Each tmp row holds s0*aligned_s2 contiguous elements.
  const int64_t tmp_row_stride = s0 * aligned_s2;

  // Handle degenerate case: s1 == 1 (no reduction needed, just copy).
  // VOR(a, a) == a for all integer types; for float types the bit pattern
  // is preserved since no NaN is involved (src == src). This is equivalent
  // to a vector copy.
  if (dichotomy_num == 0) {
    memref_t<__ubuf__ T, 2> src_2d{src0->allocated, src0->aligned,
                                   src0->offset, {s0, s2}, {stride0, 1}};
    memref_t<__ubuf__ T, 2> dst_2d{dst->allocated, dst->aligned,
                                   dst->offset, {s0, s2}, {dst_stride0, 1}};
    vector_eltwise_vv_2d<VectorOpTy::VOR, T>(&src_2d, &src_2d, &dst_2d, tmp_buf);
    return;
  }

  // ===== Fold 1: src -> tmp or dst =====
  {
    int64_t half = main_size / 2;
    bool is_only_fold = (dichotomy_num == 1);

    for (int64_t j = 0; j < half; ++j) {
      memref_t<__ubuf__ T, 2> src_row0{
          src0->allocated, src0->aligned,
          src0->offset + j * stride1,
          {s0, s2}, {stride0, 1}};
      memref_t<__ubuf__ T, 2> src_row1{
          src0->allocated, src0->aligned,
          src0->offset + (j + half) * stride1,
          {s0, s2}, {stride0, 1}};

      if (is_only_fold) {
        memref_t<__ubuf__ T, 2> dst_view{
            dst->allocated, dst->aligned, dst->offset,
            {s0, s2}, {dst_stride0, 1}};
        vector_eltwise_vv_2d<VECOP, T>(&src_row0, &src_row1, &dst_view, tmp_buf);
      } else {
        memref_t<__ubuf__ T, 2> tmp_dst{
            tmp_buf->allocated, tmp_buf->aligned,
            tmp_offset + j * tmp_row_stride,
            {s0, s2}, {aligned_s2, 1}};
        vector_eltwise_vv_2d<VECOP, T>(&src_row0, &src_row1, &tmp_dst, tmp_buf);
      }
    }
    INTRINSIC(pipe_barrier, PIPE_V);

    // Only one fold: tail goes directly into dst
    if (is_only_fold) {
      if (tail_size > 0) {
        memref_t<__ubuf__ T, 2> dst_view{
            dst->allocated, dst->aligned, dst->offset,
            {s0, s2}, {dst_stride0, 1}};
        for (int64_t t = 0; t < tail_size; ++t) {
          memref_t<__ubuf__ T, 2> src_tail{
              src0->allocated, src0->aligned,
              src0->offset + (main_size + t) * stride1,
              {s0, s2}, {stride0, 1}};
          INTRINSIC(pipe_barrier, PIPE_V);
          vector_eltwise_vv_2d<VECOP, T>(&src_tail, &dst_view, &dst_view, tmp_buf);
        }
      }
      return;
    }

    // Merge tail rows into fold-1 results for better precision.
    // Distribute tail rows across intermediate results instead of
    // accumulating them all into the final result at the end.
    if (tail_size > 0) {
      for (int64_t t = 0; t < tail_size; ++t) {
        int64_t target = t % half;
        memref_t<__ubuf__ T, 2> src_tail{
            src0->allocated, src0->aligned,
            src0->offset + (main_size + t) * stride1,
            {s0, s2}, {stride0, 1}};
        memref_t<__ubuf__ T, 2> tmp_target{
            tmp_buf->allocated, tmp_buf->aligned,
            tmp_offset + target * tmp_row_stride,
            {s0, s2}, {aligned_s2, 1}};
        INTRINSIC(pipe_barrier, PIPE_V);
        vector_eltwise_vv_2d<VECOP, T>(&src_tail, &tmp_target, &tmp_target, tmp_buf);
      }
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  }

  // ===== Fold 2..N: tmp -> tmp or dst =====
  {
    int64_t num_rows = main_size / 2;

    for (int64_t fold = 1; fold < dichotomy_num; ++fold) {
      int64_t half = num_rows / 2;
      bool is_last_fold = (fold == dichotomy_num - 1);

      for (int64_t j = 0; j < half; ++j) {
        memref_t<__ubuf__ T, 2> src_row0{
            tmp_buf->allocated, tmp_buf->aligned,
            tmp_offset + j * tmp_row_stride,
            {s0, s2}, {aligned_s2, 1}};
        memref_t<__ubuf__ T, 2> src_row1{
            tmp_buf->allocated, tmp_buf->aligned,
            tmp_offset + (j + half) * tmp_row_stride,
            {s0, s2}, {aligned_s2, 1}};

        if (is_last_fold) {
          memref_t<__ubuf__ T, 2> dst_view{
              dst->allocated, dst->aligned, dst->offset,
              {s0, s2}, {dst_stride0, 1}};
          vector_eltwise_vv_2d<VECOP, T>(&src_row0, &src_row1, &dst_view, tmp_buf);
        } else {
          // In-place write to same tmp row (safe: element-wise, reads before writes)
          memref_t<__ubuf__ T, 2> tmp_dst{
              tmp_buf->allocated, tmp_buf->aligned,
              tmp_offset + j * tmp_row_stride,
              {s0, s2}, {aligned_s2, 1}};
          vector_eltwise_vv_2d<VECOP, T>(&src_row0, &src_row1, &tmp_dst, tmp_buf);
        }
      }
      INTRINSIC(pipe_barrier, PIPE_V);
      num_rows = half;
    }
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ara_core<ReduceOpTy::REDUCE_OR, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  view_as<uint8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ara_core<ReduceOpTy::REDUCE_OR, int16_t>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ara_core<ReduceOpTy::REDUCE_AND, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  view_as<uint8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ara_core<ReduceOpTy::REDUCE_AND, int16_t>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

/// Reduce src (a0, r, a1) with stride [n0, n1, 1] to dst (a0, 1, a1) with
/// stride [n0, n1, 1] by reducing along dim1.
///
/// constraint:
/// 1. 'n1' is a1 aligned to ub_block_unit.
/// 2. 'n0' is a0 * a1 aligned to ub_block_unit.
/// 3. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 4. requires tmp_buf of size s0 * (main_size/2) * s2, where main_size is
/// the largest power of 2 <= s1. For reduce_xor, additional tmp is needed.
///
/// \param initvalue: same as reduce_ra0a1 initvalue table.
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ara(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
               memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  check_inputs_of_reduce_ara(src0, dst, initvalue);

  const int64_t s0 = src0->sizes[0];
  const int64_t s1 = src0->sizes[1];
  const int64_t s2 = src0->sizes[2];
  const int64_t stride0 = src0->strides[0];
  const int64_t stride1 = src0->strides[1];
  const int64_t stride2 = src0->strides[2];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t dst_stride2 = dst->strides[2];

  // When middle axis size > first axis size, peeling the first axis and
  // delegating to reduce_ra is more efficient than the current dichotomy
  // on the middle axis.

  const int64_t dichotomy_num = Log2(s1);
  const int64_t main_size = 1LL << dichotomy_num;
  const int64_t tail_size = s1 - main_size;
  const int64_t repeat_times = dichotomy_num + tail_size;
  if (repeat_times > s0) {
    for (int64_t i = 0; i < s0; ++i) {
      memref_t<__ubuf__ T, 2> src_slice{
          src0->allocated, src0->aligned,
          src0->offset + i * stride0,
          {s1, s2},
          {stride1, stride2}};
      memref_t<__ubuf__ T, 2> dst_slice{
          dst->allocated, dst->aligned,
          dst->offset + i * dst_stride0,
          {1, s2},
          {dst_stride1, dst_stride2}};
      reduce_ra<OP, T>(&src_slice, &dst_slice, tmp_buf, initvalue);
    }
    return;
  }

  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  bool is_stride_aligned = is32ByteAligned<T>(stride0) &&
                           is32ByteAligned<T>(stride1) &&
                           is32ByteAligned<T>(dst_stride0);
  bool is_last_axis_contiguous = (stride2 == 1) && (dst_stride2 == 1);
  bool is_offset_aligned = isAddress32ByteAligned(src0_ptr) &&
                           isAddress32ByteAligned(dst_ptr);

  if (is_last_axis_contiguous && is_stride_aligned && is_offset_aligned)
      [[likely]] {
    reduce_ara_core<OP, T>(src0, dst, tmp_buf, initvalue);
    return;
  }

  for (int64_t i = 0; i < s0; ++i){
      memref_t<__ubuf__ T, 2> src_slice{
          src0->allocated, src0->aligned,
          src0->offset + i * stride0,
          {s1, s2},
          {stride1, stride2}};
      memref_t<__ubuf__ T, 2> dst_slice{
          dst->allocated, dst->aligned,
          dst->offset + i * dst_stride0,
          {1, s2},
          {dst_stride1, dst_stride2}};
      reduce_ra<OP, T>(&src_slice, &dst_slice, tmp_buf, initvalue);
    }
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ara (3D, reduce along dim1), 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_ARA(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, half)
REGISTE_ENTIRE_REDUCE_ARA(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, float)
REGISTE_ENTIRE_REDUCE_ARA(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int16_t)

REGISTE_ENTIRE_REDUCE_ARA(reduce_max, ReduceOpTy::REDUCE_MAX, 3, half)
REGISTE_ENTIRE_REDUCE_ARA(reduce_max, ReduceOpTy::REDUCE_MAX, 3, float)
REGISTE_ENTIRE_REDUCE_ARA(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int16_t)

REGISTE_ENTIRE_REDUCE_ARA(reduce_min, ReduceOpTy::REDUCE_MIN, 3, half)
REGISTE_ENTIRE_REDUCE_ARA(reduce_min, ReduceOpTy::REDUCE_MIN, 3, float)
REGISTE_ENTIRE_REDUCE_ARA(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int16_t)

REGISTE_ENTIRE_REDUCE_ARA(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, half)
REGISTE_ENTIRE_REDUCE_ARA(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, float)
REGISTE_ENTIRE_REDUCE_ARA(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int16_t)

REGISTE_ENTIRE_REDUCE_ARA(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int8_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int16_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int64_t)

REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int8_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint8_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int16_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint16_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int64_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint64_t)

REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int8_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint8_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int16_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint16_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint32_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int64_t)
REGISTE_ENTIRE_REDUCE_ARA(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint64_t)
}
