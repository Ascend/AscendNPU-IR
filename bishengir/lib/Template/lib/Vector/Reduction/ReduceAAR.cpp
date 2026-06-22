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
#include "Vector/Reduction/ReduceARImpl.h"
#include "Vector/VecUtils.h"
#include "DMA/DMAUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_aar(memref_t<__ubuf__ T, 3> *src0,
                           memref_t<__ubuf__ T, 3> *dst, T initvalue) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src0_ptr = src0->aligned + src0->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src0->strides[1]) &&
          isSizeAlignedToBlock<T>(dst->strides[1])) &&
         "The src/dst strides[1] must be aligned to block.");
  assert((src0->strides[2] == 1 && dst->strides[2] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

/// Reduce src (a0, a1, r) with stride [n0, n1, 1] to dst (a0, a1, 1) with
/// stride [n0, n1, 1] by reducing along the last axis (dim2).
///
/// Merges the first two dims into one: [a0*(n0/n1), r], then delegates to the
/// existing 2D reduce_ar. The fast path requires n0 % n1 == 0, which allows
/// padding rows between planes. If the first two dims cannot be fused, falls
/// back to looping over dim0 and calling reduce_ar per slice.
///
/// constraint:
/// 1. 'n1' is r aligned to ub_block_unit.
/// 2. 'n0' is a multiple of 'n1' for the fast path (allows padding between
///    planes). dst stride[0] must also be a multiple of dst stride[1], with
///    the same ratio n0/n1 == dst_n0/dst_n1.
/// 3. the start pointer address should be aligned to ub_block_unit.
/// 4. tmp_buf requirement is same as reduce_ar for [a0*(n0/n1), r].
///
/// \param initvalue: same as reduce_ar initvalue table.
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_aar(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
           memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  check_inputs_of_reduce_aar(src0, dst, initvalue);

  const int64_t a0 = src0->sizes[0];
  const int64_t a1 = src0->sizes[1];
  const int64_t stride0 = src0->strides[0];
  const int64_t stride1 = src0->strides[1];
  const int64_t stride2 = src0->strides[2];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t dst_stride2 = dst->strides[2];

  bool src_fusable = (stride0 % stride1 == 0) && (stride2 == 1);
  bool dst_fusable = (dst_stride0 % dst_stride1 == 0) && (dst_stride2 == 1);

  if (src_fusable && dst_fusable) [[likely]] {
    int64_t src_rows_per_plane = stride0 / stride1;
    int64_t dst_rows_per_plane = dst_stride0 / dst_stride1;
    if (src_rows_per_plane == dst_rows_per_plane) {
      int64_t total_rows = a0 * src_rows_per_plane;
      memref_t<__ubuf__ T, 2> src_2d{src0->allocated, src0->aligned,
                                     src0->offset,
                                     {total_rows, src0->sizes[2]},
                                     {stride1, stride2}};
      memref_t<__ubuf__ T, 2> dst_2d{dst->allocated, dst->aligned,
                                     dst->offset,
                                     {total_rows, 1},
                                     {dst_stride1, dst_stride2}};
      reduce_ar<OP, T>(&src_2d, &dst_2d, tmp_buf, initvalue);
      return;
    }
  }

  // Fallback: choose loop direction to maximize reduce_ar efficiency.
  // reduce_ar dispatches to reduceAR0ToA (optimal) when dst_stride == 1,
  // otherwise falls back to reduceAR0ToAByLoopAAxis (per-row loop). Prefer the
  // direction whose dst_slice has stride[0] == 1 so results are contiguous.
  // Loop-over-a0 gives dst_stride = dst_stride1, loop-over-a1 gives dst_stride0.
  bool loop_over_a0 = (dst_stride1 == 1) || (dst_stride0 != 1 && a0 <= a1);
  if (loop_over_a0) {
    for (int64_t i = 0; i < a0; ++i) {
      memref_t<__ubuf__ T, 2> src_slice{
          src0->allocated, src0->aligned,
          src0->offset + i * stride0,
          {a1, src0->sizes[2]},
          {stride1, stride2}};
      memref_t<__ubuf__ T, 2> dst_slice{
          dst->allocated, dst->aligned,
          dst->offset + i * dst_stride0,
          {a1, 1},
          {dst_stride1, dst_stride2}};
      reduce_ar<OP, T>(&src_slice, &dst_slice, tmp_buf, initvalue);
    }
  } else {
    for (int64_t i = 0; i < a1; ++i) {
      memref_t<__ubuf__ T, 2> src_slice{
          src0->allocated, src0->aligned,
          src0->offset + i * stride1,
          {a0, src0->sizes[2]},
          {stride0, stride2}};
      memref_t<__ubuf__ T, 2> dst_slice{
          dst->allocated, dst->aligned,
          dst->offset + i * dst_stride1,
          {a0, 1},
          {dst_stride0, dst_stride2}};
      reduce_ar<OP, T>(&src_slice, &dst_slice, tmp_buf, initvalue);
    }
  }
}

/// Same as reduce_aar but delegates to reduce_ar_vcg instead of reduce_ar.
/// Used by ENABLEVCG registrations for float/half SUM/MAX/MIN.
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_aar_vcg(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
               memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  check_inputs_of_reduce_aar(src0, dst, initvalue);

  const int64_t a0 = src0->sizes[0];
  const int64_t a1 = src0->sizes[1];
  const int64_t stride0 = src0->strides[0];
  const int64_t stride1 = src0->strides[1];
  const int64_t stride2 = src0->strides[2];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t dst_stride2 = dst->strides[2];

  bool src_fusable = (stride0 % stride1 == 0) && (stride2 == 1);
  bool dst_fusable = (dst_stride0 % dst_stride1 == 0) && (dst_stride2 == 1);

  if (src_fusable && dst_fusable) [[likely]] {
    int64_t src_rows_per_plane = stride0 / stride1;
    int64_t dst_rows_per_plane = dst_stride0 / dst_stride1;
    if (src_rows_per_plane == dst_rows_per_plane) {
      int64_t total_rows = a0 * src_rows_per_plane;
      memref_t<__ubuf__ T, 2> src_2d{src0->allocated, src0->aligned,
                                     src0->offset,
                                     {total_rows, src0->sizes[2]},
                                     {stride1, stride2}};
      memref_t<__ubuf__ T, 2> dst_2d{dst->allocated, dst->aligned,
                                     dst->offset,
                                     {total_rows, 1},
                                     {dst_stride1, dst_stride2}};
      reduce_ar_vcg<OP, T>(&src_2d, &dst_2d, tmp_buf, initvalue);
      return;
    }
  }

  bool loop_over_a0 = (dst_stride1 == 1) || (dst_stride0 != 1 && a0 <= a1);
  if (loop_over_a0) {
    for (int64_t i = 0; i < a0; ++i) {
      memref_t<__ubuf__ T, 2> src_slice{
          src0->allocated, src0->aligned,
          src0->offset + i * stride0,
          {a1, src0->sizes[2]},
          {stride1, stride2}};
      memref_t<__ubuf__ T, 2> dst_slice{
          dst->allocated, dst->aligned,
          dst->offset + i * dst_stride0,
          {a1, 1},
          {dst_stride1, dst_stride2}};
      reduce_ar_vcg<OP, T>(&src_slice, &dst_slice, tmp_buf, initvalue);
    }
  } else {
    for (int64_t i = 0; i < a1; ++i) {
      memref_t<__ubuf__ T, 2> src_slice{
          src0->allocated, src0->aligned,
          src0->offset + i * stride1,
          {a0, src0->sizes[2]},
          {stride0, stride2}};
      memref_t<__ubuf__ T, 2> dst_slice{
          dst->allocated, dst->aligned,
          dst->offset + i * dst_stride1,
          {a0, 1},
          {dst_stride0, dst_stride2}};
      reduce_ar_vcg<OP, T>(&src_slice, &dst_slice, tmp_buf, initvalue);
    }
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce aar (3D, reduce along last axis), 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_ENABLEVC_AAR(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_AAR(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AAR(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AAR(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, float)
REGISTE_ENTIRE_REDUCE_AAR(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int16_t)

REGISTE_ENTIRE_REDUCE_ENABLEVC_AAR(reduce_max, ReduceOpTy::REDUCE_MAX, 3, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_AAR(reduce_max, ReduceOpTy::REDUCE_MAX, 3, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AAR(reduce_max, ReduceOpTy::REDUCE_MAX, 3, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AAR(reduce_max, ReduceOpTy::REDUCE_MAX, 3, float)
REGISTE_ENTIRE_REDUCE_AAR(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int16_t)

REGISTE_ENTIRE_REDUCE_ENABLEVC_AAR(reduce_min, ReduceOpTy::REDUCE_MIN, 3, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_AAR(reduce_min, ReduceOpTy::REDUCE_MIN, 3, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AAR(reduce_min, ReduceOpTy::REDUCE_MIN, 3, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AAR(reduce_min, ReduceOpTy::REDUCE_MIN, 3, float)
REGISTE_ENTIRE_REDUCE_AAR(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int16_t)

REGISTE_ENTIRE_REDUCE_AAR(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, half)
REGISTE_ENTIRE_REDUCE_AAR(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, float)
REGISTE_ENTIRE_REDUCE_AAR(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int16_t)

REGISTE_ENTIRE_REDUCE_AAR(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int8_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int16_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int64_t)

REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int8_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint8_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int16_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint16_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int64_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint64_t)

REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int8_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint8_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int16_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint16_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint32_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int64_t)
REGISTE_ENTIRE_REDUCE_AAR(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint64_t)
}
