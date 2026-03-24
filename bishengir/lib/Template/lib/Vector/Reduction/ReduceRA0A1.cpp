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
#include "Vector/VecUtils.h"
#include "DMA/DMAUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_ra0a1(memref_t<__ubuf__ T, 3> *src0,
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
  assert((src0->strides[1] == 1 && dst->strides[1] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

// Implement of reducing RA0A1 by scalar
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ra0a1_by_scalar(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
                       int64_t length) {
  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t size2 = src0->sizes[2];
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t src0_stride2 = src0->strides[2];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t dst_stride2 = dst->strides[2];

  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  // copy src0 from i=0 to dst to init dst
  for (int64_t j = 0; j < size1; ++j) {
    for (int64_t k = 0; k < MIN(size2, length); ++k) {
      *(dst_ptr + j * dst_stride1 + k * dst_stride2) =
        *(src0_ptr + j * src0_stride1 + k * src0_stride2);
    }
  }

  // Accumulates the results of src0 starting from i=1 into dst.
  for (int64_t i = 1; i < size0; ++i) {
    for (int64_t j = 0; j < size1; ++j) {
      for (int64_t k = 0; k < MIN(size2, length); ++k) {
        *(dst_ptr + j * dst_stride1 + k * dst_stride2) = 
        reduction_scalar_operation<OP, T>(*(dst_ptr + j * dst_stride1 + k * dst_stride2),
                                          *(src0_ptr + i * src0_stride0 + j * src0_stride1 + k * src0_stride2));
      }
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

// Implement of reducing RA0A1 by vector
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ra0a1_core(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
                  memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t size2 = src0->sizes[2];
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t src0_stride2 = src0->strides[2];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t dst_stride2 = dst->strides[2];

  memref_t<__ubuf__ T, 2> subview_src0{src0->allocated,
                                       src0->aligned,
                                       src0->offset,
                                       {size1, size2},
                                       {src0_stride1, src0_stride2}};
  memref_t<__ubuf__ T, 2> dst_2d{dst->allocated,
                                 dst->aligned,
                                 dst->offset,
                                 {size1, size2},
                                 {dst_stride1, dst_stride2}};
  vector_eltwise_vv_2d<VectorOpTy::VOR, T>(&subview_src0, &subview_src0,
                                           &dst_2d, tmp_buf);

  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();
  for (int64_t i = 1; i < size0; ++i) {
    memref_t<__ubuf__ T, 2> subview_src0{src0->allocated,
                                         src0->aligned,
                                         src0->offset + i * src0_stride0,
                                         {size1, size2},
                                         {src0_stride1, src0_stride2}};
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vv_2d<VECOP, T>(&subview_src0, &dst_2d, &dst_2d, tmp_buf);
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ra0a1_core<ReduceOpTy::REDUCE_OR, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  // convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // vector_eltwise_vv_2d does not support uint8_t, so view src as int16 to
  // process.
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  view_as<uint8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ra0a1_core<ReduceOpTy::REDUCE_OR, int16_t>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ra0a1_core<ReduceOpTy::REDUCE_AND, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  // convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // vector_eltwise_vv_2d does not support uint8_t, so view src as int16 to
  // process.
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  view_as<uint8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ra0a1_core<ReduceOpTy::REDUCE_AND, int16_t>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

/// Reduce src (r, a0, a1) with stride [n0, n1, 1] to dst (1, a0, a1) with
/// stride [n0, n1, 1].
///
/// constraint:
/// 1. 'n1' is a1 aligned to ub_block_unit.
/// 2. 'n0' is a0 * a1 aligned to ub_block_unit.
/// 3. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 4. reduce_xor requires additional tmp_buf of size a0 *
/// aligned(a1, ub_block_unit).
///
/// \param initvalue: The initvalue value is as follows
///             float16    float32    int8_t    uint8_t    int16_t    uint16_t
///             int32_t    uint32_t   int64_t   uint64_t
/// reduce_sum:  0          0          NA        NA         0          NA
///              0          NA         NA        NA
/// reduce_min:  HALF_INF   FLOAT_INF  NA        NA         INT16_MAX  NA
///              INT32_MAX  NA         NA        NA
/// reduce_max:  -HALF_INF  -FLOAT_INF NA        NA         INT16_MIN  NA
///              INT32_MIN  NA         NA        NA
/// reduce_prod: 1.0e+00f   1.0e+00f   NA        NA         1          NA
///              1          NA         NA        NA
/// reduce_xor:  NA         NA         0         NA         0          NA
///              0          NA         0         NA
/// reduce_or:   NA         NA         0         0          0          0
///              0          0          0         0
/// reduce_and:  NA         NA         1         1          1          1
///              1          1          1         1
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ra0a1(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
             memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ra0a1(src0, dst, initvalue);

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t size2 = src0->sizes[2];
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t src0_stride2 = src0->strides[2];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t dst_stride2 = dst->strides[2];

  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  bool is_stride_aligned = is32ByteAligned<T>(src0_stride0) && is32ByteAligned<T>(src0_stride1) &&
                         is32ByteAligned<T>(dst_stride1);
  bool is_last_axis_contiguous = (src0_stride2 == 1) && (dst_stride2 == 1);
  bool is_offset_aligned = isAddress32ByteAligned(src0_ptr) && isAddress32ByteAligned(dst_ptr);

  // To directly use vector reduce operations, the following conditions must be met:
  // 1. src0_stride0 & src0_stride1 & dst_stride1 32-byte alignment must be met.
  // 2. src0_stride2 & dst_stride2 must equal with 1
  // 3. The addresses of (aligned + offset) of src0 & dst 32-byte alignment must be met.
  if (is_last_axis_contiguous && is_stride_aligned && is_offset_aligned) [[likely]] {
    reduce_ra0a1_core<OP, T>(src0, dst, tmp_buf, initvalue);
    return;
  }

  // If offset alignment is not enough, could try to use scalar operation for first block to
  // move the addresses of src & dst 32-byte alignment
  if (is_last_axis_contiguous && is_stride_aligned && (!is_offset_aligned) &&
      reduction_is_same_element_nums_to_move_offset_aligned<T>(src0->offset, dst->offset)) {
    int64_t elements_calc_by_scalar = reduction_element_nums_to_move_offset_aligned<T>(src0->offset);
    reduce_ra0a1_by_scalar<OP, T>(src0, dst, elements_calc_by_scalar);

    if (size2 <= elements_calc_by_scalar) {
      return;
    }

    // For the remaining blocks, use vector operations. 
    memref_t<__ubuf__ T, 3> new_src0{src0->allocated,
                                     src0->aligned,
                                     src0->offset + elements_calc_by_scalar,
                                     {size0, size1, size2 - elements_calc_by_scalar},
                                     {src0_stride0, src0_stride1, src0_stride2}};

    memref_t<__ubuf__ T, 3> new_dst{dst->allocated,
                                    dst->aligned,
                                    dst->offset + elements_calc_by_scalar,
                                    {size0, size1, size2 - elements_calc_by_scalar},
                                    {dst_stride0, dst_stride1, dst_stride2}};

    reduce_ra0a1_core<OP, T>(&new_src0, &new_dst, tmp_buf, initvalue);
    return;
  }

  // In the remaining cases, reduction can only be performed through scalar operations.
  reduce_ra0a1_by_scalar<OP, T>(src0, dst, size2);
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ra0a1, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, half)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, float)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int16_t)

REGISTE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, half)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, float)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int16_t)

REGISTE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, half)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, float)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int16_t)

REGISTE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, half)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, float)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int16_t)

REGISTE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int8_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int16_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int64_t)

REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int8_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint8_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int16_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint16_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int64_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint64_t)

REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int8_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint8_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int16_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint16_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint32_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int64_t)
REGISTE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint64_t)
}
