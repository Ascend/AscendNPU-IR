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

#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Reduction/ReductionUtils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_ar_with_index(memref_t<__ubuf__ T, 2> *src0,
                                     memref_t<__ubuf__ T, 2> *dst_value,
                                     memref_t<__ubuf__ int32_t, 2> *dst_index,
                                     memref_t<__ubuf__ T, 1> *tmp_buf,
                                     T initvalue) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src0->aligned + src0->offset;
  auto dst_value_ptr = dst_value->aligned + dst_value->offset;
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_value_ptr) &&
         "The starting address of dst value must be 32byte aligned.");
  assert(isAddress32ByteAligned((__ubuf__ T *)dst_index_ptr) &&
         "The starting address of dst index must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_buf_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(src0->strides[0]) &&
         "The src strides[0] must be aligned to block.");
  assert(src0->strides[1] == 1 && "src last dim must be continuous.");
  assert(((isSizeAlignedToBlock<T>(dst_value->strides[0]) ||
           dst_value->strides[0] == 1) &&
          (isSizeAlignedToBlock<T>(dst_index->strides[0]) ||
           dst_index->strides[0] == 1)) &&
         "The dst strides[0] must be aligned to block or 1.");
#endif
}

template <ReduceOpTy OP, typename T,
          typename = typename std::enable_if<(std::is_same<half, T>() ||
                                              std::is_same<float, T>())>::type,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
               OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ar_with_index_special_scene(memref_t<__ubuf__ T, 2> *src0,
                                   memref_t<__ubuf__ T, 2> *dst_value,
                                   memref_t<__ubuf__ int32_t, 2> *dst_index) {
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;
  memref_t<__ubuf__ T, 1> src_1d{
      src0->aligned, src0->allocated, src0->offset, {src0->sizes[0]}, {1}};
  memref_t<__ubuf__ T, 1> dst_value_1d{dst_value->aligned,
                                       dst_value->allocated,
                                       dst_value->offset,
                                       {dst_value->sizes[0]},
                                       {1}};
  copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim(&src_1d, &dst_value_1d);
  brc_scalar_core_1d<int32_t>(0, dst_index_ptr, src0->sizes[0]);
}

/// reduce src (a, r) with stride [n, 1] to dst (a, 1) and return the reduction
/// value and index separately.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. 'n' is r aligned to ub_block_unit.
/// 4. tmp buffer size is equal to 1 block.
///
/// \param initvalue: The initvalue value is as follows
///                    float16             float32
/// reduce_min:         HALF_INF            FLOAT_INF
/// reduce_max:         -HALF_INF           -FLOAT_INF
template <ReduceOpTy OP, typename T,
          typename = typename std::enable_if<(std::is_same<half, T>() ||
                                              std::is_same<float, T>())>::type,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
               OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ar_with_index(memref_t<__ubuf__ T, 2> *src0,
                     memref_t<__ubuf__ T, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index,
                     memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ar_with_index(src0, dst_value, dst_index, tmp_buf,
                                       initvalue);

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t src_stride0 = src0->strides[0];
  const int64_t dst_value_stride0 = dst_value->strides[0];
  const int64_t dst_index_stride0 = dst_index->strides[0];
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);

  if (size1 > num_per_repeat) {
    // loop size0 and process (1, size1) each time.
    for (int64_t i = 0; i < size0; i++) {
      memref_t<__ubuf__ T, 1> subview_src0{src0->allocated,
                                           src0->aligned,
                                           src0->offset + i * src_stride0,
                                           {size1},
                                           {1}};
      memref_t<__ubuf__ T, 1> subview_dst_value{dst_value->allocated,
                                                dst_value->aligned,
                                                dst_value->offset +
                                                    i * dst_value_stride0,
                                                {size1},
                                                {1}};
      memref_t<__ubuf__ int32_t, 1> subview_dst_index{dst_index->allocated,
                                                      dst_index->aligned,
                                                      dst_index->offset +
                                                          i * dst_index_stride0,
                                                      {size1},
                                                      {1}};
      reduce_r_with_index<OP, T>(&subview_src0, &subview_dst_value,
                                 &subview_dst_index, tmp_buf, initvalue);
    }
  } else {
    // TODO: This scene is not supported later, and the upper layer converts it
    // to 1D implementation.
    if (size1 == 1 && src_stride0 == 1 && dst_value_stride0 == 1 &&
        dst_index_stride0 == 1) {
      reduce_ar_with_index_special_scene<OP, T>(src0, dst_value, dst_index);
      return;
    }

    // optimize to map size0 to repeat parameter of intrinisic.
    if (dst_value_stride0 == 1 && dst_index_stride0 == 1) {
      reduceAR0ToA<OP, T, true>(src0, dst_value, dst_index);
    } else {
      reduceAR0ToAByLoopAAxis<OP, T, true>(src0, dst_value, dst_index);
    }
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ar with index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2,
                                    float);

REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2,
                                    float);
}