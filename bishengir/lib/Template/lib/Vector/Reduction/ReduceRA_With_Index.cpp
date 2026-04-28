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
#include "Vector/Reduction/ReductionUtils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
                                     memref_t<__ubuf__ T, 2> *dst_value,
                                     memref_t<__ubuf__ int32_t, 2> *dst_index,
                                     memref_t<__ubuf__ int32_t, 1> *tmp_index) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src0->aligned + src0->offset;
  auto dst_value_ptr = dst_value->aligned + dst_value->offset;
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;
  auto tmp_index_ptr = tmp_index->aligned + tmp_index->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_value_ptr) &&
         "The starting address of dst value must be 32byte aligned.");
  assert(isAddress32ByteAligned((__ubuf__ T *)dst_index_ptr) &&
         "The starting address of dst index must be 32byte aligned.");
  assert(isAddress32ByteAligned((__ubuf__ T *)tmp_index_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(src0->strides[0]) &&
         "The src strides[0] must be aligned to block.");
  assert((src0->strides[1] == 1 && dst_value->strides[1] == 1 &&
          dst_index->strides[1] == 1) &&
         "src and dst last dim must be continuous.");
#endif
}

/// reduce src (r, a) with stride [n, 1] to dst (r, 1) and return the reduction
/// value and index separately.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. the start pointer address, namely allocated + offset, should be aligned
/// to ub_block_unit.
/// 3. tmp buffer size is equal to r * sizeof(Index) aligned to ub_block_unit +
/// 1 extra ub_block_unit
template <ReduceOpTy OP, typename T,
          typename = typename std::enable_if<(std::is_same<half, T>() ||
                                              std::is_same<float, T>())>::type,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
               OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
                     memref_t<__ubuf__ T, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index,
                     memref_t<__ubuf__ int32_t, 1> *tmp_index) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ra_with_index(src0, dst_value, dst_index, tmp_index);

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t stride0 = src0->strides[0];
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  __ubuf__ int32_t *tmp_index_ptr = tmp_index->aligned + tmp_index->offset;

  constexpr int64_t num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);

  uint16_t last_dim_align_block =
      CEIL_DIV(size1 * sizeof(T), INTR_BYTES_PER_BLOCK);
  __ubuf__ uint8_t *ub_mask_ptr =
      (__ubuf__ uint8_t *)(tmp_index->aligned + tmp_index->offset +
                           last_dim_align_block * num_per_repeat);

  // calculate the num of blocks to store the mask
  // every elements need a bit for mask
  // to align 32B, use CEIL_DIV to get aligned block
  uint16_t mask_aligned_block =
      CEIL_DIV(size1, BITS_PER_BYTE * INTR_BYTES_PER_BLOCK);
  __ubuf__ uint64_t *ub_mask_ptr_ptr =
      (__ubuf__ uint64_t *)(ub_mask_ptr +
                            mask_aligned_block * INTR_BYTES_PER_BLOCK);
  // to put the ub_mask_ptr in the CMPMASK
  ub_mask_ptr_ptr[0] =
      (uint64_t)((uint8_t *)(((uint64_t)((__ubuf__ uint8_t *)ub_mask_ptr))));
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(set_cmpmask, ((__ubuf__ uint64_t *)ub_mask_ptr_ptr));

  // to initialize the dst data
  // copy the first line to the dst
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(copy_ubuf_to_ubuf,
            dst_value_ptr,        // dst
            src_ptr,              // src
            0,                    // sid
            1,                    // nBurst
            last_dim_align_block, // lenBurst
            0,                    // srcStride
            0                     // dstStride
  );
  INTRINSIC(pipe_barrier, PIPE_V);
  brc_scalar_core_1d((int32_t)0, dst_index_ptr, size1);

  // loop size0 and process (1, size1) each time.
  for (int64_t i = 1; i < size0; i++) {
    memref_t<__ubuf__ T, 1> subview_src0{src0->allocated,
                                         src0->aligned,
                                         src0->offset + i * stride0,
                                         {size1},
                                         {1}};
    memref_t<__ubuf__ T, 1> subview_dst_value{dst_value->allocated,
                                              dst_value->aligned,
                                              dst_value->offset,
                                              {size1},
                                              {1}};

    /// to get the result of vcmpv and max value
    /// the type of T we can choose includes: int32_t, half, float
    /// execute the Op we choose, including max/min
    INTRINSIC(pipe_barrier, PIPE_V);
    if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
      vector_cmp<VectorOpTy::VCMP_LT, T>(&subview_src0, &subview_dst_value,
                                         ub_mask_ptr);
      vector_eltwise_vv_1d<VectorOpTy::VMAX, T>(
          &subview_dst_value, &subview_src0, &subview_dst_value);
    } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
      vector_cmp<VectorOpTy::VCMP_GT, T>(&subview_src0, &subview_dst_value,
                                         ub_mask_ptr);
      vector_eltwise_vv_1d<VectorOpTy::VMIN, T>(
          &subview_dst_value, &subview_src0, &subview_dst_value);
    } else {
      static_assert((OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                     OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) &&
                    "unsupported op");
    }

    // using vsel to update dst index
    INTRINSIC(pipe_barrier, PIPE_V);
    brc_scalar_core_1d((int32_t)i, tmp_index_ptr, size1);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0x0, size1);
    // use ub_mask to perform v_select
    INTRINSIC(vsel, (__ubuf__ float *)dst_index_ptr,
              (__ubuf__ float *)tmp_index_ptr, (__ubuf__ float *)dst_index_ptr,
              1,                           // repeat, auto-infered
              1,                           // dstBlockStride,
              1,                           // src0BlockStride,
              1,                           // src1BlockStride,
              INTR_BLOCKS_PER_REPEAT,      // dstRepeatStride,
              INTR_BLOCKS_PER_REPEAT,      // src0RepeatStride,
              INTR_BLOCKS_PER_REPEAT,      // src1RepeatStride,
              (uint8_t)(SelectMode::MODE2) // mode
    );
    INTRINSIC_NO_ARGS(set_mask_norm);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ra with index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2,
                                    float);

REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2,
                                    float);
}