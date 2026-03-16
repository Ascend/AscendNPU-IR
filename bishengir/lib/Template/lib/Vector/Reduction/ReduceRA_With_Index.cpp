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

template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ __attribute__((always_inline)) void
process(__ubuf__ uint8_t *ub_mask_ptr, memref_t<__ubuf__ T, 1> subview_src0,
        memref_t<__ubuf__ T, 1> subview_dst_value) {
  /// to get the result of vcmpv and max value
  /// the type of T we can choose includes: int32_t, half, float
  /// execute the Op we choose, including max/min
  if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
    if constexpr (WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT) {
      vector_cmp<VectorOpTy::VCMP_LT, T>(&subview_src0, &subview_dst_value,
                                         ub_mask_ptr);
    } else if constexpr (WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT) {
      vector_cmp<VectorOpTy::VCMP_LE, T>(&subview_src0, &subview_dst_value,
                                         ub_mask_ptr);
    } else {
      static_assert("unsupported reduce_with_index type");
    }
    vector_eltwise_vv_1d<VectorOpTy::VMAX, T>(&subview_dst_value, &subview_src0,
                                              &subview_dst_value);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
    if constexpr (WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT) {
      vector_cmp<VectorOpTy::VCMP_GT, T>(&subview_src0, &subview_dst_value,
                                         ub_mask_ptr);
    } else if constexpr (WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT) {
      vector_cmp<VectorOpTy::VCMP_GE, T>(&subview_src0, &subview_dst_value,
                                         ub_mask_ptr);
    } else {
      static_assert("unsupported reduce_with_index type");
    }

    vector_eltwise_vv_1d<VectorOpTy::VMIN, T>(&subview_dst_value, &subview_src0,
                                              &subview_dst_value);
  } else {
    static_assert((OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                   OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) &&
                  "unsupported op");
  }
}

/// reduce src (r, a) with stride [n, 1] to dst (r, 1) and return the
/// reduction value and index separately.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. the start pointer address, namely allocated + offset, should be
/// aligned to ub_block_unit.
/// 3. tmp buffer consists from 2 parts:
///   a. part required to store indices, aligned to BLOCK size
///   b. part required to store mask of indices, aligned to BLOCK size
///   Parts a and b are calucated based on size of A dim. Pointer to mask
///   calculated as 0-index of mask
template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T,
          typename = typename std::enable_if<(std::is_same<half, T>() ||
                                              std::is_same<float, T>())>::type,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
               OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)>::type,
          typename = typename std::enable_if<
              (WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT ||
               WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT)>::type>
__aiv__ __attribute__((always_inline)) void
vector_reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
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

  // FIXME: indices hardcoded as int32_t
  uint16_t num_repeats_for_a_dim_aligned =
      CEIL_DIV(size1 * sizeof(int32_t), INTR_BYTES_PER_REPEAT);
  __ubuf__ uint8_t *ub_mask_ptr =
      (__ubuf__ uint8_t *)(tmp_index_ptr + (num_repeats_for_a_dim_aligned *
                                            INTR_BYTES_PER_REPEAT) /
                                               sizeof(int32_t));

  // calculate the num of blocks to store the mask
  // every elements need a bit for mask
  // to align 32B, use CEIL_DIV to get aligned block
  uint16_t num_blocks_for_mask_aligned =
      CEIL_DIV(size1, BITS_PER_BYTE * INTR_BYTES_PER_BLOCK);
  __ubuf__ uint64_t *ub_mask_ptr_ptr =
      (__ubuf__ uint64_t *)(ub_mask_ptr +
                            num_blocks_for_mask_aligned * INTR_BYTES_PER_BLOCK);
  // to put the ub_mask_ptr in the CMPMASK
  ub_mask_ptr_ptr[0] =
      (uint64_t)((uint8_t *)(((uint64_t)((__ubuf__ uint8_t *)ub_mask_ptr))));
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(set_cmpmask, ((__ubuf__ uint64_t *)ub_mask_ptr_ptr));

  int32_t num_blocks_for_a_dim_aligned =
      CEIL_DIV(size1 * sizeof(T), INTR_BYTES_PER_BLOCK);
  // to initialize the dst data
  // copy the first line to the dst
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(copy_ubuf_to_ubuf,
            dst_value_ptr,                // dst
            src_ptr,                      // src
            0,                            // sid
            1,                            // nBurst
            num_blocks_for_a_dim_aligned, // lenBurst
            0,                            // srcStride
            0                             // dstStride
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

    INTRINSIC(pipe_barrier, PIPE_V);
    process<OP, WITH_INDEX_TYPE, T>(ub_mask_ptr, subview_src0,
                                    subview_dst_value);

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

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_stride_size_aligned_to_block(memref_t<__ubuf__ T, 2> *src0,
                                memref_t<__ubuf__ T, 2> *dst_value,
                                memref_t<__ubuf__ int32_t, 2> *dst_index) {
  if (!is32ByteAligned<T>(src0->strides[0])) {
    return false;
  }

  if ((src0->strides[1] != 1) ||
      (dst_value->strides[1] != 1) ||
      (dst_index->strides[1] != 1)) {
    return false;
  }
  return true;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_offset_address_aligned_to_block(memref_t<__ubuf__ T, 2> *src0,
                                   memref_t<__ubuf__ T, 2> *dst_value,
                                   memref_t<__ubuf__ int32_t, 2> *dst_index) {
  auto src0_ptr = src0->aligned + src0->offset;
  auto dst_value_ptr = dst_value->aligned + dst_value->offset;
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;

  return (isAddress32ByteAligned(src0_ptr) &&
          isAddress32ByteAligned(dst_value_ptr) &&
          isAddress32ByteAligned(dst_index_ptr));
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
reduce_max_with_index(__ubuf__ T *cur_src0_ptr,
                      int32_t cur_src0_index,
                      __ubuf__ T *reduce_max_val_ptr,
                      __ubuf__ int32_t *reduce_max_index_ptr) {
  if (isnan_value(*reduce_max_val_ptr)) {
    return;
  }

  if ((static_cast<float>(*cur_src0_ptr) > static_cast<float>(*reduce_max_val_ptr)) || isnan_value(*cur_src0_ptr)) {
    *reduce_max_val_ptr = *cur_src0_ptr;
    *reduce_max_index_ptr = cur_src0_index;
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
reduce_min_with_index(__ubuf__ T *cur_src0_ptr,
                      int32_t cur_src0_index,
                      __ubuf__ T *reduce_min_val_ptr,
                      __ubuf__ int32_t *reduce_min_index_ptr) {
  if (isnan_value(*reduce_min_val_ptr)) {
    return;
  }

  if ((static_cast<float>(*cur_src0_ptr) < static_cast<float>(*reduce_min_val_ptr)) || isnan_value(*cur_src0_ptr)) {
    *reduce_min_val_ptr = *cur_src0_ptr;
    *reduce_min_index_ptr = cur_src0_index;
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_operation(__ubuf__ T *cur_src0_ptr,
                 int32_t cur_src0_index,
                 __ubuf__ T *reduce_val_ptr,
                 __ubuf__ int32_t *reduce_index_ptr) {
  static_assert(OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                "ReduceOpTy not find. The return value may be incorrect.");

  if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
    reduce_max_with_index(cur_src0_ptr, cur_src0_index, reduce_val_ptr, reduce_index_ptr);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
    reduce_min_with_index(cur_src0_ptr, cur_src0_index, reduce_val_ptr, reduce_index_ptr);
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_operation_with_index_left(memref_t<__ubuf__ T, 2> *src0,
                                 int64_t col,
                                 __ubuf__ T *cur_dst_val_ptr,
                                 __ubuf__ int32_t *cur_dst_idx_ptr) {
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t src0_size0 = src0->sizes[0];

  *cur_dst_val_ptr = *(src0_ptr + col * src0_stride1);
  *cur_dst_idx_ptr = 0;
  for (int64_t row = 0; row < src0_size0; ++row) {
    __ubuf__ T *cur_src0_ptr = src0_ptr + (row * src0_stride0 + col * src0_stride1);
    scalar_operation<OP>(cur_src0_ptr, row, cur_dst_val_ptr, cur_dst_idx_ptr);
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_operation_with_index_right(memref_t<__ubuf__ T, 2> *src0,
                                  int64_t col,
                                  __ubuf__ T *cur_dst_val_ptr,
                                  __ubuf__ int32_t *cur_dst_idx_ptr) {
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t src0_size0 = src0->sizes[0];

  *cur_dst_val_ptr = *(src0_ptr + (src0_size0 - 1) * src0_stride0 +  col * src0_stride1);
  *cur_dst_idx_ptr = src0_size0 - 1;
  for (int64_t row = (src0_size0 - 1); row >= 0; --row) {
    __ubuf__ T *cur_src0_ptr = src0_ptr + (row * src0_stride0 + col * src0_stride1);
    scalar_operation<OP>(cur_src0_ptr, row, cur_dst_val_ptr, cur_dst_idx_ptr);
  }
}

template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
                            memref_t<__ubuf__ T, 2> *dst_value,
                            memref_t<__ubuf__ int32_t, 2> *dst_index) {
  WARN_SCALAR_IMPL("reduceRA with index");
  static_assert(std::is_same<half, T>() ||
                std::is_same<float, T>(),
                "T must be half or float.");
  static_assert(OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                "OP must be REDUCE_MAX_WITH_INDEX or REDUCE_MIN_WITH_INDEX.");
  static_assert(WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT ||
                WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT,
                "WITH_INDEX_TYPE must be LEFT or RIGHT.");

  const int64_t src0_size1 = src0->sizes[1];
  const int64_t dst_value_stride1 = dst_value->strides[1];
  const int64_t dst_index_stride1 = dst_index->strides[1];

  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  for (int64_t col = 0; col < src0_size1; ++col) {
    __ubuf__ T *cur_dst_val_ptr = dst_value_ptr + col * dst_value_stride1;
    __ubuf__ int32_t *cur_dst_idx_ptr = dst_index_ptr + col * dst_index_stride1;

    if (WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT) {
      scalar_operation_with_index_left<OP>(src0, col, cur_dst_val_ptr, cur_dst_idx_ptr);
    } else {
      scalar_operation_with_index_right<OP>(src0, col, cur_dst_val_ptr, cur_dst_idx_ptr);
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) int64_t
element_nums_to_move_offset_aligned(memref_t<__ubuf__ T, 2> *buf) {
  constexpr int num_per_block = UB_ALIGN_BYTES / sizeof(T);
  return CEIL_DIV(buf->offset, num_per_block) * num_per_block - buf->offset;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_same_element_nums_to_move_offset_aligned(memref_t<__ubuf__ T, 2> *src,
                                            memref_t<__ubuf__ T, 2> *dst_value,
                                            memref_t<__ubuf__ int32_t, 2> *dst_index) {
  auto src_offset_elements = element_nums_to_move_offset_aligned(src);
  auto dst_value_offset_elements = element_nums_to_move_offset_aligned(dst_value);
  auto dst_index_offset_elements = element_nums_to_move_offset_aligned(dst_index);
  if (src_offset_elements != dst_value_offset_elements) {
    return false;
  }
  return src_offset_elements * sizeof(T) == dst_index_offset_elements * sizeof(int32_t);
}

template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
                            memref_t<__ubuf__ T, 2> *dst_value,
                            memref_t<__ubuf__ int32_t, 2> *dst_index,
                            int64_t element_nums_offset_to_aligned) {
  static_assert(std::is_same<half, T>() ||
                std::is_same<float, T>(),
                "T must be half or float.");
  static_assert(OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                "OP must be REDUCE_MAX_WITH_INDEX or REDUCE_MIN_WITH_INDEX.");
  static_assert(WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT ||
                WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT,
                "WITH_INDEX_TYPE must be LEFT or RIGHT.");

  memref_t<__ubuf__ T, 2> scalar_src0 = *src0;
  memref_t<__ubuf__ T, 2> scalar_dst_value = *dst_value;
  memref_t<__ubuf__ int32_t, 2> scalar_dst_index = *dst_index;
  get_scalar_memref_t(element_nums_offset_to_aligned, src0, &scalar_src0);
  get_scalar_memref_t(element_nums_offset_to_aligned, dst_value, &scalar_dst_value);
  get_scalar_memref_t(element_nums_offset_to_aligned, dst_index, &scalar_dst_index);
  scalar_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(&scalar_src0, &scalar_dst_value, &scalar_dst_index);
}

template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ __attribute__((always_inline)) void
vector_reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
                            memref_t<__ubuf__ T, 2> *dst_value,
                            memref_t<__ubuf__ int32_t, 2> *dst_index,
                            memref_t<__ubuf__ int32_t, 1> *tmp_index,
                            int64_t element_nums_offset_to_aligned) {
  static_assert(std::is_same<half, T>() ||
                std::is_same<float, T>(),
                "T must be half or float.");
  static_assert(OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                "OP must be REDUCE_MAX_WITH_INDEX or REDUCE_MIN_WITH_INDEX.");
  static_assert(WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT ||
                WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT,
                "WITH_INDEX_TYPE must be LEFT or RIGHT.");

  memref_t<__ubuf__ T, 2> vector_src0 = *src0;
  memref_t<__ubuf__ T, 2> vector_dst_value = *dst_value;
  memref_t<__ubuf__ int32_t, 2> vector_dst_index = *dst_index;
  get_vector_memref_t(element_nums_offset_to_aligned, src0, &vector_src0);
  get_vector_memref_t(element_nums_offset_to_aligned, dst_value, &vector_dst_value);
  get_vector_memref_t(element_nums_offset_to_aligned, dst_index, &vector_dst_index);

  vector_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(&vector_src0, &vector_dst_value, &vector_dst_index, tmp_index);
}

template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ra_with_index(memref_t<__ubuf__ T, 2> *src0,
                     memref_t<__ubuf__ T, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index,
                     memref_t<__ubuf__ int32_t, 1> *tmp_index) {
  static_assert(std::is_same<half, T>() ||
                std::is_same<float, T>(),
                "T must be half or float.");
  static_assert(OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
                OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                "OP must be REDUCE_MAX_WITH_INDEX or REDUCE_MIN_WITH_INDEX.");
  static_assert(WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT ||
                WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT,
                "WITH_INDEX_TYPE must be LEFT or RIGHT.");

  bool is_stride_size_aligned = is_stride_size_aligned_to_block<T>(src0, dst_value, dst_index);
  bool is_offset_address_aligned = is_offset_address_aligned_to_block<T>(src0, dst_value, dst_index);
  bool is_same_offset_aligned = is_same_element_nums_to_move_offset_aligned(src0, dst_value, dst_index);

  // The stride sizes of source and dest are aligned.
  if (is_stride_size_aligned) {
    // The address of source, dest_value and dest_index with the offset is 32-byte aligned.
    if (is_offset_address_aligned) {
      vector_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(src0, dst_value, dst_index, tmp_index);
      return;
    }

    // The address of source, dest_value and dest_index with the offset is not 32-byte aligned.
    // After an additional same offset is added, the addresses can be 32-byte aligned.
    if (is_same_offset_aligned) {
      // source, dest_value and dest_index. The unaligned part is calculated using the scalar,
      // and the aligned part is calculated using the vector.
      int64_t scalar_element_nums = element_nums_to_move_offset_aligned<T>(src0);
      scalar_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(
        src0, dst_value, dst_index, scalar_element_nums);

      // If the scalar has processed all data, the vector does not need to be called for processing.
      auto last_dim_ub_element_nums = src0->sizes[1] * src0->strides[1];
      if (last_dim_ub_element_nums < scalar_element_nums) {
        return;
      }
      vector_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(
        src0, dst_value, dst_index, tmp_index, scalar_element_nums);
      return;
    }
    // After an additional same offset is added, the addresses can not be 32-byte aligned.
    scalar_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(src0, dst_value, dst_index);
    return;
  } 
  
  // The stride sizes of source and dest are not aligned.
  scalar_reduce_ra_with_index<OP, WITH_INDEX_TYPE>(src0, dst_value, dst_index);
}

/// reduce src (r, a) with stride [n, 1] to dst (r, 1) and return the
/// reduction value and index separately. The index is from the indices
/// input.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. the start pointer address, namely allocated + offset, should be
/// aligned to ub_block_unit.
/// 3. tmp buffer size is equal to r * sizeof(Index) aligned to
/// ub_block_unit + 1 extra ub_block_unit
template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T,
          typename = typename std::enable_if<(std::is_same<half, T>() ||
                                              std::is_same<float, T>())>::type,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX ||
               OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)>::type,
          typename = typename std::enable_if<
              (WITH_INDEX_TYPE == ReduceWithIndexOpTy::LEFT ||
               WITH_INDEX_TYPE == ReduceWithIndexOpTy::RIGHT)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ra_with_index_with_specified_index(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ int32_t, 2> *indices,
    memref_t<__ubuf__ T, 2> *dst_value,
    memref_t<__ubuf__ int32_t, 2> *dst_index,
    memref_t<__ubuf__ int32_t, 1> *tmp_index) {
  reduce_ra_with_index<OP, WITH_INDEX_TYPE, T>(src0, dst_value, dst_index,
                                               tmp_index);
  // TODO: use gather to optimize
  int64_t stride0_src = indices->strides[0];
  int64_t stride1_src = indices->strides[1];
  int64_t stride1_dst = dst_index->strides[1];
  int64_t size1 = dst_index->sizes[1];
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  __ubuf__ int32_t *indices_ptr = indices->aligned + indices->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < size1; ++i) {
    int32_t resIndex = dst_index_ptr[i * stride1_dst];
    dst_index_ptr[i * stride1_dst] =
        indices_ptr[resIndex * stride0_src + i * stride1_src];
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ra with index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

//===-------------------------------------------------------------------===//
// reduce ra with index with specified index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_max_with_index_left_with_specified_index,
    ReduceOpTy::REDUCE_MAX_WITH_INDEX, ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_max_with_index_right_with_specified_index,
    ReduceOpTy::REDUCE_MAX_WITH_INDEX, ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_max_with_index_left_with_specified_index,
    ReduceOpTy::REDUCE_MAX_WITH_INDEX, ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_max_with_index_right_with_specified_index,
    ReduceOpTy::REDUCE_MAX_WITH_INDEX, ReduceWithIndexOpTy::RIGHT, 2, float);

REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_min_with_index_left_with_specified_index,
    ReduceOpTy::REDUCE_MIN_WITH_INDEX, ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_min_with_index_right_with_specified_index,
    ReduceOpTy::REDUCE_MIN_WITH_INDEX, ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_min_with_index_left_with_specified_index,
    ReduceOpTy::REDUCE_MIN_WITH_INDEX, ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(
    reduce_min_with_index_right_with_specified_index,
    ReduceOpTy::REDUCE_MIN_WITH_INDEX, ReduceWithIndexOpTy::RIGHT, 2, float);
}
