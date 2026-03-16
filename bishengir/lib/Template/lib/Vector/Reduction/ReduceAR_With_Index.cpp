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

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_supported_stride(const memref_t<__ubuf__ T, 2> *src0,
                    const memref_t<__ubuf__ T, 2> *dst_value,
                    const memref_t<__ubuf__ int32_t, 2> *dst_index) {
  const int64_t size1 = src0->sizes[1];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  const int64_t src_stride0 = src0->strides[0];
  const int64_t dst_value_stride0 = dst_value->strides[0];
  const int64_t dst_value_stride1 = dst_value->strides[1];
  const int64_t dst_index_stride0 = dst_index->strides[0];
  const int64_t dst_index_stride1 = dst_index->strides[1];

  if (size1 > num_per_repeat) {
    bool is_src_aligned = is32ByteAligned<T>(src0->strides[0]) && (src0->strides[1] == 1);
    if (is_src_aligned) {
      return true;
    }
  }

  if (size1 == 1 && src_stride0 == 1 && dst_value_stride0 == 1 && dst_index_stride0 == 1) {
    return true;
  }

  if (is32ByteAligned<T>(src_stride0) && (src0->strides[1] == 1)) {
    return true;
  }
  return false;
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

template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ void reduce_with_index_scalar_iml(memref_t<__ubuf__ T, 2> *src,
                                          memref_t<__ubuf__ T, 2> *dst_value,
                                          memref_t<__ubuf__ int32_t, 2> *dst_index,
                                          int64_t size0, int64_t size1,
                                          T initvalue,
                                          bool need_merge = false) {
  WARN_SCALAR_IMPL("reduceAR with index");
  int64_t num_rows = size0;
  int64_t num_cols = size1;

  __ubuf__ T* src_ptr = src->aligned + src->offset;
  __ubuf__ T* dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t* dst_index_ptr = dst_index->aligned + dst_index->offset;

  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_value_stride0 = dst_value->strides[0];
  const int64_t dst_index_stride0 = dst_index->strides[0];

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < num_rows; ++i) {
    __ubuf__ T* dst_val_ptr = dst_value_ptr + i * dst_value_stride0;
    __ubuf__ int32_t* dst_idx_ptr = dst_index_ptr + i * dst_index_stride0;
    
    T tmp_extreme = *(src_ptr + i * src_stride0);
    int32_t tmp_idx = 0;

    for (int64_t j = 0; j < num_cols; ++j) {
      __ubuf__ T* val_ptr = src_ptr + i * src_stride0 + j * src->strides[1];
      if (isnan_value(tmp_extreme)) {
        break;
      }
      if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
        if ((static_cast<float>(*val_ptr) > static_cast<float>(tmp_extreme)) ||
          isnan_value(*val_ptr)) {
          tmp_extreme = *val_ptr;
          tmp_idx = static_cast<int32_t>(j);
        }
      } else { //OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX
        if (static_cast<float>(*val_ptr) < static_cast<float>(tmp_extreme) ||
          isnan_value(*val_ptr)) {
          tmp_extreme = *val_ptr;
          tmp_idx = static_cast<int32_t>(j);
        }
      }
    }
    if (need_merge && !isnan_value(tmp_extreme)) {
      T vec_extreme = *(dst_value_ptr + i * dst_value_stride0);
      int32_t vec_idx = *(dst_index_ptr + i * dst_index_stride0);
      if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
        if (static_cast<float>(vec_extreme) > (static_cast<float>(tmp_extreme)) ||
          isnan_value(vec_extreme)) {
          tmp_extreme = vec_extreme;
          tmp_idx = vec_idx;
        }
      } else { //OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX
        if (static_cast<float>(vec_extreme) < (static_cast<float>(tmp_extreme)) ||
          isnan_value(vec_extreme)) {
          tmp_extreme = vec_extreme;
          tmp_idx = vec_idx;
        }
      }
    }
    *dst_val_ptr = tmp_extreme;
    *dst_idx_ptr = tmp_idx;
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

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
vec_reduce_ar_with_index(memref_t<__ubuf__ T, 2> *src0,
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
      reduce_r_with_index<OP, WITH_INDEX_TYPE, T>(
          &subview_src0, &subview_dst_value, &subview_dst_index, tmp_buf,
          initvalue);
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
reduce_ar_with_index(memref_t<__ubuf__ T, 2> *src0,
                     memref_t<__ubuf__ T, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index,
                     memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ar_with_index(src0, dst_value, dst_index, tmp_buf,
                                       initvalue);

  const int64_t size1 = src0->sizes[1];
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;

  // if stride is not aligned, fallback to scalar implemention
  if (!is_supported_stride<T>(src0, dst_value, dst_index)) [[unlikely]] {
    reduce_with_index_scalar_iml<OP, WITH_INDEX_TYPE, T>(
      src0, dst_value, dst_index, src0->sizes[0], src0->sizes[1], initvalue);
    return;
  }

  // if stride is aligned, offset of dst is alined, offset of src is not alined, use scalar and vec instructions
  if (!isAddress32ByteAligned<T>(src_ptr)) [[unlikely]] { 
    auto address = reinterpret_cast<uintptr_t>(src_ptr);
    auto align_diff = (UB_ALIGN_BYTES - (address & 0x1F)) & 0x1F;
    int64_t scalarNum = static_cast<int64_t>(align_diff / sizeof(T));
    int64_t offset = src0->offset;
    int64_t vectorNum = size1  - scalarNum; 

    // calculate the result of vec part
    if (vectorNum > 0){
      memref_t<__ubuf__ T, 2> subview_src {
        src0->allocated,
        src0->aligned,
        src0->offset + scalarNum,
        {src0->sizes[0], vectorNum},
        {src0->strides[0], src0->strides[1]}
      };

      INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      vec_reduce_ar_with_index<OP, WITH_INDEX_TYPE, T>(&subview_src, dst_value, dst_index, tmp_buf, initvalue);
      INTRINSIC(set_flag, PIPE_V,PIPE_S,  LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_V,PIPE_S, LIB_EVENT_ID0);
      for (int i = 0; i < dst_value->sizes[0]; i++) {
        *(dst_index_ptr + i * dst_index->strides[0]) += scalarNum/src0->strides[1];
      }
    }

    // calculate the result of scalar part, and gather the result of two part
    bool need_merge = (vectorNum > 0);
    reduce_with_index_scalar_iml<OP, WITH_INDEX_TYPE, T>(
      src0, dst_value, dst_index, src0->sizes[0], scalarNum, initvalue, need_merge);
    return;
  }

  vec_reduce_ar_with_index<OP, WITH_INDEX_TYPE, T>(src0, dst_value, dst_index, tmp_buf, initvalue);

}

/// reduce src (a, r) with stride [n, 1] to dst (a, 1) and return the reduction
/// value and index separately. The index is from the indices input.
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
reduce_ar_with_index_with_specified_index(memref_t<__ubuf__ T, 2> *src0,
                                          memref_t<__ubuf__ int32_t, 2> *indices,
                                          memref_t<__ubuf__ T, 2> *dst_value,
                                          memref_t<__ubuf__ int32_t, 2> *dst_index,
                                          memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  reduce_ar_with_index<OP, WITH_INDEX_TYPE, T>(src0, dst_value, dst_index, tmp_buf, initvalue);
  // TODO: use gather to optimize
  int64_t stride0_src = indices->strides[0];
  int64_t stride1_src = indices->strides[1];
  int64_t stride0_dst = dst_index->strides[0];
  int64_t stride1_dst = dst_index->strides[1];
  int64_t size0 = dst_index->sizes[0];
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  __ubuf__ int32_t *indices_ptr = indices->aligned + indices->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < size0; ++i) {
    int32_t resIndex = dst_index_ptr[i * stride0_dst];
    dst_index_ptr[i * stride0_dst] = indices_ptr[i * stride0_src + resIndex * stride1_src];
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}


extern "C" {
//===-------------------------------------------------------------------===//
// reduce ar with index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

//===-------------------------------------------------------------------===//
// reduce ra with index with specified index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_max_with_index_left_with_specified_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_max_with_index_right_with_specified_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_max_with_index_left_with_specified_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_max_with_index_right_with_specified_index,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_min_with_index_left_with_specified_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_min_with_index_right_with_specified_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_min_with_index_left_with_specified_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(reduce_min_with_index_right_with_specified_index,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);
}
