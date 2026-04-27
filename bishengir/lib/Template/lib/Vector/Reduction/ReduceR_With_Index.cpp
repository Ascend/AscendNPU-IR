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
__aiv__ __attribute__((always_inline)) void check_inputs_of_reduce_r_with_index(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index, memref_t<__ubuf__ T, 1> *tmp_buf,
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
  assert(src0->strides[0] == 1 && "The src must be continuous vector.");
#endif
}

/// reduce src (r, ) to dst (1, ) and return the reduction value and index
/// separately.
///
/// constraint:
/// 1. dim of src/dst must be 1.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. tmp buffer size is equal to 1 block.
///
/// \param initvalue: The initvalue value is as follows
///                    float16             float32
/// reduce_min:         HALF_INF            FLOAT_INF
/// reduce_max:         -HALF_INF           -FLOAT_INF
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_with_index(memref_t<__ubuf__ T, 1> *src0,
                    memref_t<__ubuf__ T, 1> *dst_value,
                    memref_t<__ubuf__ int32_t, 1> *dst_index,
                    memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  static_assert(
      (std::is_same<half, T>::value || std::is_same<float, T>::value) &&
      "reduce_r_with_index do not support this data type");
  static_assert((OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX ||
                 OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) &&
                "reduce_r_with_index do not support this reduce op type");

  // Input parameter constraints assert.
  check_inputs_of_reduce_r_with_index(src0, dst_value, dst_index, tmp_buf,
                                      initvalue);

  const int64_t size0 = src0->sizes[0];
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  __ubuf__ T *tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0x0, size0);
  reduce_intrin<OP, T, false, Order_t::VALUE_INDEX>(reduce_intrin_args<T>{
      tmp_buf_ptr,
      nullptr, // dst_index, not work
      src_ptr,
      1, // repeat, not work
      1, // dst repeat stride
      1, // src block stride
      8  // src repeat stride
  });
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  if constexpr (sizeof(T) == 4) {
    *(__ubuf__ int32_t *)(dst_value_ptr) = GET_MAX_MIN_CNT();
  } else {
    *(__ubuf__ int16_t *)(dst_value_ptr) = GET_MAX_MIN_CNT();
  }
  *(__ubuf__ int32_t *)(dst_index_ptr) = GET_MAX_MIN_CNT() >> 32;
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce r with index, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                                   ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, half);
REGISTE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                                   ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, float);

REGISTE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                                   ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, half);
REGISTE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                                   ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, float);

//===-------------------------------------------------------------------===//
// Registe reduce_r_with_index fun, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_REDUCE_R_WITH_INDEX(ReduceOpTy::REDUCE_MAX_WITH_INDEX, half);
REGISTE_REDUCE_R_WITH_INDEX(ReduceOpTy::REDUCE_MAX_WITH_INDEX, float);
REGISTE_REDUCE_R_WITH_INDEX(ReduceOpTy::REDUCE_MIN_WITH_INDEX, half);
REGISTE_REDUCE_R_WITH_INDEX(ReduceOpTy::REDUCE_MIN_WITH_INDEX, float);
}