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
#include "Vector/Reduction/ReductionUtils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_ra(memref_t<__ubuf__ T, 2> *src0,
                          memref_t<__ubuf__ T, 2> *dst,
                          memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src0->aligned + src0->offset;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_buf_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(src0->strides[0]) &&
         "The src strides[0] must be aligned to block.");
  assert((src0->strides[1] == 1 && dst->strides[1] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

/// Reduce src (r, a) with stride [n, 1] to dst (1, a) with stride [n, 1].
///
/// constraint:
/// 1. 'n' is a aligned to ub_block_unit.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. it needs additional tmp buffer of which size is r *
/// aligned(a, ub_block_unit)/2. For reduce xor, it needs more, the size of
/// tmp buffer is (r*aligned(a, ub_block_unit)) + r1, here r1 is the max number
/// per block that intrinsic can handles.
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
vector_reduce_ra(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
                 memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ra(src0, dst, tmp_buf, initvalue);

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t src0_stride0 = src0->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();
  const int64_t dichotomy_num = Log2(size0);
  const int64_t main_size = pow((uint64_t)2, dichotomy_num);
  const int64_t tail_size = size0 - main_size;
  int64_t num = main_size;

  memref_t<__ubuf__ T, 1> *xor_additional_tmp_buf = nullptr;
  auto tmp_offset = tmp_buf->offset;
  if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
    // Reserve remaining block tmp buffer for XOR operation.
    auto size1_align_block = CEIL_FACTOR(size1, num_per_block);
    tmp_offset =
        tmp_offset + CEIL_FACTOR(size0 * size1_align_block / 2, num_per_block);
    xor_additional_tmp_buf = tmp_buf;
  }

  if (dichotomy_num > 0) {
    num = num / 2;
    // do from src0 and src1 to (tmp_buf or dst)
    memref_t<__ubuf__ T, 2> subview_src0{src0->allocated,
                                         src0->aligned,
                                         src0->offset,
                                         {num, size1},
                                         {src0_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_src1{src0->allocated,
                                         src0->aligned,
                                         src0->offset + num * src0_stride0,
                                         {num, size1},
                                         {src0_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_tmp_buf{tmp_buf->allocated,
                                            tmp_buf->aligned,
                                            tmp_offset,
                                            {num, size1},
                                            {src0_stride0, 1}};
    vector_eltwise_vv_2d<VECOP, T>(&subview_src0, &subview_src1,
                                   dichotomy_num == 1 ? dst : &subview_tmp_buf,
                                   tmp_buf);
  } else {
    // just move src0 to dst
    memref_t<__ubuf__ T, 2> subview_src{src0->allocated,
                                        src0->aligned,
                                        src0->offset,
                                        {main_size, size1},
                                        {src0_stride0, 1}};
    vector_eltwise_vv_2d<VectorOpTy::VMAX, T>(&subview_src, &subview_src, dst);
  }

  if (tail_size > 0) {
    // do tail block for the first time of dichotomy
    // do from src and (tmp_buf or dst) to (tmp_buf or dst)
    int64_t tmp_size0 = main_size / 2;
    auto loop_num = CEIL_DIV(tail_size, tmp_size0);
    for (int64_t j = 0; j < loop_num; ++j) {
      int64_t subview_size0 =
          (j == 0) ? MIN(tail_size, tmp_size0) : tail_size - tmp_size0;
      memref_t<__ubuf__ T, 2> subview_src{
          src0->allocated,
          src0->aligned,
          src0->offset + (main_size + tmp_size0 * j) * src0_stride0,
          {subview_size0, size1},
          {src0_stride0, 1}};
      memref_t<__ubuf__ T, 2> subview_tmp_buf{tmp_buf->allocated,
                                              tmp_buf->aligned,
                                              tmp_offset,
                                              {subview_size0, size1},
                                              {src0_stride0, 1}};
      INTRINSIC(pipe_barrier, PIPE_V);
      vector_eltwise_vv_2d<VECOP, T>(
          &subview_src, dichotomy_num == 1 ? dst : &subview_tmp_buf,
          dichotomy_num == 1 ? dst : &subview_tmp_buf, tmp_buf);
    }
  }

  // step 2. processs the other times of dichotomy
  for (int64_t i = 0; i < dichotomy_num - 1; ++i) {
    // do tmp_buf and tmp_buf to (tmp_buf or dst)
    num = num / 2;
    memref_t<__ubuf__ T, 2> subview_tmp_buf0{tmp_buf->allocated,
                                             tmp_buf->aligned,
                                             tmp_offset,
                                             {num, size1},
                                             {src0_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_tmp_buf1{tmp_buf->allocated,
                                             tmp_buf->aligned,
                                             tmp_offset + num * src0_stride0,
                                             {num, size1},
                                             {src0_stride0, 1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vv_2d<VECOP, T>(
        &subview_tmp_buf0, &subview_tmp_buf1,
        i == dichotomy_num - 2 ? dst : &subview_tmp_buf0, tmp_buf);
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_reduce_ra<ReduceOpTy::REDUCE_OR, uint8_t>(
    memref_t<__ubuf__ uint8_t, 2> *src0, memref_t<__ubuf__ uint8_t, 2> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  // convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // vector_eltwise_vv_2d does not support uint8_t, so view src as int16 to
  // process.
  view_as<uint8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  vector_reduce_ra<ReduceOpTy::REDUCE_OR, int16_t>(&src0_as_int16, &dst_as_int16,
                                                   &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_reduce_ra<ReduceOpTy::REDUCE_AND, uint8_t>(
    memref_t<__ubuf__ uint8_t, 2> *src0, memref_t<__ubuf__ uint8_t, 2> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  // convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // vector_eltwise_vv_2d does not support uint8_t, so view src as int16 to
  // process.
  view_as<uint8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  vector_reduce_ra<ReduceOpTy::REDUCE_AND, int16_t>(&src0_as_int16, &dst_as_int16,
                                                    &tmp_as_int16, (int16_t)initvalue);
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) T
get_scalar_operation_init_value(memref_t<__ubuf__ T, 2> *src,
                                int64_t col) {
  return *(src->aligned + src->offset + col * (src->strides[1]));
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_reduce_ra(memref_t<__ubuf__ T, 2> *src0,
                 memref_t<__ubuf__ T, 2> *dst) {
  WARN_SCALAR_IMPL("reduceRA");
  const int64_t src0_size0 = src0->sizes[0];
  const int64_t src0_size1 = src0->sizes[1];
  const int64_t src0_stride0 = src0->strides[0];
  const int64_t src0_stride1 = src0->strides[1];
  const int64_t dst_stride1 = dst->strides[1];

  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  for (int64_t col = 0; col < src0_size1; ++col) {
    __ubuf__ T *dst_val_ptr = dst_ptr + col * dst_stride1;
    *dst_val_ptr = get_scalar_operation_init_value<OP>(src0, col);

    for (int64_t row = 1; row < src0_size0; ++row) {
      __ubuf__ T *cur_src0_ptr = src0_ptr + (row * src0_stride0 + col * src0_stride1);
      *dst_val_ptr = reduction_scalar_operation<OP, T>(*dst_val_ptr, *cur_src0_ptr);
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_stride_size_aligned_to_block(memref_t<__ubuf__ T, 2> *src0,
                                memref_t<__ubuf__ T, 2> *dst) {
  if (!is32ByteAligned<T>(src0->strides[0])) {
    return false;
  }

  if ((src0->strides[1] != 1) || (dst->strides[1] != 1)) {
    return false;
  }
  return true;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_offset_address_aligned_to_block(memref_t<__ubuf__ T, 2> *src0,
                                   memref_t<__ubuf__ T, 2> *dst) {
  auto src0_ptr = src0->aligned + src0->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  return (isAddress32ByteAligned(src0_ptr) &&
          isAddress32ByteAligned(dst_ptr));
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_reduce_ra(memref_t<__ubuf__ T, 2> *src0,
                 memref_t<__ubuf__ T, 2> *dst,
                 int64_t element_nums_offset_to_aligned) {
  memref_t<__ubuf__ T, 2> scalar_src0 = *src0;
  memref_t<__ubuf__ T, 2> scalar_dst_value = *dst;
  get_scalar_memref_t(element_nums_offset_to_aligned, src0, &scalar_src0);
  get_scalar_memref_t(element_nums_offset_to_aligned, dst, &scalar_dst_value);
  scalar_reduce_ra<OP>(&scalar_src0, &scalar_dst_value);
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_reduce_ra(memref_t<__ubuf__ T, 2> *src0,
                 memref_t<__ubuf__ T, 2> *dst,
                 memref_t<__ubuf__ T, 1> *tmp_buf,
                 T initvalue,
                 int64_t element_nums_offset_to_aligned) {
  memref_t<__ubuf__ T, 2> vector_src0 = *src0;
  memref_t<__ubuf__ T, 2> vector_dst_value = *dst;
  get_vector_memref_t(element_nums_offset_to_aligned, src0, &vector_src0);
  get_vector_memref_t(element_nums_offset_to_aligned, dst, &vector_dst_value);

  vector_reduce_ra<OP>(&vector_src0, &vector_dst_value, tmp_buf, initvalue);
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ra(memref_t<__ubuf__ T, 2> *src0,
          memref_t<__ubuf__ T, 2> *dst,
          memref_t<__ubuf__ T, 1> *tmp_buf,
          T initvalue) {
  bool is_stride_size_aligned = is_stride_size_aligned_to_block<T>(src0, dst);
  bool is_offset_address_aligned = is_offset_address_aligned_to_block<T>(src0, dst);
  bool is_same_offset_aligned = reduction_is_same_element_nums_to_move_offset_aligned<T>(src0->offset, dst->offset);

  // The stride sizes of source and dest are aligned.
  if (is_stride_size_aligned) {
    // The address of source and dest with the offset is 32-byte aligned.
    if(is_offset_address_aligned) {
      vector_reduce_ra<OP>(src0, dst, tmp_buf, initvalue);
      return;
    }

    // The address of source and dest with the offset is not 32-byte aligned.
    // After an additional same offset is added, the addresses can be 32-byte aligned.
    if (is_same_offset_aligned) {
      // source and dest. The unaligned part is calculated using the scalar,
      // and the aligned part is calculated using the vector.
      int64_t element_nums_offset_to_aligned = reduction_element_nums_to_move_offset_aligned<T>(src0->offset);
      scalar_reduce_ra<OP>(src0, dst, element_nums_offset_to_aligned);

      // If the scalar has processed all data, the vector does not need to be called for processing.
      auto last_dim_ub_element_nums = src0->sizes[1] * src0->strides[1];
      if (last_dim_ub_element_nums < element_nums_offset_to_aligned) {
        return;
      }
      vector_reduce_ra<OP>(src0, dst, tmp_buf, initvalue, element_nums_offset_to_aligned);
      return;
    }

    // After an additional same offset is added, the addresses can not be 32-byte aligned.
    scalar_reduce_ra<OP>(src0, dst);
    return;
  }

  // The stride sizes of source and dest are not aligned.
  scalar_reduce_ra<OP>(src0, dst);
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ra, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int64_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int64_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint64_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int64_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint64_t)
}