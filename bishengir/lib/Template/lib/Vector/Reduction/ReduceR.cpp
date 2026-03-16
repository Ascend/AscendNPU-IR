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
check_inputs_of_reduce_r(memref_t<__ubuf__ T, 1> *src0,
                         memref_t<__ubuf__ T, 1> *dst,
                         memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_buf_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(src0->strides[0] == 1 && "The src must be continuous vector.");
#endif
}

/// reduce src (r,) to dst (1,), here r is reduction axis.
/// For float reduce_sum/min/max, we can use vector cross intrinsic, namely
/// vcadd/vcmin/vcmax.
///
/// constraint:
/// 1. dim of src/dst must be 1.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. r <= r0, if r > r0, dst will be overwritten and user should reserve
/// r / r0 * 32B more size for dst. here r0 is the max number per repeat that
/// intrinsic can handles.
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_by_vc(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
               T initvalue) {
  // check T and op type
  static_assert(
      ((std::is_same<T, half>::value || std::is_same<T, float>::value) &&
       (OP == ReduceOpTy::REDUCE_SUM || OP == ReduceOpTy::REDUCE_MIN ||
        OP == ReduceOpTy::REDUCE_MAX)) &&
      "reduce_r_enablevc not support this data type or operation type");
  const int64_t size0 = src0->sizes[0];
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0x0, size0);

  // use vector cross intrinsic, namely vcadd/vcmin/vcmax through one
  // instruction in counting mode.
  reduce_intrin<OP, T, true, Order_t::ONLY_VALUE>(
      reduce_intrin_args<T>{dst_ptr,
                            nullptr, // dst_index, not work
                            src0_ptr,
                            1,   // repeat, not work
                            1,   // dst repeat stride
                            1,   // src block stride
                            8}); // src repeat stride
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    if constexpr (sizeof(T) == 4) {
      *(__ubuf__ int32_t *)dst_ptr = GET_ACC_VAL();
    } else if constexpr (sizeof(T) == 2) {
      *(__ubuf__ int16_t *)dst_ptr = GET_ACC_VAL();
    } else {
      static_assert((sizeof(T) == 4 || sizeof(T) == 2) &&
                    "unsupport reduce dtype.");
    }
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN ||
                       OP == ReduceOpTy::REDUCE_MAX) {
    if constexpr (sizeof(T) == 4) {
      *(__ubuf__ int32_t *)dst_ptr = GET_MAX_MIN_CNT();
    } else if constexpr (sizeof(T) == 2) {
      *(__ubuf__ int16_t *)dst_ptr = GET_MAX_MIN_CNT();
    } else {
      static_assert((sizeof(T) == 4 || sizeof(T) == 2) &&
                    "unsupport reduce dtype.");
    }
  } else {
    static_assert((OP == ReduceOpTy::REDUCE_SUM ||
                   OP == ReduceOpTy::REDUCE_MIN ||
                   OP == ReduceOpTy::REDUCE_MAX) &&
                  "unsupport reduce op.");
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// reduce src (r,) to dst (1,), here r is reduction axis.
/// For float reduce_sum/min/max, we can use vector cross intrinsic, namely
/// vcadd/vcmin/vcmax.
///
/// constraint:
/// 1. dim of src/dst must be 1.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. the tmp buffer size is as follows:
///    For reduce_min/reduce_max:
///        r <= r0: tmp_buf_size = 0
///        r > r0: tmp_buf_size = r0
///    For reduce_sum:
///        r <= r1:
///            r == 2^n: tmp_buf_size = 0
///            r != 2^n: tmp_buf_size = aligned(r, ub_block_unit) / 2
///        r1 < r <= r0: tmp_buf_size = 0
///        r > r0:
///            tmp_buf_size = max(r0, aligned(r, ub_block_unit) / 2)
/// 4. r0 is the max number per repeat that intrinsic can handles.
/// 5. r1 is the max number per block that intrinsic can handles.
///
/// \param initvalue: The initvalue value is as follows
///             float16             float32             int16_t          int32_t
/// reduce_sum:  0                   0                   NA               NA
/// reduce_min:  HALF_INF            FLOAT_INF           NA               NA
/// reduce_max:  -HALF_INF           -FLOAT_INF          NA               NA
/// reduce_prod: NA                  NA                  NA               NA
template <ReduceOpTy OP, typename T,
          typename = typename std::enable_if<(std::is_same<half, T>() ||
                                              std::is_same<float, T>())>::type,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_SUM || OP == ReduceOpTy::REDUCE_MIN ||
               OP == ReduceOpTy::REDUCE_MAX)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_r_core(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
              memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // check T and op type
  static_assert(
      ((std::is_same<T, half>::value || std::is_same<T, float>::value) &&
       (OP != ReduceOpTy::REDUCE_PROD && OP != ReduceOpTy::REDUCE_XOR)) &&
      "reduce_r without tmp_buf not support this data type or operation type");
  const int64_t size0 = src0->sizes[0];
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;

  // Input parameter constraints assert.
  check_inputs_of_reduce_r(src0, dst, tmp_buf, initvalue);

  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
    memref_t<__ubuf__ T, 2> src_2d{
        src0->allocated,
        src0->aligned,
        src0->offset,
        {1, src0->sizes[0]},
        {CEIL_FACTOR(src0->sizes[0], num_per_block), 1}};
    memref_t<__ubuf__ T, 2> dst_2d{dst->allocated,
                                   dst->aligned,
                                   dst->offset,
                                   {1, dst->sizes[0]},
                                   {dst->strides[0], 1}};
    // reduce src (1, r) to temp buffer (1, r0) by dichotomy and reduce
    // temp buffer (1, r0) to dst (1, 1) with strides [1, 1] by
    // vcmax/vcmin/vcadd with one repeat, here r0 is the max number per repeat
    // that intrinsic can handles.
    reduce_ar<OP, T>(&src_2d, &dst_2d, tmp_buf, initvalue);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN ||
                       OP == ReduceOpTy::REDUCE_MAX) {
    constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();
    constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
    if (size0 > num_per_repeat) {
      memref_t<__ubuf__ T, 1> subview_tmp{tmp_buf->allocated,
                                          tmp_buf->aligned,
                                          tmp_buf->offset,
                                          {num_per_repeat},
                                          {1}};
      // Attach initial value to temp buffer (r0,), here r0 is the max number
      // per repeat that intrinsic can handles.
      auto tmp_buf_ptr = subview_tmp.aligned + subview_tmp.offset;
      brc_scalar_core_1d(initvalue, tmp_buf_ptr, num_per_repeat);
      INTRINSIC(pipe_barrier, PIPE_V);
      reduceRToR0<VECOP, T, true>(src0, &subview_tmp, nullptr);
      INTRINSIC(pipe_barrier, PIPE_V);
      reduce_r_by_vc<OP, T>(&subview_tmp, dst, initvalue);
    } else {
      reduce_r_by_vc<OP, T>(src0, dst, initvalue);
    }
  }
}

/// reduce src (r,) to dst (1,), here r is reduction axis.
/// For int reduce_sum/min/max or any type of reduce_prod, there is no
/// corresponding vector cross intrinsic and we can only use vector element
/// intrinsic by dichotomy to reduce (r0,) to (1,).
///
/// constraint:
/// 1. dim of src/dst must be 1.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. the tmp buffer size is as follows:
///    * r > num_per_repeat:
///        For reduce_xor: tmp_buf_size = r0 * 2 + r1 / 2
///        For the others: tmp_buf_size = r0 + r1 / 2
///    * r <= num_per_repeat:
///        For reduce_xor: tmp_buf_size = 2.5 * src_size
///        For the others: tmp_buf_size = 1.5 * src_size
/// 4. r0 is the max number per repeat that intrinsic can handles.
/// 5. r1 is the max number per block that intrinsic can handles.
///
/// \param initvalue: The initvalue value is as follows
///             float16    float32    int8_t    uint8_t    int16_t    uint16_t
///             int32_t    uint32_t   int64_t   uint64_t
/// reduce_sum:  NA         NA         NA        NA         0          NA
///              0          NA         NA        NA
/// reduce_min:  NA         NA         NA        NA         INT8_MAX   NA
///              INT32_MAX  NA         NA        NA
/// reduce_max:  NA         NA         NA        NA         INT8_MIN   NA
///              INT32_MIN  NA         NA        NA
/// reduce_prod: 1.0e+00f   1.0e+00f   NA        NA         1          NA
///              1          NA         NA        NA
/// reduce_xor:  NA         NA         0         NA         0          NA
///              0          NA         0         NA
/// reduce_or:   NA         NA         0         0          0          0
///              0          0          0         0
/// reduce_and:  NA         NA         1         1          1          1
///              1          1          1         1
template <
    ReduceOpTy OP, typename T,
    typename = typename std::enable_if<
        (std::is_same<int8_t, T>() || std::is_same<int16_t, T>() ||
         std::is_same<int32_t, T>() || std::is_same<int64_t, T>() ||
         std::is_same<uint8_t, T>() || std::is_same<uint16_t, T>() ||
         std::is_same<uint32_t, T>() || std::is_same<uint64_t, T>()) ||
        (OP == ReduceOpTy::REDUCE_PROD || OP == ReduceOpTy::REDUCE_XOR ||
         OP == ReduceOpTy::REDUCE_OR || OP == ReduceOpTy::REDUCE_AND)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_r_core(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
              memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_r(src0, dst, tmp_buf, initvalue);

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  memref_t<__ubuf__ T, 2> src_2d{
      src0->allocated,
      src0->aligned,
      src0->offset,
      {1, src0->sizes[0]},
      {CEIL_FACTOR(src0->sizes[0], num_per_block), 1}};
  memref_t<__ubuf__ T, 2> dst_2d{dst->allocated,
                                 dst->aligned,
                                 dst->offset,
                                 {1, dst->sizes[0]},
                                 {dst->strides[0], 1}};
  reduce_ar<OP, T>(&src_2d, &dst_2d, tmp_buf, initvalue);
}

/// USING VCG
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_vcg_core(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
             memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // // Input parameter constraints assert.
  check_inputs_of_reduce_r(src0, dst, tmp_buf, initvalue);

  const int64_t size0 = src0->sizes[0];

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;

  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  const int64_t size_aligned = CEIL_FACTOR(size0, num_per_block);

  if (num_per_repeat < size0) {
    // Fast vcgadd impl
    const int64_t k = Log2(num_per_block);
    const int64_t div_num =
        divisions_needed_by_pow2(size_aligned, num_per_repeat, k);
    const int64_t divisor = pow(num_per_block, div_num);
    const int64_t size_tmp = (size0 + divisor - 1) / divisor;
    reduceRByBlocks<OP, T>(src0, tmp_buf, initvalue);
    INTRINSIC(pipe_barrier, PIPE_V);
    memref_t<__ubuf__ T, 1> tmp_buf_1d{
        tmp_buf->allocated, tmp_buf->aligned, tmp_buf->offset, {size_tmp}, {1}};
    reduce_r_by_vc<OP, T>(&tmp_buf_1d, dst, initvalue);
  } else {
    INTRINSIC(pipe_barrier, PIPE_V);
    reduce_r_by_vc<OP, T>(src0, dst, initvalue);
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_core_on_scalar(memref_t<__ubuf__ T, 1> *src0,
                        memref_t<__ubuf__ T, 1> *dst,
                        int64_t scalar_element_num, bool need_merge) {
  WARN_SCALAR_IMPL("reduceR");
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_value_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  if (need_merge) {
    *dst_value_ptr = reduction_scalar_operation<OP, T>(*dst_value_ptr, *src_ptr);
  } else {
    *dst_value_ptr = *src_ptr;
  }

  if (scalar_element_num > 1) {
    for (int64_t i = 1; i < scalar_element_num; i++){
      T val = *(src_ptr + i * src0->strides[0]);
      *dst_value_ptr = reduction_scalar_operation<OP, T>(*dst_value_ptr, val);
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_vcg(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
             memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  int64_t scalar_element_num = reduction_get_element_nums_on_scalar_1d<T>(src0);
  int64_t vector_element_num = src0->sizes[0] - scalar_element_num;

  if (vector_element_num != 0) [[likely]] {
    memref_t<__ubuf__ T, 1> src0_subview = *src0;
    src0_subview.offset += scalar_element_num;
    src0_subview.sizes[0] = vector_element_num;
    reduce_r_vcg_core<OP, T>(&src0_subview, dst, tmp_buf, initvalue);
  }

  if (scalar_element_num != 0) {
    check_inputs_of_reduce_r(src0, dst, tmp_buf, initvalue);
    bool need_merge = !(vector_element_num == 0);
    reduce_r_core_on_scalar<OP, T>(src0, dst, scalar_element_num, need_merge);
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
         memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  int64_t scalar_element_num = reduction_get_element_nums_on_scalar_1d<T>(src0);
  int64_t vector_element_num = src0->sizes[0] - scalar_element_num;

  if (vector_element_num != 0) [[likely]] {
    memref_t<__ubuf__ T, 1> src0_subview = *src0;
    src0_subview.offset += scalar_element_num;
    src0_subview.sizes[0] = vector_element_num;
    reduce_r_core<OP, T>(&src0_subview, dst, tmp_buf, initvalue);
  }

  if (scalar_element_num != 0) {
    check_inputs_of_reduce_r(src0, dst, tmp_buf, initvalue);
    bool need_merge = !(vector_element_num == 0);
    reduce_r_core_on_scalar<OP, T>(src0, dst, scalar_element_num, need_merge);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce r, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_ENABLEVC_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, float)
REGISTE_ENTIRE_REDUCE_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, int16_t)

REGISTE_ENTIRE_REDUCE_ENABLEVC_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, float)
REGISTE_ENTIRE_REDUCE_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, int16_t)

REGISTE_ENTIRE_REDUCE_ENABLEVC_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, float)
REGISTE_ENTIRE_REDUCE_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, int16_t)

REGISTE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, half)
REGISTE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, float)
REGISTE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, int16_t)

REGISTE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int8_t)
REGISTE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int16_t)
REGISTE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int64_t)

REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int8_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint8_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int16_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint16_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int64_t)
REGISTE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint64_t)

REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int8_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint8_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int16_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint16_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint32_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int64_t)
REGISTE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint64_t)
}
