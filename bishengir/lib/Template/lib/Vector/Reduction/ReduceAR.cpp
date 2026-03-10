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


template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) T
get_scalar_operation_init_value(__ubuf__ T *src0_ptr,
                                int64_t offset) {
  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    return 0;
  } else if constexpr (OP == ReduceOpTy::REDUCE_MAX) {
    return *(src0_ptr + offset);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN) {
    return *(src0_ptr + offset);
  } else if constexpr (OP == ReduceOpTy::REDUCE_PROD) {
    return 1;
  } else if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
    return 0;
  } else if constexpr (OP == ReduceOpTy::REDUCE_OR) {
    return 0;
  } else if constexpr (OP == ReduceOpTy::REDUCE_AND) {
    return *(src0_ptr + offset);
  } else {
    static_assert("unsupported reduce op");
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ void reduce_scalar_iml(memref_t<__ubuf__ T, 2> *src,
                               memref_t<__ubuf__ T, 2> *dst_value,
                               int64_t size0, int64_t size1,
                               T /*initvalue*/,
                               bool USE_DST_FOR_INIT = false) {
  cce::printf("Warning: This implementation uses scalar instructions, which may result in suboptimal performance");
  if (size1 <= 0) return;

  __ubuf__ T* src_ptr       = src->aligned + src->offset;
  __ubuf__ T* dst_value_ptr = dst_value->aligned + dst_value->offset;

  const int64_t src_stride0  = src->strides[0];
  const int64_t src_stride1  = src->strides[1];
  const int64_t dst_stride0  = dst_value->strides[0];

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < size0; ++i) {
    T acc = USE_DST_FOR_INIT ? *(dst_value_ptr + i * dst_stride0) :
            get_scalar_operation_init_value<OP, T>(src_ptr, i * src_stride0);
    for (int64_t j = 0; j < size1; ++j) {
      T val = *(src_ptr + i * src_stride0 + j * src_stride1);
      acc = reduction_scalar_operation<OP, T>(acc, val);
    }
    *(dst_value_ptr + i * dst_stride0) = acc;
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vcpaddCountMode(__ubuf__ T *dst_ptr, __ubuf__ T *src_ptr, int64_t length) {
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, length);
  INTRINSIC(vcpadd, dst_ptr, src_ptr,
            1, // repeat
            1, // dst_block_stride
            1, // src_block_stride
            8  // src_repeat_stride
  );
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_reduce_ar(memref_t<__ubuf__ T, 2> *src0,
                          memref_t<__ubuf__ T, 2> *dst,
                          memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src0->aligned + src0->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_buf_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(src0->strides[1] == 1 && "The src last dim must be continuous.");
  assert((isSizeAlignedToBlock<T>(dst->strides[0]) || dst->strides[0] == 1) &&
         "The dst strides[0] must be aligned to block or 1.");
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  // Except for vcpadd and vcgadd, the remaining stride0 needs to be aligned
  // with the block.
  if (!((std::is_same<half, T>() || std::is_same<float, T>()) &&
        OP == ReduceOpTy::REDUCE_SUM && src0->sizes[0] <= num_per_block &&
        src0->sizes[1] == src0->strides[0] && dst->strides[0] == 1)) {
    assert(isSizeAlignedToBlock<T>(src0->strides[0]) &&
           "The src strides[0] must be aligned to block.");
  }
#endif
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ar_vcg_unalign(memref_t<__ubuf__ T, 2> *src0,
                      memref_t<__ubuf__ T, 2> *dst,
                      memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t src_stride0 = src0->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  int64_t tmp_buf_1d_row_size =
      CEIL_FACTOR(src_stride0 / num_per_block, num_per_block);
  for (int64_t i = 0; i < size0; i++) {
    memref_t<__ubuf__ T, 1> subview_src0{src0->allocated,
                                         src0->aligned,
                                         src0->offset + i * src_stride0,
                                         {size1},
                                         {1}};
    memref_t<__ubuf__ T, 1> subview_dst{
        dst->allocated, dst->aligned, dst->offset + i * dst_stride0, {1}, {1}};
    memref_t<__ubuf__ T, 1> tmp_buf_1d{tmp_buf->allocated,
                                       tmp_buf->aligned,
                                       tmp_buf->offset +
                                           i * (tmp_buf_1d_row_size),
                                       {tmp_buf_1d_row_size},
                                       {1}};
    reduce_r_vcg<OP, T>(&subview_src0, &subview_dst, &tmp_buf_1d, initvalue);
  }
  return;
}
 
/// reduce src (a, r) with stride [n1, 1] to dst (a, 1) with stride [n2, 1],
/// here r is reduction axis, a is non-reduce axis. dtype is float and op is
/// reduce_sum/reduce_min/reduce_max reduce by vcgadd/vcgmin/vcgmax.
///
/// USING VCG
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ar_vcg(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
              memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ar<OP, T>(src0, dst, tmp_buf, initvalue);
 
  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t src_stride0 = src0->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
 
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
 
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
 
  if (num_per_repeat < size1) {
    if (size1 != src_stride0) {
      // Reduce each row seperately. lower performance
      reduce_ar_vcg_unalign<OP, T>(src0, dst, tmp_buf, initvalue);
      return;
    }
    const int64_t k = Log2(num_per_block);
    const int64_t div_num =
        divisions_needed_by_pow2(src_stride0, num_per_repeat, k);
    const int64_t divisor = pow(num_per_block, div_num);
    const int64_t size_tmp = (size1 + divisor - 1) / divisor;
    // Fast vcg reduceAR Impl
    memref_t<__ubuf__ T, 2> tmp_buf_2d{tmp_buf->allocated,
                                       tmp_buf->aligned,
                                       tmp_buf->offset,
                                       {size0, size_tmp},
                                       {size_tmp, 1}};
    // note: size tmp may not be aligned
    reduceARByBlocks<OP, T>(src0, &tmp_buf_2d, initvalue);
    // reduce temp buffer (a, r0) to dst (a, 1) with strides [n, 1] by
    // vcmax/vcmin/vcadd, here r0 is <= max number per repeat that intrinsic
    // can handles, n is 1 or aligned to a block.
    INTRINSIC(pipe_barrier, PIPE_V);
    if (dst->strides[0] == 1) {
      reduceAR0ToA<OP, T, false>(&tmp_buf_2d, dst, nullptr);
    } else {
      reduceAR0ToAByLoopAAxis<OP, T, false>(&tmp_buf_2d, dst, nullptr);
    }
  } else if (size1 <= num_per_repeat) {
    reduce_ar_core<OP, T>(src0, dst, tmp_buf, initvalue);
  }
}
 
/// reduce src (a, r) with stride [n1, 1] to dst (a, 1) with stride [n2, 1],
/// here r is reduction axis, a is non-reduce axis. dtype is float and op is
/// reduce_sum/reduce_min/reduce_max reduce by vcadd/vcmin/vcmax.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. the tmp buffer size is as follows:
///    * r <= num_per_block:
///        r == 2^n: tmp_buf_size = 0
///        r != 2^n: tmp_buf_size = a * aligned(r, ub_block_unit) / 2
///    * num_per_block < r <= num_per_repeat:
///        tmp_buf_size = 0
///    * r > num_per_repeat:
///        tmp_buf_size =
///            max(a * num_per_repeat, a * aligned(r, ub_block_unit) / 2)
/// 4. 'n1' is r aligned to ub_block_unit, 'n2' is 1 or aligned to a block.
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
reduce_ar_core(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
               memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ar<OP, T>(src0, dst, tmp_buf, initvalue);

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t src_stride0 = src0->strides[0];
  const int64_t dst_stride0 = dst->strides[0];

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;

  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();

  if (num_per_repeat < size1) {
    memref_t<__ubuf__ T, 2> tmp_buf_2d{tmp_buf->allocated,
                                       tmp_buf->aligned,
                                       tmp_buf->offset,
                                       {size0, num_per_repeat},
                                       {num_per_repeat, 1}};
    // reduce src (a, r) to temp buffer (a, r0) by dichotomy.
    // here r0 is the max number per repeat that intrinsic can handles.
    reduceARxToARyByDichotomy<OP, T>(src0, &tmp_buf_2d, nullptr);

    // reduce temp buffer (a, r0) to dst (a, 1) with strides [n, 1] by
    // vcmax/vcmin/vcadd, here r0 is the max number per repeat that intrinsic
    // can handles, n is 1 or aligned to a block.
    INTRINSIC(pipe_barrier, PIPE_V);
    if (dst->strides[0] == 1) {
      reduceAR0ToA<OP, T, false>(&tmp_buf_2d, dst, nullptr);
    } else {
      reduceAR0ToAByLoopAAxis<OP, T, false>(&tmp_buf_2d, dst, nullptr);
    }
  } else if (size1 <= num_per_block && size1 == src_stride0 &&
             dst_stride0 == 1 && OP == ReduceOpTy::REDUCE_SUM) [[unlikely]] {
    int64_t length = size0 * size1;
    if (size1 == num_per_block) {
      // reduce (a, r) to (a, 1) by vcgadd without additional temp buffer
      // when r is ub_block_unit
      INTRINSIC_NO_ARGS(set_mask_count);
      INTRINSIC(set_vector_mask, 0, length);
      INTRINSIC(vcgadd, dst_ptr, src_ptr,
                1, // repeat
                1, // dst_repeat_stride
                1, // src_block_stride
                8  // src_repeat_stride
      );
      INTRINSIC_NO_ARGS(set_mask_norm);
    } else if ((size1 & (size1 - 1)) == 0 && size1 < num_per_block) {
      // reduce (a, r) to (a, 1) by vcpadd if r < ub_block_unit and
      // extent == stride, tmp buffer needs half memory of src
      if (size1 == 2) {
        vcpaddCountMode(dst_ptr, src_ptr, length);
      } else {
        for (int64_t i = size1; i > 1; i >>= 1) {
          if (i == size1) {
            vcpaddCountMode<T>(tmp_buf_ptr, src_ptr, length);
          } else if (i == 2) {
            vcpaddCountMode<T>(dst_ptr, tmp_buf_ptr, length);
          } else {
            vcpaddCountMode<T>(tmp_buf_ptr, tmp_buf_ptr, length);
          }
          INTRINSIC(pipe_barrier, PIPE_V);
          length >>= 1;
        }
      }
    }
  } else {
    // reduce src (a, r) to dst (a, 1) with strides [n, 1] by vcmax/vcmin/vcadd,
    // here n is 1 or aligned to a block.
    INTRINSIC(pipe_barrier, PIPE_V);
    if (dst->strides[0] == 1) {
      reduceAR0ToA<OP, T, false>(src0, dst, nullptr);
    } else {
      reduceAR0ToAByLoopAAxis<OP, T, false>(src0, dst, nullptr);
    }
  }
}

/// reduce src (a, r) with stride [n1, 1] to dst (a, 1) with stride [n2, 1],
/// here r is reduction axis, a is non-reduce axis. dtype is int32 or op =
/// reduce_prod reduce by element-wise vector operations.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. the tmp buffer size is as follows:
///    * r > num_per_repeat:
///        For reduce_xor: tmp_buf_size = a * r0 * 2 + a * r1
///        For the others: tmp_buf_size = a * r0 + a * r1
///    * r <= num_per_repeat:
///        For reduce_xor: tmp_buf_size = 3 * a * aligned(r, ub_block_unit)
///        For the others: tmp_buf_size = 2 * a * aligned(r, ub_block_unit)
/// 4. r0 is the max number per repeat that intrinsic can handles, r1 is the
/// max number per block that intrinsic can handles.
/// 5. 'n1' is r aligned to ub_block_unit, 'n2' is 1 or aligned to a block.
///
/// \param initvalue: The initvalue value is as follows
///             float16    float32    int8_t    uint8_t    int16_t    uint16_t
///             int32_t    uint32_t   int64_t   uint64_t
/// reduce_sum:  NA         NA         NA        NA         0          NA
///              0          NA         NA        NA
/// reduce_min:  NA         NA         NA        NA         INT16_MAX  NA
///              INT32_MAX  NA         NA        NA
/// reduce_max:  NA         NA         NA        NA         INT16_MIN  NA
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
    ReduceOpTy OP, typename T, int64_t REDUCE_TO = 1, int64_t MOVE_TO_DST = 1,
    typename = typename std::enable_if<
        (std::is_same<int8_t, T>() || std::is_same<uint8_t, T>() ||
         std::is_same<int16_t, T>() || std::is_same<uint16_t, T>() ||
         std::is_same<int32_t, T>() || std::is_same<uint32_t, T>() ||
         std::is_same<int64_t, T>() || std::is_same<uint64_t, T>()) ||
        (OP == ReduceOpTy::REDUCE_PROD || OP == ReduceOpTy::REDUCE_XOR ||
         OP == ReduceOpTy::REDUCE_OR || OP == ReduceOpTy::REDUCE_AND)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
               memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  // Input parameter constraints assert.
  check_inputs_of_reduce_ar<OP, T>(src0, dst, tmp_buf, initvalue);

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();
  memref_t<__ubuf__ T, 1> *xor_additional_tmp_buf = nullptr;
  auto tmp_offset = tmp_buf->offset;
  __ubuf__ T *tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;

  if (num_per_repeat < size1) {
    if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
      // Reserve remaining block tmp buffer for XOR operation.
      xor_additional_tmp_buf = tmp_buf;
      tmp_offset = tmp_offset + size0 * num_per_repeat;
      tmp_buf_ptr = tmp_buf->aligned + tmp_offset;
    }

    // tmp_buf_for_block_reduce is a (a*r1/2) subview of tmp_buf with
    // offset (a*r0 or a*r0*2 depends on reduceop). here r0 is the max number
    // per repeat that intrinsic can handles. r1 is the max number per block
    // that intrinsic can handles.
    __ubuf__ T *tmp_buf_for_block_reduce_ptr =
        tmp_buf_ptr + size0 * num_per_repeat;

    // Attach initial value to temp buffer (a, r0), here r0 is the max number
    // per repeat that intrinsic can handles.
    brc_scalar_core_1d(initvalue, tmp_buf_ptr, size0 * num_per_repeat);

    memref_t<__ubuf__ T, 2> tmp_buf_2d{tmp_buf->allocated,
                                       tmp_buf->aligned,
                                       tmp_offset,
                                       {size0, num_per_repeat},
                                       {num_per_repeat, 1}};
    // reduce src (a,r) to temp buffer (a,r0) by element-wise vector operation,
    // here r0 is the max number per repeat that intrinsic can handles.
    reduceARToAR0<VECOP, T>(src0, &tmp_buf_2d, xor_additional_tmp_buf);

    memref_t<__ubuf__ T, 2> subview_tmp_buf_2d{tmp_buf->allocated,
                                               tmp_buf->aligned,
                                               tmp_offset,
                                               {size0, num_per_block},
                                               {num_per_block, 1}};
    // reduce temp buffer (a, r0) to temp buffer (a, r1) by element-wise vector
    // operation, here r0 is the max number per repeat that intrinsic can
    // handles, r1 is the max number per block that inrtinsic can handles.
    reduceARxToARyByDichotomy<OP, T, num_per_repeat, num_per_block>(
        &tmp_buf_2d, &subview_tmp_buf_2d, xor_additional_tmp_buf);

    // reduce temp buffer (a, r1) to dst (a, 1) by element-wise vector
    // operation, here r1 is the max number per block that inrtinsic can
    // handles.
    reduceAR1ToAByDichotomy<OP, T, REDUCE_TO, MOVE_TO_DST>(
        &tmp_buf_2d, dst, xor_additional_tmp_buf, tmp_buf_for_block_reduce_ptr);
  } else {
    if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
      auto size1_align_block = CEIL_FACTOR(size1, num_per_block);
      // Reserve remaining block tmp buffer for XOR operation.
      xor_additional_tmp_buf = tmp_buf;
      tmp_offset = tmp_offset + size0 * size1_align_block;
      tmp_buf_ptr = tmp_buf->aligned + tmp_offset;
    }

    memref_t<__ubuf__ T, 2> tmp_buf_2d{tmp_buf->allocated,
                                       tmp_buf->aligned,
                                       tmp_offset,
                                       {size0, num_per_block},
                                       {num_per_block, 1}};
    // tmp_buf_for_block_reduce is a (a*r1/2) subview of tmp_buf, here r1 is
    // the max number per block that intrinsic can handles.
    __ubuf__ T *tmp_buf_for_block_reduce_ptr;
    if (size1 > num_per_block) {
      // reduce src (a, r) to temp buffer (a, r1) by element-wise
      // vector operation, here r1 is the max number per block that
      // can handles.
      reduceARxToARyByDichotomy<OP, T>(src0, &tmp_buf_2d,
                                       xor_additional_tmp_buf);
      tmp_buf_for_block_reduce_ptr =
          tmp_buf_ptr + size0 * (&tmp_buf_2d)->strides[0];
    } else {
      tmp_buf_for_block_reduce_ptr = tmp_buf_ptr + size0 * num_per_block;

      // Attach initial value to temp buffer (a, r1),
      // here r1 is the max number per block that intrinsic can handles.
      brc_scalar_core_1d(initvalue, tmp_buf_ptr, size0 * num_per_block);

      // copy src (a, r) to temp buffer (a, r1) by element-wise vector
      // operation, here r1 is the max number per block that intrinsic can
      // handles.
      INTRINSIC(pipe_barrier, PIPE_V);
      tmp_buf_2d.sizes[1] = size1;
      vector_eltwise_vv_2d<VECOP, T>(src0, &tmp_buf_2d, &tmp_buf_2d, tmp_buf);
      tmp_buf_2d.sizes[1] = num_per_block;
    }

    // reduce temp buffer (a, r1) to dst (a, 1) by element-wise vector
    // operation, here r1 is the max number per block that intrinsic can
    // handles.
    reduceAR1ToAByDichotomy<OP, T, REDUCE_TO, MOVE_TO_DST>(
        &tmp_buf_2d, dst, xor_additional_tmp_buf, tmp_buf_for_block_reduce_ptr);
  }
}

template <ReduceOpTy OP,
          typename = typename std::enable_if<
              (OP == ReduceOpTy::REDUCE_XOR || OP == ReduceOpTy::REDUCE_OR ||
               OP == ReduceOpTy::REDUCE_AND)>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core(memref_t<__ubuf__ int8_t, 2> *src0,
               memref_t<__ubuf__ int8_t, 2> *dst,
               memref_t<__ubuf__ int8_t, 1> *tmp_buf, int8_t initvalue) {

  if (OP == ReduceOpTy::REDUCE_AND && src0->sizes[1] % 2) {
    auto src0_ptr = src0->aligned + src0->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    constexpr int8_t pad_value = 0xFF;
    for (int i = 0; i < src0->sizes[0]; i++) {
      *(src0_ptr + i * src0->strides[0] + src0->sizes[1]) = pad_value;
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  }

  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  view_as<int8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 1>(tmp_buf, &tmp_as_int16);
  // no intrinsic to process int8, so view src as int16 to process.
  // after int16 reduce, dst should be (size0, 1) x int16
  // tmp_dst as intermediate temporary output.
  memref_t<__ubuf__ int16_t, 2> tmp_dst{(__ubuf__ int16_t *)tmp_buf->allocated,
                                        (__ubuf__ int16_t *)tmp_buf->aligned,
                                        tmp_buf->offset,
                                        {dst->sizes[0], 1},
                                        {dst->strides[0], 1}};

  if (src0->sizes[1] > 2) {
    int16_t init = (OP == ReduceOpTy::REDUCE_AND) ? (int16_t)UINT16_MAX : 0;
    // no intris, view as int16 to process.
    reduce_ar_core<OP, int16_t, 1, 0>(&src0_as_int16, &tmp_dst, &tmp_as_int16,
                                      (int16_t)init);
  } else {
    view_as<int8_t, int16_t, 2>(src0, &tmp_dst);
  }

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  // reduce (a, 1)xint16 to (a, 1)xint8 by scalar operation.
  __ubuf__ int16_t *tmp_dst_ptr = tmp_dst.aligned + tmp_dst.offset;
  __ubuf__ int8_t *dst_ptr = dst->aligned + dst->offset;
  for (int64_t i = 0; i < src0->sizes[0]; ++i) {
    int16_t scalar = *(tmp_dst_ptr + i * tmp_dst.strides[0]);
    uint16_t high = (uint16_t)(scalar >> 8) & 0xFF;
    uint16_t low = scalar & 0xFF;
    if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
      *(dst_ptr + i * dst->strides[0]) = (~(high & low)) & (high | low);
    } else if constexpr (OP == ReduceOpTy::REDUCE_OR) {
      *(dst_ptr + i * dst->strides[0]) = high | low;
    } else if constexpr (OP == ReduceOpTy::REDUCE_AND) {
      *(dst_ptr + i * dst->strides[0]) = high & low;
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_XOR, int32_t>(
    memref_t<__ubuf__ int32_t, 2> *src0, memref_t<__ubuf__ int32_t, 2> *dst,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf, int32_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process int32, so view src as int16 to process.
  view_as<int32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<int32_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_XOR, int16_t, 2, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_XOR, int64_t>(
    memref_t<__ubuf__ int64_t, 2> *src0, memref_t<__ubuf__ int64_t, 2> *dst,
    memref_t<__ubuf__ int64_t, 1> *tmp_buf, int64_t initvalue) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process int64, so view src as int16 to process.
  view_as<int64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<int64_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_XOR, int16_t, 4, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <ReduceOpTy OP, typename = typename std::enable_if<
                             OP == ReduceOpTy::REDUCE_OR ||
                             OP == ReduceOpTy::REDUCE_AND>::type>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core(memref_t<__ubuf__ uint8_t, 2> *src0,
               memref_t<__ubuf__ uint8_t, 2> *dst,
               memref_t<__ubuf__ uint8_t, 1> *tmp_buf, uint8_t initvalue) {
  // convert uint8_t memref to int8 memref
  memref_t<__ubuf__ int8_t, 2> src0_as_int8;
  memref_t<__ubuf__ int8_t, 2> dst_as_int8;
  memref_t<__ubuf__ int8_t, 1> tmp_as_int8;

  view_as<uint8_t, int8_t, 2>(src0, &src0_as_int8);
  view_as<uint8_t, int8_t, 2>(dst, &dst_as_int8);
  view_as<uint8_t, int8_t, 1>(tmp_buf, &tmp_as_int8);

  reduce_ar_core<OP>(&src0_as_int8, &dst_as_int8, &tmp_as_int8,
                     (int8_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_OR, uint16_t>(
    memref_t<__ubuf__ uint16_t, 2> *src0, memref_t<__ubuf__ uint16_t, 2> *dst,
    memref_t<__ubuf__ uint16_t, 1> *tmp_buf, uint16_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process uint16, so view src as int16 to process.
  view_as<uint16_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint16_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint16_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_OR, int16_t, 1, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_AND, uint16_t>(
    memref_t<__ubuf__ uint16_t, 2> *src0, memref_t<__ubuf__ uint16_t, 2> *dst,
    memref_t<__ubuf__ uint16_t, 1> *tmp_buf, uint16_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process uint16, so view src as int16 to process.
  view_as<uint16_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint16_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint16_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_AND, int16_t, 1, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_OR, int32_t>(
    memref_t<__ubuf__ int32_t, 2> *src0, memref_t<__ubuf__ int32_t, 2> *dst,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf, int32_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process int32, so view src as int16 to process.
  view_as<int32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<int32_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_OR, int16_t, 2, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_AND, int32_t>(
    memref_t<__ubuf__ int32_t, 2> *src0, memref_t<__ubuf__ int32_t, 2> *dst,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf, int32_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process int32, so view src as int16 to process.
  view_as<int32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<int32_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_AND, int16_t, 2, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_OR, uint32_t>(
    memref_t<__ubuf__ uint32_t, 2> *src0, memref_t<__ubuf__ uint32_t, 2> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, uint32_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process uint32, so view src as int16 to process.
  view_as<uint32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint32_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_OR, int16_t, 2, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_AND, uint32_t>(
    memref_t<__ubuf__ uint32_t, 2> *src0, memref_t<__ubuf__ uint32_t, 2> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, uint32_t initvalue) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process uint32, so view src as int16 to process.
  view_as<uint32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint32_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_AND, int16_t, 2, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_OR, int64_t>(
    memref_t<__ubuf__ int64_t, 2> *src0, memref_t<__ubuf__ int64_t, 2> *dst,
    memref_t<__ubuf__ int64_t, 1> *tmp_buf, int64_t initvalue) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process int64, so view src as int16 to process.
  view_as<int64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<int64_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_OR, int16_t, 4, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_AND, int64_t>(
    memref_t<__ubuf__ int64_t, 2> *src0, memref_t<__ubuf__ int64_t, 2> *dst,
    memref_t<__ubuf__ int64_t, 1> *tmp_buf, int64_t initvalue) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process int64, so view src as int16 to process.
  view_as<int64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<int64_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_AND, int16_t, 4, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_OR, uint64_t>(
    memref_t<__ubuf__ uint64_t, 2> *src0, memref_t<__ubuf__ uint64_t, 2> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, uint64_t initvalue) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process uint64, so view src as int16 to process.
  view_as<uint64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint64_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_OR, int16_t, 4, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <>
__aiv__ __attribute__((always_inline)) void
reduce_ar_core<ReduceOpTy::REDUCE_AND, uint64_t>(
    memref_t<__ubuf__ uint64_t, 2> *src0, memref_t<__ubuf__ uint64_t, 2> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, uint64_t initvalue) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  memref_t<__ubuf__ int16_t, 1> tmp_as_int16;
  // no intrinsic to process uint64, so view src as int16 to process.
  view_as<uint64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 2>(dst, &dst_as_int16);
  view_as<uint64_t, int16_t, 1>(tmp_buf, &tmp_as_int16);

  reduce_ar_core<ReduceOpTy::REDUCE_AND, int16_t, 4, 1>(
      &src0_as_int16, &dst_as_int16, &tmp_as_int16, (int16_t)initvalue);
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vec_reduce_ar(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
          memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  if constexpr ((OP == ReduceOpTy::REDUCE_XOR || OP == ReduceOpTy::REDUCE_OR ||
                 OP == ReduceOpTy::REDUCE_AND) &&
                (std::is_same<T, int8_t>::value ||
                 std::is_same<T, uint8_t>::value)) {
    reduce_ar_core<OP>(src0, dst, tmp_buf, initvalue);
  } else {
    reduce_ar_core<OP, T>(src0, dst, tmp_buf, initvalue);
  }
}


template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ar(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
          memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue) {
  __ubuf__ T *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  const int64_t size1 = src0->sizes[1];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  const int64_t src_stride0 = src0->strides[0];
  const int64_t dst_stride0 = dst->strides[0];

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  // if stride is not aligned, fallback to scalar implemention
  if constexpr ((OP == ReduceOpTy::REDUCE_MAX || OP == ReduceOpTy::REDUCE_MIN ||
                 OP == ReduceOpTy::REDUCE_SUM) &&
                (std::is_same<T, half>::value ||
                 std::is_same<T, float>::value)){
                bool is_src_aligned = is32ByteAligned<T>(src0->strides[0]) && (src0->strides[1] == 1);
                bool is_dst_aligned = (dst->strides[1] == 1) && ((dst->strides[0] == 1) ||
                is32ByteAligned<T>(dst->strides[0]));
    bool is_special_aligned_case_use_vcgadd_vcpadd = size1 <= num_per_repeat && src_stride0 == size1 &&
    dst_stride0 == 1 && OP == ReduceOpTy::REDUCE_SUM;

    if(!(is_src_aligned && is_dst_aligned || is_special_aligned_case_use_vcgadd_vcpadd)) [[unlikely]] {
      reduce_scalar_iml<OP, T>(src0, dst, src0->sizes[0], src0->sizes[1], initvalue);
      return;
    }
  } else if constexpr ((OP == ReduceOpTy::REDUCE_XOR || OP == ReduceOpTy::REDUCE_OR ||
                        OP == ReduceOpTy::REDUCE_AND) &&
                        std::is_same<T, int8_t>::value ||
                        (OP == ReduceOpTy::REDUCE_OR ||
                        OP == ReduceOpTy::REDUCE_AND) &&
                        std::is_same<T, uint8_t>::value){
      bool is_special_aligned_case_for_uint8 = src0->sizes[0] <= 2 || is32ByteAligned<T>(src0->strides[0]) &&
      src0->strides[1] == 1;
      if(!is_special_aligned_case_for_uint8) [[unlikely]] {
        reduce_scalar_iml<OP, T>(src0, dst, src0->sizes[0], src0->sizes[1], initvalue);
        return;
      }
  } else { // uint/int (8||16||32||64) ||(or||and||xor||prod) || xor&&(int32|| int64)
    bool is_src_aligned = is32ByteAligned<T>(src0->strides[0]) && (src0->strides[1] == 1);
    bool is_dst_aligned = (dst->strides[1] == 1);

    if(!is_src_aligned && is_dst_aligned) [[unlikely]] {
      reduce_scalar_iml<OP, T>(src0, dst, src0->sizes[0], src0->sizes[1], initvalue);
      return;
    }
  }

  //if stride is aligned, offset of dst is not alined, implement the reduction operation using scalar instructions
  if (!isAddress32ByteAligned<T>(dst_ptr)) [[unlikely]] {
    reduce_scalar_iml<OP, T>(src0, dst, src0->sizes[0], src0->sizes[1], initvalue);
    return;
  }

  //if stride is aligned, offset of dst is alined, offset of src is not alined, use scalar and vec instructions
  if (!isAddress32ByteAligned<T>(src_ptr)) [[unlikely]] {
    auto address = reinterpret_cast<uintptr_t>(src_ptr);
    auto align_diff = (32 - (address & 0x1F)) & 0x1F;
    int64_t scalarNum = static_cast<int64_t>(align_diff / sizeof(T));
    int64_t offset = src0->offset;
    int64_t vectorNum = size1  - scalarNum; 

    // calculate the result of vec part
    if (vectorNum > 0){
      memref_t<__ubuf__ T, 2> subview_src{
        src0->allocated,
        src0->aligned,
        src0->offset + scalarNum,
        {src0->sizes[0], vectorNum},
        {src0->strides[0], src0->strides[1]}
      };

      INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      vec_reduce_ar<OP, T>(&subview_src, dst, tmp_buf, initvalue);
    }
    INTRINSIC(set_flag, PIPE_V,PIPE_S,  LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V,PIPE_S, LIB_EVENT_ID0);

    // calculate the result of scalar part, and gather the result of two part
    bool USE_DST_FOR_INIT = (vectorNum > 0);
    reduce_scalar_iml<OP, T>(src0, dst, src0->sizes[0], scalarNum, initvalue, USE_DST_FOR_INIT);
    return;
  }

  //stride and offset are all aligned
  vec_reduce_ar<OP, T>(src0, dst, tmp_buf, initvalue);

}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ar, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, float)
REGISTE_ENTIRE_REDUCE_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int16_t)

REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, float)
REGISTE_ENTIRE_REDUCE_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int16_t)

REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half)
REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, float)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half)
REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, float)
REGISTE_ENTIRE_REDUCE_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int16_t)

REGISTE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, half)
REGISTE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, float)
REGISTE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int16_t)

REGISTE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int8_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int16_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int64_t)

REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int8_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint8_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int16_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint16_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int64_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint64_t)

REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int8_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint8_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int16_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint16_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint32_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int64_t)
REGISTE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint64_t)
}