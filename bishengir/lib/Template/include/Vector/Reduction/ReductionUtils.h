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

#ifndef HIVM_MLIR_TEMPLATE_VECTOR_REDUCTION_UTILS_H
#define HIVM_MLIR_TEMPLATE_VECTOR_REDUCTION_UTILS_H

#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/VecUtils.h"

enum class ReduceOpTy : uint32_t {
  REDUCE_BEGIN = 100,
  REDUCE_SUM = 101,
  REDUCE_MAX = 102,
  REDUCE_MAX_WITH_INDEX = 103,
  REDUCE_MIN = 104,
  REDUCE_MIN_WITH_INDEX = 105,
  REDUCE_PROD = 106,
  REDUCE_XOR = 107,
  REDUCE_OR = 108,
  REDUCE_AND = 109,
  REDUCE_END = 199,
};

enum class ReduceWithIndexOpTy : uint32_t {
  REDUCE_WITH_INDEX_TYPE_BEGIN = 200,
  LEFT = 201,
  RIGHT = 202,
  REDUCE_WITH_INDEX_TYPE_END = 299,
};

// Applicable to vcadd, vcmin, vcmax.
template <typename T>
struct reduce_intrin_args {
  __ubuf__ T *dst_value;
  __ubuf__ int32_t *dst_index;
  __ubuf__ T *src;
  uint8_t repeat;
  uint16_t dstRepeatStride;
  uint16_t srcBlockStride;
  uint16_t srcRepeatStride;
};

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) constexpr VectorOpTy GetVectorOpTy() {
  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    return VectorOpTy::VADD;
  } else if constexpr (OP == ReduceOpTy::REDUCE_MAX) {
    return VectorOpTy::VMAX;
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN) {
    return VectorOpTy::VMIN;
  } else if constexpr (OP == ReduceOpTy::REDUCE_PROD) {
    return VectorOpTy::VMUL;
  } else if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
    return VectorOpTy::VXOR;
  } else if constexpr (OP == ReduceOpTy::REDUCE_OR) {
    return VectorOpTy::VOR;
  } else if constexpr (OP == ReduceOpTy::REDUCE_AND) {
    return VectorOpTy::VAND;
  }
}

// MODE adapt for vcadd.
// If #mode=0, in each iteration this instruction will generate one accumulation
// result and write to Unified Buffer and SPR.ACC_VAL remains untouched.
// If #mode=1, this instruction will only generate a final accumulation result
// across all iterations and write to SPR.ACC_VAL.
// ORDER adapt for vcmin/vcmax.
// ORDER corresponds to four modes VALUE_INDEX/INDEX_VALUE/ONLY_VALUE/ONLY_INDEX
template <ReduceOpTy OP, typename T, bool MODE, Order_t ORDER>
__aiv__ __attribute__((always_inline)) void
reduce_intrin(reduce_intrin_args<T> args) {
#define REDUCE_ARGS                                                            \
  args.dst_value, args.src, args.repeat, args.dstRepeatStride,                 \
      args.srcBlockStride, args.srcRepeatStride

#define REDUCE_INDEX_ONLY_ARGS                                                 \
  (__ubuf__ T *)args.dst_index, args.src, args.repeat, args.dstRepeatStride,   \
      args.srcBlockStride, args.srcRepeatStride

  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    INTRINSIC(vcadd, REDUCE_ARGS, MODE);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MAX) {
    INTRINSIC(vcmax, REDUCE_ARGS, ORDER);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN) {
    INTRINSIC(vcmin, REDUCE_ARGS, ORDER);
  } else if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
    if constexpr (ORDER == Order_t::ONLY_INDEX) {
      INTRINSIC(vcmax, REDUCE_INDEX_ONLY_ARGS, ORDER);
    } else {
      INTRINSIC(vcmax, REDUCE_ARGS, ORDER);
    }
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
    if constexpr (ORDER == Order_t::ONLY_INDEX) {
      INTRINSIC(vcmin, REDUCE_INDEX_ONLY_ARGS, ORDER);
    } else {
      INTRINSIC(vcmin, REDUCE_ARGS, ORDER);
    }
  }
}

/// Calculate mask value bipartite interval jump.
/// example:
/// float32: set_vector_mask(0, 0xffffffffffffffff) ->
///          set_vector_mask(0, 0x0f0f0f0f0f0f0f0f) ->
///          set_vector_mask(0, 0x0303030303030303) ->
///          set_vector_mask(0, 0x0101010101010101)
/// float16: set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff) ->
///          set_vector_mask(0x00ff00ff00ff00ff, 0x00ff00ff00ff00ff) ->
///          set_vector_mask(0x000f000f000f000f, 0x000f000f000f000f) ->
///          set_vector_mask(0x0003000300030003, 0x0003000300030003) ->
///          set_vector_mask(0x0001000100010001, 0x0001000100010001)
/// \param cur_fold_rounds: Current number of folding rounds.
/// \param size: The number of rows processed in one repeat When it is the main
/// block, size=8, when it is the tail block, size<8, here 8 is the maximum
/// number of blocks in a repeat.
/// \param is_tail: Whether the tail block.
template <typename T>
__aiv__ __attribute__((always_inline)) void
SetMaskValueForByDot(uint16_t cur_fold_rounds, int64_t size, bool is_tail) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if constexpr (sizeof(T) == 2) {
    uint64_t value = (1 << (num_per_block / (1 << (cur_fold_rounds + 1)))) - 1;
    if (!is_tail) {
      INTRINSIC(set_vector_mask,
                value * (MAX_UINT64 / ((1 << num_per_block) - 1)),
                value * (MAX_UINT64 / ((1 << num_per_block) - 1)));
    } else {
      if (size > INTR_BLOCKS_PER_REPEAT / 2) {
        INTRINSIC(
            set_vector_mask,
            (((((uint64_t)1 << (size * num_per_block - 65)) * 2) - 1) / 65535) *
                value,
            value * (MAX_UINT64 / ((1 << num_per_block) - 1)));
      } else {
        INTRINSIC(
            set_vector_mask, 0,
            (((((uint64_t)1
                << ((size + INTR_BLOCKS_PER_REPEAT / 2) * num_per_block - 65)) *
               2) -
              1) /
             65535) *
                value);
      }
    }
  } else {
    uint64_t value = (1 << (num_per_block / (1 << (cur_fold_rounds + 1)))) - 1;
    if (!is_tail) {
      INTRINSIC(set_vector_mask, 0,
                value * (MAX_UINT64 / ((1 << num_per_block) - 1)));
    } else {
      INTRINSIC(
          set_vector_mask, 0,
          (((((uint64_t)1 << (size * num_per_block - 1)) * 2) - 1) / 255) *
              value);
    }
  }
}

/// reduce src (a, r) to dst (a, 1) use vector cross intrinsic, namely
/// vcadd/vcmin/vcmax.
///
/// constraints:
/// 1. r < r0.
/// 2. r0 is equal or less than the max number per repeat that intrinsic can
/// handles.
template <ReduceOpTy OP, typename T, bool HASINDEX = false>
__aiv__ __attribute__((always_inline)) void
reduceAR0ToA(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst_value,
             memref_t<__ubuf__ int32_t, 2> *dst_index) {
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  const int64_t src_size0 = src->sizes[0];
  const int64_t src_size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_value_stride0 = dst_value->strides[0];
  const int64_t dst_index_stride0 = dst_index->strides[0];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  SetMaskValueByCount(src_size1);
  for (int64_t i = 0; i < src_size0 / INTR_MAX_REPEAT_CNTS; i++) {
    INTRINSIC(pipe_barrier, PIPE_V);
    reduce_intrin<OP, T, false, Order_t::ONLY_VALUE>(reduce_intrin_args<T>{
        dst_value_ptr + i * INTR_MAX_REPEAT_CNTS,
        nullptr, // dst_index, not work
        src_ptr + i * INTR_MAX_REPEAT_CNTS * src_stride0,
        INTR_MAX_REPEAT_CNTS,                              // repeat
        1,                                                 // dst repeat stride
        1,                                                 // src block stride
        static_cast<uint16_t>(src_stride0 / num_per_block) // src repeat stride
    });
    if constexpr (HASINDEX) {
      reduce_intrin<OP, T, false, Order_t::ONLY_INDEX>(reduce_intrin_args<T>{
          nullptr, // dst_value, not work
          dst_index_ptr + i * INTR_MAX_REPEAT_CNTS,
          src_ptr + i * INTR_MAX_REPEAT_CNTS * src_stride0,
          INTR_MAX_REPEAT_CNTS, // repeat
          1,                    // dst repeat stride
          1,                    // src block stride
          static_cast<uint16_t>(src_stride0 /
                                num_per_block) // src repeat stride
      });
    }
  }
  if (src_size0 % INTR_MAX_REPEAT_CNTS != 0) {
    INTRINSIC(pipe_barrier, PIPE_V);
    reduce_intrin<OP, T, false, Order_t::ONLY_VALUE>(reduce_intrin_args<T>{
        dst_value_ptr + src_size0 / INTR_MAX_REPEAT_CNTS * INTR_MAX_REPEAT_CNTS,
        nullptr, // dst_index, not work
        src_ptr + src_size0 / INTR_MAX_REPEAT_CNTS *
                      (INTR_MAX_REPEAT_CNTS * src_stride0),
        static_cast<uint8_t>(src_size0 % INTR_MAX_REPEAT_CNTS), // repeat
        1,                                                 // dst repeat stride
        1,                                                 // src block stride
        static_cast<uint16_t>(src_stride0 / num_per_block) // src repeat stride
    });
    if constexpr (HASINDEX) {
      reduce_intrin<OP, T, false, Order_t::ONLY_INDEX>(reduce_intrin_args<T>{
          nullptr, // dst_value, not work
          dst_index_ptr +
              src_size0 / INTR_MAX_REPEAT_CNTS * INTR_MAX_REPEAT_CNTS,
          src_ptr + src_size0 / INTR_MAX_REPEAT_CNTS *
                        (INTR_MAX_REPEAT_CNTS * src_stride0),
          static_cast<uint8_t>(src_size0 % INTR_MAX_REPEAT_CNTS), // repeat
          1, // dst repeat stride
          1, // src block stride
          static_cast<uint16_t>(src_stride0 /
                                num_per_block) // src repeat stride
      });
    }
  }
}

/// reduce src (a, r) to dst (a, 1) with stride [n, 1] use vector cross
/// intrinsic namely vcadd/vcmin/vcmax, here n is aligned to UB block unit.
///
/// constraints:
/// 1. r < r0.
/// 2. r0 is equal or less than the max number per repeat that intrinsic can
/// handles.
template <ReduceOpTy OP, typename T, bool HASINDEX = false>
__aiv__ __attribute__((always_inline)) void
reduceAR0ToAByLoopAAxis(memref_t<__ubuf__ T, 2> *src,
                        memref_t<__ubuf__ T, 2> *dst_value,
                        memref_t<__ubuf__ int32_t, 2> *dst_index) {
  const int64_t src_size0 = src->sizes[0];
  const int64_t src_size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_value_stride0 = dst_value->strides[0];
  const int64_t dst_index_stride0 = dst_index->strides[0];
  for (int64_t i = 0; i < src_size0; i++) {
    memref_t<__ubuf__ T, 2> subview_src{src->allocated,
                                        src->aligned,
                                        src->offset + i * src_stride0,
                                        {1, src_size1},
                                        {src_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_dst_value{dst_value->allocated,
                                              dst_value->aligned,
                                              dst_value->offset +
                                                  i * dst_value_stride0,
                                              {1, 1},
                                              {dst_value_stride0, 1}};
    if constexpr (HASINDEX) {
      memref_t<__ubuf__ int32_t, 2> subview_dst_index{dst_index->allocated,
                                                      dst_index->aligned,
                                                      dst_index->offset +
                                                          i * dst_index_stride0,
                                                      {1, 1},
                                                      {dst_index_stride0, 1}};
      reduceAR0ToA<OP, T, true>(&subview_src, &subview_dst_value,
                                &subview_dst_index);
    } else {
      reduceAR0ToA<OP, T, false>(&subview_src, &subview_dst_value, nullptr);
    }
  }
}

/// reduce src (a, rx) to dst (a, ry) by dichotomy.
///
/// constraint:
/// 1. dim of src/dst must be 2.
/// 2. rx >= ry > r1.
/// 3. ry must be the power of 2.
/// 4. r1 is the max number per block that intrinsic can handles.
template <ReduceOpTy OP, typename T, int64_t RX_SIZE = -1, int64_t RY_SIZE = -1>
__aiv__ __attribute__((always_inline)) void
reduceARxToARyByDichotomy(memref_t<__ubuf__ T, 2> *src,
                          memref_t<__ubuf__ T, 2> *dst,
                          memref_t<__ubuf__ T, 1> *tmp_buf) {
  const int64_t src_size0 = src->sizes[0];
  const int64_t src_size1 = RX_SIZE != -1 ? RX_SIZE : src->sizes[1];
  const int64_t dst_size1 = RY_SIZE != -1 ? RY_SIZE : dst->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();

  const int64_t n = Log2(src_size1);
  const int64_t dichotomy_num = n - Log2(dst_size1);
  const int64_t main_size = pow((uint64_t)2, n);
  const int64_t tail_size = src_size1 - main_size;
  dst->strides[0] = src_size1 >= num_per_repeat * 2 ? main_size / 2 : main_size;
  int64_t num = main_size;

  // step 1. specially process the first time of dichotomy
  // 1. dichotomy from src to dst
  // 2. process the tail block

  // do main block for the first time of dichotomy
  if (dichotomy_num > 0) {
    num = num / 2;
    // do from src0 and src1 to dst
    memref_t<__ubuf__ T, 2> subview_src0{src->allocated,
                                         src->aligned,
                                         src->offset,
                                         {src_size0, num},
                                         {src_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_src1{src->allocated,
                                         src->aligned,
                                         src->offset + num,
                                         {src_size0, num},
                                         {src_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_dst{dst->allocated,
                                        dst->aligned,
                                        dst->offset,
                                        {src_size0, num},
                                        {dst->strides[0], 1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vv_2d<VECOP, T>(&subview_src0, &subview_src1, &subview_dst,
                                   tmp_buf);
  } else {
    // just move src0 to dst
    memref_t<__ubuf__ T, 2> subview_src{src->allocated,
                                        src->aligned,
                                        src->offset,
                                        {src_size0, main_size},
                                        {src_stride0, 1}};
    memref_t<__ubuf__ T, 2> subview_dst{dst->allocated,
                                        dst->aligned,
                                        dst->offset,
                                        {src_size0, main_size},
                                        {main_size, 1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vv_2d<VectorOpTy::VMAX, T>(&subview_src, &subview_src,
                                              &subview_dst);
  }

  if (tail_size > 0) {
    // do tail block for the first time of dichotomy
    // do from src1 and dst to dst
    auto loop_num = CEIL_DIV(tail_size, num);
    for (int64_t j = 0; j < loop_num; ++j) {
      memref_t<__ubuf__ T, 2> subview_src{
          src->allocated,
          src->aligned,
          src->offset + main_size + num * j,
          {src_size0,
           j == 0 ? (tail_size < num ? tail_size : num) : (tail_size - num)},
          {src_stride0, 1}};
      memref_t<__ubuf__ T, 2> subview_dst{
          dst->allocated,
          dst->aligned,
          dst->offset,
          {src_size0,
           j == 0 ? (tail_size < num ? tail_size : num) : (tail_size - num)},
          {dst->strides[0], 1}};
      INTRINSIC(pipe_barrier, PIPE_V);
      vector_eltwise_vv_2d<VECOP, T>(&subview_src, &subview_dst, &subview_dst,
                                     tmp_buf);
    }
  }

  // step 2. processs the other times of dichotomy
  for (int64_t i = 0; i < dichotomy_num - 1; ++i) {
    // do dst0 and dst1 to dst
    num = num / 2;
    memref_t<__ubuf__ T, 2> subview_dst0{dst->allocated,
                                         dst->aligned,
                                         dst->offset,
                                         {src_size0, num},
                                         {dst->strides[0], 1}};
    memref_t<__ubuf__ T, 2> subview_dst1{dst->allocated,
                                         dst->aligned,
                                         dst->offset + num,
                                         {src_size0, num},
                                         {dst->strides[0], 1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    vector_eltwise_vv_2d<VECOP, T>(&subview_dst0, &subview_dst1, &subview_dst0,
                                   tmp_buf);
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vcgOP(__ubuf__ T *dst_ptr,
                                                  __ubuf__ T *src_ptr) {
  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    INTRINSIC(vcgadd, dst_ptr, src_ptr,
              1, // repeat
              1, // dst_repeat_stride
              1, // src_block_stride
              8  // src_repeat_stride
    );
  } else if constexpr (OP == ReduceOpTy::REDUCE_MAX) {
    INTRINSIC(vcgmax, dst_ptr, src_ptr,
              1, // repeat
              1, // dst_repeat_stride
              1, // src_block_stride
              8  // src_repeat_stride
    );
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN) {
    INTRINSIC(vcgmin, dst_ptr, src_ptr,
              1, // repeat
              1, // dst_repeat_stride
              1, // src_block_stride
              8  // src_repeat_stride
    );
  }
}

// 3 pipe_V barriers
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduceARByBlocks(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                 T initvalue) {
  const int64_t src_size0 = src->sizes[0];
  const int64_t src_size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t k = Log2(num_per_block);
  int64_t num = src_stride0;
  const int64_t div_num = divisions_needed_by_pow2(num, num_per_repeat, k);
  // step 1. specially process the first division
  if (div_num > 0) {
    // Need to broadcast initvalue to dst
    broadcast_scalar<T, 2>(initvalue, dst);
    // Need to pad align if src_stride0 != src_size1
    int64_t length = src_size0 * src_stride0;
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, length);
    vcgOP<OP, T>(dst_ptr, src_ptr);
    INTRINSIC_NO_ARGS(set_mask_norm);
    num = num / num_per_block;
  } else if (div_num <= 0) {
    return;
  }

  // step 2. processs the other divisions
  for (int64_t i = 0; i < div_num - 1; ++i) {
    // we keep reducing staying in same buffer
    int64_t length = src_size0 * num;
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, length);
    vcgOP<OP, T>(dst_ptr, src_ptr);
    INTRINSIC_NO_ARGS(set_mask_norm);
    num = num / num_per_block;
  }
}

// 3 pipe_V barriers. could slow down performance
template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduceRByBlocks(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                T initvalue) {
  const int64_t src_size0 = src->sizes[0];
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t size_aligned = CEIL_FACTOR(src_size0, num_per_block);
  const int64_t k = Log2(num_per_block);
  int64_t num = size_aligned;
  const int64_t div_num = divisions_needed_by_pow2(num, num_per_repeat, k);
  // step 1. specially process the first division
  if (div_num > 0) {
    // Need to broadcast initvalue to dst
    broadcast_scalar<T, 1>(initvalue, dst);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, src_size0);
    vcgOP<OP, T>(dst_ptr, src_ptr);
    INTRINSIC_NO_ARGS(set_mask_norm);
    num = num / num_per_block;
  } else if (div_num <= 0) {
    return;
  }

  // step 2. processs the other divisions
  for (int64_t i = 0; i < div_num - 1; ++i) {
    // from dst to dst we keep reducing
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, num);
    vcgOP<OP, T>(dst_ptr, src_ptr);
    INTRINSIC_NO_ARGS(set_mask_norm);
    num = num / num_per_block;
  }
}

/// reduce src buffer (a, r1) with stride [n, 1] to dst (a, 1) by element-wise
/// vector operation.
/// 32Bits: Fold in half thress times (a, 8)->(a, 4)->(a, 2)->(a, 1)
/// 16Bits: Fold in half four times (a, 16)->(a, 8)->(a, 4)->(a, 2)->(a, 1)
///
/// Constraints:
/// 1. For reduce_xor tmp buffer size must be larger than a * r1, the rest must
/// be larger than a * r1.
/// 2. 'r1' is the max number per block that inrtinsic can handles.
/// 3. n is block aligned.
///
/// \param tmp_buf: Reserve extra buf for xor operations.
template <ReduceOpTy OP, typename T, int64_t REDUCE_TO = 1,
          int64_t MOVE_TO_DST = 1>
__aiv__ __attribute__((always_inline)) void
reduceAR1ToAByDichotomy(memref_t<__ubuf__ T, 2> *src,
                        memref_t<__ubuf__ T, 2> *dst,
                        memref_t<__ubuf__ T, 1> *tmp_buf,
                        __ubuf__ T *tmp_buf_for_block_reduce_ptr) {
  const int64_t src_size0 = src->sizes[0];
  const int64_t src_stride0 = src->strides[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  constexpr VectorOpTy VECOP = GetVectorOpTy<OP, T>();
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  uint16_t block_stride = src_stride0 / num_per_block;
  uint16_t repeat_stride = INTR_BLOCKS_PER_REPEAT * block_stride;
  auto num = num_per_block;
  int64_t dim_a = src_size0;
  int dichotomy_num = Log2<num_per_block>() - Log2<REDUCE_TO>();

  // Maximum number of lines processed by one instruction.
  constexpr int rows_per_ins = INTR_BLOCKS_PER_REPEAT * INTR_MAX_REPEAT_CNTS;
  for (int64_t i = 0; i < dichotomy_num; ++i) {
    num = num >> 1;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    // Divide the scalar data in src buffer (a, num*2) into (a, num)
    // and move the second half to the tmp buffer, here num is the valid
    // number processed in one row after folding.
    for (int64_t j = 0; j < dim_a; ++j) {
      for (int64_t k = 0; k < num; ++k) {
        *(tmp_buf_for_block_reduce_ptr + j * num_per_block + k) =
            *(src_ptr + j * src_stride0 + k + num);
      }
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

    // Perform element-wise operations on tmp buffer(a, num) and
    // src buffer(a, num) [n, 1], and store the results back to src buffer,
    // here num is the valid number processed in one row after folding.
    // One row has one block of data.
    // One repeat processing 8 blocks means 8 lines.
    if (INTR_BLOCKS_PER_REPEAT <= dim_a) {
      SetMaskValueForByDot<T>(i, INTR_BLOCKS_PER_REPEAT, false);
      // One instruction repeats 255 times means 8*255=2040 lines.
      if (rows_per_ins <= dim_a) {
        for (int64_t j = 0; j < dim_a / rows_per_ins; ++j) {
          INTRINSIC(pipe_barrier, PIPE_V);
          intrin_args<2, T> args{
              src_ptr + j * rows_per_ins * src_stride0,
              {tmp_buf_for_block_reduce_ptr + j * rows_per_ins * num_per_block,
               src_ptr + j * rows_per_ins * src_stride0},
              0,                                        // scalar, not work
              INTR_MAX_REPEAT_CNTS,                     // repeat
              block_stride,                             // dst block stride
              {1, block_stride},                        // src block stride
              repeat_stride,                            // dst repeat stride
              {INTR_BLOCKS_PER_REPEAT, repeat_stride}}; // src repeat stride
          if (OP == ReduceOpTy::REDUCE_XOR) {
            vector_eltwise_xor_intrin<T>(args, tmp_buf);
          } else {
            vector_eltwise_vv_intrin<VECOP, T>(args);
          }
        }
      }
      if (dim_a < rows_per_ins ||
          dim_a / INTR_BLOCKS_PER_REPEAT % rows_per_ins != 0) {
        INTRINSIC(pipe_barrier, PIPE_V);
        intrin_args<2, T> args{
            src_ptr + FLOOR_FACTOR(dim_a, rows_per_ins) * src_stride0,
            {tmp_buf_for_block_reduce_ptr +
                 FLOOR_FACTOR(dim_a, rows_per_ins) * num_per_block,
             src_ptr + FLOOR_FACTOR(dim_a, rows_per_ins) * src_stride0},
            0, // scalar, not work
            static_cast<uint64_t>(dim_a / INTR_BLOCKS_PER_REPEAT %
                                  rows_per_ins),      // repeat
            block_stride,                             // dst block stride
            {1, block_stride},                        // src block stride
            repeat_stride,                            // dst repeat stride
            {INTR_BLOCKS_PER_REPEAT, repeat_stride}}; // src repeat stride
        if (OP == ReduceOpTy::REDUCE_XOR) {
          vector_eltwise_xor_intrin<T>(args, tmp_buf);
        } else {
          vector_eltwise_vv_intrin<VECOP, T>(args);
        }
      }
    }
    if (dim_a % INTR_BLOCKS_PER_REPEAT != 0) {
      // In the scenario of dim_a = 1, the valid bits are continuous,
      // otherwise they are spaced.
      SetMaskValueForByDot<T>(i, dim_a % INTR_BLOCKS_PER_REPEAT, true);
      INTRINSIC(pipe_barrier, PIPE_V);
      intrin_args<2, T> args{
          src_ptr + FLOOR_FACTOR(dim_a, INTR_BLOCKS_PER_REPEAT) * src_stride0,
          {tmp_buf_for_block_reduce_ptr +
               FLOOR_FACTOR(dim_a, INTR_BLOCKS_PER_REPEAT) * num_per_block,
           src_ptr + FLOOR_FACTOR(dim_a, INTR_BLOCKS_PER_REPEAT) * src_stride0},
          0,                 // scalar, not work
          1,                 // repeat
          block_stride,      // dst block stride
          {1, block_stride}, // src block stride
          0,                 // dst repeat stride
          {0, 0}};           // src repeat stride
      if (OP == ReduceOpTy::REDUCE_XOR) {
        vector_eltwise_xor_intrin<T>(args, tmp_buf);
      } else {
        vector_eltwise_vv_intrin<VECOP, T>(args);
      }
    }
  }
  if constexpr (MOVE_TO_DST) {
    const int64_t dst_stride0 = dst->strides[0];
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < dim_a; ++i) {
      for (int64_t j = 0; j < dst->sizes[1]; ++j) {
        *(dst_ptr + i * dst_stride0 + j) = *(src_ptr + i * src_stride0 + j);
      }
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  } else {
    *dst = {src->allocated,
            src->aligned,
            src->offset,
            {src->sizes[0], src->sizes[1]},
            {src->strides[0], src->strides[1]}};
  }
}

/// reduce src (r,) to dst (r0,).
///
/// constraint:
/// 1. r0 is the max number per repeat that intrinsic can handles.
/// 2. r > r0.
/// 3. Considering that repeated setting of mask will affect performance. When
/// NEED_MASK=false, the mask is no longer set inside the function and needs to
/// be guaranteed before calling the function.
template <VectorOpTy OP, typename T, bool NEED_MASK = true>
__aiv__ __attribute__((always_inline)) void
reduceRToR0(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
            memref_t<__ubuf__ T, 1> *tmp_buf) {
  const int64_t size0 = src->sizes[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);

  // part 1: main block
  int64_t main_size = size0 / num_per_repeat;
  if constexpr (NEED_MASK) {
    SetMaskValueByCount(num_per_repeat);
  }
  for (int64_t i = 0; i < main_size / INTR_MAX_REPEAT_CNTS; ++i) {
    INTRINSIC(pipe_barrier, PIPE_V);
    intrin_args<2, T> args{
        dst_ptr,
        {src_ptr + i * INTR_MAX_REPEAT_CNTS * num_per_repeat, dst_ptr},
        0,                            // scalar, not work
        INTR_MAX_REPEAT_CNTS,         // repeat
        1,                            // dst block stride
        {1, 1},                       // src block stride
        0,                            // dst repeat stride
        {INTR_BLOCKS_PER_REPEAT, 0}}; // src repeat stride
    if constexpr (OP == VectorOpTy::VXOR) {
      vector_eltwise_xor_intrin<T>(args, tmp_buf);
    } else {
      vector_eltwise_vv_intrin<OP, T>(args);
    }
  }
  if (main_size % INTR_MAX_REPEAT_CNTS != 0) {
    INTRINSIC(pipe_barrier, PIPE_V);
    intrin_args<2, T> args{
        dst_ptr,
        {src_ptr +
             FLOOR_FACTOR(main_size, INTR_MAX_REPEAT_CNTS) * num_per_repeat,
         dst_ptr},
        0, // scalar, not work
        static_cast<uint64_t>(main_size % INTR_MAX_REPEAT_CNTS), // repeat
        1,                            // dst block stride
        {1, 1},                       // src block stride
        0,                            // dst repeat stride
        {INTR_BLOCKS_PER_REPEAT, 0}}; // src repeat stride
    if constexpr (OP == VectorOpTy::VXOR) {
      vector_eltwise_xor_intrin<T>(args, tmp_buf);
    } else {
      vector_eltwise_vv_intrin<OP, T>(args);
    }
  }
  // part 2: tail block
  int64_t tail_size = size0 % num_per_repeat;
  if constexpr (NEED_MASK) {
    SetMaskValueByCount(tail_size);
  }
  if (tail_size != 0) {
    INTRINSIC(pipe_barrier, PIPE_V);
    intrin_args<2, T> args{dst_ptr,
                           {src_ptr + main_size * num_per_repeat, dst_ptr},
                           0,                            // scalar, not work
                           1,                            // repeat
                           1,                            // dst block stride
                           {1, 1},                       // src block stride
                           0,                            // dst repeat stride
                           {INTR_BLOCKS_PER_REPEAT, 0}}; // src repeat stride
    if constexpr (OP == VectorOpTy::VXOR) {
      vector_eltwise_xor_intrin<T>(args, tmp_buf);
    } else {
      vector_eltwise_vv_intrin<OP, T>(args);
    }
  }
}

/// reduce src (a, r) to dst (a, r0).
///
/// constraint:
/// 1. r0 is the max number per repeat that intrinsic can handles.
/// 2. r > r0.
template <VectorOpTy VECOP, typename T>
__aiv__ __attribute__((always_inline)) void
reduceARToAR0(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
              memref_t<__ubuf__ T, 1> *tmp_buf) {
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  int64_t main_size = FLOOR_FACTOR(size1, num_per_repeat);
  int64_t tail_size = size1 % num_per_repeat;
  if (size0 > size1 / num_per_repeat || VECOP == VectorOpTy::VXOR) {
    memref_t<__ubuf__ T, 2> subview_dst{dst->allocated,
                                        dst->aligned,
                                        dst->offset,
                                        {size0, num_per_repeat},
                                        {dst_stride0, dst_stride1}};
    // part 1: main block
    for (int64_t i = 0; i < size1 / num_per_repeat; ++i) {
      memref_t<__ubuf__ T, 2> subview_src{src->allocated,
                                          src->aligned,
                                          src->offset + i * num_per_repeat,
                                          {size0, num_per_repeat},
                                          {src_stride0, src_stride1}};
      INTRINSIC(pipe_barrier, PIPE_V);
      vector_eltwise_vv_2d<VECOP, T>(&subview_src, &subview_dst, &subview_dst,
                                     tmp_buf);
    }
    // part 2: tail block
    if (tail_size != 0) {
      memref_t<__ubuf__ T, 2> subview_src{src->allocated,
                                          src->aligned,
                                          src->offset + main_size,
                                          {size0, tail_size},
                                          {src_stride0, src_stride1}};
      INTRINSIC(pipe_barrier, PIPE_V);
      subview_dst.sizes[1] = tail_size;
      vector_eltwise_vv_2d<VECOP, T>(&subview_src, &subview_dst, &subview_dst,
                                     tmp_buf);
    }
  } else {
    // part 1: main block
    SetMaskValueByCount(num_per_repeat);
    for (int64_t i = 0; i < size0; ++i) {
      memref_t<__ubuf__ T, 1> subview_src{src->allocated,
                                          src->aligned,
                                          src->offset + i * src_stride0,
                                          {main_size},
                                          {1}};
      memref_t<__ubuf__ T, 1> subview_dst{dst->allocated,
                                          dst->aligned,
                                          dst->offset + i * num_per_repeat,
                                          {num_per_repeat},
                                          {1}};
      reduceRToR0<VECOP, T, false>(&subview_src, &subview_dst, tmp_buf);
    }
    // part 2: tail block
    if (tail_size) {
      SetMaskValueByCount(tail_size);
      for (int64_t i = 0; i < size0; ++i) {
        memref_t<__ubuf__ T, 1> subview_src{src->allocated,
                                            src->aligned,
                                            src->offset + i * src_stride0 +
                                                main_size,
                                            {tail_size},
                                            {1}};
        memref_t<__ubuf__ T, 1> subview_dst{dst->allocated,
                                            dst->aligned,
                                            dst->offset + i * num_per_repeat,
                                            {num_per_repeat},
                                            {1}};
        reduceRToR0<VECOP, T, false>(&subview_src, &subview_dst, tmp_buf);
      }
    }
  }
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ar(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
          memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue);

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_ar_vcg(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
              memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue);

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
         memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue);

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_by_vc(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
               T initvalue);

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_vcg(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
             memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue);

/// reduce src (r, ) to dst (1, ) and return the reduction value and index
/// separately.
///
/// constraint:
/// 1. dim of src/dst must be 1.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
/// 3. tmp buffer size is equal to 1 block.
template <ReduceOpTy OP, ReduceWithIndexOpTy WITH_INDEX_TYPE, typename T>
__aiv__ __attribute__((always_inline)) void
reduce_r_with_index(memref_t<__ubuf__ T, 1> *src0,
                    memref_t<__ubuf__ T, 1> *dst_value,
                    memref_t<__ubuf__ int32_t, 1> *dst_index,
                    memref_t<__ubuf__ T, 1> *tmp_buf, T initvalue);

#define REGISTE_REDUCE_R_WITH_INDEX(op, with_index_type, type)                 \
  template __aiv__ __attribute__((always_inline)) void                         \
  reduce_r_with_index<op, with_index_type, type>(                              \
      memref_t<__ubuf__ type, 1> * src0,                                       \
      memref_t<__ubuf__ type, 1> * dst_value,                                  \
      memref_t<__ubuf__ int32_t, 1> * dst_index,                               \
      memref_t<__ubuf__ type, 1> * tmp_buf, type initvalue)

/// Reduce src (a, r) to tmp (a, r0) and then reduce tmp (a, r0) to dst (a, 1)
/// by vector cross intrinsic. For float reduce_sum/min/max, we can use vector
/// cross intrinsic, namely vcadd/vcmin/vcmax, to reduce (a, r0) to (a, 1).
#define DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(op_name, op_type, dim, dtype)        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_enablevc_##op_name##_##ar##_##dtype(                        \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

#define DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(op_name, op_type, dim, dtype)       \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_enablevcg_##op_name##_##ar##_##dtype(                       \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

/// Reduce src (a, r) to tmp (a, r0) and then reduce tmp (a, r0) to dst (a, 1)
/// by vector element intrinsic. For int reduce_sum/min/max or any type of
/// reduce_prod/reduce_xor, there is no corresponding vector cross intrinsic and
/// we can only use vector element intrinsic by dichotomy to reduce (a, r0) to
/// (a, 1).
#define DECLARE_ENTIRE_REDUCE_AR(op_name, op_type, dim, dtype)                 \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ar##_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

/// Reduce src (a, r) to dst (a, 1) by vector cross intrinsic. Only support
/// reduce_min and reduce_max of half and float types, namely vcmin/vcmax.
/// Return the value after reduction and the index corresponding to the value.
#define DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(op_name, op_type, with_index_type, \
                                            dim, dtype)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ar##_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index,                          \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

#define REGISTE_ENTIRE_REDUCE_AR(op_name, op_type, dim, dtype)                 \
  DECLARE_ENTIRE_REDUCE_AR(op_name, op_type, dim, dtype) {                     \
    reduce_ar<op_type, dtype>(src0, dst, tmp_buf, initvalue);                  \
  }

#define REGISTE_ENTIRE_REDUCE_ENABLEVC_AR(op_name, op_type, dim, dtype)        \
  DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(op_name, op_type, dim, dtype) {            \
    reduce_ar<op_type, dtype>(src0, dst, tmp_buf, initvalue);                  \
  }

#define REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX(op_name, op_type, with_index_type, \
                                            dim, dtype)                        \
  DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(op_name, op_type, with_index_type, dim,  \
                                      dtype) {                                 \
    reduce_ar_with_index<op_type, with_index_type, dtype>(                     \
        src0, dst_value, dst_index, tmp_buf, initvalue);                       \
      }

#define REGISTE_ENTIRE_REDUCE_ENABLEVCG_AR(op_name, op_type, dim, dtype)       \
  DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(op_name, op_type, dim, dtype) {           \
    reduce_ar_vcg<op_type, dtype>(src0, dst, tmp_buf, initvalue);              \
  }

#define DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(op_name, op_type, with_index_type, \
                                            dim, dtype)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ra##_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index,                          \
          memref_t<__ubuf__ int32_t, 1> *tmp_index)

#define REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX(op_name, op_type, with_index_type, \
                                            dim, dtype)                        \
  DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(op_name, op_type, with_index_type, dim,  \
                                      dtype) {                                 \
    reduce_ra_with_index<op_type, with_index_type, dtype>(                     \
        src0, dst_value, dst_index, tmp_index);                                \
  }

/// Reduce src (r,) to dst (1,). For float reduce_sum/min/max, we can use vector
/// cross intrinsic, namely vcadd/vcmin/vcmax.
#define DECLARE_ENTIRE_REDUCE_ENABLEVC_R(op_name, op_type, dim, dtype)         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_enablevc_##op_name##_##r##_##dtype(                         \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

#define DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(op_name, op_type, dim, dtype)        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_enablevcg_##op_name##_##r##_##dtype(                        \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

/// Reduce src (r,) to tmp (r0,) and then reduce tmp (r0,) to dst (1,) by vector
/// element intrinsic. For int reduce_sum/min/max or any type of
/// reduce_prod/reduce_xor, there is no corresponding vector cross intrinsic and
/// we can only use vector element intrinsic by dichotomy to reduce (r0,) to
/// (1,).
#define DECLARE_ENTIRE_REDUCE_R(op_name, op_type, dim, dtype)                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##r##_##dtype(                                  \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

/// Reduce src (r,) to dst (1,). by vector cross intrinsic. Only support
/// reduce_min and reduce_max of half and float types, namely vcmin/vcmax.
/// Return the value after reduction and the index corresponding to the value.
#define DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(op_name, op_type, with_index_type,  \
                                           dim, dtype)                         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##r##_##dtype(                                  \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index,                          \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

#define REGISTE_ENTIRE_REDUCE_ENABLEVC_R(op_name, op_type, dim, dtype)         \
  DECLARE_ENTIRE_REDUCE_ENABLEVC_R(op_name, op_type, dim, dtype) {             \
    reduce_r<op_type, dtype>(src0, dst, tmp_buf, initvalue);                   \
  }

#define REGISTE_ENTIRE_REDUCE_ENABLEVCG_R(op_name, op_type, dim, dtype)        \
  DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(op_name, op_type, dim, dtype) {            \
    reduce_r_vcg<op_type, dtype>(src0, dst, tmp_buf, initvalue);               \
  }

#define REGISTE_ENTIRE_REDUCE_R(op_name, op_type, dim, dtype)                  \
  DECLARE_ENTIRE_REDUCE_R(op_name, op_type, dim, dtype) {                      \
    reduce_r<op_type, dtype>(src0, dst, tmp_buf, initvalue);                   \
  }

#define REGISTE_ENTIRE_REDUCE_R_WITH_INDEX(op_name, op_type, with_index_type,  \
                                           dim, dtype)                         \
  DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(op_name, op_type, with_index_type, dim,   \
                                     dtype) {                                  \
    reduce_r_with_index<op_type, with_index_type, dtype>(                      \
        src0, dst_value, dst_index, tmp_buf, initvalue);                       \
  }

/// Reduce src (r, a) with stride [n, 1] to dst (1, a) with stride [n, 1] by
/// vector element intrinsic, here 'n' is a aligned to ub_block_unit.
#define DECLARE_ENTIRE_REDUCE_RA(op_name, op_type, dim, dtype)                 \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ra##_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

/// Reduce src (r, a0, a1) with stride [n, n1, 1] to dst (1, a0, a1) with stride
// [n, n1, 1] by vector element intrinsic, here 'n' is a0*a1 aligned to
// ub_block_unit, 'n1' is a1 aligned to ub_block_unit.
#define DECLARE_ENTIRE_REDUCE_RA0A1(op_name, op_type, dim, dtype)              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ra0a1##_##dtype(                              \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp, dtype initvalue)

#define REGISTE_ENTIRE_REDUCE_RA(op_name, op_type, dim, dtype)                 \
  DECLARE_ENTIRE_REDUCE_RA(op_name, op_type, dim, dtype) {                     \
    reduce_ra<op_type, dtype>(src0, dst, tmp_buf, initvalue);                  \
  }

#define REGISTE_ENTIRE_REDUCE_RA0A1(op_name, op_type, dim, dtype)              \
  DECLARE_ENTIRE_REDUCE_RA0A1(op_name, op_type, dim, dtype) {                  \
    reduce_ra0a1<op_type, dtype>(src0, dst, tmp, initvalue);                   \
  }

#define REGISTE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name,      \
                                                                 op_type,      \
                                                                 with_index_type, \
                                                                 dim,          \
                                                                 dtype)        \
  DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name, op_type,   \
                                                           with_index_type,    \
                                                           dim,                \
                                                           dtype) {            \
    reduce_ar_with_index_with_specified_index<op_type, with_index_type, dtype>( \
        src0, indices, dst_value, dst_index, tmp_buf, initvalue);              \
  }

#define DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name,      \
                                                                 op_type,      \
                                                                 with_index_type, \
                                                                 dim, dtype)   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ar##_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ int32_t, dim> *indices,                            \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index,                          \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

#define REGISTE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name,      \
                                                                 op_type,      \
                                                                 with_index_type, \
                                                                 dim,          \
                                                                 dtype)        \
  DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name, op_type,   \
                                                           with_index_type,    \
                                                           dim,                \
                                                           dtype) {            \
    reduce_ra_with_index_with_specified_index<op_type, with_index_type, dtype>( \
        src0, indices, dst_value, dst_index, tmp_index);                        \
  }

#define DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name,      \
                                                                 op_type,      \
                                                                 with_index_type, \
                                                                 dim, dtype)   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##ra##_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ int32_t, dim> *indices,                            \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index,                          \
          memref_t<__ubuf__ int32_t, 1> *tmp_index)

#define REGISTE_ENTIRE_REDUCE_R_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name,       \
                                                                 op_type,      \
                                                                 with_index_type, \
                                                                 dim,          \
                                                                 dtype)        \
  DECLARE_ENTIRE_REDUCE_R_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name, op_type,    \
                                                           with_index_type,    \
                                                           dim,                \
                                                           dtype) {            \
    reduce_r_with_index_with_specified_index<op_type, with_index_type, dtype>( \
        src0, indices, dst_value, dst_index, tmp_buf, initvalue);              \
  }

#define DECLARE_ENTIRE_REDUCE_R_WITH_INDEX_WITH_SPECIFIED_INDEX(op_name,      \
                                                                 op_type,      \
                                                                 with_index_type, \
                                                                 dim, dtype)   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##r##_##dtype(                                  \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ int32_t, dim> *indices,                            \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index,                          \
          memref_t<__ubuf__ dtype, 1> *tmp_buf, dtype initvalue)

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline))
std::enable_if_t<!std::is_same_v<T, half>, T>
reduction_scalar_operation(T src0_oprand, T src1_oprand) {
  static_assert(OP == ReduceOpTy::REDUCE_SUM ||
                OP == ReduceOpTy::REDUCE_MAX ||
                OP == ReduceOpTy::REDUCE_MIN ||
                OP == ReduceOpTy::REDUCE_PROD ||
                OP == ReduceOpTy::REDUCE_XOR ||
                OP == ReduceOpTy::REDUCE_OR ||
                OP == ReduceOpTy::REDUCE_AND,
                "ReduceOpTy not find. The return value may be incorrect.");

  if constexpr (OP == ReduceOpTy::REDUCE_SUM) {
    return src0_oprand + src1_oprand;
  } else if constexpr (OP == ReduceOpTy::REDUCE_MAX) {
    // check if one of operands is NAN
    if (src0_oprand != src0_oprand || src1_oprand != src1_oprand) return NAN;
    return (src0_oprand > src1_oprand) ? src0_oprand : src1_oprand;
  } else if constexpr (OP == ReduceOpTy::REDUCE_MIN) {
    // check if one of operands is NAN
    if (src0_oprand != src0_oprand || src1_oprand != src1_oprand) return NAN;
    return (src0_oprand < src1_oprand) ? src0_oprand : src1_oprand;
  } else if constexpr (OP == ReduceOpTy::REDUCE_PROD) {
    return src0_oprand * src1_oprand;
  } else if constexpr (OP == ReduceOpTy::REDUCE_XOR) {
    return src0_oprand ^ src1_oprand;
  } else if constexpr (OP == ReduceOpTy::REDUCE_OR) {
    return src0_oprand | src1_oprand;
  } else if constexpr (OP == ReduceOpTy::REDUCE_AND) {
    return src0_oprand & src1_oprand;
  }
  return src0_oprand;
}

template <ReduceOpTy OP, typename T>
__aiv__ __attribute__((always_inline))
std::enable_if_t<std::is_same_v<T, half>, T>
reduction_scalar_operation(T src0_oprand, T src1_oprand) {
  float src0_oprand_float = static_cast<float>(src0_oprand);
  float src1_oprand_float = static_cast<float>(src1_oprand);
  float result_float = reduction_scalar_operation<OP, float>(src0_oprand_float, src1_oprand_float);
  return static_cast<T>(result_float);
}

template <typename T>
__aiv__ __attribute__((always_inline)) int64_t reduction_element_nums_to_move_offset_aligned(
    int64_t offset) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  return CEIL_DIV(offset, num_per_block) * num_per_block - offset;
}

template <typename T>
__aiv__ __attribute__((always_inline)) int64_t reduction_get_element_nums_on_scalar_1d(
      memref_t<__ubuf__ T, 1> *src0) {
  int64_t scalar_element_num = 0;
  auto src0_ptr = src0->aligned + src0->offset;

  if (src0->strides[0] == 1) [[likely]] {
    if (isAddress32ByteAligned(src0_ptr)) [[likely]] {
      return 0;
    }

    scalar_element_num = reduction_element_nums_to_move_offset_aligned<T>(src0->offset);
    if (scalar_element_num > src0->sizes[0]) {
      scalar_element_num = src0->sizes[0];
    }
  } else {
    scalar_element_num = src0->sizes[0];
  }

  return scalar_element_num;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
get_scalar_memref_t(int64_t element_nums_offset_to_aligned,
                    memref_t<__ubuf__ T, 2> *src,
                    memref_t<__ubuf__ T, 2> *scalar_memref) {
  *scalar_memref = *src;
  if (src->sizes[1] * src->strides[1] < element_nums_offset_to_aligned) {
    return;
  }
  scalar_memref->sizes[1] = CEIL_DIV(element_nums_offset_to_aligned, src->strides[1]);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
get_vector_memref_t(int64_t element_nums_offset_to_aligned,
                    memref_t<__ubuf__ T, 2> *src,
                    memref_t<__ubuf__ T, 2> *vector_memref) {
  *vector_memref = *src;
  if (src->sizes[1] * src->strides[1] < element_nums_offset_to_aligned) {
    return;
  }

  vector_memref->allocated += vector_memref->offset + element_nums_offset_to_aligned;
  vector_memref->aligned += vector_memref->offset + element_nums_offset_to_aligned;
  vector_memref->offset = 0;
  vector_memref->sizes[1] -= CEIL_DIV(element_nums_offset_to_aligned, src->strides[1]);
  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool reduction_is_same_element_nums_to_move_offset_aligned(
    int64_t src_offset, int64_t dst_offset) {
  return reduction_element_nums_to_move_offset_aligned<T>(src_offset) ==
         reduction_element_nums_to_move_offset_aligned<T>(dst_offset);
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
isnan_value(T data) {
  static_assert(std::is_same<half, T>() ||
                std::is_same<float, T>(),
                "T must be half or float.");
  return static_cast<float>(data) != static_cast<float>(data);
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ar, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half);
DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, float);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2,
                                   float);
DECLARE_ENTIRE_REDUCE_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int16_t);

DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half);
DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, float);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2,
                                   float);
DECLARE_ENTIRE_REDUCE_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int16_t);

DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half);
DECLARE_ENTIRE_REDUCE_ENABLEVC_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, float);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2,
                                   float);
DECLARE_ENTIRE_REDUCE_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int16_t);

DECLARE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, half);
DECLARE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, float);
DECLARE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int16_t);

// TODO support uint16/uint32/int64/uint64
DECLARE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int8_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int16_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int64_t);

DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int8_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint8_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int16_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint16_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int64_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint64_t);

DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int8_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint8_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int16_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint16_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint32_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int64_t);
DECLARE_ENTIRE_REDUCE_AR(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint64_t);

//===-------------------------------------------------------------------===//
// reduce r, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_ENABLEVC_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, half);
DECLARE_ENTIRE_REDUCE_ENABLEVC_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, float);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, half);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, float);
DECLARE_ENTIRE_REDUCE_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_sum, ReduceOpTy::REDUCE_SUM, 1, int16_t);

DECLARE_ENTIRE_REDUCE_ENABLEVC_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, half);
DECLARE_ENTIRE_REDUCE_ENABLEVC_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, float);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, half);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, float);
DECLARE_ENTIRE_REDUCE_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_max, ReduceOpTy::REDUCE_MAX, 1, int16_t);

DECLARE_ENTIRE_REDUCE_ENABLEVC_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, half);
DECLARE_ENTIRE_REDUCE_ENABLEVC_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, float);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, half);
DECLARE_ENTIRE_REDUCE_ENABLEVCG_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, float);
DECLARE_ENTIRE_REDUCE_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_min, ReduceOpTy::REDUCE_MIN, 1, int16_t);

DECLARE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, half);
DECLARE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, float);
DECLARE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_prod, ReduceOpTy::REDUCE_PROD, 1, int16_t);

// TODO support uint16/uint32/int64/uint64
DECLARE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int8_t);
DECLARE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int16_t);
DECLARE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_xori, ReduceOpTy::REDUCE_XOR, 1, int64_t);

DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int8_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint8_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int16_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint16_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, int64_t);
DECLARE_ENTIRE_REDUCE_R(reduce_ori, ReduceOpTy::REDUCE_OR, 1, uint64_t);

DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int8_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint8_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int16_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint16_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint32_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, int64_t);
DECLARE_ENTIRE_REDUCE_R(reduce_andi, ReduceOpTy::REDUCE_AND, 1, uint64_t);

//===-------------------------------------------------------------------===//
// reduce ra, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half);
DECLARE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, float);
DECLARE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int16_t);

DECLARE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half);
DECLARE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, float);
DECLARE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int16_t);

DECLARE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half);
DECLARE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, float);
DECLARE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int16_t);

DECLARE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, half);
DECLARE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, float);
DECLARE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int16_t);

// TODO support uint16/uint32/int64/uint64
DECLARE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int8_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int16_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int64_t);

DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int8_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint8_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int16_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint16_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int64_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint64_t);

DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int8_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint8_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int16_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint16_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint32_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int64_t);
DECLARE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint64_t);

//===-------------------------------------------------------------------===//
// reduce ra0a1, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, half);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, float);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_sum, ReduceOpTy::REDUCE_SUM, 3, int16_t);

DECLARE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, half);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, float);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_max, ReduceOpTy::REDUCE_MAX, 3, int16_t);

DECLARE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, half);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, float);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_min, ReduceOpTy::REDUCE_MIN, 3, int16_t);

DECLARE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, half);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, float);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_prod, ReduceOpTy::REDUCE_PROD, 3, int16_t);

// TODO support uint16/uint32/int64/uint64
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int8_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int16_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_xori, ReduceOpTy::REDUCE_XOR, 3, int64_t);

DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int8_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint8_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int16_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint16_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, int64_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_ori, ReduceOpTy::REDUCE_OR, 3, uint64_t);

DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int8_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint8_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int16_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint16_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint32_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, int64_t);
DECLARE_ENTIRE_REDUCE_RA0A1(reduce_andi, ReduceOpTy::REDUCE_AND, 3, uint64_t);

//===-------------------------------------------------------------------===//
// reduce r with index, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_max_with_index_left,
                                   ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                   ReduceWithIndexOpTy::LEFT, 1, half);
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_max_with_index_right,
                                   ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                   ReduceWithIndexOpTy::RIGHT, 1, half);
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_max_with_index_left,
                                   ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                   ReduceWithIndexOpTy::LEFT, 1, float);
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_max_with_index_right,
                                   ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                   ReduceWithIndexOpTy::RIGHT, 1, float);

DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_min_with_index_left,
                                   ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                   ReduceWithIndexOpTy::LEFT, 1, half);
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_min_with_index_right,
                                   ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                   ReduceWithIndexOpTy::RIGHT, 1, half);
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_min_with_index_left,
                                   ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                   ReduceWithIndexOpTy::LEFT, 1, float);
DECLARE_ENTIRE_REDUCE_R_WITH_INDEX(reduce_min_with_index_right,
                                   ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                   ReduceWithIndexOpTy::RIGHT, 1, float);

//===-------------------------------------------------------------------===//
// reduce ar with index, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
DECLARE_ENTIRE_REDUCE_AR_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

//===-------------------------------------------------------------------===//
// reduce ra with index, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_left,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_max_with_index_right,
                                    ReduceOpTy::REDUCE_MAX_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);

DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, half);
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, half);
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_left,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::LEFT, 2, float);
DECLARE_ENTIRE_REDUCE_RA_WITH_INDEX(reduce_min_with_index_right,
                                    ReduceOpTy::REDUCE_MIN_WITH_INDEX,
                                    ReduceWithIndexOpTy::RIGHT, 2, float);
}

#endif // HIVM_MLIR_TEMPLATE_VECTOR_REDUCTION_UTILS_H
