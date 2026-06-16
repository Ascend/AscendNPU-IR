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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H

#include "DMA/DMAUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"
#if defined(__DAV_C310__)

template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                 memref_t<__ubuf__ T, 1> *temp, bool reverse);

template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                  memref_t<__ubuf__ T, 1> *temp, bool reverse);

extern "C" {
#define DECLARE_CUMPROD(DIM, dtype, cum_dim)                                   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_cumprod_##DIM##d_##dtype##_dim##cum_dim(                    \
          memref_t<__ubuf__ dtype, DIM> *src,                                  \
          memref_t<__ubuf__ dtype, DIM> *dst,                                  \
          memref_t<__ubuf__ dtype, DIM> *temp, bool reverse = false)

#define DECLARE_CUMPROD_WITHOUT_DEFAULT_VALUE(DIM, dtype, cum_dim)             \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_cumprod_##DIM##d_##dtype##_dim##cum_dim(                    \
          memref_t<__ubuf__ dtype, DIM> *src,                                  \
          memref_t<__ubuf__ dtype, DIM> *dst,                                  \
          memref_t<__ubuf__ dtype, DIM> *temp, bool reverse)

#define REGISTER_CUMPROD(DIM, dtype, cum_dim)                                  \
  DECLARE_CUMPROD_WITHOUT_DEFAULT_VALUE(DIM, dtype, cum_dim) {                 \
    vector_cumprod_##DIM##d<dtype, cum_dim>(src, dst, nullptr, reverse);       \
  }

#define REGISTER_CUMPROD_WITH_TEMP(DIM, dtype, cum_dim)                        \
  DECLARE_CUMPROD_WITHOUT_DEFAULT_VALUE(DIM, dtype, cum_dim) {                 \
    vector_cumprod_##DIM##d<dtype, cum_dim>(src, dst, temp, reverse);          \
  }
}

extern "C" {
#define DECLARE_CUMSUM(DIM, dtype, cum_dim)                                    \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_cumsum_##DIM##d_##dtype##_dim##cum_dim(                     \
          memref_t<__ubuf__ dtype, DIM> *src,                                  \
          memref_t<__ubuf__ dtype, DIM> *dst,                                  \
          memref_t<__ubuf__ dtype, DIM> *temp, bool reverse = false)

#define DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(DIM, dtype, cum_dim)              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_cumsum_##DIM##d_##dtype##_dim##cum_dim(                     \
          memref_t<__ubuf__ dtype, DIM> *src,                                  \
          memref_t<__ubuf__ dtype, DIM> *dst,                                  \
          memref_t<__ubuf__ dtype, DIM> *temp, bool reverse)

#define REGISTER_CUMSUM(DIM, dtype, cum_dim)                                   \
  DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(DIM, dtype, cum_dim) {                  \
    vector_cumsum_##DIM##d<dtype, cum_dim>(src, dst, nullptr, reverse);        \
  }

#define REGISTER_CUMSUM_WITH_TEMP(DIM, dtype, cum_dim)                         \
  DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(DIM, dtype, cum_dim) {                  \
    vector_cumsum_##DIM##d<dtype, cum_dim>(src, dst, temp, reverse);           \
  }
}

template <typename SRC_T, typename DST_T = SRC_T> struct cumulative_args {
  memref_t<__ubuf__ DST_T, 1> *dst;
  memref_t<__ubuf__ SRC_T, 1> *src[2];
  int64_t dst_stride;
  int64_t src_stride[2];
  int64_t src_size;
  bool reverse;
};

template <typename SRC_TYPE, typename DST_TYPE>
__aiv__ __attribute__((always_inline)) void
cumsum_2d(cumulative_args<SRC_TYPE, DST_TYPE> args);

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_for_cum_op(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {

    int64_t total_size = src->sizes[0] * src->strides[0];
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset,
        {total_size}, {1}};
    memref_t<__ubuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset,
        {total_size}, {1}};
    copy_ubuf_to_ubuf_1d_core<T>(&src_1d, &dst_1d);
    return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
transpose_dim_01(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
  if (is_no_op<3>(src->sizes))
    return;
  const int64_t src_size0 = src->sizes[0];
  const int64_t src_size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const uint32_t burst_len = CEIL_DIV(src->sizes[2], num_per_block);

  bool is_loop_src_size0 = src_size0 < src_size1;
  int64_t loop_num = is_loop_src_size0 ? src_size0 : src_size1;
  int64_t nburst = is_loop_src_size0 ? src_size1 : src_size0;
  int64_t src_loop_stride = is_loop_src_size0 ? src_stride0 : src_stride1;
  int64_t dst_loop_stride = is_loop_src_size0 ? dst_stride1 : dst_stride0;
  int64_t src_repeat_stride = is_loop_src_size0 ? src_stride1 : src_stride0;
  int64_t dst_repeat_stride = is_loop_src_size0 ? dst_stride0 : dst_stride1;

  int64_t burst_cnt_main = nburst / INTRIN_MAX_BURST_CNT;
  int64_t burst_cnt_tail = nburst % INTRIN_MAX_BURST_CNT;

  for (int64_t i = 0; i < loop_num; ++i) {
    __ubuf__ T *cur_src = src_ptr + i * src_loop_stride;
    __ubuf__ T *cur_dst = dst_ptr + i * dst_loop_stride;
    if (burst_cnt_main > 0) {
      copy_ubuf_to_ubuf_intrin_core<T>(cur_dst, cur_src, INTRIN_MAX_BURST_CNT,
                                       burst_len, src_repeat_stride,
                                       dst_repeat_stride);
    }
    if (burst_cnt_tail > 0) {
      copy_ubuf_to_ubuf_intrin_core<T>(
          cur_dst + burst_cnt_main * INTRIN_MAX_BURST_CNT * dst_repeat_stride,
          cur_src + burst_cnt_main * INTRIN_MAX_BURST_CNT * src_repeat_stride,
          burst_cnt_tail, burst_len, src_repeat_stride, dst_repeat_stride);
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
transpose_ar2ra_i8(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  uint32_t M = src->sizes[0];
  uint16_t N = src->sizes[1];
  int32_t srcStride0 = src->strides[0];
  int32_t dstStride0 = dst->strides[0];
  uint32_t num_per_reg = REG_REGISTER_SIZE / sizeof(int16_t);
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  uint16_t mLoop = CEIL_DIV((uint16_t)M, (uint16_t)num_per_reg);
  using gatherT =
      std::conditional_t<std::is_same<T, uint8_t>::value, uint16_t, int16_t>;
  __VEC_SCOPE__ {
    VectorReg<gatherT> dataReg;
    VectorReg<uint8_t> packReg;
    VectorReg<int16_t> idx_reg;
    vector_bool full_mask, full_mask_i8;
    uint32_t full_mask_size = num_per_reg;
    CREATE_MASK_BY_SIZE(full_mask, int16_t, full_mask_size);
    vci(idx_reg, 0);
    vmuls(idx_reg, idx_reg, srcStride0, full_mask);
    for (uint16_t m = 0; m < mLoop; m++) {
      uint32_t mask_size = (m == mLoop - 1) ? M - m * num_per_reg : num_per_reg;
      uint32_t mask_size4i8 =
          (m == mLoop - 1) ? M - m * num_per_reg : num_per_reg;
      CREATE_MASK_BY_SIZE(full_mask, int16_t, mask_size);
      CREATE_MASK_BY_SIZE(full_mask_i8, T, mask_size4i8);
      __ubuf__ T *src_block = src_ptr + (int32_t)m * num_per_reg * srcStride0;
      for (uint16_t n = 0; n < N; n++) {
        vgather2(dataReg, src_block + n, (VectorReg<uint16_t> &)idx_reg,
                 full_mask);
        vpack((VectorReg<uint8_t> &)packReg, dataReg, LOWER, MODE_ZEROING);
        vsts((VectorReg<T> &)packReg, dst_ptr, n * dstStride0 + m * num_per_reg,
             NORM_B32, full_mask_i8);
      }
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
transpose_ar2ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  uint32_t M = src->sizes[0];
  uint16_t N = src->sizes[1];
  int32_t srcStride0 = src->strides[0];
  int32_t dstStride0 = dst->strides[0];
  uint32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  uint16_t mLoop = CEIL_DIV((uint16_t)M, (uint16_t)num_per_reg);
  using IdxT = std::conditional_t<sizeof(T) == 2, int16_t, int32_t>;
  using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
  __VEC_SCOPE__{
    VectorReg<T> dataReg;
    VectorReg<IdxT> idx_reg;
    vector_bool full_mask;
    uint32_t full_mask_size = num_per_reg;
    CREATE_MASK_BY_SIZE(full_mask, T, full_mask_size);
    vci(idx_reg, 0);
    vmuls(idx_reg, idx_reg, srcStride0, full_mask);
    for (uint16_t m = 0; m < mLoop; m++) {
      uint32_t mask_size = (m == mLoop - 1) ? M - m * num_per_reg : num_per_reg;
      CREATE_MASK_BY_SIZE(full_mask, T, mask_size);
      __ubuf__ T *src_block = src_ptr + (int32_t)m * num_per_reg * srcStride0;
      for (uint16_t n = 0; n < N; n++) {
        vgather2(dataReg, src_block + n, (VectorReg<uIdxT> &)idx_reg, full_mask);
        vsts(dataReg, dst_ptr, n * dstStride0 + m * num_per_reg, NORM_B32, full_mask);
      }
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
suitable_to_fold(memref_t<__ubuf__ T, 3> *src) {
  constexpr int32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  constexpr int32_t num_per_block = 32 / sizeof(T);
  int32_t m = src->sizes[0];
  int32_t n = src->sizes[2];
  return CEIL_DIV(n, num_per_block) * m <
         CEIL_DIV(CEIL_FACTOR(n, num_per_block) * m, num_per_reg);
}

struct sklansky_param_t {
  int32_t startOffset;
  int32_t group_offset;
  int32_t realDupSize;
  int32_t flag;
  int32_t mFactor;
  int32_t mFold;
  int32_t nAddFactor;
  int32_t groupMainCount;
  int32_t groupTailCount;
  int32_t addCount;
  int32_t addTailCount;
};

extern "C" {
DECLARE_CUMPROD(1, int8_t, 0);
DECLARE_CUMPROD(1, uint8_t, 0);
DECLARE_CUMPROD(1, int16_t, 0);
DECLARE_CUMPROD(1, uint16_t, 0);
DECLARE_CUMPROD(1, int32_t, 0);
DECLARE_CUMPROD(1, uint32_t, 0);
DECLARE_CUMPROD(1, half, 0);
DECLARE_CUMPROD(1, float, 0);
DECLARE_CUMPROD(1, bfloat16_t, 0);

DECLARE_CUMPROD(2, int8_t, 0);
DECLARE_CUMPROD(2, uint8_t, 0);
DECLARE_CUMPROD(2, int16_t, 0);
DECLARE_CUMPROD(2, uint16_t, 0);
DECLARE_CUMPROD(2, int32_t, 0);
DECLARE_CUMPROD(2, uint32_t, 0);
DECLARE_CUMPROD(2, half, 0);
DECLARE_CUMPROD(2, float, 0);
DECLARE_CUMPROD(2, bfloat16_t, 0);

DECLARE_CUMPROD(2, int8_t, 1);
DECLARE_CUMPROD(2, uint8_t, 1);
DECLARE_CUMPROD(2, int16_t, 1);
DECLARE_CUMPROD(2, uint16_t, 1);
DECLARE_CUMPROD(2, int32_t, 1);
DECLARE_CUMPROD(2, uint32_t, 1);
DECLARE_CUMPROD(2, half, 1);
DECLARE_CUMPROD(2, float, 1);
DECLARE_CUMPROD(2, bfloat16_t, 1);

DECLARE_CUMPROD(3, int8_t, 0);
DECLARE_CUMPROD(3, uint8_t, 0);
DECLARE_CUMPROD(3, int16_t, 0);
DECLARE_CUMPROD(3, uint16_t, 0);
DECLARE_CUMPROD(3, int32_t, 0);
DECLARE_CUMPROD(3, uint32_t, 0);
DECLARE_CUMPROD(3, half, 0);
DECLARE_CUMPROD(3, float, 0);
DECLARE_CUMPROD(3, bfloat16_t, 0);

DECLARE_CUMPROD(3, int8_t, 1);
DECLARE_CUMPROD(3, uint8_t, 1);
DECLARE_CUMPROD(3, int16_t, 1);
DECLARE_CUMPROD(3, uint16_t, 1);
DECLARE_CUMPROD(3, int32_t, 1);
DECLARE_CUMPROD(3, uint32_t, 1);
DECLARE_CUMPROD(3, half, 1);
DECLARE_CUMPROD(3, float, 1);
DECLARE_CUMPROD(3, bfloat16_t, 1);

DECLARE_CUMSUM(1, int8_t, 0);
DECLARE_CUMSUM(1, uint8_t, 0);
DECLARE_CUMSUM(1, int16_t, 0);
DECLARE_CUMSUM(1, uint16_t, 0);
DECLARE_CUMSUM(1, int32_t, 0);
DECLARE_CUMSUM(1, uint32_t, 0);
DECLARE_CUMSUM(1, half, 0);
DECLARE_CUMSUM(1, float, 0);
DECLARE_CUMSUM(1, bfloat16_t, 0);

DECLARE_CUMSUM(2, int8_t, 0);
DECLARE_CUMSUM(2, uint8_t, 0);
DECLARE_CUMSUM(2, int16_t, 0);
DECLARE_CUMSUM(2, uint16_t, 0);
DECLARE_CUMSUM(2, int32_t, 0);
DECLARE_CUMSUM(2, uint32_t, 0);
DECLARE_CUMSUM(2, half, 0);
DECLARE_CUMSUM(2, float, 0);
DECLARE_CUMSUM(2, bfloat16_t, 0);

DECLARE_CUMSUM(2, int8_t, 1);
DECLARE_CUMSUM(2, uint8_t, 1);
DECLARE_CUMSUM(2, int16_t, 1);
DECLARE_CUMSUM(2, uint16_t, 1);
DECLARE_CUMSUM(2, int32_t, 1);
DECLARE_CUMSUM(2, uint32_t, 1);
DECLARE_CUMSUM(2, half, 1);
DECLARE_CUMSUM(2, float, 1);
DECLARE_CUMSUM(2, bfloat16_t, 1);

DECLARE_CUMSUM(3, int8_t, 0);
DECLARE_CUMSUM(3, uint8_t, 0);
DECLARE_CUMSUM(3, int16_t, 0);
DECLARE_CUMSUM(3, uint16_t, 0);
DECLARE_CUMSUM(3, int32_t, 0);
DECLARE_CUMSUM(3, uint32_t, 0);
DECLARE_CUMSUM(3, half, 0);
DECLARE_CUMSUM(3, float, 0);
DECLARE_CUMSUM(3, bfloat16_t, 0);

DECLARE_CUMSUM(3, int8_t, 1);
DECLARE_CUMSUM(3, uint8_t, 1);
DECLARE_CUMSUM(3, int16_t, 1);
DECLARE_CUMSUM(3, uint16_t, 1);
DECLARE_CUMSUM(3, int32_t, 1);
DECLARE_CUMSUM(3, uint32_t, 1);
DECLARE_CUMSUM(3, half, 1);
DECLARE_CUMSUM(3, float, 1);
DECLARE_CUMSUM(3, bfloat16_t, 1);
}
#endif // defined(__DAV_C310__)
#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMULATIVE_UTILS_H