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

#ifndef HIVM_MLIR_TEMPLATE_DMA_UTILS_H
#define HIVM_MLIR_TEMPLATE_DMA_UTILS_H

#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

constexpr uint16_t INTRIN_MAX_BURST_CNT = (1 << 12) - 1;             // 4095
constexpr uint32_t INTRIN_MAX_BURST_LEN = ((1 << 21) - 1) / 32 * 32; // 2097120
constexpr uint32_t INTRIN_MAX_GAP = (1lu << 32) - 1; // 4294967295
#if defined(__DAV_C310__)
constexpr uint64_t INTRIN_MAX_STRIDE = (1lu << 40) - 1; // 1099511627775
#endif

enum class PadMode : uint32_t {
  Null = 0, // if null, means not set
  FirstElem = 1,
  Value = 2
};

enum class EvictionPolicy : uint32_t { EvictFirst = 0, EvictLast = 1 };

enum class AtomicKind : uint32_t { None = 0, Add = 1, Max = 2, Min = 3 };

template <typename T>
__aiv__ __attribute__((always_inline)) bool isStrideAligned(int64_t s) {
  constexpr int64_t bytes = sizeof(T);
  return s * bytes % INTR_BYTES_PER_BLOCK == 0;
}

template <typename T>
__aiv__ __attribute__((always_inline)) int64_t calc_ub_gap(int64_t stride,
                                                           int64_t size) {
  constexpr int32_t bytes = sizeof(T);
  int32_t factor = 1;
  if constexpr (bytes == BYTES_B64) {
    factor = 2;
  }

  const int64_t n_elems_of_aligned_unit = UB_ALIGN_BYTES / bytes * factor;
  const int64_t gap = CEIL_DIV(stride * factor, n_elems_of_aligned_unit) -
                      CEIL_DIV(size * factor, n_elems_of_aligned_unit);
  return gap;
}

// Copy gm to cbuf intrin for 1d and 2d, all dtypes are supported.
// For copy 1d, burst_cnt is 1 and gap is 0.
// For copy 2d, burst_cnt, burst_len and gap are all needed to compute.
template <typename T>
__aicore__ __attribute__((always_inline)) void load_gm_to_cbuf_intrin_core(
    __gm__ T *gm_ptr, int64_t gm_offset, __cbuf__ T *cbuf_ptr,
    int64_t cbuf_offset, uint16_t burst_cnt, uint32_t burst_len,
    uint32_t src_gap, uint32_t dst_gap, pad_t pad_mode) {

  INTRINSIC(copy_gm_to_cbuf,
            cbuf_ptr + cbuf_offset, // dst
            gm_ptr + gm_offset,     // src
            0,                      // sid
            burst_cnt,              // burst_cnt
            burst_len,              // burst_len
            src_gap,                // src_gap
            dst_gap,                // dst_gap
            pad_t::PAD_NONE);       // pad_mode
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_1d_core_with_contiguous_last_dim(memref_t<__gm__ T, 1> *gm,
                                                 memref_t<__cbuf__ T, 1> *cbuf,
                                                 pad_t pad_mode) {
  auto src_ptr = gm->aligned + gm->offset;
  auto dst_ptr = cbuf->aligned + cbuf->offset;
  const int64_t size0 = gm->sizes[0];
  constexpr int bytes = sizeof(T);
  // burst_len of gm <-> l1 is in units of 32bytes.
  uint32_t burst_len = CEIL_DIV(size0 * bytes, L1_ALIGN_BYTES);
  load_gm_to_cbuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, burst_len, 0, 0,
                              pad_mode);
}

// Copy 2d, where the last dimension is contiguous i.e. stride1 = 1.
template <typename T>
__aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_2d_core_with_contiguous_last_dim(memref_t<__gm__ T, 2> *gm,
                                                 memref_t<__cbuf__ T, 2> *cbuf,
                                                 pad_t pad_mode) {
  int64_t size1 = gm->sizes[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride0_cbuf = cbuf->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t cbuf_gap = (stride0_cbuf - size1) * bytes / L1_ALIGN_BYTES;
  const int64_t gm_gap = (stride0_gm - size1) * bytes / L1_ALIGN_BYTES;

  if (gm_gap > INTRIN_MAX_GAP || 0 > cbuf_gap || 0 > stride0_gm - size1)
      [[unlikely]] {
    // special cases of stride, unlikely used
    const int64_t size0 = gm->sizes[0];
    for (int64_t i = 0; i < size0; i++) {
      memref_t<__gm__ T, 1> gm_1d = {gm->allocated,
                                     gm->aligned,
                                     gm->offset + stride0_gm * i,
                                     {size1},
                                     {1}};
      memref_t<__cbuf__ T, 1> cbuf_1d = {cbuf->allocated,
                                         cbuf->aligned,
                                         cbuf->offset + stride0_cbuf * i,
                                         {size1},
                                         {1}};
      load_gm_to_cbuf_1d_core_with_contiguous_last_dim<T>(&gm_1d, &cbuf_1d,
                                                          pad_mode);
    }
  } else {
    // split burst cnt into main and tail part
    int64_t size0 = gm->sizes[0];
    auto gm_ptr = gm->aligned + gm->offset;
    auto cbuf_ptr = cbuf->aligned + cbuf->offset;
    // burst_len of gm <-> l1 is in units of 32bytes.
    const int64_t burst_len = CEIL_DIV(size1 * bytes, L1_ALIGN_BYTES);
    const int32_t burst_cnt_main = size0 / INTRIN_MAX_BURST_CNT;
    const int32_t burst_cnt_tail = size0 % INTRIN_MAX_BURST_CNT;
    if (burst_cnt_main > 0) [[unlikely]] {
      // considering the max. ub size, burst_cnt_main can only <= 1
      load_gm_to_cbuf_intrin_core<T>(gm_ptr, 0, cbuf_ptr, 0,
                                     INTRIN_MAX_BURST_CNT, burst_len, gm_gap,
                                     cbuf_gap, pad_mode);
    }

    if (burst_cnt_tail > 0) [[likely]] {
      // tail part is often used, as tail part can cover more than half of
      // ub size
      const int64_t main_gm_offset =
          burst_cnt_main * INTRIN_MAX_BURST_CNT * stride0_gm;
      const int64_t main_cbuf_offset =
          burst_cnt_main * INTRIN_MAX_BURST_CNT * stride0_cbuf;
      load_gm_to_cbuf_intrin_core<T>(gm_ptr, main_gm_offset, cbuf_ptr,
                                     main_cbuf_offset, burst_cnt_tail,
                                     burst_len, gm_gap, cbuf_gap, pad_mode);
    }
  }
}

// Copy 3d, where the last dim is contiguous i.e. stride2=1
template <typename T>
__aicore__ __attribute__((always_inline)) void
load_gm_to_cbuf_3d_core_with_contiguous_last_dim(memref_t<__gm__ T, 3> *gm,
                                                 memref_t<__cbuf__ T, 3> *cbuf,
                                                 pad_t pad_mode) {
  int64_t size0 = gm->sizes[0];
  int64_t size1 = gm->sizes[1];
  int64_t size2 = gm->sizes[2];
  int64_t stride0_cbuf = cbuf->strides[0];
  int64_t stride1_cbuf = cbuf->strides[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride1_gm = gm->strides[1];
  if (size0 > size1) {
    size0 = size1;
    size1 = gm->sizes[0];
    stride0_cbuf = cbuf->strides[1];
    stride1_cbuf = cbuf->strides[0];
    stride0_gm = gm->strides[1];
    stride1_gm = gm->strides[0];
  }

  for (int64_t i = 0; i < size0; i++) {
    memref_t<__gm__ T, 2> gm_2d = {gm->allocated,
                                   gm->aligned,
                                   gm->offset + stride0_gm * i,
                                   {size1, size2},
                                   {stride1_gm, 1}};
    memref_t<__cbuf__ T, 2> cbuf_2d = {cbuf->allocated,
                                       cbuf->aligned,
                                       cbuf->offset + stride0_cbuf * i,
                                       {size1, size2},
                                       {stride1_cbuf, 1}};

    load_gm_to_cbuf_2d_core_with_contiguous_last_dim<T>(&gm_2d, &cbuf_2d,
                                                        pad_mode);
  }
}

#if defined(__DAV_C310__)
// Copy gm to ubuf intrin for 1d and 2d, all dtypes are supported.
// For copy 1d, burst_cnt is 1 and gap is 0.
// For copy 2d, burst_cnt, burst_len and gap are all needed to compute.
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_intrin_core(
    __gm__ T *gm_ptr, int64_t gm_offset, __ubuf__ T *ub_ptr, int64_t ub_offset,
    uint16_t burst_cnt, uint32_t burst_len, int64_t left_padding_num,
    uint64_t src_gap, uint32_t dst_gap, uint8_t l2_cache_ctl) {
  constexpr int32_t bytes = sizeof(T);
  // Use rightPaddingNum to pad and the pad_value is 0 as default.
  auto sizeWithPad = burst_len + (left_padding_num * bytes);
  auto rightPaddingNum =
      (CEIL_FACTOR(sizeWithPad, UB_ALIGN_BYTES) - sizeWithPad) / bytes;
  uint32_t dst_align_gap = dst_gap / UB_ALIGN_BYTES;
  // TODO: figure out what are these fields for
  bool data_select_bit = true;
  const uint64_t src_stride = src_gap + burst_len;
  const uint32_t dst_stride = dst_gap + burst_len;
  if (left_padding_num == 0)
    rightPaddingNum = 0;
  if constexpr (bytes == BYTES_B8 || bytes == BYTES_B16 || bytes == BYTES_B32) {
    INTRINSIC(copy_gm_to_ubuf_align_v2, (ub_ptr + ub_offset - left_padding_num),
              gm_ptr + gm_offset, 0, burst_cnt, burst_len, left_padding_num,
              rightPaddingNum, data_select_bit, l2_cache_ctl, src_stride,
              dst_stride);
  } else if constexpr (bytes == BYTES_B64) {
    INTRINSIC(copy_gm_to_ubuf_align_v2,
              reinterpret_cast<__ubuf__ uint32_t *>(ub_ptr + ub_offset -
                                                    left_padding_num),
              reinterpret_cast<__gm__ uint32_t *>(gm_ptr + gm_offset), 0,
              burst_cnt, burst_len, left_padding_num, rightPaddingNum,
              data_select_bit, l2_cache_ctl, src_stride, dst_stride);
  }
}
#else
///========================================
/// m300 arch
///========================================
// Copy gm to ubuf intrin for 1d and 2d, all dtypes are supported.
// For copy 1d, burst_cnt is 1 and gap is 0.
// For copy 2d, burst_cnt, burst_len and gap are all needed to compute.
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_intrin_core(
    __gm__ T *gm_ptr, int64_t gm_offset, __ubuf__ T *ub_ptr, int64_t ub_offset,
    uint16_t burst_cnt, uint32_t burst_len, int64_t left_padding_num,
    uint64_t src_gap, uint32_t dst_gap, uint8_t l2_cache_ctl) {
  constexpr int32_t bytes = sizeof(T);
  // Use rightPaddingNum to pad and the pad_value is 0 as default.
  auto sizeWithPad = burst_len + (left_padding_num * bytes);
  auto rightPaddingNum =
      (CEIL_FACTOR(sizeWithPad, UB_ALIGN_BYTES) - sizeWithPad) / bytes;
  uint32_t dst_align_gap = dst_gap / UB_ALIGN_BYTES;
  if (left_padding_num == 0)
    rightPaddingNum = 0;
  if constexpr (bytes == BYTES_B8 || bytes == BYTES_B16 || bytes == BYTES_B32) {
    INTRINSIC(copy_gm_to_ubuf_align, (ub_ptr + ub_offset - left_padding_num),
              gm_ptr + gm_offset, 0, burst_cnt, burst_len, left_padding_num,
              rightPaddingNum, src_gap, dst_align_gap);
  } else if constexpr (bytes == BYTES_B64) {
    INTRINSIC(copy_gm_to_ubuf_align,
              reinterpret_cast<__ubuf__ uint32_t *>(ub_ptr + ub_offset -
                                                    left_padding_num),
              reinterpret_cast<__gm__ uint32_t *>(gm_ptr + gm_offset), 0,
              burst_cnt, burst_len, left_padding_num, rightPaddingNum, src_gap,
              dst_align_gap);
  }
}
#endif

#if defined(__DAV_C310__)
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_intrin_core(__ubuf__ T *ub_ptr, int64_t ub_offset,
                             __gm__ T *gm_ptr, int64_t gm_offset,
                             uint16_t burst_cnt, uint32_t burst_len,
                             uint32_t src_gap, uint64_t dst_gap) {
  constexpr int32_t bytes = sizeof(T);
  uint32_t src_align_gap = src_gap / UB_ALIGN_BYTES;
  uint8_t l2_cache_ctl = 0;
  const uint32_t src_stride = src_gap + burst_len;
  const uint64_t dst_stride = dst_gap + burst_len;
  if constexpr (bytes == BYTES_B8 || bytes == BYTES_B16 || bytes == BYTES_B32) {
    INTRINSIC(copy_ubuf_to_gm_align_v2, gm_ptr + gm_offset, ub_ptr + ub_offset,
              0, burst_cnt, burst_len, l2_cache_ctl, dst_stride, src_stride);
  } else if constexpr (bytes == BYTES_B64) {
    int32_t factor = 2;
    INTRINSIC(copy_ubuf_to_gm_align_v2,
              reinterpret_cast<__gm__ uint32_t *>(gm_ptr) + gm_offset * factor,
              reinterpret_cast<__ubuf__ uint32_t *>(ub_ptr) +
                  ub_offset * factor,
              0, burst_cnt, burst_len, l2_cache_ctl, dst_stride, src_stride);
  }
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_intrin_core(__ubuf__ T *ub_ptr, int64_t ub_offset,
                             __gm__ T *gm_ptr, int64_t gm_offset,
                             uint16_t burst_cnt, uint32_t burst_len,
                             uint32_t src_gap, uint64_t dst_gap) {
  constexpr int32_t bytes = sizeof(T);
  uint32_t src_align_gap = src_gap / UB_ALIGN_BYTES;
  if constexpr (bytes == BYTES_B8 || bytes == BYTES_B16 || bytes == BYTES_B32) {
    INTRINSIC(copy_ubuf_to_gm_align, gm_ptr + gm_offset, ub_ptr + ub_offset, 0,
              burst_cnt, burst_len, src_align_gap, dst_gap);
  } else if constexpr (bytes == BYTES_B64) {
    int32_t factor = 2;
    INTRINSIC(copy_ubuf_to_gm_align,
              reinterpret_cast<__gm__ uint32_t *>(gm_ptr) + gm_offset * factor,
              reinterpret_cast<__ubuf__ uint32_t *>(ub_ptr) +
                  ub_offset * factor,
              0, burst_cnt, burst_len, src_align_gap, dst_gap);
  }
}
#endif

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_1d_core(memref_t<__ubuf__ T, 1> *src,
                          memref_t<__ubuf__ T, 1> *dst);
#if defined(__DAV_C310__)
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_1d_core(memref_t<__ubuf__ T, 1> *src,
                          memref_t<__cbuf__ T, 1> *dst);
#endif

#define REGISTE_COPY_UB_TO_UB_1D_FUNC(type)                                    \
  template __aiv__ __attribute__((always_inline)) void                         \
  copy_ubuf_to_ubuf_1d_core<type>(memref_t<__ubuf__ type, 1> * src,            \
                                  memref_t<__ubuf__ type, 1> * dst)

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_1d_by_scalar(memref_t<__gm__ T, 1> *src,
                             memref_t<__ubuf__ T, 1> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < src->sizes[0]; i++) {
    *(dst_ptr + i * dst->strides[0]) = *(src_ptr + i * src->strides[0]);
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_2d_by_scalar(memref_t<__gm__ T, 2> *src,
                             memref_t<__ubuf__ T, 2> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < src->sizes[0]; ++i) {
    for (int j = 0; j < src->sizes[1]; ++j) {
      *(dst_ptr + i * dst->strides[0] + j * dst->strides[1]) =
          *(src_ptr + i * src->strides[0] + j * src->strides[1]);
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_3d_by_scalar(memref_t<__gm__ T, 3> *src,
                             memref_t<__ubuf__ T, 3> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  for (int i = 0; i < src->sizes[0]; ++i) {
    memref_t<__gm__ T, 2> src_2d{src->allocated,
                                 src->aligned,
                                 src->offset + i * src->strides[0],
                                 {src->sizes[1], src->sizes[2]},
                                 {src->strides[1], src->strides[2]}};
    memref_t<__ubuf__ T, 2> dst_2d{dst->allocated,
                                   dst->aligned,
                                   dst->offset + i * dst->strides[0],
                                   {dst->sizes[1], dst->sizes[2]},
                                   {dst->strides[1], dst->strides[2]}};
    load_gm_to_ubuf_2d_by_scalar<T>(&src_2d, &dst_2d);
  }
}

// TODO: After "removing the extract slice and inserting it into Deinterleave
// on A5", issues may arise such as tail shaft strides not being 1 or block
// misalignment. Temporarily use scalar to circumvent.
// TODO: when multiple cores simultaneously write data to a section of GM buffer
// with cacheline overlapped, the written buffer may be overwritten by following
// cores. Need to use st_dev intrinsic.
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_1d_by_scalar(memref_t<__ubuf__ T, 1> *src,
                              memref_t<__gm__ T, 1> *dst) {
  constexpr int bytes = sizeof(T);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < src->sizes[0]; i++) {
    __gm__ T *gm_ptr = dst_ptr + i * dst->strides[0];
    __ubuf__ T *ub_ptr = src_ptr + i * src->strides[0];
#if defined(__DAV_C310__)
    if constexpr (bytes == BYTES_B32 || bytes == BYTES_B64) {
      INTRINSIC(st_dev, (T)*ub_ptr, (__gm__ T *)gm_ptr, 0);
    } else if constexpr (bytes == BYTES_B16) {
      auto *ub_ptr_u = reinterpret_cast<__ubuf__ uint16_t *>(ub_ptr);
      auto *gm_ptr_u = reinterpret_cast<__gm__ uint16_t *>(gm_ptr);
      INTRINSIC(st_dev, *ub_ptr_u, gm_ptr_u, 0);
    } else if constexpr (bytes == BYTES_B8) {
      auto *ub_ptr_u = reinterpret_cast<__ubuf__ uint8_t *>(ub_ptr);
      auto *gm_ptr_u = reinterpret_cast<__gm__ uint8_t *>(gm_ptr);
      INTRINSIC(st_dev, *ub_ptr_u, gm_ptr_u, 0);
    }
#else
    *gm_ptr = *ub_ptr;
#endif
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE3, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE3, LIB_EVENT_ID0);
}

// TODO: when multiple cores simultaneously write data to a section of GM buffer
// with cacheline overlapped, the written buffer may be overwritten by following
// cores. Need to use st_dev intrinsic.
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_2d_by_scalar(memref_t<__ubuf__ T, 2> *src,
                              memref_t<__gm__ T, 2> *dst) {
  constexpr int bytes = sizeof(T);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE3, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < src->sizes[0]; i++) {
    for (int j = 0; j < src->sizes[1]; j++) {
      __gm__ T *gm_ptr = dst_ptr + i * dst->strides[0] + j * dst->strides[1];
      __ubuf__ T *ub_ptr = src_ptr + i * src->strides[0] + j * src->strides[1];
#if defined(__DAV_C310__)
      if constexpr (bytes == BYTES_B32 || bytes == BYTES_B64) {
        INTRINSIC(st_dev, (T)*ub_ptr, (__gm__ T *)gm_ptr, 0);
      } else if constexpr (bytes == BYTES_B16) {
        auto *ub_ptr_u = reinterpret_cast<__ubuf__ uint16_t *>(ub_ptr);
        auto *gm_ptr_u = reinterpret_cast<__gm__ uint16_t *>(gm_ptr);
        INTRINSIC(st_dev, *ub_ptr_u, gm_ptr_u, 0);
      } else if constexpr (bytes == BYTES_B8) {
        auto *ub_ptr_u = reinterpret_cast<__ubuf__ uint8_t *>(ub_ptr);
        auto *gm_ptr_u = reinterpret_cast<__gm__ uint8_t *>(gm_ptr);
        INTRINSIC(st_dev, *ub_ptr_u, gm_ptr_u, 0);
      }
#else
      *gm_ptr = *ub_ptr;
#endif
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE3, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE3, LIB_EVENT_ID0);
}

// TODO: when multiple cores simultaneously write data to a section of GM buffer
// with cacheline overlapped, the written buffer may be overwritten by following
// cores. Need to use st_dev intrinsic.
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_3d_by_scalar(memref_t<__ubuf__ T, 3> *src,
                              memref_t<__gm__ T, 3> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  for (int i = 0; i < src->sizes[0]; ++i) {
    memref_t<__ubuf__ T, 2> src_2d{src->allocated,
                                   src->aligned,
                                   src->offset + i * src->strides[0],
                                   {src->sizes[1], src->sizes[2]},
                                   {src->strides[1], src->strides[2]}};
    memref_t<__gm__ T, 2> dst_2d{dst->allocated,
                                 dst->aligned,
                                 dst->offset + i * dst->strides[0],
                                 {dst->sizes[1], dst->sizes[2]},
                                 {dst->strides[1], dst->strides[2]}};
    store_ubuf_to_gm_2d_by_scalar<T>(&src_2d, &dst_2d);
  }
}

/// `load_gm_to_ubuf_intrin_core` with `numBurst = 1, burst_len = size[0] *
/// dtypeBytes` Prerequisites: `strideUB[0] == strideGM[0] == 1`
template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_1d_core_with_contiguous_last_dim(memref_t<__gm__ T, 1> *gm,
                                                 memref_t<__ubuf__ T, 1> *ub,
                                                 int64_t left_padding_num,
                                                 uint8_t l2_cache_ctl) {
  auto src_ptr = gm->aligned + gm->offset;
  auto dst_ptr = ub->aligned + ub->offset;
  const int64_t size0 = gm->sizes[0];
  constexpr int bytes = sizeof(T);

  load_gm_to_ubuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, size0 * bytes,
                              left_padding_num, 0, 0, l2_cache_ctl);
}

/// `store_ubuf_to_gm_intrin_core` with `numBurst = 1, burst_len = size[0] *
/// dtypeBytes` Prerequisites: `strideUB[0] == strideGM[0] == 1`
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_1d_core_with_contiguous_last_dim(memref_t<__ubuf__ T, 1> *ub,
                                                  memref_t<__gm__ T, 1> *gm) {
  auto src_ptr = ub->aligned + ub->offset;
  auto dst_ptr = gm->aligned + gm->offset;
  const int64_t size0 = ub->sizes[0];
  constexpr int bytes = sizeof(T);

  store_ubuf_to_gm_intrin_core(src_ptr, 0, dst_ptr, 0, 1, size0 * bytes, 0, 0);
}

#if defined(__DAV_C310__)
/// 2D GM -> UB core func mainly using DMA
/// Prerequisites: `strideUB[1] == strideGM[1] == 1` and startUBAddr is 32B
/// aligned
/// + `UB|GM stride[0] < size[1] | GM stride[0] > hardware maximum limit`
///   Decompose into for-loop, converting to 1D each time
///   - UB `stride2D[0] * dtypeBytes % 32 == 0`
///      ```
///      for i = [0, size2D[0]) {
///        offset1D = offset2D + i * stride2D[0], size1D = [size2D[1]], stride1D
///        = [stride2D[1]] load_gm_to_ubuf_1d_core_with_contiguous_last_dim
///      }
///      ```
///   - UB `stride2D[0] * dtypeBytes % 32 != 0`
///     `load_gm_to_ubuf_2d_by_scalar`
///     Fall back to looped scalar transfer (~~consider directly delegating to
///     1D scenario, relying on non-aligned starting address handling in 1D
///     scenario function to fall back to looped scalar transfer; no judgment
///     made here, uniformly decompose into for-loop~~, not considered, because
///     looped scalar transfer implementation requires inserting synchronization
///     before and after the for-loop; emitting multi-dimensional loops together
///     is more efficient)
/// + Default (`UB&GM stride[0] >= size[1] & GM stride[0] <= hardware maximum
/// limit`)
///   `numBurst = size[0], burst_len = size[1] * dtypeBytes`
///   - consider `size[0] > hardware-allowed max burst cnt for a single
///   instruction emission`
///     Split `size[0]` into multiple groups of max burst cnt, handle the tail
///     block separately. max burst cnt = 4096, which guarantees UB `stride[0] *
///     max burst cnt * dtypeBytes % 32B == 0`.
///     ```
///     for i = [0, size[0] / max burst cnt) {
///       numBurst = max burst cnt, burst_len = size[1] * dtypeBytes
///       load_gm_to_ubuf_intrin_core
///     }
///     Tail block: numBurst = size[0] % max burst cnt, burst_len = size[1] *
///     dtypeBytes
///                 load_gm_to_ubuf_intrin_core
///     ```
template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_2d_core_with_contiguous_last_dim(memref_t<__gm__ T, 2> *gm,
                                                 memref_t<__ubuf__ T, 2> *ub,
                                                 int64_t left_padding_num,
                                                 uint8_t l2_cache_ctl) {
  const int64_t size0 = gm->sizes[0];
  const int64_t size1 = gm->sizes[1];
  const int64_t stride0_gm = gm->strides[0];
  const int64_t stride0_ub = ub->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t ub_gap = (stride0_ub - size1) * bytes;
  const int64_t gm_gap = (stride0_gm - size1) * bytes;
  bool is_over_gap = stride0_gm * bytes > INTRIN_MAX_STRIDE;
  if (is_over_gap || 0 > ub_gap || 0 > gm_gap) [[unlikely]] {
    // special cases of stride, unlikely used
    if (stride0_ub * bytes % INTR_BYTES_PER_BLOCK != 0) {
      load_gm_to_ubuf_2d_by_scalar<T>(gm, ub);
      return;
    }
    // then we ensure call of load_gm_to_ubuf_1d_core_with_contiguous_last_dim
    // with 32B aligned start UB address.
    for (int64_t i = 0; i < size0; i++) {
      memref_t<__gm__ T, 1> gm_1d = {gm->allocated,
                                     gm->aligned,
                                     gm->offset + stride0_gm * i,
                                     {size1},
                                     {1}};
      memref_t<__ubuf__ T, 1> ub_1d = {ub->allocated,
                                       ub->aligned,
                                       ub->offset + stride0_ub * i,
                                       {size1},
                                       {1}};
      load_gm_to_ubuf_1d_core_with_contiguous_last_dim<T>(
          &gm_1d, &ub_1d, left_padding_num, l2_cache_ctl);
    }
    return;
  }

  // split burst cnt into main and tail part
  auto gm_ptr = gm->aligned + gm->offset;
  auto ub_ptr = ub->aligned + ub->offset;
  const int64_t burst_len = size1 * bytes;
  uint16_t num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  uint16_t alignd_intrin_max_burst_cnt =
      FLOOR_FACTOR(INTRIN_MAX_BURST_CNT, num_per_block);
  const int32_t burst_cnt_main = size0 / alignd_intrin_max_burst_cnt;
  const int32_t burst_cnt_tail = size0 % alignd_intrin_max_burst_cnt;
  if (burst_cnt_main > 0) [[unlikely]] {
    // when 1d->2d, the main_gm_offset maybe larger than 1
    for (int64_t i = 0; i < burst_cnt_main; i++) {
      const int64_t main_gm_offset =
          i * alignd_intrin_max_burst_cnt * stride0_gm;
      const int64_t main_ub_offset =
          i * alignd_intrin_max_burst_cnt * stride0_ub;
      load_gm_to_ubuf_intrin_core<T>(
          gm_ptr, main_gm_offset, ub_ptr, main_ub_offset,
          alignd_intrin_max_burst_cnt, burst_len, left_padding_num, gm_gap,
          ub_gap, l2_cache_ctl);
    }
  }
  if (burst_cnt_tail > 0) [[likely]] {
    // tail part is often used, as tail part can cover more than half of
    // ub size
    const int64_t main_gm_offset =
        burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_gm;
    const int64_t main_ub_offset =
        burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_ub;
    load_gm_to_ubuf_intrin_core<T>(
        gm_ptr, main_gm_offset, ub_ptr, main_ub_offset, burst_cnt_tail,
        burst_len, left_padding_num, gm_gap, ub_gap, l2_cache_ctl);
  }
  return;
}
#else
///========================================
/// m300 arch
///========================================
// Copy 2d, where the last dimension is contiguous i.e. stride1 = 1.
template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_2d_core_with_contiguous_last_dim(memref_t<__gm__ T, 2> *gm,
                                                 memref_t<__ubuf__ T, 2> *ub,
                                                 int64_t left_padding_num,
                                                 uint8_t l2_cache_ctl) {
  int64_t size1 = gm->sizes[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride0_ub = ub->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t ub_gap = (stride0_ub - size1) * bytes;
  const int64_t gm_gap = (stride0_gm - size1) * bytes;
  bool is_over_gap = gm_gap > INTRIN_MAX_GAP;
  if (is_over_gap || 0 > ub_gap || 0 > stride0_gm - size1) [[unlikely]] {
    // special cases of stride, unlikely used
    const int64_t size0 = gm->sizes[0];
    for (int64_t i = 0; i < size0; i++) {

      memref_t<__gm__ T, 1> gm_1d = {gm->allocated,
                                     gm->aligned,
                                     gm->offset + stride0_gm * i,
                                     {size1},
                                     {1}};
      memref_t<__ubuf__ T, 1> ub_1d = {ub->allocated,
                                       ub->aligned,
                                       ub->offset + stride0_ub * i,
                                       {size1},
                                       {1}};
      load_gm_to_ubuf_1d_core_with_contiguous_last_dim<T>(
          &gm_1d, &ub_1d, left_padding_num, l2_cache_ctl);
    }
  } else {
    // split burst cnt into main and tail part
    int64_t size0 = gm->sizes[0];
    auto gm_ptr = gm->aligned + gm->offset;
    auto ub_ptr = ub->aligned + ub->offset;
    const int64_t burst_len = size1 * bytes;
    uint16_t num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
    uint16_t alignd_intrin_max_burst_cnt =
        FLOOR_FACTOR(INTRIN_MAX_BURST_CNT, num_per_block);
    const int32_t burst_cnt_main = size0 / alignd_intrin_max_burst_cnt;
    const int32_t burst_cnt_tail = size0 % alignd_intrin_max_burst_cnt;
    if (burst_cnt_main > 0) [[unlikely]] {
      // when 1d->2d, the main_gm_offset maybe larger than 1
      for (int64_t i = 0; i < burst_cnt_main; i++) {
        const int64_t main_gm_offset =
            i * alignd_intrin_max_burst_cnt * stride0_gm;
        const int64_t main_ub_offset =
            i * alignd_intrin_max_burst_cnt * stride0_ub;
        load_gm_to_ubuf_intrin_core<T>(
            gm_ptr, main_gm_offset, ub_ptr, main_ub_offset,
            alignd_intrin_max_burst_cnt, burst_len, left_padding_num, gm_gap,
            ub_gap, l2_cache_ctl);
      }
    }
    if (burst_cnt_tail > 0) [[likely]] {
      // tail part is often used, as tail part can cover more than half of
      // ub size
      const int64_t main_gm_offset =
          burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_gm;
      const int64_t main_ub_offset =
          burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_ub;
      load_gm_to_ubuf_intrin_core<T>(
          gm_ptr, main_gm_offset, ub_ptr, main_ub_offset, burst_cnt_tail,
          burst_len, left_padding_num, gm_gap, ub_gap, l2_cache_ctl);
    }
  }
}
#endif

#if defined(__DAV_C310__)
/// 2D UB -> GM core func mainly using DMA
/// Prerequisites: `strideUB[1] == strideGM[1] == 1` and startUBAddr is 32B
/// aligned
/// + `UB|GM stride[0] < size[1] | GM stride[0] > hardware maximum limit`
///   Decompose into for-loop, converting to 1D each time
///   - UB `stride2D[0] * dtypeBytes % 32 == 0`
///      ```
///      for i = [0, size2D[0]) {
///        offset1D = offset2D + i * stride2D[0], size1D = [size2D[1]], stride1D
///        = [stride2D[1]] store_ubuf_to_gm_1d_core_with_contiguous_last_dim
///      }
///      ```
///   - UB `stride2D[0] * dtypeBytes % 32 != 0`
///     Fall back to looped scalar transfer
///     `store_ubuf_to_gm_2d_by_scalar`
/// + Default (`UB&GM stride[0] >= size[1] & GM stride[0] <= hardware maximum
/// limit`)
///   `numBurst = size[0], burst_len = size[1] * dtypeBytes`
///   - consider `size[0] > hardware-allowed max burst cnt for a single
///   instruction emission`
///     Split `size[0]` into multiple groups of max burst cnt, handle the tail
///     block separately. max burst cnt = 4096, which guarantees UB `stride[0] *
///     max burst cnt * dtypeBytes % 32B == 0`.
///     ```
///     for i = [0, size[0] / max burst cnt) {
///       numBurst = max burst cnt, burst_len = size[1] * dtypeBytes
///       store_ubuf_to_gm_intrin_core
///     }
///     Tail block: numBurst = size[0] % max burst cnt, burst_len = size[1] *
///     dtypeBytes
///                 store_ubuf_to_gm_intrin_core
///     ```
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_2d_core_with_contiguous_last_dim(memref_t<__ubuf__ T, 2> *ub,
                                                  memref_t<__gm__ T, 2> *gm) {
  const int64_t size0 = ub->sizes[0];
  const int64_t size1 = ub->sizes[1];
  const int64_t stride0_ub = ub->strides[0];
  const int64_t stride0_gm = gm->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t ub_gap = (stride0_ub - size1) * bytes;
  const int64_t gm_gap = (stride0_gm - size1) * bytes;
  bool is_over_gap = stride0_gm * bytes > INTRIN_MAX_STRIDE;
  if (is_over_gap || 0 > ub_gap || 0 > gm_gap) [[unlikely]] {
    // unlikely used, special cases of stride
    if (stride0_ub * bytes % INTR_BYTES_PER_BLOCK != 0) {
      store_ubuf_to_gm_2d_by_scalar<T>(ub, gm);
      return;
    }
    // then we ensure call of store_ubuf_to_gm_1d_core_with_contiguous_last_dim
    // with 32B aligned start UB address.
    for (int64_t i = 0; i < size0; i++) {
      memref_t<__ubuf__ T, 1> ub_1d = {ub->allocated,
                                       ub->aligned,
                                       ub->offset + stride0_ub * i,
                                       {size1},
                                       {1}};
      memref_t<__gm__ T, 1> gm_1d = {gm->allocated,
                                     gm->aligned,
                                     gm->offset + stride0_gm * i,
                                     {size1},
                                     {1}};
      store_ubuf_to_gm_1d_core_with_contiguous_last_dim<T>(&ub_1d, &gm_1d);
    }
    return;
  }

  auto ub_ptr = ub->aligned + ub->offset;
  auto gm_ptr = gm->aligned + gm->offset;
  const int64_t burst_len = size1 * bytes;
  uint16_t num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  uint16_t alignd_intrin_max_burst_cnt =
      FLOOR_FACTOR(INTRIN_MAX_BURST_CNT, num_per_block);
  const int32_t burst_cnt_main = size0 / alignd_intrin_max_burst_cnt;
  const int32_t burst_cnt_tail = size0 % alignd_intrin_max_burst_cnt;
  if (burst_cnt_main > 0) [[unlikely]] {
    // when 1d->2d, the main_gm_offset maybe larger than 1
    for (int64_t i = 0; i < burst_cnt_main; i++) {
      const int64_t main_gm_offset =
          i * alignd_intrin_max_burst_cnt * stride0_gm;
      const int64_t main_ub_offset =
          i * alignd_intrin_max_burst_cnt * stride0_ub;
      store_ubuf_to_gm_intrin_core<T>(
          ub_ptr, main_ub_offset, gm_ptr, main_gm_offset,
          alignd_intrin_max_burst_cnt, burst_len, ub_gap, gm_gap);
    }
  }

  if (burst_cnt_tail > 0) [[likely]] {
    // tail part is often used, as tail part can cover more than half of
    // ub size
    const int64_t main_gm_offset =
        burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_gm;
    const int64_t main_ub_offset =
        burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_ub;
    store_ubuf_to_gm_intrin_core<T>(ub_ptr, main_ub_offset, gm_ptr,
                                    main_gm_offset, burst_cnt_tail, burst_len,
                                    ub_gap, gm_gap);
  }
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_2d_core_with_contiguous_last_dim(memref_t<__ubuf__ T, 2> *ub,
                                                  memref_t<__gm__ T, 2> *gm) {
  int64_t size1 = ub->sizes[1];
  int64_t stride0_ub = ub->strides[0];
  int64_t stride0_gm = gm->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t ub_gap = (stride0_ub - size1) * bytes;
  const int64_t gm_gap = (stride0_gm - size1) * bytes;
  bool is_over_gap = gm_gap > INTRIN_MAX_GAP;
  if (is_over_gap || 0 > ub_gap || 0 > stride0_gm - size1) [[unlikely]] {
    // unlikely used, special cases of stride
    const int64_t size0 = ub->sizes[0];
    for (int64_t i = 0; i < size0; i++) {

      memref_t<__ubuf__ T, 1> ub_1d = {ub->allocated,
                                       ub->aligned,
                                       ub->offset + stride0_ub * i,
                                       {size1},
                                       {1}};
      memref_t<__gm__ T, 1> gm_1d = {gm->allocated,
                                     gm->aligned,
                                     gm->offset + stride0_gm * i,
                                     {size1},
                                     {1}};
      store_ubuf_to_gm_1d_core_with_contiguous_last_dim<T>(&ub_1d, &gm_1d);
    }
  } else {
    int64_t size0 = ub->sizes[0];
    auto ub_ptr = ub->aligned + ub->offset;
    auto gm_ptr = gm->aligned + gm->offset;
    const int64_t burst_len = size1 * bytes;
    uint16_t num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
    uint16_t alignd_intrin_max_burst_cnt =
        FLOOR_FACTOR(INTRIN_MAX_BURST_CNT, num_per_block);
    const int32_t burst_cnt_main = size0 / alignd_intrin_max_burst_cnt;
    const int32_t burst_cnt_tail = size0 % alignd_intrin_max_burst_cnt;
    if (burst_cnt_main > 0) [[unlikely]] {
      // when 1d->2d, the main_gm_offset maybe larger than 1
      for (int64_t i = 0; i < burst_cnt_main; i++) {
        const int64_t main_gm_offset =
            i * alignd_intrin_max_burst_cnt * stride0_gm;
        const int64_t main_ub_offset =
            i * alignd_intrin_max_burst_cnt * stride0_ub;
        store_ubuf_to_gm_intrin_core<T>(
            ub_ptr, main_ub_offset, gm_ptr, main_gm_offset,
            alignd_intrin_max_burst_cnt, burst_len, ub_gap, gm_gap);
      }
    }

    if (burst_cnt_tail > 0) [[likely]] {
      // tail part is often used, as tail part can cover more than half of
      // ub size
      const int64_t main_gm_offset =
          burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_gm;
      const int64_t main_ub_offset =
          burst_cnt_main * alignd_intrin_max_burst_cnt * stride0_ub;
      store_ubuf_to_gm_intrin_core<T>(ub_ptr, main_ub_offset, gm_ptr,
                                      main_gm_offset, burst_cnt_tail, burst_len,
                                      ub_gap, gm_gap);
    }
  }
}
#endif

#if defined(__DAV_C310__)
/// 3D GM -> UB core func mainly using DMA
/// + `UB&GM stride[1] == stride[2] * size[2]`: dimension 1 is contiguous
///   - Convert to 2D: `offset2D = offset3D, size2D = [size3D[0], size3D[1] *
///   size3D[2]], stride2D = [stride3D[0], stride[2]]`
///     `load_gm_to_ubuf_2d_core_with_contiguous_last_dim`
/// + `UB|GM stride[1] != stride[2] * size[2]`: dimension 1 is non-contiguous
///   + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 is contiguous
///     - Convert to 2D: `offset2D = offset3D, size2D = [size3D[0] * size3D[1],
///     size3D[2]], stride2D = [stride3D[1], stride[2]]`
///       `load_gm_to_ubuf_2d_core_with_contiguous_last_dim`
///   + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 is non-contiguous
///     + Decompose into for-loop, converting to 2D each iteration: since both
///     dimension 0 and 1 are non-contiguous anyway, the outer loop can be
///     hoisted along the axis with `min(size3D[0], size3D[1])`.
///       However, the corresponding `stride3D[0]` or `stride3D[1]` must be a
///       multiple of 32B to allow decomposition into a for-loop. Because if not
///       a multiple of 32B, it would fall back to looped scalar transfer; thus
///       the 32B alignment condition takes priority. If stride of dimension 1
///       is a multiple of 32B, then stride of dimension 0 must also be (due to
///       multiplicative relationship). Conversely, if stride of dimension 0 is
///       not a multiple of 32B, stride of dimension 1 cannot be either.
///       Classification as follows:
///       + UB `stride3D[1] * dtypeBytes % 32B == 0`:
///         - `size3D[0] <= size3D[1]`: use dimension 0 as loop axis
///           ```
///           for i = [0, size3D[0]) {
///             offset2D = offset3D + i * stride3D[0], size2D = [size3D[1],
///             size3D[2]], stride2D = [stride3D[1], stride[2]]
///             load_gm_to_ubuf_2d_core_with_contiguous_last_dim
///           }
///           ```
///         - `size3D[0] > size3D[1]`: use dimension 1 as loop axis
///           ```
///           for i = [0, size3D[1]) {
///             offset2D = offset3D + i * stride3D[1], size2D = [size3D[0],
///             size3D[2]], stride2D = [stride3D[0], stride[2]]
///             load_gm_to_ubuf_2d_core_with_contiguous_last_dim
///           }
///           ```
///       + UB `stride3D[1] * dtypeBytes % 32B != 0 && stride3D[0] * dtypeBytes
///       % 32B == 0`: use dimension 0 as loop axis
///         ```
///         for i = [0, size3D[0]) {
///           offset2D = offset3D + i * stride3D[0], size2D = [size3D[1],
///           size3D[2]], stride2D = [stride3D[1], stride[2]]
///           load_gm_to_ubuf_2d_core_with_contiguous_last_dim
///         }
///         ```
///       + UB `stride3D[0] * dtypeBytes % 32B != 0`: no axis with 32B-aligned
///       stride
///         fall back to looped scalar transfer
template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_3d_core_with_contiguous_last_dim(memref_t<__gm__ T, 3> *gm,
                                                 memref_t<__ubuf__ T, 3> *ub,
                                                 int64_t left_padding_num,
                                                 uint8_t l2_cache_ctl) {
  int64_t size0 = gm->sizes[0];
  int64_t size1 = gm->sizes[1];
  int64_t size2 = gm->sizes[2];
  int64_t stride0_ub = ub->strides[0];
  int64_t stride1_ub = ub->strides[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride1_gm = gm->strides[1];

  if (stride1_ub == size2 && stride1_gm == size2) {
    // deal scenarios where the lower two dimensions can collapse.
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset,
                                {size0, size1 * size2},
                                {stride0_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset,
                                  {size0, size1 * size2},
                                  {stride0_ub, 1}};
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
    return;
  }
  // axis 1 is non-contiguous
  // axis 0 is contiguous
  if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset,
                                {size0 * size1, size2},
                                {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset,
                                  {size0 * size1, size2},
                                  {stride1_ub, 1}};
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
    return;
  }
  // axis 0 is non-contiguous
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if (stride0_ub % num_per_block != 0) {
    // stride1_ub is not 32B aligned either
    load_gm_to_ubuf_3d_by_scalar<T>(gm, ub);
    return;
  }

  if (stride1_ub % num_per_block == 0) {
    // stride0_ub is also 32B aligned
    // then we select min(size0, size1) as the loop axis
    if (size0 <= size1) {
      for (int64_t i = 0; i < size0; ++i) {
        memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                    gm->aligned,
                                    gm->offset + i * stride0_gm,
                                    {size1, size2},
                                    {stride1_gm, 1}};
        memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                      ub->aligned,
                                      ub->offset + i * stride0_ub,
                                      {size1, size2},
                                      {stride1_ub, 1}};
        load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
            &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
      }
      return;
    }
    // size0 > size1
    for (int64_t i = 0; i < size1; ++i) {
      memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                  gm->aligned,
                                  gm->offset + i * stride1_gm,
                                  {size0, size2},
                                  {stride0_gm, 1}};
      memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                    ub->aligned,
                                    ub->offset + i * stride1_ub,
                                    {size0, size2},
                                    {stride0_ub, 1}};
      load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
          &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
    }
    return;
  }

  // here the remaining case is stride1_ub % num_per_block != 0 &&
  // stride0_ub % num_per_block == 0.
  // thus we can only loop axis 0
  for (int64_t i = 0; i < size0; ++i) {
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset + i * stride0_gm,
                                {size1, size2},
                                {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset + i * stride0_ub,
                                  {size1, size2},
                                  {stride1_ub, 1}};
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
  }
  return;
}
#else
///========================================
/// m300 arch
///========================================
// Copy 3d, where the last dim is contiguous i.e. stride2=1
template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_3d_core_with_contiguous_last_dim(memref_t<__gm__ T, 3> *gm,
                                                 memref_t<__ubuf__ T, 3> *ub,
                                                 int64_t left_padding_num,
                                                 uint8_t l2_cache_ctl) {
  int64_t size0 = gm->sizes[0];
  int64_t size1 = gm->sizes[1];
  int64_t size2 = gm->sizes[2];
  int64_t stride0_ub = ub->strides[0];
  int64_t stride1_ub = ub->strides[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride1_gm = gm->strides[1];

  if (stride1_ub == size2 && stride1_gm == size2) {
    // deal scenarios where the lower two dimensions can collapse.
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset,
                                {size0, size1 * size2},
                                {stride0_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset,
                                  {size0, size1 * size2},
                                  {stride0_ub, 1}};
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
    return;
  }

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  if (size1 == 1 && ub->offset % num_per_block == 0) {
    // Always swap when size1 == 1 to enable coalesced access
    size0 = size1;
    size1 = gm->sizes[0];
    stride0_ub = ub->strides[1];
    stride1_ub = ub->strides[0];
    stride0_gm = gm->strides[1];
    stride1_gm = gm->strides[0];
  } else if (size0 > size1 && stride1_ub % num_per_block == 0) {
    // Optimize memory access pattern by swapping dimensions when beneficial:
    // - Condition 1: size0 > size1
    // - Condition 2: The stride of dim1 in Unified Buffer (UB) is block-aligned
    size0 = size1;
    size1 = gm->sizes[0];
    stride0_ub = ub->strides[1];
    stride1_ub = ub->strides[0];
    stride0_gm = gm->strides[1];
    stride1_gm = gm->strides[0];
  }

  for (int64_t i = 0; i < size0; i++) {
    memref_t<__gm__ T, 2> gm_2d = {gm->allocated,
                                   gm->aligned,
                                   gm->offset + stride0_gm * i,
                                   {size1, size2},
                                   {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d = {ub->allocated,
                                     ub->aligned,
                                     ub->offset + stride0_ub * i,
                                     {size1, size2},
                                     {stride1_ub, 1}};

    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
  }
}
#endif

#if defined(__DAV_C310__)
/// 3D UB -> GM core func mainly using DMA
/// + `UB&GM stride[1] == stride[2] * size[2]`: dimension 1 is contiguous
///   - Convert to 2D: `offset2D = offset3D, size2D = [size3D[0], size3D[1] *
///   size3D[2]], stride2D = [stride3D[0], stride[2]]`
///     `store_ubuf_to_gm_2d_core_with_contiguous_last_dim`
/// + `UB|GM stride[1] != stride[2] * size[2]`: dimension 1 is non-contiguous
///   + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 is contiguous
///     - Convert to 2D: `offset2D = offset3D, size2D = [size3D[0] * size3D[1],
///     size3D[2]], stride2D = [stride3D[1], stride[2]]`
///       `store_ubuf_to_gm_2d_core_with_contiguous_last_dim`
///   + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 is non-contiguous
///     + Decompose into for-loop, converting to 2D each iteration: since both
///     dimension 0 and 1 are non-contiguous anyway, the outer loop can be
///     hoisted along the axis with `min(size3D[0], size3D[1])`.
///       However, the corresponding `stride3D[0]` or `stride3D[1]` must be a
///       multiple of 32B to allow decomposition into a for-loop. Because if not
///       a multiple of 32B, it would fall back to looped scalar transfer; thus
///       the 32B alignment condition takes priority. If stride of dimension 1
///       is a multiple of 32B, then stride of dimension 0 must also be (due to
///       multiplicative relationship). Conversely, if stride of dimension 0 is
///       not a multiple of 32B, stride of dimension 1 cannot be either.
///       Classification as follows:
///       + UB `stride3D[1] * dtypeBytes % 32B == 0`:
///         - `size3D[0] <= size3D[1]`: use dimension 0 as loop axis
///           ```
///           for i = [0, size3D[0]) {
///             offset2D = offset3D + i * stride3D[0], size2D = [size3D[1],
///             size3D[2]], stride2D = [stride3D[1], stride[2]]
///             store_ubuf_to_gm_2d_core_with_contiguous_last_dim
///           }
///           ```
///         - `size3D[0] > size3D[1]`: use dimension 1 as loop axis
///           ```
///           for i = [0, size3D[1]) {
///             offset2D = offset3D + i * stride3D[1], size2D = [size3D[0],
///             size3D[2]], stride2D = [stride3D[0], stride[2]]
///             store_ubuf_to_gm_2d_core_with_contiguous_last_dim
///           }
///           ```
///       + UB `stride3D[1] * dtypeBytes % 32B != 0 && stride3D[0] * dtypeBytes
///       % 32B == 0`: use dimension 0 as loop axis
///         ```
///         for i = [0, size3D[0]) {
///           offset2D = offset3D + i * stride3D[0], size2D = [size3D[1],
///           size3D[2]], stride2D = [stride3D[1], stride[2]]
///           store_ubuf_to_gm_2d_core_with_contiguous_last_dim
///         }
///         ```
///       + UB `stride3D[0] * dtypeBytes % 32B != 0`: no axis with 32B-aligned
///       stride
///         fall back to looped scalar transfer
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_3d_core_with_contiguous_last_dim(memref_t<__ubuf__ T, 3> *ub,
                                                  memref_t<__gm__ T, 3> *gm) {
  int64_t size0 = ub->sizes[0];
  int64_t size1 = ub->sizes[1];
  int64_t size2 = ub->sizes[2];
  int64_t stride0_ub = ub->strides[0];
  int64_t stride1_ub = ub->strides[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride1_gm = gm->strides[1];

  if (stride1_ub == size2 && stride1_gm == size2) {
    // deal scenarios where the lower two dimensions can collapse.
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset,
                                  {size0, size1 * size2},
                                  {stride0_ub, 1}};
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset,
                                {size0, size1 * size2},
                                {stride0_gm, 1}};
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
    return;
  }
  // axis 1 is non-contiguous
  // axis 0 is contiguous
  if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset,
                                {size0 * size1, size2},
                                {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset,
                                  {size0 * size1, size2},
                                  {stride1_ub, 1}};
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
    return;
  }
  // axis 0 is non-contiguous
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if (stride0_ub % num_per_block != 0) {
    // stride1_ub is not 32B aligned either
    store_ubuf_to_gm_3d_by_scalar<T>(ub, gm);
    return;
  }

  if (stride1_ub % num_per_block == 0) {
    // stride0_ub is also 32B aligned
    // then we select min(size0, size1) as the loop axis
    if (size0 <= size1) {
      for (int64_t i = 0; i < size0; ++i) {
        memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                    gm->aligned,
                                    gm->offset + i * stride0_gm,
                                    {size1, size2},
                                    {stride1_gm, 1}};
        memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                      ub->aligned,
                                      ub->offset + i * stride0_ub,
                                      {size1, size2},
                                      {stride1_ub, 1}};
        store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
      }
      return;
    }
    // size0 > size1
    for (int64_t i = 0; i < size1; ++i) {
      memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                  gm->aligned,
                                  gm->offset + i * stride1_gm,
                                  {size0, size2},
                                  {stride0_gm, 1}};
      memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                    ub->aligned,
                                    ub->offset + i * stride1_ub,
                                    {size0, size2},
                                    {stride0_ub, 1}};
      store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
    }
    return;
  }

  // here the remaining case is stride1_ub % num_per_block != 0 &&
  // stride0_ub % num_per_block == 0.
  // thus we can only loop axis 0
  for (int64_t i = 0; i < size0; ++i) {
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset + i * stride0_gm,
                                {size1, size2},
                                {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset + i * stride0_ub,
                                  {size1, size2},
                                  {stride1_ub, 1}};
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
  }
  return;
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_3d_core_with_contiguous_last_dim(memref_t<__ubuf__ T, 3> *ub,
                                                  memref_t<__gm__ T, 3> *gm) {
  int64_t size0 = ub->sizes[0];
  int64_t size1 = ub->sizes[1];
  int64_t size2 = ub->sizes[2];
  int64_t stride0_ub = ub->strides[0];
  int64_t stride1_ub = ub->strides[1];
  int64_t stride0_gm = gm->strides[0];
  int64_t stride1_gm = gm->strides[1];

  if (stride1_ub == size2 && stride1_gm == size2) {
    // deal scenarios where the lower two dimensions can collapse.
    memref_t<__ubuf__ T, 2> ub_2d{ub->allocated,
                                  ub->aligned,
                                  ub->offset,
                                  {size0, size1 * size2},
                                  {stride0_ub, 1}};
    memref_t<__gm__ T, 2> gm_2d{gm->allocated,
                                gm->aligned,
                                gm->offset,
                                {size0, size1 * size2},
                                {stride0_gm, 1}};
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim(&ub_2d, &gm_2d);
    return;
  }

  if (size0 > size1) {
    size0 = size1;
    size1 = gm->sizes[0];
    stride0_ub = ub->strides[1];
    stride1_ub = ub->strides[0];
    stride0_gm = gm->strides[1];
    stride1_gm = gm->strides[0];
  }

  for (int64_t i = 0; i < size0; i++) {
    memref_t<__ubuf__ T, 2> ub_2d = {ub->allocated,
                                     ub->aligned,
                                     ub->offset + stride0_ub * i,
                                     {size1, size2},
                                     {stride1_ub, 1}};
    memref_t<__gm__ T, 2> gm_2d = {gm->allocated,
                                   gm->aligned,
                                   gm->offset + stride0_gm * i,
                                   {size1, size2},
                                   {stride1_gm, 1}};

    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
  }
}
#endif

template <typename T>
__aicore__ __attribute__((always_inline)) void copy_gm_to_cbuf_intrin_direct(
    __gm__ T *gm_ptr, int64_t gm_offset, __cbuf__ T *cbuf_ptr,
    int64_t cbuf_offset, uint16_t burst_cnt, uint32_t burst_len, pad_t pad_mode,
    uint16_t srcStride, uint16_t dstStride) {

  // Support PAD_NONE mode
  INTRINSIC(copy_gm_to_cbuf, cbuf_ptr + cbuf_offset, gm_ptr + gm_offset, 0,
            burst_cnt, burst_len, srcStride, dstStride, pad_t::PAD_NONE);
}

/// func of copy ub <-> ub intrin for 1d and 2d, all dtypes are supported.
/// eg.
///  if copy 1d, burst_cnt is 1 and gap is 0.
///  if copy 2d, burst_cnt, burst_len and gap are all needed to compute.
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_intrin_core(__ubuf__ T *dst_ptr, __ubuf__ T *src_ptr,
                              uint16_t burst_cnt, uint32_t burst_len,
                              uint16_t src_stride, uint16_t dst_stride) {
  constexpr int32_t bytes = sizeof(T);
  int16_t src_gap = CEIL_DIV(src_stride, UB_ALIGN_BYTES / bytes) - burst_len;
  int16_t dst_gap = CEIL_DIV(dst_stride, UB_ALIGN_BYTES / bytes) - burst_len;
  INTRINSIC(copy_ubuf_to_ubuf, dst_ptr, src_ptr, 0, // sid
            burst_cnt, burst_len, MAX(src_gap, 0), MAX(dst_gap, 0));
}

#if defined(__DAV_C310__)
/// Requirement: `stride[0] == 1`
/// - `src|dst` starting address not 32B aligned:
/// `vector_dma_unalign_offset_vv_1d_vf`
/// - `src&dst` starting address 32B aligned
///   - size[0] not a multiple of 32B: `vector_dma_unalign_size_vv_1d_vf`
///   - size[0] is a multiple of 32B: `copy_ubuf_to_ubuf_intrin_core`
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int burst_len = CEIL_DIV(size0, num_per_block);

  if (!isAddress32ByteAligned(src_ptr) || !isAddress32ByteAligned(dst_ptr)) {
    vector_dma_unalign_offset_vv_1d_vf<T>(src, dst);
    return;
  }
  // src&dst starting addresses are 32B aligned
  if (size0 % num_per_block != 0) {
    vector_dma_unalign_size_vv_1d_vf<T>(src, dst);
    return;
  }

  copy_ubuf_to_ubuf_intrin_core(dst_ptr, src_ptr,
                                1,         // burst_cnt
                                burst_len, // burst_len
                                0,         // src_stride
                                0);        // dst_stride
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int burst_len = CEIL_DIV(size0, num_per_block);

  copy_ubuf_to_ubuf_intrin_core(dst_ptr, src_ptr,
                                1,         // burst_cnt
                                burst_len, // burst_len
                                0,         // src_stride
                                0);        // dst_stride
}
#endif

/// calculate src_gap and dst_gap for copy ub <-> ub, 2d
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_intrin_core(memref_t<__ubuf__ T, 2> *src,
                                 memref_t<__ubuf__ T, 2> *dst) {
  int64_t size1 = src->sizes[1];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t src_gap = calc_ub_gap<T>(src_stride0, size1);
  const int64_t dst_gap = calc_ub_gap<T>(dst_stride0, size1);
  int64_t size0 = src->sizes[0];
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int burst_len = CEIL_DIV(size1, num_per_block);
  const int32_t burst_cnt_main = size0 / INTRIN_MAX_BURST_CNT;
  const int32_t burst_cnt_tail = size0 % INTRIN_MAX_BURST_CNT;
  if (burst_cnt_main > 0) [[unlikely]] {
    // considering the max. ub size, burst_cnt_main can only <= 1
    copy_ubuf_to_ubuf_intrin_core<T>(dst_ptr, src_ptr, INTRIN_MAX_BURST_CNT,
                                     burst_len, src_stride0, dst_stride0);
  }

  if (burst_cnt_tail > 0) [[likely]] {
    // tail part is often used, as tail part can cover more than half of
    // ub size
    const int64_t main_dst_offset =
        burst_cnt_main * INTRIN_MAX_BURST_CNT * dst_stride0;
    const int64_t main_src_offset =
        burst_cnt_main * INTRIN_MAX_BURST_CNT * src_stride0;
    copy_ubuf_to_ubuf_intrin_core<T>(dst_ptr + main_dst_offset,
                                     src_ptr + main_src_offset, burst_cnt_tail,
                                     burst_len, src_stride0, dst_stride0);
  }
}

/// Decompose the 2D src and dst into a for-loop, invoking
/// `copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim` in each iteration
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  for (int64_t i = 0; i < size0; i++) {
    memref_t<__ubuf__ T, 1> src_1d = {src->allocated,
                                      src->aligned,
                                      src->offset + src_stride0 * i,
                                      {size1},
                                      {1}};
    memref_t<__ubuf__ T, 1> dst_1d = {dst->allocated,
                                      dst->aligned,
                                      dst->offset + dst_stride0 * i,
                                      {size1},
                                      {1}};
    copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(&src_1d, &dst_1d);
  }
}

#if defined(__DAV_C310__)
/// Requirement: `stride[1] == 1`
/// - `src|dst` starting address not 32B aligned:
///   - `copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim`
/// - `src&dst` starting address 32B aligned
///   - `src|dst stride[0] < size[1]`: overlap scenario, decompose into loop:
///   `copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim`
///   - `src|dst stride[0] >= size[1]`: default scenario
///     - `src&dst stride[0] == stride[1] * size[1]`: dimension 0 contiguous
///       - Convert to 1D: `offset1D = offset2D, size1D = [size2D[0] *
///       size2D[1]], stride1D = [stride2D[1]]`
///         `copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim`
///     - `src|dst stride[0] != stride[1] * size[1]`: dimension 0 non-contiguous
///       - `size[1] * dtypeBytes % 32 != 0 || src|dst stride[0] * dtypeBytes %
///       32 != 0`: neither burstLen nor burstGap satisfy 32B alignment,
///       hardware `MOV_UB_TO_UB` instruction cannot be used
///         - `copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim`
///       - Both burstLen and burstGap satisfy 32B alignment:
///       `copy_ubuf_to_ubuf_2d_intrin_core`
/// Note: impl is simplified to merge multiple cases of
/// copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride1 = dst->strides[1];
  constexpr int32_t bytes = sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / bytes;
  // the start addr is 32B aligned
  if (isAddress32ByteAligned(src_ptr) && isAddress32ByteAligned(dst_ptr)) {
    // axis 0 is contiguous
    if (src_stride0 == src_stride1 * size1 &&
        dst_stride0 == dst_stride1 * size1) {
      memref_t<__ubuf__ T, 1> src_1d = {
          src->allocated, src->aligned, src->offset, {size0 * size1}, {1}};
      memref_t<__ubuf__ T, 1> dst_1d = {
          dst->allocated, dst->aligned, dst->offset, {size0 * size1}, {1}};
      copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(&src_1d, &dst_1d);
      return;
    }
    // size & strides are 32B aligned
    if ((src_stride0 % num_per_block == 0 &&
         dst_stride0 % num_per_block == 0) &&
        size1 % num_per_block == 0) {
      copy_ubuf_to_ubuf_2d_intrin_core<T>(src, dst);
      return;
    }
  }
  // remaining cases
  copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim<T>(src, dst);
  return;
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  const int64_t size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t src_gap = calc_ub_gap<T>(src_stride0, size1);
  const int64_t dst_gap = calc_ub_gap<T>(dst_stride0, size1);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  if (0 > src_gap || 0 > dst_gap) [[unlikely]] {
    copy_ubuf_to_ubuf_2d_to_1d_core_with_contiguous_last_dim(src, dst);
  } else {
    copy_ubuf_to_ubuf_2d_intrin_core<T>(src, dst);
  }
}
#endif

/// Directly call `copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim` in a
/// for-loop, but incorporate logic to select the shorter axis between
/// axis 0 and axis 1 as the loop axis.
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_3d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  const int64_t size2 = src->sizes[2];
  constexpr int32_t bytes = sizeof(T);
  const int64_t src_stride0 = src->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t dst_stride1 = dst->strides[1];
  // choose the shorter axis as the loop num
  bool is_repeat_size0 = size0 > size1;
  int repeat_size = is_repeat_size0 ? size0 : size1;
  int64_t loop_num = is_repeat_size0 ? size1 : size0;
  int64_t src_loop_stride = is_repeat_size0 ? src_stride1 : src_stride0;
  int64_t dst_loop_stride = is_repeat_size0 ? dst_stride1 : dst_stride0;
  int64_t src_repeat_stride = is_repeat_size0 ? src_stride0 : src_stride1;
  int64_t dst_repeat_stride = is_repeat_size0 ? dst_stride0 : dst_stride1;

  for (int64_t i = 0; i < loop_num; i++) {
    memref_t<__ubuf__ T, 2> src_2d = {src->allocated,
                                      src->aligned,
                                      src->offset + i * src_loop_stride,
                                      {repeat_size, size2},
                                      {src_repeat_stride, 1}};
    memref_t<__ubuf__ T, 2> dst_2d = {dst->allocated,
                                      dst->aligned,
                                      dst->offset + i * dst_loop_stride,
                                      {repeat_size, size2},
                                      {dst_repeat_stride, 1}};

    copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src_2d, &dst_2d);
  }
}

#if defined(__DAV_C310__)
/// func of copy ub -> cbuf intrin for 1d and 2d, all dtypes are supported.
/// eg.
///  if copy 1d, burst_cnt is 1 and gap is 0.
///  if copy 2d, burst_cnt, burst_len and gap are all needed to compute.
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_intrin_core(__cbuf__ T *dst_ptr, __ubuf__ T *src_ptr,
                              uint16_t burst_cnt, uint16_t burst_len,
                              uint16_t src_stride, uint16_t dst_stride) {
  constexpr int32_t bytes = sizeof(T);
  int16_t src_gap = CEIL_DIV(src_stride, UB_ALIGN_BYTES / bytes) - burst_len;
  int16_t dst_gap = CEIL_DIV(dst_stride, L1_ALIGN_BYTES / bytes) - burst_len;
  INTRINSIC(copy_ubuf_to_cbuf, dst_ptr, src_ptr, 0, // sub_blockid
            burst_cnt, burst_len, MAX(src_gap, 0), MAX(dst_gap, 0));
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_1d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 1> *src, memref_t<__cbuf__ T, 1> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  uint16_t burst_len = CEIL_DIV(size0, num_per_block);

  // TODO: handle !isAddress32ByteAligned if possible
  copy_ubuf_to_cbuf_intrin_core(dst_ptr, src_ptr,
                                1,         // burst_cnt
                                burst_len, // burst_len
                                0,         // src_stride
                                0);        // dst_stride
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_2d_intrin_core(memref_t<__ubuf__ T, 2> *src,
                                 memref_t<__cbuf__ T, 2> *dst) {
  int64_t size1 = src->sizes[1];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  constexpr int32_t bytes = sizeof(T);
  int64_t size0 = src->sizes[0];
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t burst_len = CEIL_DIV(size1 * bytes, L1_ALIGN_BYTES);
  const int32_t burst_cnt_main = size0 / INTRIN_MAX_BURST_CNT;
  const int32_t burst_cnt_tail = size0 % INTRIN_MAX_BURST_CNT;
  if (burst_cnt_main > 0) [[unlikely]] {
    copy_ubuf_to_cbuf_intrin_core<T>(dst_ptr, src_ptr,
                                     INTRIN_MAX_BURST_CNT, // burst_cnt
                                     burst_len,            // burst_len
                                     src_stride0,          // src_stride
                                     dst_stride0);         // dst_stride
  }

  if (burst_cnt_tail > 0) [[likely]] {
    const int64_t main_dst_offset =
        burst_cnt_main * INTRIN_MAX_BURST_CNT * dst_stride0;
    const int64_t main_src_offset =
        burst_cnt_main * INTRIN_MAX_BURST_CNT * src_stride0;
    copy_ubuf_to_cbuf_intrin_core<T>(dst_ptr + main_dst_offset,
                                     src_ptr + main_src_offset,
                                     burst_cnt_tail, // burst_cnt
                                     burst_len,      // burst_len
                                     src_stride0,    // src_stride
                                     dst_stride0);   // dst_stride
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_2d_to_1d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 2> *src, memref_t<__cbuf__ T, 2> *dst) {
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  for (int64_t i = 0; i < size0; i++) {
    memref_t<__ubuf__ T, 1> src_1d = {src->allocated,
                                      src->aligned,
                                      src->offset + src_stride0 * i,
                                      {size1},
                                      {1}};
    memref_t<__cbuf__ T, 1> dst_1d = {dst->allocated,
                                      dst->aligned,
                                      dst->offset + dst_stride0 * i,
                                      {size1},
                                      {1}};
    copy_ubuf_to_cbuf_1d_core_with_contiguous_last_dim<T>(&src_1d, &dst_1d);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_2d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 2> *src, memref_t<__cbuf__ T, 2> *dst) {
  int64_t size1 = src->sizes[1];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  constexpr int32_t bytes = sizeof(T);
  const int64_t src_gap = calc_ub_gap<T>(src_stride0, size1);
  const int64_t dst_gap = (dst_stride0 - size1) * bytes / L1_ALIGN_BYTES;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  if (0 > src_gap || 0 > dst_gap) [[unlikely]] {
    copy_ubuf_to_cbuf_2d_to_1d_core_with_contiguous_last_dim(src, dst);
  } else {
    // TODO: handle !isAddress32ByteAligned if possible
    copy_ubuf_to_cbuf_2d_intrin_core<T>(src, dst);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_3d_core_with_contiguous_last_dim(
    memref_t<__ubuf__ T, 3> *src, memref_t<__cbuf__ T, 3> *dst) {
  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  constexpr int32_t bytes = sizeof(T);
  int64_t src_stride0 = src->strides[0];
  int64_t src_stride1 = src->strides[1];
  int64_t dst_stride0 = dst->strides[0];
  int64_t dst_stride1 = dst->strides[1];

  bool is_repeat_size0 = size0 > size1;
  int repeat_size = is_repeat_size0 ? size0 : size1;
  int64_t loop_num = is_repeat_size0 ? size1 : size0;
  int64_t src_loop_stride = is_repeat_size0 ? src_stride1 : src_stride0;
  int64_t dst_loop_stride = is_repeat_size0 ? dst_stride1 : dst_stride0;
  int64_t src_repeat_stride = is_repeat_size0 ? src_stride0 : src_stride1;
  int64_t dst_repeat_stride = is_repeat_size0 ? dst_stride0 : dst_stride1;

  for (int64_t i = 0; i < loop_num; i++) {
    memref_t<__ubuf__ T, 2> src_2d = {src->allocated,
                                      src->aligned,
                                      src->offset + i * src_loop_stride,
                                      {repeat_size, size2},
                                      {src_repeat_stride, 1}};
    memref_t<__cbuf__ T, 2> dst_2d = {dst->allocated,
                                      dst->aligned,
                                      dst->offset + i * dst_loop_stride,
                                      {repeat_size, size2},
                                      {dst_repeat_stride, 1}};

    copy_ubuf_to_cbuf_2d_core_with_contiguous_last_dim<T>(&src_2d, &dst_2d);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_2d_by_nddma(memref_t<__gm__ T, 2> *src,
                            memref_t<__ubuf__ T, 2> *dst) {
  static_assert(
      sizeof(T) <= 8,
      "DMA only supports types up to 64-bit. "
      "Larger types (e.g., long double, __int128) are not supported.");
  // nddma_out_to_ub does not support FP8, and this is only data movement which
  // has no impact on precision. Conversion to U8 is feasible as long as the bit
  // width is the same.
  using DmaType = std::conditional_t<std::is_same_v<T, float8_e4m3_t> ||
                                         std::is_same_v<T, float8_e5m2_t>,
                                     uint8_t, T>;

  __gm__ DmaType *src_ptr =
      reinterpret_cast<__gm__ DmaType *>(src->aligned + src->offset);
  __ubuf__ DmaType *dst_ptr =
      reinterpret_cast<__ubuf__ DmaType *>(dst->aligned + dst->offset);

  if constexpr (sizeof(T) <= 4) {
    // 32-bit and below version
    nddma_desc::loop_desc loop0(dst->sizes[1], dst->strides[1],
                                src->strides[1]);
    nddma_desc::loop_desc loop1(dst->sizes[0], dst->strides[0],
                                src->strides[0]);
    nddma_desc desc(loop1, loop0);

    INTRINSIC(nddma_out_to_ub, dst_ptr, src_ptr, 0, desc, 0, NEAREST_PADDING,
              0);

  } else if constexpr (sizeof(T) == 8) {
    // 64-bit version
    using T32 = typename std::conditional<std::is_same_v<T, int64_t>, int32_t,
                                          uint32_t>::type;

    __gm__ T32 *src_32 = reinterpret_cast<__gm__ T32 *>(src_ptr);
    __ubuf__ T32 *dst_32 = reinterpret_cast<__ubuf__ T32 *>(dst_ptr);

    nddma_desc::loop_desc loop0(dst->sizes[1] * 2, dst->strides[1] * 2,
                                src->strides[1] * 2);
    nddma_desc::loop_desc loop1(dst->sizes[0] * 2, dst->strides[0] * 2,
                                src->strides[0] * 2);
    nddma_desc desc(loop1, loop0);

    INTRINSIC(nddma_out_to_ub, dst_32, src_32, 0, desc, 0, NEAREST_PADDING, 0);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_3d_by_nddma(memref_t<__gm__ T, 3> *src,
                            memref_t<__ubuf__ T, 3> *dst) {
  static_assert(
      sizeof(T) <= 8,
      "DMA only supports types up to 64-bit. "
      "Larger types (e.g., long double, __int128) are not supported.");

  using DmaType = std::conditional_t<std::is_same_v<T, float8_e4m3_t> ||
                                         std::is_same_v<T, float8_e5m2_t>,
                                     uint8_t, T>;

  __gm__ DmaType *src_ptr =
      reinterpret_cast<__gm__ DmaType *>(src->aligned + src->offset);
  __ubuf__ DmaType *dst_ptr =
      reinterpret_cast<__ubuf__ DmaType *>(dst->aligned + dst->offset);

  if constexpr (sizeof(T) <= 4) {
    // 32-bit and below version
    nddma_desc::loop_desc loop0(dst->sizes[2], dst->strides[2],
                                src->strides[2]);
    nddma_desc::loop_desc loop1(dst->sizes[1], dst->strides[1],
                                src->strides[1]);
    nddma_desc::loop_desc loop2(dst->sizes[0], dst->strides[0],
                                src->strides[0]);
    nddma_desc desc(loop2, loop1, loop0);

    INTRINSIC(nddma_out_to_ub, dst_ptr, src_ptr, 0, desc, 0, NEAREST_PADDING,
              0);

  } else if constexpr (sizeof(T) == 8) {
    // 64-bit version
    using T32 = typename std::conditional<std::is_same_v<T, int64_t>, int32_t,
                                          uint32_t>::type;

    __gm__ T32 *src_32 = reinterpret_cast<__gm__ T32 *>(src_ptr);
    __ubuf__ T32 *dst_32 = reinterpret_cast<__ubuf__ T32 *>(dst_ptr);

    nddma_desc::loop_desc loop0(dst->sizes[2] * 2, dst->strides[2] * 2,
                                src->strides[2] * 2);
    nddma_desc::loop_desc loop1(dst->sizes[1] * 2, dst->strides[1] * 2,
                                src->strides[1] * 2);
    nddma_desc::loop_desc loop2(dst->sizes[0] * 2, dst->strides[0] * 2,
                                src->strides[0] * 2);
    nddma_desc desc(loop2, loop1, loop0);

    INTRINSIC(nddma_out_to_ub, dst_32, src_32, 0, desc, 0, NEAREST_PADDING, 0);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_2d_by_nddma(memref_t<__gm__ T, 2> *src,
                            memref_t<__ubuf__ T, 2> *dst);

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_3d_by_nddma(memref_t<__gm__ T, 3> *src,
                            memref_t<__ubuf__ T, 3> *dst);

#endif

template <typename T>
__aiv__ __attribute__((always_inline)) void set_atomic_type() {
  if constexpr (std::is_same<T, bfloat16_t>::value) {
    INTRINSIC_NO_ARGS(set_atomic_bf16);
  } else if constexpr (std::is_same<T, half>::value) {
    INTRINSIC_NO_ARGS(set_atomic_f16);
  } else if constexpr (std::is_same<T, float>::value) {
    INTRINSIC_NO_ARGS(set_atomic_f32);
  } else if constexpr (std::is_same<T, int8_t>::value) {
    INTRINSIC_NO_ARGS(set_atomic_s8);
  } else if constexpr (std::is_same<T, int16_t>::value) {
    INTRINSIC_NO_ARGS(set_atomic_s16);
  } else if constexpr (std::is_same<T, int32_t>::value) {
    INTRINSIC_NO_ARGS(set_atomic_s32);
  } else {
    static_assert("unsupport atomic type");
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
set_atomic_kind(AtomicKind atomic_kind) {
  set_atomic_type<T>();

  if (atomic_kind == AtomicKind::Add) {
    INTRINSIC_NO_ARGS(set_atomic_add);
  }
  if (atomic_kind == AtomicKind::Max) {
    INTRINSIC_NO_ARGS(set_atomic_max);
  }
  if (atomic_kind == AtomicKind::Min) {
    INTRINSIC_NO_ARGS(set_atomic_min);
  } else {
    static_assert("unsupport atomic kind");
  }
}

__aiv__ __attribute__((always_inline)) void
set_store_atomic_none(AtomicKind atomic_kind);

template <typename T>
__aiv__ __attribute__((always_inline)) T set_pad_value_null() {
  if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
    return {0};
  } else if constexpr (std::is_same_v<T, float>) {
    return 0.0;
  } else if constexpr (std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, uint8_t>) {
    return '\0';
  } else if constexpr (std::is_same_v<T, int16_t> ||
                       std::is_same_v<T, uint16_t> ||
                       std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, uint32_t> ||
                       std::is_same_v<T, int64_t> ||
                       std::is_same_v<T, uint64_t>) {
    return 0;
  } else {
    static_assert("unsupport pad value");
  }
}

#if defined(__DAV_C310__)
template <typename T, int DIM,
          typename = std::enable_if_t<std::is_same_v<T, int64_t> ||
                                      std::is_same_v<T, uint64_t>>>
__aiv__ __attribute__((always_inline)) void
apply_padding_b64(memref_t<__ubuf__ T, DIM> *dst, int64_t offset,
                  int64_t pad_value, int64_t repeat,
                  int64_t padding_shift_count) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t size = dst->sizes[DIM - 1];
  int64_t size_aligned =
      (DIM == 2) ? dst->strides[0] : CEIL_FACTOR(size, num_per_block);
  int64_t align_pad = size_aligned - size;
  if (align_pad == 0) {
    return;
  }
  __ubuf__ T *base_ptr = dst->aligned + dst->offset;
  for (int64_t i = 0; i < repeat; ++i) {
    __ubuf__ T *row_ptr = base_ptr + i * size_aligned;
    for (int64_t j = 0; j < align_pad; ++j) {
      *(row_ptr + size + j) = static_cast<T>(pad_value);
    }
  }
}
#else
///========================================
/// m300 arch
///========================================
template <typename T, int DIM,
          typename = std::enable_if_t<std::is_same_v<T, int64_t> ||
                                      std::is_same_v<T, uint64_t>>>
__aiv__ __attribute__((always_inline)) void
apply_padding_b64(memref_t<__ubuf__ T, DIM> *dst, int64_t offset,
                  int64_t pad_value, int64_t repeat,
                  int64_t padding_shift_count) {
  const int64_t factor = sizeof(T) / sizeof(uint32_t);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t size = dst->sizes[DIM - 1];
  int64_t size_aligned =
      (DIM == 2) ? dst->strides[0] : CEIL_FACTOR(size, num_per_block);
}
#endif

// The following padValueType is used to convert the FP8 PadValue to the float
// type. The reason for doing so is that if the PadValue is of FP8 type, an FP8
// constant will be automatically generated, which is not allowed in the LLVM IR
// received by bisheng. Thus, the conversion here is only to avoid generating
// FP8 constants.
template <typename T> struct PadValueType {
  using type = T;
};

#if defined(__DAV_C310__)
template <> struct PadValueType<float8_e4m3_t> {
  using type = float;
};

template <> struct PadValueType<float8_e5m2_t> {
  using type = float;
};
#endif

template <typename T> using PadValueT = typename PadValueType<T>::type;

#define DECLARE_DMA_LOAD(dim, type)                                            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_load_gm_to_ubuf_##dim##d_##type(                            \
          memref_t<__gm__ type, dim> *src, memref_t<__ubuf__ type, dim> *dst,  \
          PadMode pad_mode, type pad_value, int64_t left_padding_num,          \
          EvictionPolicy eviction_policy)

#define REGISTE_DMA_LOAD(dim, type)                                            \
  DECLARE_DMA_LOAD(dim, type) {                                                \
    load_gm_to_ubuf_##dim##d_core<type>(src, dst, pad_mode, pad_value,         \
                                        left_padding_num, eviction_policy);    \
  }

#if defined(__DAV_C310__)

#define DECLARE_DMA_LOAD_FP8(dim, type)                                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_load_gm_to_ubuf_##dim##d_##type(                            \
          memref_t<__gm__ type, dim> *src, memref_t<__ubuf__ type, dim> *dst,  \
          PadMode pad_mode, float pad_value, int64_t left_padding_num,         \
          EvictionPolicy eviction_policy)

#define REGISTE_DMA_LOAD_FP8(dim, type)                                        \
  DECLARE_DMA_LOAD_FP8(dim, type) {                                            \
    load_gm_to_ubuf_##dim##d_core<type>(src, dst, pad_mode, pad_value,         \
                                        left_padding_num, eviction_policy);    \
  }

#endif

#define DECLARE_DMA_STORE(dim, type)                                           \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_store_ubuf_to_gm_##dim##d_##type(                           \
          memref_t<__ubuf__ type, dim> *src, memref_t<__gm__ type, dim> *dst,  \
          AtomicKind atomic_kind)

#define REGISTE_DMA_STORE(dim, type)                                           \
  DECLARE_DMA_STORE(dim, type) {                                               \
    store_ubuf_to_gm_##dim##d_core<type>(src, dst, atomic_kind);               \
  }

#define REGISTE_DMA_CBUF_COPY(src_scope, dst_scope, dim, type)                 \
  DECLARE_DMA_CBUF_COPY(src_scope, dst_scope, dim, type) {                     \
    copy_##src_scope##_to_##dst_scope##_##dim##d_core<type>(src, dst,          \
                                                            pad_mode);         \
  }

#define DECLARE_DMA_UB_COPY(src_scope, dst_scope, dim, type)                   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_copy_##src_scope##_to_##dst_scope##_##dim##d_##type(        \
          memref_t<__##src_scope##__ type, dim> *src,                          \
          memref_t<__##dst_scope##__ type, dim> *dst)

#define REGISTE_DMA_UB_COPY(src_scope, dst_scope, dim, type)                   \
  DECLARE_DMA_UB_COPY(src_scope, dst_scope, dim, type) {                       \
    copy_##src_scope##_to_##dst_scope##_##dim##d_core<type>(src, dst);         \
  }

#define REGISTE_DMA_L1_LOAD(dim, type)                                         \
  DECLARE_DMA_L1_LOAD(dim, type) {                                             \
    load_gm_to_cbuf_##dim##d_core<type>(src, dst, pad_mode);                   \
  }

#define DECLARE_DMA_L1_LOAD(dim, type)                                         \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_load_gm_to_cbuf_##dim##d_##type(                            \
          memref_t<__gm__ type, dim> *src, memref_t<__cbuf__ type, dim> *dst,  \
          pad_t pad_mode)

extern "C" {
//===-------------------------------------------------------------------===//
// load gm to ub, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_LOAD(1, int8_t);
DECLARE_DMA_LOAD(1, uint8_t);
DECLARE_DMA_LOAD(1, int16_t);
DECLARE_DMA_LOAD(1, uint16_t);
DECLARE_DMA_LOAD(1, int32_t);
DECLARE_DMA_LOAD(1, uint32_t);
DECLARE_DMA_LOAD(1, int64_t);
DECLARE_DMA_LOAD(1, uint64_t);
DECLARE_DMA_LOAD(1, half);
DECLARE_DMA_LOAD(1, float);
DECLARE_DMA_LOAD(1, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_LOAD_FP8(1, float8_e4m3_t);
DECLARE_DMA_LOAD_FP8(1, float8_e5m2_t);
#endif

//===-------------------------------------------------------------------===//
// load gm to ub, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_LOAD(2, int8_t);
DECLARE_DMA_LOAD(2, uint8_t);
DECLARE_DMA_LOAD(2, int16_t);
DECLARE_DMA_LOAD(2, uint16_t);
DECLARE_DMA_LOAD(2, int32_t);
DECLARE_DMA_LOAD(2, uint32_t);
DECLARE_DMA_LOAD(2, int64_t);
DECLARE_DMA_LOAD(2, uint64_t);
DECLARE_DMA_LOAD(2, half);
DECLARE_DMA_LOAD(2, float);
DECLARE_DMA_LOAD(2, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_LOAD_FP8(2, float8_e4m3_t);
DECLARE_DMA_LOAD_FP8(2, float8_e5m2_t);
#endif

//===-------------------------------------------------------------------===//
// load gm to ub, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_LOAD(3, int8_t);
DECLARE_DMA_LOAD(3, uint8_t);
DECLARE_DMA_LOAD(3, int16_t);
DECLARE_DMA_LOAD(3, uint16_t);
DECLARE_DMA_LOAD(3, int32_t);
DECLARE_DMA_LOAD(3, uint32_t);
DECLARE_DMA_LOAD(3, int64_t);
DECLARE_DMA_LOAD(3, uint64_t);
DECLARE_DMA_LOAD(3, half);
DECLARE_DMA_LOAD(3, float);
DECLARE_DMA_LOAD(3, bfloat16_t);
DECLARE_DMA_LOAD(3, int8_t);
DECLARE_DMA_LOAD(3, uint8_t);
#if defined(__DAV_C310__)
DECLARE_DMA_LOAD_FP8(3, float8_e4m3_t);
DECLARE_DMA_LOAD_FP8(3, float8_e5m2_t);
#endif
//===-------------------------------------------------------------------===//
// load gm to cbuf, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_L1_LOAD(1, half);
DECLARE_DMA_L1_LOAD(1, float);
DECLARE_DMA_L1_LOAD(1, bfloat16_t);
DECLARE_DMA_L1_LOAD(1, int32_t);
DECLARE_DMA_L1_LOAD(1, uint32_t);
DECLARE_DMA_L1_LOAD(1, int16_t);
DECLARE_DMA_L1_LOAD(1, uint16_t);
DECLARE_DMA_L1_LOAD(1, int8_t);
DECLARE_DMA_L1_LOAD(1, uint8_t);

//===-------------------------------------------------------------------===//
// load gm to cbuf, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_L1_LOAD(2, half);
DECLARE_DMA_L1_LOAD(2, float);
DECLARE_DMA_L1_LOAD(2, bfloat16_t);
DECLARE_DMA_L1_LOAD(2, int32_t);
DECLARE_DMA_L1_LOAD(2, uint32_t);
DECLARE_DMA_L1_LOAD(2, int16_t);
DECLARE_DMA_L1_LOAD(2, uint16_t);
DECLARE_DMA_L1_LOAD(2, int8_t);
DECLARE_DMA_L1_LOAD(2, uint8_t);

//===-------------------------------------------------------------------===//
// load gm to cbuf, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_L1_LOAD(3, half);
DECLARE_DMA_L1_LOAD(3, float);
DECLARE_DMA_L1_LOAD(3, bfloat16_t);
DECLARE_DMA_L1_LOAD(3, int32_t);
DECLARE_DMA_L1_LOAD(3, uint32_t);
DECLARE_DMA_L1_LOAD(3, int16_t);
DECLARE_DMA_L1_LOAD(3, uint16_t);
DECLARE_DMA_L1_LOAD(3, int8_t);
DECLARE_DMA_L1_LOAD(3, uint8_t);

//===-------------------------------------------------------------------===//
// store ub to gm, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_STORE(1, int8_t);
DECLARE_DMA_STORE(1, uint8_t);
DECLARE_DMA_STORE(1, int16_t);
DECLARE_DMA_STORE(1, uint16_t);
DECLARE_DMA_STORE(1, int32_t);
DECLARE_DMA_STORE(1, uint32_t);
DECLARE_DMA_STORE(1, int64_t);
DECLARE_DMA_STORE(1, uint64_t);
DECLARE_DMA_STORE(1, half);
DECLARE_DMA_STORE(1, float);
DECLARE_DMA_STORE(1, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_STORE(1, float8_e4m3_t);
DECLARE_DMA_STORE(1, float8_e5m2_t);
#endif

//===-------------------------------------------------------------------===//
// store ub to gm, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_STORE(2, int8_t);
DECLARE_DMA_STORE(2, uint8_t);
DECLARE_DMA_STORE(2, int16_t);
DECLARE_DMA_STORE(2, uint16_t);
DECLARE_DMA_STORE(2, int32_t);
DECLARE_DMA_STORE(2, uint32_t);
DECLARE_DMA_STORE(2, int64_t);
DECLARE_DMA_STORE(2, uint64_t);
DECLARE_DMA_STORE(2, half);
DECLARE_DMA_STORE(2, float);
DECLARE_DMA_STORE(2, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_STORE(2, float8_e4m3_t);
DECLARE_DMA_STORE(2, float8_e5m2_t);
#endif
//===-------------------------------------------------------------------===//
// store ub to gm, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_STORE(3, int8_t);
DECLARE_DMA_STORE(3, uint8_t);
DECLARE_DMA_STORE(3, int16_t);
DECLARE_DMA_STORE(3, uint16_t);
DECLARE_DMA_STORE(3, int32_t);
DECLARE_DMA_STORE(3, uint32_t);
DECLARE_DMA_STORE(3, int64_t);
DECLARE_DMA_STORE(3, uint64_t);
DECLARE_DMA_STORE(3, half);
DECLARE_DMA_STORE(3, float);
DECLARE_DMA_STORE(3, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_STORE(3, float8_e4m3_t);
DECLARE_DMA_STORE(3, float8_e5m2_t);
#endif
//===-------------------------------------------------------------------===//
// ub to ub, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, bool);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, int8_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, uint8_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, int16_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, uint16_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, int32_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, uint32_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, int64_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, uint64_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, half);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, float);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, float8_e4m3_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 1, float8_e5m2_t);
#endif

//===-------------------------------------------------------------------===//
// ub to ub, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, bool);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, int8_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, uint8_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, int16_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, uint16_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, int32_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, uint32_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, int64_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, uint64_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, half);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, float);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, float8_e4m3_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 2, float8_e5m2_t);
#endif

//===-------------------------------------------------------------------===//
// ub to ub, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, bool);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, int8_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, uint8_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, int16_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, uint16_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, int32_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, uint32_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, int64_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, uint64_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, half);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, float);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, bfloat16_t);
#if defined(__DAV_C310__)
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, float8_e4m3_t);
DECLARE_DMA_UB_COPY(ubuf, ubuf, 3, float8_e5m2_t);
#endif

#if defined(__DAV_C310__)
//===-------------------------------------------------------------------===//
// ub to cbuf, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, int8_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, uint8_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, int16_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, uint16_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, int32_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, uint32_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, float8_e4m3_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, float8_e5m2_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, half);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, float);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 1, bfloat16_t);

//===-------------------------------------------------------------------===//
// ub to cbuf, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, int8_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, uint8_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, int16_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, uint16_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, int32_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, uint32_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, float8_e4m3_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, float8_e5m2_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, half);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, float);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 2, bfloat16_t);

//===-------------------------------------------------------------------===//
// ub to cbuf, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, int8_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, uint8_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, int16_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, uint16_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, int32_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, uint32_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, float8_e4m3_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, float8_e5m2_t);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, half);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, float);
DECLARE_DMA_UB_COPY(ubuf, cbuf, 3, bfloat16_t);
#endif
}

#endif // HIVM_MLIR_TEMPLATE_DMA_UTILS_H
