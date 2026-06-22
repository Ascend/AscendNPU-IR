/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "Vector/Gather/GatherMaskUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_mask_1d(memref_t<__ubuf__ T, 1> *src,
               memref_t<__ubuf__ bool, 1> *mask,
               memref_t<__ubuf__ T, 1> *dst,
               memref_t<__ubuf__ int64_t, 1> *dst_size)
{
    // Check alignment conditions
    if (is_unaligned_gather_mask_1d(src, mask, dst)) [[unlikely]] {
        gather_mask_1d_scalar<T>(src, mask, dst, dst_size);
        return;
    }
    
    int64_t element_count = src->sizes[0];
    int64_t mask_count = mask->sizes[0];

    if (element_count <= 0 || element_count != mask_count) {
        INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
        return;
    }

    constexpr int factor = bitwidthOf<int8_t>() / bitwidthOf<bool>();
    __ubuf__ T *src_ptr = src->aligned + src->offset;
    __ubuf__ int8_t *mask_ptr = (__ubuf__ int8_t *)mask->aligned + mask->offset/factor;
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    __ubuf__ int64_t *dst_size_ptr = dst_size->aligned + dst_size->offset;

    constexpr bool is_16bit = (sizeof(T) == 2);
    constexpr bool is_32bit = (sizeof(T) == 4);

    static_assert(is_16bit || is_32bit, "Only 16-bit or 32-bit types are supported");

    if constexpr (is_16bit) {
        INTRINSIC_NO_ARGS(set_mask_count);
        INTRINSIC(set_vector_mask, 0, element_count);

        INTRINSIC(vreducev2,
                  (__ubuf__ uint16_t*)dst_ptr,
                  (__ubuf__ uint16_t*)src_ptr,
                  (__ubuf__ uint16_t*)mask_ptr,
                  static_cast<uint16_t>(1),
                  static_cast<uint8_t>(1),
                  static_cast<uint8_t>(PatternMode::NORMAL_MODE),
                  static_cast<uint16_t>(8),
                  static_cast<uint8_t>(1));

    } else if constexpr (is_32bit) {
        INTRINSIC_NO_ARGS(set_mask_count);
        INTRINSIC(set_vector_mask, 0, element_count);

        INTRINSIC(vreducev2,
                  (__ubuf__ uint32_t*)dst_ptr,
                  (__ubuf__ uint32_t*)src_ptr,
                  (__ubuf__ uint32_t*)mask_ptr,
                  static_cast<uint16_t>(1),
                  static_cast<uint8_t>(1),
                  static_cast<uint8_t>(PatternMode::NORMAL_MODE),
                  static_cast<uint16_t>(8),
                  static_cast<uint8_t>(1));
    }

    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    *(__ubuf__ int64_t *)(dst_size_ptr) = get_rsvd_cnt();
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
 
    INTRINSIC_NO_ARGS(set_mask_norm);
    INTRINSIC(pipe_barrier, PIPE_V);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_mask_1d_scalar(memref_t<__ubuf__ T, 1> *src,
                      memref_t<__ubuf__ bool, 1> *mask,
                      memref_t<__ubuf__ T, 1> *dst,
                      memref_t<__ubuf__ int64_t, 1> *dst_size) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("gather_mask_1d");
#endif
  int64_t element_count = src->sizes[0];
  int64_t mask_count = mask->sizes[0];
  
  if (element_count <= 0 || element_count != mask_count) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    return;
  }

  constexpr int factor = bitwidthOf<int8_t>() / bitwidthOf<bool>();
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ bool *mask_ptr = mask->aligned;
  int64_t bit_offset = mask->offset;
  int64_t byte_offset = bit_offset / factor;
  int64_t start_bit = bit_offset % factor;
  __ubuf__ int8_t* mask_pt = (__ubuf__ int8_t*)mask_ptr + byte_offset;

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ int64_t *dst_size_ptr = dst_size->aligned + dst_size->offset;
  const int64_t src_stride = src->strides[0];
  const int64_t dst_stride = dst->strides[0];

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t write_idx = 0;
  for (int64_t i = 0; i < element_count; ++i) {
      int64_t global_bit = start_bit + i;
      int64_t byte_idx = global_bit / 8;
      int bit_idx = global_bit % 8;
      
      bool mask_value = (mask_pt[byte_idx] >> bit_idx) & 1;
      
      if (mask_value) {
          dst_ptr[write_idx * dst_stride] = src_ptr[i * src_stride];
          write_idx++;
      }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);

  *dst_size_ptr = write_idx;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_unaligned_gather_mask_1d(memref_t<__ubuf__ T, 1> *src,
                            memref_t<__ubuf__ bool, 1> *mask,
                            memref_t<__ubuf__ T, 1> *dst) {
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  int64_t mask_offset = mask->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  
  bool is_offset_aligned = isAddress32ByteAligned<T>(src_ptr) &&
                          (mask_offset % INTR_BYTES_PER_REPEAT == 0) &&
                          isAddress32ByteAligned<T>(dst_ptr);
  bool is_stride_aligned = (src->strides[0] == 1) &&
                          (mask->strides[0] == 1) &&
                          (dst->strides[0] == 1);
  
  return !is_offset_aligned || !is_stride_aligned;
}

extern "C" {
REGISTE_GATHER_MASK(1, uint16_t)
REGISTE_GATHER_MASK(1, uint32_t)
REGISTE_GATHER_MASK(1, int16_t)
REGISTE_GATHER_MASK(1, int32_t)
REGISTE_GATHER_MASK(1, half)
REGISTE_GATHER_MASK(1, float)
REGISTE_GATHER_MASK(1, bfloat16_t)
}