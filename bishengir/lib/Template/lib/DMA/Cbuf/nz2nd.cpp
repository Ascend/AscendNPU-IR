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

#include "DMA/NZ2ND.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_copy_cbuf_to_gm_4d_to_2d_core(memref_t<__cbuf__ T, 4> *src,
                                              memref_t<__gm__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride3_src = src->strides[3];
  assert((stride3_src == 1) && "Last dimension of src must be contiguous.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) __gm__ T * convert_cbuf_array_to_gm(__cbuf__ T *ptr, __gm__ T *tmp, size_t count) {
    __cbuf__ uint8_t *aligned_ptr = (__cbuf__ uint8_t *)((uint64_t)ptr & ~0xF);
    uint32_t offset = (uint64_t)ptr - (uint64_t)aligned_ptr;
    const int64_t unit_size = 32;
    INTRINSIC(pipe_barrier, PIPE_ALL);
    INTRINSIC(dcci, tmp, 0);
    INTRINSIC(copy_cbuf_to_gm, tmp, aligned_ptr,
              0,
              1,
              (sizeof(T) * count + offset) / unit_size,
              0,
              0
    );
    INTRINSIC(pipe_barrier, PIPE_ALL);
    return (__gm__ T*)((__gm__ uint8_t*)tmp + offset);
};

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_gm_4d_to_2d_core(memref_t<__cbuf__ T, 4> *src,
                              memref_t<__gm__ T, 2> *dst) {
  check_inputs_of_copy_cbuf_to_gm_4d_to_2d_core(src, dst);

  __cbuf__ T *src_ptr = src->aligned + src->offset;
  __gm__ T *dst_ptr = dst->aligned + dst->offset;

  const int64_t max_chunk_size = 16;

  if (src->strides[3] == 1 && src->sizes[3] <= max_chunk_size) [[likely]] { // keep this because check is only on when
                                                            // ENABLE_CPU_TRACE_INTRINSIC
    T tmp[max_chunk_size]; // temporary array for first items
    int64_t tmp_size = 0;
    for (int64_t x = (dst->sizes[1] / src->sizes[3]) * src->sizes[3]; x >= 0; x = x - src->sizes[3]) {
      for (int64_t y = dst->sizes[0] - 1; y >=0; y--) {
        int64_t dst_offset = y * dst->sizes[1] + x;
        int64_t a = y / src->sizes[2];
        int64_t b = y - a * src->sizes[2];
        int64_t c = x / src->sizes[3];
        int64_t chunk_size = c == src->sizes[0] - 1 ? dst->sizes[1] - x : src->sizes[3];
        int64_t src_offset = a * src->sizes[2] * src->sizes[3] + b * src->sizes[3] + c * src->sizes[1] * src->sizes[2] * src->sizes[3];
        __gm__ T * converted = convert_cbuf_array_to_gm(src_ptr + src_offset, dst->aligned, src->sizes[3]);
        for (int64_t i = chunk_size - 1; i >= 0; i--) {
          int64_t dst_address = dst_offset + i;
          if (dst_address < max_chunk_size) {
            tmp[dst_address] = converted[i];
            if (tmp_size < dst_address + 1) {
              tmp_size = dst_address + 1;
            }
          } else {
            dst_ptr[dst_address] = converted[i];
          }
        }
      }
    }
    for (int64_t i = 0; i < tmp_size; i++) {
      dst_ptr[i] = tmp[i];
    }
  }
  // in other cases, the destination will not be updated and will contain
  // undefined contents
}

extern "C" {
REGISTER_NZ2ND(float, 4, 2)
REGISTER_NZ2ND(bfloat16_t, 4, 2)
REGISTER_NZ2ND(half, 4, 2)
REGISTER_NZ2ND(int8_t, 4, 2)
REGISTER_NZ2ND(uint8_t, 4, 2)
REGISTER_NZ2ND(int16_t, 4, 2)
REGISTER_NZ2ND(uint16_t, 4, 2)
REGISTER_NZ2ND(int32_t, 4, 2)
REGISTER_NZ2ND(uint32_t, 4, 2)
REGISTER_NZ2ND(int64_t, 4, 2)
REGISTER_NZ2ND(uint64_t, 4, 2)
}
