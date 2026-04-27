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

#include "DMA/L12UB.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_copy_cbuf_to_ub_4d_to_2d_core(memref_t<__cbuf__ T, 4> *src,
                                              memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride3_src = src->strides[3];
  assert((stride3_src == 1) && "Last dimension of src must be contiguous.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_ub_4d_to_2d_core(memref_t<__cbuf__ T, 4> *src,
                              memref_t<__ubuf__ T, 2> *dst) {
  check_inputs_of_copy_cbuf_to_ub_4d_to_2d_core(src, dst);

  __cbuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  int64_t size3 = src->sizes[3];
  int64_t stride0 = src->strides[0];
  int64_t stride1 = src->strides[1];
  int64_t stride2 = src->strides[2];
  int64_t stride3 = src->strides[3];

  if (stride3 == 1) [[likely]] { // keep this because check is only on when
                                 // ENABLE_CPU_TRACE_INTRINSIC
    for (int64_t i = 0; i < size0; i++) {
      for (int64_t j = 0; j < size1; j++) {
        for (int64_t k = 0; k < size2; k++) {
          int64_t src_offset = i * stride0 + j * stride1 + k * stride2;
          __cbuf__ void *cur_src =
              static_cast<__cbuf__ void *>(src_ptr + src_offset);
          int64_t dst_m = j * size2 + k;
          int64_t dst_n = i * size3;
          int64_t dst_offset =
              dst_m * dst->strides[0] + dst_n * dst->strides[1];
          __ubuf__ void *cur_dst =
              static_cast<__ubuf__ void *>(dst_ptr + dst_offset);
          int64_t unit = 32; // assuming size3 * sizeof(T) is at least 32 and is
                             // divisible by 32
          // TODO: use nBurst to simplify the loop
          INTRINSIC(copy_cbuf_to_ubuf, cur_dst, cur_src,
                    0,                        // sid
                    1,                        // nBurst
                    size3 * sizeof(T) / unit, // lenBurst
                    0,                        // srcGap
                    0                         // dstGap
          );
        }
      }
    }
  }
  // in other cases, the destination will not be updated and will contain
  // undefined contents
}

extern "C" {
REGISTER_L12UB(float, 4, 2)
REGISTER_L12UB(bfloat16_t, 4, 2)
REGISTER_L12UB(half, 4, 2)
REGISTER_L12UB(int8_t, 4, 2)
REGISTER_L12UB(uint8_t, 4, 2)
REGISTER_L12UB(int16_t, 4, 2)
REGISTER_L12UB(uint16_t, 4, 2)
REGISTER_L12UB(int32_t, 4, 2)
REGISTER_L12UB(uint32_t, 4, 2)
REGISTER_L12UB(int64_t, 4, 2)
REGISTER_L12UB(uint64_t, 4, 2)
}
