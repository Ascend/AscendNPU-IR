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

#include "Vector/Gather/GatherUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
flip_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst) {
  const int64_t size0 = src->sizes[0];
  const int64_t stride0 = src->strides[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < size0; ++i) {
    *(dst_ptr + stride0 * i) = *(src_ptr + stride0 * (size0 - 1 - i));
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

extern "C" {
//===-------------------------------------------------------------------===//
// flip, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_FLIP(1, uint16_t)
REGISTE_FLIP(1, uint32_t)
REGISTE_FLIP(1, int16_t)
REGISTE_FLIP(1, int32_t)
REGISTE_FLIP(1, int8_t)
REGISTE_FLIP(1, uint8_t)
REGISTE_FLIP(1, int64_t)
REGISTE_FLIP(1, uint64_t)
REGISTE_FLIP(1, half)
REGISTE_FLIP(1, float)
REGISTE_FLIP(1, bfloat16_t)
}
