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
#include "RegBase/Gather/GatherUtils.h"
#include "RegBase/VecUtils.h"
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)
/// gather op description:
/// retrieve some elements from src according to indices,
/// and store them into dst
///
/// \param src (type: memref<a x T>)
/// \param indices (type: memref<b x int32_t>)
/// \param dst (type: memref<b x T>)

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_1d(memref_t<__ubuf__ T, 1> *src,
          memref_t<__ubuf__ int32_t, 1> *indices,
          memref_t<__ubuf__ T, 1> *dst) {
  const int64_t size_indices = indices->sizes[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ int32_t *inc_ptr = indices->aligned + indices->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int i = 0; i < size_indices; i++) {
    dst_ptr[i] = src_ptr[inc_ptr[i]];
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

extern "C" {
//===-------------------------------------------------------------------===//
// gather, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_GATHER(1, uint16_t)
REGISTE_GATHER(1, uint32_t)
REGISTE_GATHER(1, int8_t)
REGISTE_GATHER(1, int16_t)
REGISTE_GATHER(1, int32_t)
REGISTE_GATHER(1, half)
REGISTE_GATHER(1, float)
REGISTE_GATHER(1, bfloat16_t)
REGISTE_GATHER(1, float8_e4m3_t)
REGISTE_GATHER(1, float8_e5m2_t)
}
#endif