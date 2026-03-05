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
#include "Vector/Gather/GatherUtils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_core(__ubuf__ T *dst_ptr, __ubuf__ uint32_t *src_ptr, int64_t count) {
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0x0, count);
  INTRINSIC(vgather,
            dst_ptr,                // dst
            src_ptr,                // indices_src
            0,                      // indices_offset_address
            INTR_BLOCKS_PER_REPEAT, // dst rep stride
            1);                     // repeat
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// gather op description:
/// retrieve some elements from src according to indices,
/// and store them into dst
///
/// \param src (type: memref<a x T>)
/// \param indices (type: memref<b x int32_t>)
/// \param dst (type: memref<b x T>)
/// \param tmp_buf (type: memref<b x int32_t>)
///
/// Constraints:
/// 1. only accept b16 and b32 data type for T
/// 2. tmp_buf size should be at least b x int32_t

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *indices,
          memref_t<__ubuf__ T, 1> *dst,
          memref_t<__ubuf__ int32_t, 1> *tmp_buf) {
  const int64_t size_src = src->sizes[0];
  const int64_t size_indices = indices->sizes[0];
  // step 1: compute the gather index ub addrs of which unit is bytes
  // the gather ub addr is src_start_pointer + indices * sizeof(T)
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  if (size_src == 1) {
    // src: memref<1xT, strided[N]>
    // dst: memref<nxT, strided[N]>
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    const int64_t stride_dst = dst->strides[0];
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for(int64_t i = 0; i < size_indices; ++i) {
      dst_ptr[i * stride_dst] = src_ptr[0];
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  const int32_t addr_src = (intptr_t)src_ptr;
  constexpr int bytes = sizeof(T);
  // indices_offset = vmuls(indices, sizeof(T))
  memref_t<__ubuf__ int32_t, 1> tmp_indices_offset{tmp_buf->allocated,
                                                   tmp_buf->aligned,
                                                   tmp_buf->offset,
                                                   {size_indices},
                                                   {1}};
  vector_eltwise_vs_1d<VectorOpTy::VMULS>(indices, bytes, &tmp_indices_offset);
  INTRINSIC(pipe_barrier, PIPE_V);

  // indices_elements_addr = vadds(indices_offset, addr_src)
  vector_eltwise_vs_1d<VectorOpTy::VADDS>(&tmp_indices_offset, addr_src,
                                          &tmp_indices_offset);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step 2: use vgather intr to gather data
  __ubuf__ int32_t *tmp_indices_offset_ptr =
      tmp_indices_offset.aligned + tmp_indices_offset.offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  if constexpr (bytes == BYTES_B32) {
    // b32 type
    gather_core((__ubuf__ uint32_t *)dst_ptr,
                (__ubuf__ uint32_t *)tmp_indices_offset_ptr, size_indices);
  } else if constexpr (bytes == BYTES_B16) {
    // b16 type
    gather_core((__ubuf__ uint16_t *)dst_ptr,
                (__ubuf__ uint32_t *)tmp_indices_offset_ptr, size_indices);
  } else {
    static_assert("vgather unsupports this data type");
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// gather, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_GATHER(1, uint16_t)
REGISTE_GATHER(1, uint32_t)
REGISTE_GATHER(1, int16_t)
REGISTE_GATHER(1, int32_t)
REGISTE_GATHER(1, half)
REGISTE_GATHER(1, float)
REGISTE_GATHER(1, bfloat16_t)
}
