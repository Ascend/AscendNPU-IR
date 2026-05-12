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

#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Deinterleave/DeinterleaveUtils.h"
#include "Vector/VecUtils.h"

// Deinterleave [c0,c1,...] along logical axis with element stride s:
// logical ch0 indices 0,2,4,... lie at offsets 0, 2s, 4s,... — model as 2D
// (pairs,1) with strides {2*s,1} and base offset 0;
// logical ch1 indices 1,3,5,... lie at offsets s, 3s, 5s,... — model as 2D
// (pairs,1) with strides {2*s,1} and base offset +s;
// copy_ubuf_to_ubuf_2d_core uses repeat+gap (vector when 32B-aligned in
// element strides, else scalar).
template <DeinterleaveMode MODE, typename T>
__aiv__ __attribute__((always_inline)) void
vector_deinterleave_1d_aligned(memref_t<__ubuf__ T, 1> *src,
                               memref_t<__ubuf__ T, 1> *dst) {
  int64_t src_size0 = src->sizes[0];
  int64_t dst_size0 = dst->sizes[0];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];
  memref_t<__ubuf__ T, 2> src_2d = {src->allocated,
                                    src->aligned,
                                    src->offset,
                                    {src_size0, 1},
                                    {2 * src_stride0, 1}};
  if constexpr (MODE == DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS) {
    src_2d.offset += src_stride0;
  }
  memref_t<__ubuf__ T, 2> dst_2d = {dst->allocated,
                                    dst->aligned,
                                    dst->offset,
                                    {dst_size0, 1},
                                    {dst_stride0, 1}};
  copy_ubuf_to_ubuf_2d_core<T>(&src_2d, &dst_2d);
}

/// deinterleave op description:
/// 1. deinterleave src (a,) to dst (a / 2) odd or even depending on mode,
/// 'a' is size of src, and size of dst is always 'a' / 2
/// 2. deinterleave src (a,) with stride [n] to dst (a) with stride [1] ,
/// 'a' is size of src and dst, and 'n' is stride
///
/// \param src (type: memref<a x T>)
/// \param dst (type: memref<(a/2) x T>)
/// or
/// \param src (type: memref<axT, strided<[n]>>)
/// \param dst (type: memref<axT, strided<[1]>>)
///
/// Constraints:
/// 1. dst will store odd or even elements of src depending on mode
/// 2. deinterleave n to 1 only support stride is 32 bytes align

template <DeinterleaveMode MODE, typename T>
__aiv__ __attribute__((always_inline)) void
vector_deinterleave_1d(memref_t<__ubuf__ T, 1> *src,
                       memref_t<__ubuf__ T, 1> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_v_1d(src, dst);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t src_size0 = src->sizes[0];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];

  if constexpr (MODE == DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS) {
    if (src_stride0 % num_per_block == 0) {
      if (src_size0 == dst->sizes[0] && src_stride0 == dst->strides[0]) {
        // src: memref<axT, strided<[n]>>
        // dst: memref<axT, strided<[n]>>
        copy_ubuf_to_ubuf_1d_core(src, dst);
        return;
      }

      // src: memref<axT, strided<[n]>>
      // dst: memref<axT, strided<[1]>>
      int64_t repeat_times = src_size0;
      int64_t src_repeat_stride = src_stride0 / num_per_block;
      select_channel0_from_block<T, 1>(src, dst, repeat_times,
                                       src_repeat_stride);
      return;
    }
  }

  if (src_stride0 % num_per_block == 0 && dst_stride0 % num_per_block == 0) {
    vector_deinterleave_1d_aligned<MODE, T>(src, dst);
    return;
  }

  memref_t<__ubuf__ T, 1> new_src = *src;
  memref_t<__ubuf__ T, 1> new_dst = *dst;
  if (src_stride0 != 1) {
    new_src.sizes[0] = src_size0 * src_stride0;
    new_src.strides[0] = 1;
    new_dst.sizes[0] = dst->sizes[0] * dst->strides[0];
    new_dst.strides[0] = 1;
  }

  if constexpr (MODE == DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS) {
    vreducev2_1d_with_pattern_mode<T, PatternMode::INDEX_0_FROM_2_ELEMENTS>(
        &new_src, &new_dst);
  } else if constexpr (MODE == DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS) {
    vreducev2_1d_with_pattern_mode<T, PatternMode::INDEX_1_FROM_2_ELEMENTS>(
        &new_src, &new_dst);
  } else {
    static_assert("deinterleave op's unsupported mode");
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// deinterleave op, 1 dim, select 1 from 2
//===-------------------------------------------------------------------===//
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, half)
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1,
                      bfloat16_t)
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, float)
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, int16_t)
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, int32_t)
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, uint16_t)
REGISTER_DEINTERLEAVE(channel_0_from_2_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, uint32_t)

REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, half)
REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1,
                      bfloat16_t)
REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, float)
REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, int16_t)
REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, int32_t)
REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, uint16_t)
REGISTER_DEINTERLEAVE(channel_1_from_2_channels,
                      DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, uint32_t)

//===-------------------------------------------------------------------===//
// deinterleave op, 1 dim, select 1 from n
//===-------------------------------------------------------------------===//
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1, half)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1,
                      bfloat16_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1, float)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1, int16_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1, int32_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1, uint16_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 1, uint32_t)
}
