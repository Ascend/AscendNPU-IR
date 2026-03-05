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
#include "Vector/Deinterleave/DeinterleaveUtils.h"
#include "Vector/VecUtils.h"

/// deinterleave op description:
/// deinterleave channel0 from N channels support 2 scenarios
/// 1. src (a, b) with stride [m, n] to dst (a, b) with stride [p, 1]
/// 'a' and 'b' are size of src, 'n' and 'm' are stride of src, 'p' is stride of
/// dst
/// 2. src src (a, b0*n) with stride [m, 1] to dst (a, b0) with stride [p, 1]
/// 'a' and 'b' are size of src, 'm' and 'p' are stride of src, 'n' is
/// channel_num
///
/// \param src (type: memref<axbxT, strided<[m, n]>>)
/// \param dst (type: memref<axbxT, strided<[p, 1]>>)
/// or
/// \param src (type: memref<axmxT, strided<[m, 1]>>)
/// \param dst (type: memref<axm/nxT, strided<[p, 1]>)
///
/// Constraints:
/// 1. only support select channel0 from N channels

template <DeinterleaveMode MODE, typename T>
__aiv__ __attribute__((always_inline)) void
vector_deinterleave_2d(memref_t<__ubuf__ T, 2> *src,
                       memref_t<__ubuf__ T, 2> *dst) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t src_stride0 = src->strides[0];
  int64_t src_stride1 = src->strides[1];
  int64_t src_size0 = src->sizes[0];
  int64_t src_size1 = src->sizes[1];
  int64_t dst_size0 = dst->sizes[0];
  int64_t dst_stride0 = dst->strides[0];

  if constexpr (MODE == DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS) {
    if (src_stride1 % num_per_block == 0 &&
        src_stride0 == dst_stride0 * src_stride1) {
      // src: memref<axbxT, strided<[m, n]>>
      // dst: memref<axbxT, strided<[m/n, 1]>>
      int64_t repeat_times = src_size0 * dst_stride0;
      int64_t src_repeat_stride = src_stride1 / num_per_block;
      select_channel0_from_block<T, 2>(src, dst, repeat_times,
                                       src_repeat_stride);
      return;
    }
    if (src_stride1 == 1 && src_stride0 % dst_stride0 == 0 &&
        (src_stride0 / dst_stride0) % num_per_block == 0) {
      // src: memref<axnxT, strided<[n, 1]>>
      // dst: memref<axm/NxT, strided<[m/N, 1]>>
      int64_t repeat_times = dst_size0 * dst_stride0;
      int64_t src_repeat_stride = src_stride0 / dst_stride0 / num_per_block;
      select_channel0_from_block<T, 2>(src, dst, repeat_times,
                                       src_repeat_stride);
      return;
    }

    // TODO support channel0 from channels when N is not 32 bytes align
  }
  static_assert("deinterleave op's unsupported mode");
}

extern "C" {
//===-------------------------------------------------------------------===//
// deinterleave op, 2 dim
//===-------------------------------------------------------------------===//
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, half)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2,
                      bfloat16_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, float)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, int16_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, int32_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, uint16_t)
REGISTER_DEINTERLEAVE(channel_0_from_n_channels,
                      DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, uint32_t)
}