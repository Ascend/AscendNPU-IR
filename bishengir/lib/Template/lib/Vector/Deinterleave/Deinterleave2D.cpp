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

template <DeinterleaveMode MODE, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_deinterleave_2d(memref_t<__ubuf__ T, 2> *src,
                       memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  WARN_SCALAR_IMPL("vector_deinterleave_2d");
#endif
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  int64_t src_stride0 = src->strides[0];
  int64_t src_stride1 = src->strides[1];
  int64_t src_size0 = src->sizes[0];
  int64_t src_size1 = src->sizes[1];
  int64_t dst_size0 = dst->sizes[0];
  int64_t dst_size1 = dst->sizes[1];
  int64_t dst_stride0 = dst->strides[0];
  int64_t dst_stride1 = dst->strides[1];

  // For scenario 2 (src_stride1 == 1, channel count encoded in size1), the
  // effective stride between consecutive channel-0 elements along the inner
  // dim is N = src_size1 / dst_size1. For scenario 1, src_stride1 already
  // equals N, so eff_src_stride1 = src_stride1.
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t eff_src_stride1 = src_stride1;
  if (src_stride1 == 1 && src_stride0 % dst_stride0 == 0 &&
      (src_stride0 / dst_stride0) % num_per_block == 0) {
    eff_src_stride1 = src_size1 / dst_size1;
  }

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  for (int64_t i = 0; i < dst_size0; ++i) {
    for (int64_t j = 0; j < dst_size1; ++j) {
      dst_ptr[i * dst_stride0 + j * dst_stride1] =
          src_ptr[i * src_stride0 + j * eff_src_stride1];
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

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
    // When N is not 32 Byte aligned, fallback to scalar deinterleave
    __ubuf__ T *src_ptr = src->aligned + src->offset;
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    bool is_offset_aligned = isAddress32ByteAligned<T>(src_ptr) &&
                             isAddress32ByteAligned<T>(dst_ptr);
    if (!is_offset_aligned) {
      scalar_deinterleave_2d<MODE, T>(src, dst);
      return;
    }
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
