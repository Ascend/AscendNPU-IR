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
#include "Vector/VecUtils.h"
#include <type_traits>

/// Core func of vcopy intrin : (a, 1, c) -> (a, b, c)
/// \param src (type: memref<a x 1 x c xT, stride[c, c, 1]>)
/// \param dst (type: memref<a x b x c xT, stride[b*c, c, 1]>)
///
/// Constraints:
/// 1. 'c' must be num_per_block, num_per_block means element number of UB block
/// unit.
/// 2. 'b' <= MAX_VCOPY_DST_REPEAT_STRIDE(4096).
/// 3. The intrin needs count mode.
template <typename T, typename = typename std::enable_if<sizeof(T) == 2 ||
                                                         sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
brc_mid_axis_3d_vcopy_specialization(memref_t<__ubuf__ T, 3> *src,
                                     memref_t<__ubuf__ T, 3> *dst) {
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t dst_repeat_stride = dst->sizes[1];
  const int64_t vcopy_main_cnt = dst->sizes[0] / INTR_MAX_REPEAT_CNTS;

  INTRINSIC(set_vector_mask, 0x0, dst->strides[0]); // count mode
  int64_t src_offset = 0;
  int64_t dst_offset = 0;
  if (vcopy_main_cnt > 0) [[unlikely]] {
    // deal full mask main
    for (int64_t i = 0; i < vcopy_main_cnt; ++i) {
      if constexpr (sizeof(T) == BYTES_B16) {
        // TODO:vcopy only support uint16 and uint32, type of
        // dst_addr/src_addr will be T in the future.
        INTRINSIC(vcopy,
                  (__ubuf__ uint16_t *)dst_ptr + dst_offset, // dst
                  (__ubuf__ uint16_t *)src_ptr + src_offset, // src
                  INTR_MAX_REPEAT_CNTS,                      // repeat
                  1,                                         // dst blk stride
                  0,                                         // src blk stride
                  dst_repeat_stride,
                  1); // src rep stride
      }
      if constexpr (sizeof(T) == BYTES_B32) {
        INTRINSIC(vcopy,
                  (__ubuf__ uint32_t *)dst_ptr + dst_offset, // dst
                  (__ubuf__ uint32_t *)src_ptr + src_offset, // src
                  INTR_MAX_REPEAT_CNTS,                      // repeat
                  1,                                         // dst blk stride
                  0,                                         // src blk stride
                  dst_repeat_stride,
                  1); // src rep stride
      }

      dst_offset = dst_offset + INTR_MAX_REPEAT_CNTS * dst->strides[0];
      src_offset = src_offset + INTR_MAX_REPEAT_CNTS * src->strides[0];
    }
  }
  // deal tail
  const int64_t vcopy_tail_cnt = dst->sizes[0] % INTR_MAX_REPEAT_CNTS;
  if (vcopy_tail_cnt > 0) [[likely]] {
    if constexpr (sizeof(T) == BYTES_B16) {
      INTRINSIC(vcopy,
                (__ubuf__ uint16_t *)dst_ptr + dst_offset, // dst
                (__ubuf__ uint16_t *)src_ptr + src_offset, // src
                vcopy_tail_cnt,                            // repeat
                1,                                         // dst blk stride
                0,                                         // src blk stride
                dst_repeat_stride,                         // dst rep stride
                1);                                        // src rep stride
    }
    if constexpr (sizeof(T) == BYTES_B32) {
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)dst_ptr + dst_offset, // dst
                (__ubuf__ uint32_t *)src_ptr + src_offset, // src
                vcopy_tail_cnt,                            // repeat
                1,                                         // dst blk stride
                0,                                         // src blk stride
                dst_repeat_stride,                         // dst rep stride
                1);                                        // src rep stride
    }
  }
}

/// Core func of vcopy intrin : (a, 1, c) -> (a, b, c)
/// \param src (type: memref<a x 1 x c xT, stride[B, B, 1]>)
/// \param dst (type: memref<a x b x c xT, stride[b*B, B, 1]>)
///
/// Constraints:
/// 1. 'c' must be ub_block_unit aligned, which is multiple of UB block unit.
/// 2. The intrin needs count mode.
template <typename T, typename = typename std::enable_if<
                          sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                          sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
brc_mid_axis_3d_vcopy(memref_t<__ubuf__ T, 3> *src,
                      memref_t<__ubuf__ T, 3> *dst) {
  // brc middle align support b8/b64
  if constexpr (sizeof(T) == BYTES_B8 || sizeof(T) == BYTES_B64) {
    memref_t<__ubuf__ T, 2> src_2d;
    memref_t<__ubuf__ T, 2> dst_2d;
    throw_Nd_to_Md_args<T, 3, 2>(src, &src_2d);
    throw_Nd_to_Md_args<T, 3, 2>(dst, &dst_2d);
    for (int64_t loop_num = 0; loop_num < dst->sizes[0]; ++loop_num) {
      brc_first_axis_2d_align_core(&src_2d, &dst_2d);
      src_2d.offset = src_2d.offset + src->strides[0];
      dst_2d.offset = dst_2d.offset + dst->strides[0];
    }
    return;
  }
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const int64_t dst_repeat_stride = CEIL_DIV(
      dst->strides[1],
      num_per_block); // dstRepeatSize=aligned(c, ub_block_unit)/num_per_block
  if constexpr (sizeof(T) == BYTES_B16 || sizeof(T) == BYTES_B32) {
    if (dst->strides[1] == num_per_block &&
        dst->strides[0] == dst->sizes[1] * dst->sizes[2] &&
        dst->sizes[1] <= MAX_VCOPY_DST_REPEAT_STRIDE) {
      // map a to repeat axis and map c to block axis
      brc_mid_axis_3d_vcopy_specialization<T>(src, dst);
      return;
    }
  }

  const int64_t vcopy_main_cnt = dst->sizes[1] / INTR_MAX_REPEAT_CNTS;
  INTRINSIC(set_vector_mask, 0x0, dst->sizes[2]); // count mode
  int64_t loop_src_offset = 0;
  int64_t loop_dst_offset = 0;
  int64_t src_offset = 0;
  int64_t dst_offset = 0;
  // loop a axis and then map b to repeat axis and map c to burst length of one
  // repeat
  for (int64_t loop_num = 0; loop_num < dst->sizes[0]; ++loop_num) {
    src_offset = loop_src_offset;
    dst_offset = loop_dst_offset;
    if (vcopy_main_cnt > 0) [[unlikely]] {
      // deal full mask main
      for (int64_t i = 0; i < vcopy_main_cnt; ++i) {
        if constexpr (sizeof(T) == BYTES_B16) {
          // TODO:vcopy only support uint16 and uint32, type of
          // dst_addr/src_addr will be T in the future.
          INTRINSIC(vcopy,
                    (__ubuf__ uint16_t *)dst_ptr + dst_offset, // dst
                    (__ubuf__ uint16_t *)src_ptr + src_offset, // src
                    INTR_MAX_REPEAT_CNTS,                      // repeat
                    1,                                         // dst blk stride
                    1,                                         // src blk stride
                    dst_repeat_stride, // dst rep stride=aligned(c,
                                       // ub_block_unit)/num_per_block
                    0);                // src rep stride
        }
        if constexpr (sizeof(T) == BYTES_B32) {
          INTRINSIC(vcopy,
                    (__ubuf__ uint32_t *)dst_ptr + dst_offset, // dst
                    (__ubuf__ uint32_t *)src_ptr + src_offset, // src
                    INTR_MAX_REPEAT_CNTS,                      // repeat
                    1,                                         // dst blk stride
                    1,                                         // src blk stride
                    dst_repeat_stride, // dst rep stride = aligned(c,
                                       // ub_block_unit)/num_per_block
                    0);                // src rep stride
        }

        dst_offset = dst_offset + INTR_MAX_REPEAT_CNTS * dst->strides[1];
      }
    }
    // deal tail
    const int64_t vcopy_tail_cnt = dst->sizes[1] % INTR_MAX_REPEAT_CNTS;
    if (vcopy_tail_cnt > 0) [[likely]] {
      if constexpr (sizeof(T) == BYTES_B16) {
        INTRINSIC(vcopy,
                  (__ubuf__ uint16_t *)dst_ptr + dst_offset, // dst
                  (__ubuf__ uint16_t *)src_ptr + src_offset, // src
                  vcopy_tail_cnt,                            // repeat
                  1,                                         // dst blk stride
                  1,                                         // src blk stride
                  dst_repeat_stride,                         // dst rep stride
                  0);                                        // src rep stride
      }
      if constexpr (sizeof(T) == BYTES_B32) {
        INTRINSIC(vcopy,
                  (__ubuf__ uint32_t *)dst_ptr + dst_offset, // dst
                  (__ubuf__ uint32_t *)src_ptr + src_offset, // src
                  vcopy_tail_cnt,                            // repeat
                  1,                                         // dst blk stride
                  1,                                         // src blk stride
                  dst_repeat_stride,                         // dst rep stride
                  0);                                        // src rep stride
      }
    }

    loop_src_offset = loop_src_offset + src->strides[0];
    loop_dst_offset = loop_dst_offset + dst->strides[0];
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_broadcast_mid_axis_align_3d(
    memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(dst->strides[1]) &&
         "The dst strides[1] must be aligned to block.");
#endif
}

/// Core func of aligned, broadcast src (a, 1, c) to dst (a, b, c), 'a' and 'c'
/// may be merged axis of multiple elementwise axes and b is broadcast axis.
/// Here the size of 'c' is aligned to UB block unit.
///
/// \param src (type: memref<a x 1 x c xT, stride[B, B, 1]>)
/// \param dst (type: memref<a x b x c xT, stride[b*B, B, 1]>)
///
/// Constraints:
/// 1. 'c' must be ub_block_unit aligned, which is multiple of UB block unit
///
/// Example:
/// memref<20x1x16xf16, strided<[16, 16, 1]>> => memref<20x10x16xf16,
/// strided<[160, 16, 1]>>
template <typename T,
          typename = typename std::enable_if<
              std::is_same<T, bool>::value || sizeof(T) == 1 ||
              sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_mid_axis_align_3d(memref_t<__ubuf__ T, 3> *src,
                                   memref_t<__ubuf__ T, 3> *dst) {
  // Input parameter constraints assert.
  check_inputs_of_vector_broadcast_mid_axis_align_3d(src, dst);

  INTRINSIC_NO_ARGS(set_mask_count);
  brc_mid_axis_3d_vcopy<T>(src, dst);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

// full template specialization for bool.
template <>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_mid_axis_align_3d(memref_t<__ubuf__ bool, 3> *src,
                                   memref_t<__ubuf__ bool, 3> *dst) {
  memref_t<__ubuf__ half, 3> half_src;
  memref_t<__ubuf__ half, 3> half_dst;
  view_as<bool, half, 3>(src, &half_src);
  view_as<bool, half, 3>(dst, &half_dst);
  vector_broadcast_mid_axis_align_3d(&half_src, &half_dst);
}

extern "C" {
//===-------------------------------------------------------------------===//
// broadcast mid axis align, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_BROADCAST_MID_AXIS_ALIGN(bfloat16_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(int16_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(uint16_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(half);
REGISTE_BROADCAST_MID_AXIS_ALIGN(int32_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(uint32_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(float);
REGISTE_BROADCAST_MID_AXIS_ALIGN(bool);
REGISTE_BROADCAST_MID_AXIS_ALIGN(int8_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(uint8_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(int64_t);
REGISTE_BROADCAST_MID_AXIS_ALIGN(uint64_t);
}