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
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/VecUtils.h"
#include <cstdint>
#include <type_traits>

/// Core func of vbrcb intrin : (m, 1) -> (n, p)
/// \param src (type: memref<m x 1 xT, stride[1, 1]>)
/// \param dst (type: memref<n x p xT, stride[p, 1]>)
///
/// Constraints:
/// 1. 'n' must be aligned(m, src_num_per_vbrcb_repeat), here
/// src_num_per_vbrcb_repeat means the number of src that vbrcb intr handle per
/// repeat
/// 2. 'p' must be dst_num_per_vbrcb_block, here dst_num_per_vbrcb_block means
/// the number of dst that vbrcb intr handle per block
template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_last_axis_2d_vbrcb(memref_t<__ubuf__ T, 2> *src,
                       memref_t<__ubuf__ T, 2> *dst) {
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  int64_t repeat_times = CEIL_DIV(
      src->sizes[0], kSrcNumPerRepeatOfVBRCB); // repeat = aligned(a, 8)/8

  constexpr int32_t max_vbrcb_repeat_times = get_vbrcb_max_repeat_cnts<T>();
  int64_t loop_num = repeat_times / max_vbrcb_repeat_times;

  if (repeat_times >= max_vbrcb_repeat_times) [[unlikely]] {
    for (int64_t i = 0; i < loop_num; ++i) {
      vbrcb_intrin<T>(src_ptr +
                          i * kSrcNumPerRepeatOfVBRCB * max_vbrcb_repeat_times,
                      dst_ptr + i * num_per_repeat * max_vbrcb_repeat_times,
                      1, // dst_block_stride
                      8, // dst_repeat_stride
                      max_vbrcb_repeat_times);
    }
  }

  if (repeat_times % max_vbrcb_repeat_times != 0) [[likely]] {
    int64_t remain_repeat_times = repeat_times % max_vbrcb_repeat_times;
    vbrcb_intrin<T>(
        src_ptr + loop_num * kSrcNumPerRepeatOfVBRCB * max_vbrcb_repeat_times,
        dst_ptr + loop_num * num_per_repeat * max_vbrcb_repeat_times,
        1, // dst_block_stride
        8, // dst_repeat_stride
        remain_repeat_times);
  }
}

/// Core func of vcopy intrin : (m, n) -> (m, multipled(n)), here multipled(n)
/// means p*n, which p is constant.
/// \param src (type: memref<m x n xT, stride[n, 1]>)
/// \param dst (type: memref<m x multipled(n) xT, stride[multipled(n), 1]>)
///
/// Constraints:
/// 1. 'n' must be num_per_block, num_per_block means element number of UB block
/// unit.
/// 2. The intrin needs count mode.
template <typename T, typename = typename std::enable_if<sizeof(T) == 2 ||
                                                         sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
brc_last_axis_2d_vcopy(memref_t<__ubuf__ T, 2> *src,
                       memref_t<__ubuf__ T, 2> *dst) {
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  // TODO: assert dst->strides[0] is aligned
  int64_t dst_repeat_stride =
      dst->strides[0] * sizeof(T) /
      UB_ALIGN_BYTES; // dstRepeatSize=aligned(b, ub_block_unit)/num_per_block
  int64_t repeat = dst->sizes[0]; // repeat=a

  INTRINSIC(set_vector_mask, 0x0, dst->sizes[1]); // count mode
  __ubuf__ T *dst_addr = dst_ptr;
  __ubuf__ T *src_addr = src_ptr;
  if (repeat >= INTR_MAX_REPEAT_CNTS) [[unlikely]] {
    // deal full mask main
    int64_t loop_num = repeat / INTR_MAX_REPEAT_CNTS;
    for (int64_t i = 1; i <= loop_num; i++) {
      if constexpr (sizeof(T) == BYTES_B16) {
        // TODO:vcopy only support uint16 and uint32, type of
        // dst_addr/src_addr will be T in the future.
        INTRINSIC(vcopy,
                  (__ubuf__ uint16_t *)dst_addr, // dst
                  (__ubuf__ uint16_t *)src_addr, // src
                  INTR_MAX_REPEAT_CNTS,          // repeat
                  1,                             // dst blk stride
                  0,                             // src blk stride
                  dst_repeat_stride,             // dst rep stride=aligned(b,
                                                 // ub_block_unit)/num_per_block
                  1);                            // src rep stride
      }
      if constexpr (sizeof(T) == BYTES_B32) {
        INTRINSIC(vcopy,
                  (__ubuf__ uint32_t *)dst_addr, // dst
                  (__ubuf__ uint32_t *)src_addr, // src
                  INTR_MAX_REPEAT_CNTS,          // repeat
                  1,                             // dst blk stride
                  0,                             // src blk stride
                  dst_repeat_stride,             // dst rep stride = aligned(b,
                                                 // ub_block_unit)/num_per_block
                  1);                            // src rep stride
      }

      dst_addr = dst_ptr +
                 i * dst_repeat_stride * num_per_block * INTR_MAX_REPEAT_CNTS;
      src_addr = src_ptr + i * num_per_block * INTR_MAX_REPEAT_CNTS;
    }
  }
  // deal tail
  int64_t repeat_tail =
      repeat - repeat / INTR_MAX_REPEAT_CNTS * INTR_MAX_REPEAT_CNTS;
  if (repeat_tail > 0) [[likely]] {
    if constexpr (sizeof(T) == BYTES_B16) {
      INTRINSIC(vcopy,
                (__ubuf__ uint16_t *)dst_addr, // dst
                (__ubuf__ uint16_t *)src_addr, // src
                repeat_tail,                   // repeat
                1,                             // dst blk stride
                0,                             // src blk stride
                dst_repeat_stride,             // dst rep stride
                1);                            // src rep stride
    }
    if constexpr (sizeof(T) == BYTES_B32) {
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)dst_addr, // dst
                (__ubuf__ uint32_t *)src_addr, // src
                repeat_tail,                   // repeat
                1,                             // dst blk stride
                0,                             // src blk stride
                dst_repeat_stride,             // dst rep stride
                1);                            // src rep stride
    }
  }
}

/// Core func of vreducev2 intrin : gather the subview (m, p) with stride[n, 1]
/// of src (m, n) together to dst (m, p).
/// \param src (type: memref<m x n xT, stride[n, 1]>)
/// \param dst (type: memref<m x p xT, stride[p, 1]>)
///
/// Constraints:
/// 1. 'n' must be ub_block_unit aligned.
/// 2. 'p' <= 'n'.
/// 3. The intrin needs count mode.
template <typename T, typename = typename std::enable_if<
                          std::is_same<T, bool>::value || sizeof(T) == 2 ||
                          sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
brc_last_axis_2d_vreducev2(memref_t<__ubuf__ T, 2> *src,
                           memref_t<__ubuf__ T, 2> *dst) {
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;

  INTRINSIC(set_vector_mask, 0x0, dst->sizes[1]); // count mode
  // maximum repeat of vreducev2 intr is enough large and no need to consider
  // the repeat overflow case
  if constexpr (sizeof(T) == BYTES_B16) {
    // TODO:vreducev2 only support uint16 and uint32, type of dst_addr/src_addr
    // will be T in the future.
    INTRINSIC(
        vreducev2,
        (__ubuf__ uint16_t *)dst_ptr, // dst
        (__ubuf__ uint16_t *)src_ptr, // src0
        (__ubuf__ uint16_t *)src_ptr, // src1_pattern, not work
        dst->sizes[0],                // repeat = a
        1,                            // src0 blk stride
        7, // pattern mode, 7 means pattern values are all 1, would not use src1
        src->strides[0] * sizeof(T) /
            UB_ALIGN_BYTES, // src0 rep stride = aligned(b,
                            // ub_block_unit)/num_per_block
        0);                 // src1 rep stride
  }
  if constexpr (sizeof(T) == BYTES_B32) {
    INTRINSIC(
        vreducev2,
        (__ubuf__ uint32_t *)dst_ptr, // dst
        (__ubuf__ uint32_t *)src_ptr, // src0
        (__ubuf__ uint32_t *)src_ptr, // src1_pattern, not work
        dst->sizes[0],                // repeat = a
        1,                            // src0 blk stride
        7, // pattern mode, 7 means pattern values are all 1, would not use src1
        src->strides[0] * sizeof(T) /
            UB_ALIGN_BYTES, // src0 rep stride = aligned(b,
                            // ub_block_unit)/num_per_block
        0);                 // src1 rep stride
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_broadcast_last_axis_align_2d(
    memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  auto tmp_ptr = tmp->aligned + tmp->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(dst->strides[0]) &&
         "The dst strides[0] must be aligned to block.");
#endif
}

/// Core func of aligned, broadcast src (a, 1) with stride [n, 1] to dst (a, b),
/// here n is 1 or aligned to a block. 'a' may be merged axis of multiple
/// elementwise axes and b is broadcast axis. Here the size of 'b' is aligned
/// to UB block unit, it need additional tmp buffer with size 'c'.
///
/// \param src (type: memref<a x 1 xT, stride[1, 1]>)
/// \param dst (type: memref<a x b xT, stride[b, 1]>)
/// \param tmp (type: memref<c xT, stride[1]>)
///
/// Constraints:
/// 1. 'b' must be ub_block_unit aligned, which is multiple of UB block unit
/// 2. 'c' must be larger than (aligned(a,
/// src_num_per_vbrcb_repeat)*num_per_block), here src_num_per_vbrcb_repeat
/// means the number of src that vbrcb intr handle per repeat, and num_per_block
/// means element number of UB block unit.
///
/// Example:
/// memref<20x1xf16, strided<[1, 1]>> => memref<20x10xf16, strided<[16, 1]>>
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_align_2d(memref_t<__ubuf__ T, 2> *src,
                                    memref_t<__ubuf__ T, 2> *dst,
                                    memref_t<__ubuf__ T, 1> *tmp) {
  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_vector_broadcast_last_axis_align_2d(src, dst, tmp);

  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  // broadcast src:(1, 1) -> dst:(1, b)
  if (src->sizes[0] == 1 && src->strides[0] == 1) [[unlikely]] {
    // TODO : remove it after refactoring the brc_scalar api to use scalar as
    // input and move scalar load outside library
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    // brc (1, 1) to (1, b), the count is b(dst->sizes[1]).
    brc_scalar_core_1d<T>(src_ptr[0], dst_ptr, dst->sizes[1]);
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // When broadcast (a, 1) with stride [n, 1] to (a, b) with stride
  // [aligned(b, num_per_block), 1], and b != aligned(b, num_per_block), here
  // n is 1 or aligned to a block, use for + 1d brc to deal.
  if (src->strides[0] != 1 || dst->strides[0] != dst->sizes[1]) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    for (int64_t i = 0; i < dst->sizes[0]; ++i) {
      brc_scalar_core_1d<T>(src->strides[0] != 1 ? src_ptr[i * src->strides[0]]
                                                 : src_ptr[i],
                            dst_ptr + i * dst->strides[0], dst->sizes[1]);
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // When a % 8 == 0 and b == num_per_block, broadcast (a, 1) to (a, b) do not
  // need temp_buf.
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if (dst->sizes[0] % 8 == 0 && dst->sizes[1] == num_per_block) {
    INTRINSIC_NO_ARGS(set_mask_count);
    brc_last_axis_2d_vbrcb<T>(src, dst);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }

  // convert tmp_1d to tmp_2d (type: memref<aligned(a,
  // src_num_per_vbrcb_repeat) x num_per_block xT, stride[num_per_block, 1]>)
  memref_t<__ubuf__ T, 2> tmp_2d;
  convert_1d_to_2d_args<T>(tmp, &tmp_2d,
                           CEIL_FACTOR(dst->sizes[0], kSrcNumPerRepeatOfVBRCB),
                           INTR_BYTES_PER_BLOCK / sizeof(T));

  INTRINSIC_NO_ARGS(set_mask_count);
  // (a, 1) -> (aligned(a, 8), num_per_block)
  brc_last_axis_2d_vbrcb<T>(src, &tmp_2d);
  INTRINSIC(pipe_barrier, PIPE_V);
  // (aligned(a, 8), num_per_block) -> (a, b)
  brc_last_axis_2d_vcopy<T>(&tmp_2d, dst);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// Core func of aligned b64, broadcast src (a, 1) with stride [n, 1] to
/// dst (a, b), here n is 1 or aligned to a block. 'a' may be merged axis of
/// multiple elementwise axes and b is broadcast axis. Here the size of 'b' is
/// aligned to UB block unit.
///
/// \param src (type: memref<a x 1 xT, stride[1, 1]>)
/// \param dst (type: memref<a x b xT, stride[b, 1]>)
///
/// Constraints:
/// 1. 'b' must be ub_block_unit aligned, which is multiple of UB block unit
template <typename T, typename = typename std::enable_if<sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_b64_align_2d(memref_t<__ubuf__ T, 2> *src,
                                        memref_t<__ubuf__ T, 2> *dst) {
  const int64_t src_size0 = src->sizes[0];
  const int64_t dst_size1 = dst->sizes[1];
  auto src_ptr = src->aligned + src->offset;
  for (int i = 0; i < src_size0; i++) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    T scalar = *(src_ptr + i * src->strides[0]);
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    memref_t<__ubuf__ T, 1> dst_1d{(__ubuf__ T *)dst->allocated,
                                   (__ubuf__ T *)dst->aligned,
                                   dst->offset + i * dst->strides[0],
                                   {dst_size1},
                                   {1}};
    broadcast_scalar_b64(scalar, &dst_1d);
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_align_2d(memref_t<__ubuf__ int64_t, 2> *src,
                                    memref_t<__ubuf__ int64_t, 2> *dst,
                                    memref_t<__ubuf__ int64_t, 1> *tmp) {
  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }
  vector_broadcast_last_axis_b64_align_2d<int64_t>(src, dst);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_align_2d(memref_t<__ubuf__ uint64_t, 2> *src,
                                    memref_t<__ubuf__ uint64_t, 2> *dst,
                                    memref_t<__ubuf__ uint64_t, 1> *tmp) {
  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }
  vector_broadcast_last_axis_b64_align_2d<uint64_t>(src, dst);
}

__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_bool_uncontinuous_2d(
    memref_t<__ubuf__ bool, 2> *src, memref_t<__ubuf__ bool, 2> *dst) {
  memref_t<__ubuf__ uint16_t, 2> src_as_uint16;
  memref_t<__ubuf__ uint16_t, 2> dst_as_uint16;
  view_as<bool, uint16_t, 2>(src, &src_as_uint16);
  view_as<bool, uint16_t, 2>(dst, &dst_as_uint16);
  __ubuf__ uint16_t *src_ptr = src_as_uint16.aligned + src_as_uint16.offset;
  __ubuf__ uint16_t *dst_ptr = dst_as_uint16.aligned + dst_as_uint16.offset;

  const int64_t dst_size0 = dst->sizes[0];
  const int64_t dst_size1 = dst->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];

  const int64_t bits_factor = bitwidthOf<uint16_t>() / bitwidthOf<bool>();
  int64_t brc_size_as_uint16 = CEIL_FACTOR(dst_size1, bits_factor);
  int64_t src_stride0_as_uint16 = src_stride0 / bits_factor;
  int64_t dst_stride0_as_uint16 = dst_stride0 / bits_factor;
  for (int i = 0; i < dst_size0; ++i) {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    uint16_t scalar =
        ((uint16_t)(src_ptr[i * src_stride0_as_uint16]) & (uint16_t)1) == 1
            ? MAX_UINT16
            : 0;
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    brc_scalar_core_1d(scalar, dst_ptr + i * dst_stride0_as_uint16,
                       brc_size_as_uint16);
  }
}

// full template specialization for bool.
template <>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_align_2d(memref_t<__ubuf__ bool, 2> *src,
                                    memref_t<__ubuf__ bool, 2> *dst,
                                    memref_t<__ubuf__ bool, 1> *tmp) {
  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }
  const int64_t src_size0 = src->sizes[0];
  const int64_t dst_size1 = dst->sizes[1];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t src_stride0 = src->strides[0];
  const int64_t src_size1 = src->sizes[1];
  if (dst_size1 != dst_stride0 || src_size1 != src_stride0) {
    return vector_broadcast_last_axis_bool_uncontinuous_2d(src, dst);
  }
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(half);
  // tmp buffer size required for half types.
  const int64_t base_tmp_offset =
      CEIL_FACTOR(src_size0, kSrcNumPerRepeatOfVBRCB) * num_per_block;
  constexpr int bits_half = bitwidthOf<half>();

  // Step 1: brc src (a, 1) i1 -> (a, 16) i1 src_after_step1
  // use vsel to implement it by viewing (a, 16) i1 src_after_step1 ->
  // (a, 1) half src_after_step2.
  memref_t<__ubuf__ uint16_t, 1> tmp_zero{(__ubuf__ uint16_t *)tmp->allocated,
                                          (__ubuf__ uint16_t *)tmp->aligned,
                                          tmp->offset / bits_half +
                                              base_tmp_offset,
                                          {num_per_block},
                                          {1}};
  __ubuf__ uint16_t *tmp_zero_ptr = (&tmp_zero)->aligned + (&tmp_zero)->offset;
  brc_scalar_core_1d((uint16_t)0, tmp_zero_ptr, num_per_block);

  memref_t<__ubuf__ uint16_t, 1> tmp_ones{(__ubuf__ uint16_t *)tmp->allocated,
                                          (__ubuf__ uint16_t *)tmp->aligned,
                                          tmp->offset / bits_half +
                                              base_tmp_offset + num_per_block,
                                          {num_per_block},
                                          {1}};
  __ubuf__ uint16_t *tmp_ones_ptr = (&tmp_ones)->aligned + (&tmp_ones)->offset;
  brc_scalar_core_1d(MAX_UINT16, tmp_ones_ptr, num_per_block);

  memref_t<__ubuf__ half, 2> half_dst_tmp{
      (__ubuf__ half *)tmp->allocated,
      (__ubuf__ half *)tmp->aligned,
      tmp->offset / bits_half + base_tmp_offset + num_per_block * 2,
      {src_size0, 1},
      {1, 1}};
  __ubuf__ half *half_dst_tmp_ptr =
      (&half_dst_tmp)->aligned + (&half_dst_tmp)->offset;

  __ubuf__ bool *src_ptr = src->aligned + src->offset;
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0x0, src_size0);
  INTRINSIC(set_cmpmask, (__ubuf__ bool *)src_ptr);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vsel, half_dst_tmp_ptr, (__ubuf__ half *)tmp_ones_ptr,
            (__ubuf__ half *)tmp_zero_ptr,
            1,  // repeat, not work
            1,  // dst block stride
            0,  // stc0 block stride
            0,  // src1 block stride
            8,  // dst repeat stride
            0,  // src0 repeat stride
            0); // src1 repeat stride
  INTRINSIC_NO_ARGS(set_mask_norm);

  // Step 2: brc (a, 1) half src_after_step2 -> (a, b/16) half dst
  // (namely, (a, b) i1 dst).
  memref_t<__ubuf__ half, 2> half_dst;
  view_as<bool, half, 2>(dst, &half_dst);

  memref_t<__ubuf__ half, 1> half_tmp{(__ubuf__ half *)tmp->allocated,
                                      (__ubuf__ half *)tmp->aligned,
                                      tmp->offset / bits_half,
                                      {base_tmp_offset},
                                      {1}};

  INTRINSIC(pipe_barrier, PIPE_V);
  vector_broadcast_last_axis_align_2d(&half_dst_tmp, &half_dst, &half_tmp);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_broadcast_last_axis_unalign_2d(
    memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  auto tmp_ptr = tmp->aligned + tmp->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_ptr) &&
         "The starting address of tmp must be 32byte aligned.");
#endif
}

/// Core func of unaligned, broadcast src (a, 1) to dst (a, b), 'a' may be
/// merged axis of multiple elementwise axes and 'b' is broadcast axis. Here the
/// size of 'b' is unaligned to UB block unit, it need additional tmp buffer
/// with size 'c'.
///
/// \param src (type: memref<a x 1 xT, stride[1, 1]>)
/// \param dst (type: memref<a x b xT, stride[b, 1]>)
/// \param tmp (type: memref<c xT, stride[1]>)
///
/// Constraints:
/// 1. 'b' must be ub_block_unit unaligned, which is multiple of UB block unit
/// 2. The tmp size is as shown below:
///      * 'c' > a * align(b, ub_block_unit), where sizesof(T) = 8.
///      * 'c' > (aligned(a, src_num_per_vbrcb_repeat) * aligned(b,
///           ub_block_unit)), where sizesof(T) != 8.
/// 3. src_num_per_vbrcb_repeat means the number of src that vbrcb intr handle
/// per repeat.
///
/// Example:
/// memref<20x1xf16, strided<[1,1]>> => memref<20x10xf16, strided<[10,1]>>
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_unalign_2d(memref_t<__ubuf__ T, 2> *src,
                                      memref_t<__ubuf__ T, 2> *dst,
                                      memref_t<__ubuf__ T, 1> *tmp) {
  static_assert(
      (std::is_same<half, T>::value || std::is_same<bfloat16_t, T>::value ||
       std::is_same<float, T>::value || std::is_same<int16_t, T>::value ||
       std::is_same<uint16_t, T>::value || std::is_same<int32_t, T>::value ||
       std::is_same<uint32_t, T>::value || std::is_same<int64_t, T>::value ||
       std::is_same<uint64_t, T>::value) &&
      "vector_broadcast_last_axis_unalign do not support this data type");

  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_vector_broadcast_last_axis_unalign_2d(src, dst, tmp);

  // broadcast src:(1, 1) -> dst:(1, b)
  if (src->sizes[0] == 1 && src->strides[0] == 1) [[unlikely]] {
    __ubuf__ T *src_ptr = src->aligned + src->offset;
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    brc_scalar_core_1d<T>(src_ptr[0], dst_ptr, dst->sizes[0] * dst->strides[0]);
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // convert tmp_1d to tmp_2d (type: memref<aligned(a,
  // src_num_per_vbrcb_repeat) x aligned(b, ub_block_unit) xT,
  // stride[aligned(b, ub_block_unit), 1]>)
  memref_t<__ubuf__ T, 2> tmp_2d;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  convert_1d_to_2d_args<T>(tmp, &tmp_2d,
                           CEIL_FACTOR(dst->sizes[0], kSrcNumPerRepeatOfVBRCB),
                           CEIL_FACTOR(dst->sizes[1], num_per_block));

  INTRINSIC_NO_ARGS(set_mask_count);
  // when b <= num_per_block, broadcast src (a, 1) to tmp (a, num_per_block)
  // by vbrcb and then to dst (a, b) by vreducev2.

  // when b > num_per_block, broadcast src (a, 1) to dst (a, num_per_block) by
  // vbrcb and then to tmp (a, aligned(b, ub_block_unit)) by vcopy and then to
  // dst (a, b) by vreducev2.
  bool is_larger_than_block = dst->sizes[1] > num_per_block;
  brc_last_axis_2d_vbrcb<T>(src, is_larger_than_block ? dst : &tmp_2d);

  if (is_larger_than_block) {
    INTRINSIC(pipe_barrier, PIPE_V);
    brc_last_axis_2d_vcopy<T>(dst, &tmp_2d);
  }
  INTRINSIC(pipe_barrier, PIPE_V);
  brc_last_axis_2d_vreducev2<T>(&tmp_2d, dst); // src1 is neglected
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// Core func of unaligned b64, broadcast src (a, 1) to dst (a, b), 'a' may be
/// merged axis of multiple elementwise axes and 'b' is broadcast axis. Here the
/// size of 'b' is unaligned to UB block unit, it need additional tmp buffer
/// with size 'c'.
///
/// \param src (type: memref<a x 1 xT, stride[1, 1]>)
/// \param dst (type: memref<a x b xT, stride[b, 1]>)
/// \param tmp (type: memref<c xT, stride[1]>)
///
/// Constraints:
/// 1. 'b' must be ub_block_unit unaligned, which is multiple of UB block unit
/// 2. tmp_buf size need large than a * align(b, ub_block_unit).
template <typename T, typename = typename std::enable_if<sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_b64_unalign_2d(memref_t<__ubuf__ T, 2> *src,
                                          memref_t<__ubuf__ T, 2> *dst,
                                          memref_t<__ubuf__ T, 1> *tmp) {
  const int64_t src_size0 = src->sizes[0];
  const int64_t dst_size1 = dst->sizes[1];
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  // broadcast src:(1, 1) -> dst:(1, b)
  if (src->sizes[0] == 1 && src->strides[0] == 1) [[unlikely]] {
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    T scalar = *src_ptr;
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    memref_t<__ubuf__ T, 1> dst_1d{(__ubuf__ T *)dst->allocated,
                                   (__ubuf__ T *)dst->aligned,
                                   dst->offset,
                                   {dst_size1},
                                   {1}};
    broadcast_scalar_b64(scalar, &dst_1d);
    return;
  }

  // step1: broadcast src:(a, 1) -> tmp:(a, align(b, ub_block_unit))
  auto dst_size1_align_b32 = CEIL_FACTOR(dst_size1, num_per_block);
  memref_t<__ubuf__ T, 2> tmp_2d{(__ubuf__ T *)tmp->allocated,
                                 (__ubuf__ T *)tmp->aligned,
                                 tmp->offset,
                                 {src_size0, dst_size1_align_b32},
                                 {dst_size1_align_b32, 1}};
  vector_broadcast_last_axis_b64_align_2d(src, &tmp_2d);

  // step2: tmp:(a, align(b, ub_block_unit)) -> dst:(a, b) by vreducev2.
  INTRINSIC_NO_ARGS(set_mask_count);
  memref_t<__ubuf__ uint32_t, 2> dst_2d_b32;
  view_as<T, uint32_t, 2>(dst, &dst_2d_b32);
  memref_t<__ubuf__ uint32_t, 2> tmp_2d_b32;
  view_as<T, uint32_t, 2>(&tmp_2d, &tmp_2d_b32);
  INTRINSIC(pipe_barrier, PIPE_V);
  brc_last_axis_2d_vreducev2<uint32_t>(&tmp_2d_b32, &dst_2d_b32);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_unalign_2d(memref_t<__ubuf__ int64_t, 2> *src,
                                      memref_t<__ubuf__ int64_t, 2> *dst,
                                      memref_t<__ubuf__ int64_t, 1> *tmp) {
  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }
  vector_broadcast_last_axis_b64_unalign_2d<int64_t>(src, dst, tmp);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_unalign_2d(memref_t<__ubuf__ uint64_t, 2> *src,
                                      memref_t<__ubuf__ uint64_t, 2> *dst,
                                      memref_t<__ubuf__ uint64_t, 1> *tmp) {
  if (!is_offset_aligned(src)) {
    vector_broadcast_last_axis_by_scalar(src, dst);
    return;
  }
  vector_broadcast_last_axis_b64_unalign_2d<uint64_t>(src, dst, tmp);
}

/// Core func of unknown aligned, broadcast src (a, 1) to dst (a, b), 'a' may be
/// merged axis of multiple elementwise axes and 'b' is broadcast axis. Here the
/// size of 'b' is not sure to be aligned to UB block unit, it need additional
/// tmp buffer with size 'c'.
///
/// \param src (type: memref<a x 1 xT, stride[1, 1]>)
/// \param dst (type: memref<a x b xT, stride[b, 1]>)
/// \param tmp (type: memref<c xT, stride[1]>)
///
/// Constraints:
/// 1. 'b' must be ub_block_unit unaligned, which is multiple of UB block unit
/// 2. 'c' must be larger than (aligned(a, src_num_per_vbrcb_repeat)*aligned(b,
/// ub_block_unit)), here src_num_per_vbrcb_repeat means the number of src that
/// vbrcb intr handle per repeat. If it is b64 type, b <= ub_block_unit: the
/// size of 'c' must be larger than a * ub_block_unit, and for
/// b > ub_block_unit: the size of 'c' must be larger than
/// a * align(b, ub_block_unit) + a * ub_block_unit.
template <typename T,
          typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 ||
                                             sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
vector_broadcast_last_axis_unknown_align_2d(memref_t<__ubuf__ T, 2> *src,
                                            memref_t<__ubuf__ T, 2> *dst,
                                            memref_t<__ubuf__ T, 1> *tmp) {
  if (is_stride_aligned(dst)) {
    vector_broadcast_last_axis_align_2d<T>(src, dst, tmp);
    return;
  }
  vector_broadcast_last_axis_unalign_2d<T>(src, dst, tmp);
}

extern "C" {
//===-------------------------------------------------------------------===//
// broadcast last axis, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_BROADCAST_LAST_AXIS(bfloat16_t);
REGISTE_BROADCAST_LAST_AXIS(int16_t);
REGISTE_BROADCAST_LAST_AXIS(uint16_t);
REGISTE_BROADCAST_LAST_AXIS(half);
REGISTE_BROADCAST_LAST_AXIS(int32_t);
REGISTE_BROADCAST_LAST_AXIS(uint32_t);
REGISTE_BROADCAST_LAST_AXIS(float);
REGISTE_BROADCAST_LAST_AXIS(int64_t);
REGISTE_BROADCAST_LAST_AXIS(uint64_t);
REGISTE_BROADCAST_LAST_AXIS_ALIGN(bool);

REGISTE_BRC_LAST_AXIS_2D_UNALIGN(half);
REGISTE_BRC_LAST_AXIS_2D_UNALIGN(float);
}
