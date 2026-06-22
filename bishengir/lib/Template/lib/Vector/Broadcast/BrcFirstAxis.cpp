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

/// vcopy src to aligned dst. copy src size1 each repeat. support b16/b32
/// currently.
/// \param dst dst stride0 is aligned(stride0, ub_block_unit)
///
/// Constraints:
/// 1. dst stride0 is aligned(stride0, ub_block_unit).
/// 2. The intrin needs count mode.
template <typename T, typename = typename std::enable_if<sizeof(T) == 2 ||
                                                         sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_2d_vcopy(memref_t<__ubuf__ T, 2> *src,
                        memref_t<__ubuf__ T, 2> *dst) {
  // vcopy for aligned buffer. e.g. memref<1x28xf16, strided<[32, 1]>> ->
  // memref<5x28xf16, strided<[32, 1]>>
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int32_t vcopy_main_cnt = dst->sizes[0] / INTR_MAX_REPEAT_CNTS;

  INTRINSIC(set_vector_mask, 0x0, src->sizes[1]); // count mode
  int64_t dst_offset = 0;
  if (vcopy_main_cnt > 0) {
    for (int i = 0; i < vcopy_main_cnt; ++i) {
      // uint16_t for b16, uint32_t for b32
      if constexpr (sizeof(T) == BYTES_B16) {
        INTRINSIC(
            vcopy, (__ubuf__ uint16_t *)dst_ptr + dst_offset,
            (__ubuf__ uint16_t *)src_ptr,
            INTR_MAX_REPEAT_CNTS,                         // repeat
            1, 1,                                         // blk stride
            dst->strides[0] * sizeof(T) / UB_ALIGN_BYTES, // dst rep stride
            0                                             // src rep stride
        );
      } else {
        INTRINSIC(
            vcopy, (__ubuf__ uint32_t *)dst_ptr + dst_offset,
            (__ubuf__ uint32_t *)src_ptr,
            INTR_MAX_REPEAT_CNTS,                         // repeat
            1, 1,                                         // blk stride
            dst->strides[0] * sizeof(T) / UB_ALIGN_BYTES, // dst rep stride
            0                                             // src rep stride
        );
      }

      dst_offset += INTR_MAX_REPEAT_CNTS * dst->strides[0];
    }
  }

  const int32_t vcopy_tail_cnt = dst->sizes[0] % INTR_MAX_REPEAT_CNTS;
  if (vcopy_tail_cnt > 0) {
    if constexpr (sizeof(T) == BYTES_B16) {
      INTRINSIC(vcopy, (__ubuf__ uint16_t *)dst_ptr + dst_offset,
                (__ubuf__ uint16_t *)src_ptr, vcopy_tail_cnt, 1, 1,
                dst->strides[0] * sizeof(T) / UB_ALIGN_BYTES, 0);
    } else {
      INTRINSIC(vcopy, (__ubuf__ uint32_t *)dst_ptr + dst_offset,
                (__ubuf__ uint32_t *)src_ptr, vcopy_tail_cnt, 1, 1,
                dst->strides[0] * sizeof(T) / UB_ALIGN_BYTES, 0);
    }
  }
}

/// vreducev2 to get unaligned dst, e.g.
/// memref<5x28xf16, strided<[32, 1]>> -> memref<5x28xf16, strided<[28, 1]>>
/// reduce get dst size1 from src stride0.
/// \param src src stride0 is aligned(stride0, ub_block_unit)
///
/// Constraints:
/// 1. src stride0 is aligned(stride0, ub_block_unit).
/// 2. The intrin needs count mode.
template <typename T, typename = typename std::enable_if<sizeof(T) == 2 ||
                                                         sizeof(T) == 4>::type>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_2d_vreducev2(memref_t<__ubuf__ T, 2> *src,
                            memref_t<__ubuf__ T, 2> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_vector_mask, 0x0, dst->sizes[1]); // counter mode

  // uint16_t for b16, uint32_t for b32
  if constexpr (sizeof(T) == BYTES_B16) {
    INTRINSIC(
        vreducev2,
        (__ubuf__ uint16_t *)dst_ptr, // dst
        (__ubuf__ uint16_t *)src_ptr, // src0
        (__ubuf__ uint16_t *)src_ptr, // src1_pattern, not work
        dst->sizes[0],                // repeat
        1,                            // src0 blk stride
        7, // pattern mode, 7 means pattern values are all 1, would not use src1
        src->strides[0] * sizeof(T) / UB_ALIGN_BYTES, // src0 rep stride
        0                                             // src1 rep stride
    );
  } else {
    INTRINSIC(
        vreducev2,
        (__ubuf__ uint32_t *)dst_ptr, // dst
        (__ubuf__ uint32_t *)src_ptr, // src
        (__ubuf__ uint32_t *)src_ptr, // src1_pattern, not work
        dst->sizes[0],                // repeat
        1,                            // src0 blk stride
        7, // pattern mode, 7 means pattern values are all 1, would not use src1
        src->strides[0] * sizeof(T) / UB_ALIGN_BYTES, // src0 rep stride
        0                                             // src1 rep stride
    );
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_brc_first_axis_2d_align_core(memref_t<__ubuf__ T, 2> *src,
                                             memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(dst->strides[0]) &&
         "The dst strides[0] must be aligned to block.");
#endif
}

/// first axis of dst is aligned
/// Constraints:
/// 1. b8/b64 brc first align will step dummy data, size[1] will be aligned to
/// num_pre_block.
template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_2d_align_core(memref_t<__ubuf__ T, 2> *src,
                             memref_t<__ubuf__ T, 2> *dst) {
  static_assert(
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
      "brc_first_axis_2d_align_core do not support this data type");

  if (!is_offset_aligned(src)) {
    vector_broadcast_first_axis_by_scalar(src, dst);
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_brc_first_axis_2d_align_core(src, dst);

  INTRINSIC_NO_ARGS(set_mask_count);
  if constexpr (sizeof(T) == BYTES_B8) {
    // b8 scene
    memref_t<__ubuf__ uint16_t, 2> src_tmp;
    memref_t<__ubuf__ uint16_t, 2> dst_tmp;
    // convert small type to big type. eg. memref<3x9xi8, stride<[32, 1]>> to
    // memref<3x16xi16, stride<[16, 1]>>
    view_as<T, uint16_t, 2>(src, &src_tmp);
    view_as<T, uint16_t, 2>(dst, &dst_tmp);
    brc_first_axis_2d_vcopy<uint16_t>(&src_tmp, &dst_tmp);
  } else if constexpr (sizeof(T) == BYTES_B16 || sizeof(T) == BYTES_B32) {
    brc_first_axis_2d_vcopy(src, dst);
  } else {
    // b64 scene
    memref_t<__ubuf__ uint32_t, 2> src_tmp;
    memref_t<__ubuf__ uint32_t, 2> dst_tmp;
    // convert big type to small type. eg. memref<3x6xi64, stride<[8, 1]>> to
    // memref<3x16xi32, stride<[16, 1]>>
    view_as<T, uint32_t, 2>(src, &src_tmp);
    view_as<T, uint32_t, 2>(dst, &dst_tmp);
    brc_first_axis_2d_vcopy<uint32_t>(&src_tmp, &dst_tmp);
  }
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_brc_first_axis_2d_unalign_core(memref_t<__ubuf__ T, 2> *src,
                                               memref_t<__ubuf__ T, 2> *dst,
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

/// last axis of dst is unaligned
/// \param tmp tmp buf stride0 is aligned(stride0_dst, ub_block_unit). allocated
/// buffer size should be size0*stride0
template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_2d_unalign_core(memref_t<__ubuf__ T, 2> *src,
                               memref_t<__ubuf__ T, 2> *dst,
                               memref_t<__ubuf__ T, 1> *tmp) {
  static_assert(
      (sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
      "brc_first_axis_2d_unalign_core do not support this data type.");

  if (!is_offset_aligned(src)) {
    vector_broadcast_first_axis_by_scalar(src, dst);
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_brc_first_axis_2d_unalign_core(src, dst, tmp);

  if (dst->sizes[1] == 1 && dst->strides[0] == 1) [[unlikely]] {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    if constexpr (sizeof(T) == BYTES_B64) {
      memref_t<__ubuf__ T, 1> dst_1d{
          dst->allocated, dst->aligned, dst->offset, {dst->sizes[0]}, {1}};
      broadcast_scalar_b64(src_ptr[0], &dst_1d);
    } else {
      brc_scalar_core_1d(src_ptr[0], dst_ptr, dst->sizes[0]);
    }
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if constexpr (sizeof(T) == BYTES_B16 || sizeof(T) == BYTES_B32) {
    INTRINSIC_NO_ARGS(set_mask_count);
    // convert tmp_1d to tmp_2d (type: memref<size0_dst x aligned(size1_dst,
    // num_per_block) xT, stride[aligned(size1_dst, num_per_block), 1]>)
    memref_t<__ubuf__ T, 2> tmp_2d;
    convert_1d_to_2d_args<T>(tmp, &tmp_2d, dst->sizes[0],
                             CEIL_FACTOR(dst->sizes[1], num_per_block));
    brc_first_axis_2d_vcopy(src, &tmp_2d);
    INTRINSIC(pipe_barrier, PIPE_V);
    brc_first_axis_2d_vreducev2(&tmp_2d, dst);
    INTRINSIC_NO_ARGS(set_mask_norm);
  } else if constexpr (sizeof(T) == BYTES_B64) {
    memref_t<__ubuf__ T, 2> tmp_2d{
        tmp->allocated,
        tmp->aligned,
        tmp->offset,
        {dst->sizes[0], dst->sizes[1]},
        {CEIL_FACTOR(dst->sizes[1], num_per_block), 1}};
    // step1: (1, b_align) -> (a, b_align), where b_align is b aligned to
    // ub_block_unit.
    brc_first_axis_2d_align_core(src, &tmp_2d);

    // step2: (a, b_align) -> (a, b) , where b_align is b aligned to
    // ub_block_unit.
    INTRINSIC_NO_ARGS(set_mask_count);
    memref_t<__ubuf__ uint32_t, 2> tmp_2d_b32;
    view_as<T, uint32_t, 2>(&tmp_2d, &tmp_2d_b32);
    memref_t<__ubuf__ uint32_t, 2> dst_b32;
    view_as<T, uint32_t, 2>(dst, &dst_b32);
    INTRINSIC(pipe_barrier, PIPE_V);
    brc_first_axis_2d_vreducev2(&tmp_2d_b32, &dst_b32);
    INTRINSIC_NO_ARGS(set_mask_norm);
  }
}

/// core func of unknown aligned, consists of align and unalign branches.
/// aligned: means last dim of src or dst is storage aligned
/// unaligned: means last dim is not aligned
/// note that memref<16x16xf16, strided<16, 1>> is aligned and continuous
/// \param src e.g. memref<1x10xf16>
/// \param tmp tmp buf stride0 is aligned(stride0_dst, ub_block_unit). allocated
/// buffer size should be size0*stride0
template <typename T>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_2d_unknown_align_core(memref_t<__ubuf__ T, 2> *src,
                                     memref_t<__ubuf__ T, 2> *dst,
                                     memref_t<__ubuf__ T, 1> *tmp) {
  static_assert(
      (sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
      "brc_first_axis_2d_unknown_align_core do not support this data type.");
  if (is_stride_aligned(dst)) {
    brc_first_axis_2d_align_core(src, dst);
  } else {
    brc_first_axis_2d_unalign_core(src, dst, tmp);
  }
}

/// convert src (x0, x1, x2) with stride [m0, m1, 1]
/// to dst (x0, y1) with stride [m0, 1], where y1 = x1 * m1
template <typename T>
__aiv__ __attribute__((always_inline)) void
convert_3d_to_2d_args(memref_t<__ubuf__ T, 3> *src_3d,
                      memref_t<__ubuf__ T, 2> *dst_2d) {
  // TODO avoid broadcast with dirty data in 3D scenario
  // src: memref<axbxcxT, strided<[m, n, 1]>>
  // dst: memref<axb*nxT, strided<[m, 1]>>
  *dst_2d = {src_3d->allocated,
             src_3d->aligned,
             src_3d->offset,
             {src_3d->sizes[0], src_3d->sizes[1] * src_3d->strides[1]},
             {src_3d->strides[0], 1}};
}

template <typename T, typename = typename std::enable_if_t<sizeof(T) == 2 ||
                                                           sizeof(T) == 4>>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_3d_vcopy(memref_t<__ubuf__ T, 3> *src,
                        memref_t<__ubuf__ T, 3> *dst) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  // INT32 example
  // src: sizes [1, 5, 16], strides [32 * 5, 32, 1]
  // OOOOOOOOOOOOOOOOXXXXXXXXXXXXXXXX
  // OOOOOOOOOOOOOOOOXXXXXXXXXXXXXXXX
  // OOOOOOOOOOOOOOOOXXXXXXXXXXXXXXXX
  // OOOOOOOOOOOOOOOOXXXXXXXXXXXXXXXX
  // OOOOOOOOOOOOOOOOXXXXXXXXXXXXXXXX
  // dst: sizes [3, 5, 16], strides [16 * 5, 16, 1]
  // OOOOOOOOOOOOOOOO
  // OOOOOOOOOOOOOOOO
  // OOOOOOOOOOOOOOOO
  // OOOOOOOOOOOOOOOO
  // OOOOOOOOOOOOOOOO
  // Goal: we want to copy "O"s from source to destination dst->sizes[0] (i.e. 3) times
  // Each iteration we have to shift dst address forward at dst->strides[1]( i.e 16 * 5 )
  // parameters for vcopy:
  //     repeats                                      : 5
  //     elements, processed per repeat (IN ELEMENTS) : 16
  //     stride between repeats for dst (IN BLOCKS!)  : 2
  //     stride between repeats for src (IN BLOCKS!)  : 4 
  INTRINSIC(set_vector_mask, 0x0, MIN(src->strides[1], dst->strides[1])); // in count mode
  for (int i = 0; i < dst->sizes[0]; ++i) {
    auto offset = dst->strides[0] * i;
    if constexpr (sizeof(T) == BYTES_B16) {
      INTRINSIC(vcopy,
                (__ubuf__ uint16_t *)dst_ptr + offset,
                (__ubuf__ uint16_t *)src_ptr,
                dst->sizes[1], // repeats
                1,             // dst block stride
                1,             // src block stride
                dst->strides[1] / (INTR_BYTES_PER_BLOCK / sizeof(T)), // dst rep stride
                src->strides[1] / (INTR_BYTES_PER_BLOCK / sizeof(T)));  // src rep stride
    } else {
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)dst_ptr + offset,
                (__ubuf__ uint32_t *)src_ptr,
                dst->sizes[1], // repeats
                1,             // dst block stride
                1,             // src block stride
                dst->strides[1] / (INTR_BYTES_PER_BLOCK / sizeof(T)), // dst rep stride
                src->strides[1] / (INTR_BYTES_PER_BLOCK / sizeof(T)));  // src rep stride
    }
  }
}

/// broadcast src (1, a) to dst (b, a), here b is broadcast axis and a may be
/// merged axis of multiple elementwise axes. Here the size of a is aligned to
/// ub_block_unit.
///
/// constraint:
/// 1. dim of src/dst must be 2d or 3d
///
/// 3d reused 2d core function.
/// example1:
/// memref<1x10xf16, strided<[16,1]>> => memref<4x10xf16, strided<[16,1]>>
/// example2:
/// memref<1x5x10xf16, strided<[80,16,1]>> =>
/// memref<4x5x10xf16, strided<[80,16,1]>>
/// converts to memref<1x80xf16, strided<[80,1]>> =>
/// memref<4x80xf16, strided<[80,1]>>
template <typename T, size_t Dim,
          typename = typename std::enable_if<
              std::is_same<T, bool>::value || sizeof(T) == 1 ||
              sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>::type,
          typename = typename std::enable_if<Dim == 2 || Dim == 3>::type>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_align_core(memref_t<__ubuf__ T, Dim> *src,
                          memref_t<__ubuf__ T, Dim> *dst) {
  if constexpr (Dim == 2) {
    brc_first_axis_2d_align_core(src, dst);
  } else {
    // Process 3D case
    if (!is_offset_aligned(src)) {
      // offset is unaligned for 3D case => fallback to scalar copy scenario
      vector_broadcast_first_axis_by_scalar(src, dst);
      return;
    }
    if (src->strides[1] == dst->strides[1]) {
      // if strides are equal for src/dst of [a, b, c] dimensions we can flatten
      // tensor to [a, b * c] and reuse 2D case
      memref_t<__ubuf__ T, 2> src_2d;
      memref_t<__ubuf__ T, 2> dst_2d;
      convert_3d_to_2d_args(src, &src_2d);
      convert_3d_to_2d_args(dst, &dst_2d);
      brc_first_axis_2d_align_core(&src_2d, &dst_2d);
      return;
    }
    // 3D case and strides are not equal
    INTRINSIC_NO_ARGS(set_mask_count);
    if constexpr (sizeof(T) == BYTES_B16 || sizeof(T) == BYTES_B32){
      brc_first_axis_3d_vcopy(src, dst);
    } else {
      // b64 scene
      memref_t<__ubuf__ uint32_t, 3> src_tmp;
      memref_t<__ubuf__ uint32_t, 3> dst_tmp;
      view_as<T, uint32_t, 3>(src, &src_tmp);
      view_as<T, uint32_t, 3>(dst, &dst_tmp);
      brc_first_axis_3d_vcopy<uint32_t>(&src_tmp, &dst_tmp);
    }
    INTRINSIC_NO_ARGS(set_mask_norm);
  }
}

// Partial template specialization for bool
template <size_t Dim,
          typename = typename std::enable_if<Dim == 2 || Dim == 3>::type>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_align_core(memref_t<__ubuf__ bool, Dim> *src,
                          memref_t<__ubuf__ bool, Dim> *dst) {
  memref_t<__ubuf__ half, Dim> half_src;
  memref_t<__ubuf__ half, Dim> half_dst;
  view_as<bool, half, Dim>(src, &half_src);
  view_as<bool, half, Dim>(dst, &half_dst);
  brc_first_axis_align_core(&half_src, &half_dst);
}

/// broadcast src (1, a) to dst (b, a), here b is broadcast axis and a may be
/// merged axis of multiple elementwise axes. Here the size of a is unaligned to
/// ub_block_unit, it need additional tmp buffer with size (b*aligned(a,
/// ub_block_unit)).
///
/// constraint:
/// 1. dim of src/dst must be 2d or 3d
/// 2. tmp buffer size must be larger than dst_size0*aligned(dst_stride0,
/// ub_block_unit), tmp buffer stride0 is aligned(dst_stride0, ub_block_unit)
///
/// 3d reuse 2d core function.
/// example1:
/// memref<1x5x10xf16, strided<[50,10,1]>> =>
/// memref<4x5x10xf16, strided<[50,10,1]>>
/// converts to memref<1x50xf16, strided<[50,1]>> =>
/// memref<4x50xf16, strided<[50,1]>>.
/// example2:
/// memref<1x5x10xf16, strided<[100,20,1]>> =>
/// memref<2x5x10xf16, strided<[100,20,1]>>
/// converts to memref<1x50xf16, strided<[100,1]>> =>
/// memref<2x50xf16, strided<[100,1]>>.
/// \param tmp tmp buf stride0 is aligned(stride0_dst, ub_block_unit). allocated
/// buffer size should be size0*stride0
template <typename T, size_t Dim,
          typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 ||
                                             sizeof(T) == 8>::type,
          typename = typename std::enable_if<Dim == 2 || Dim == 3>::type>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_unalign_core(memref_t<__ubuf__ T, Dim> *src,
                            memref_t<__ubuf__ T, Dim> *dst,
                            memref_t<__ubuf__ T, 1> *tmp) {
  if constexpr (Dim == 2) {
    brc_first_axis_2d_unalign_core(src, dst, tmp);
  } else {
    // reuse 2d core
    memref_t<__ubuf__ T, 2> src_2d;
    memref_t<__ubuf__ T, 2> dst_2d;
    convert_3d_to_2d_args(src, &src_2d);
    convert_3d_to_2d_args(dst, &dst_2d);
    brc_first_axis_2d_unalign_core(&src_2d, &dst_2d, tmp);
  }
}

/// broadcast src (1, a) to dst (b, a), here b is broadcast axis and a may be
/// merged axis of multiple elementwise axes. Here the size of a is not sure to
/// be aligned, it need additional tmp buffer with size (b*aligned(a,
/// ub_block_unit)).
//
/// constraint:
/// 1. dim of src/dst must be 2d or 3d
/// 2. tmp buffer size must be larger than dst_size0*aligned(dst_stride0,
/// ub_block_unit), tmp buffer stride0 is aligned(dst_stride0, ub_block_unit)
///
/// \param tmp tmp buf stride0 is aligned(stride0_dst, ub_block_unit). allocated
/// buffer size should be size0*stride0
template <typename T, size_t Dim,
          typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 ||
                                             sizeof(T) == 8>::type,
          typename = typename std::enable_if<Dim == 2 || Dim == 3>::type>
__aiv__ __attribute__((always_inline)) void
brc_first_axis_unknown_align_core(memref_t<__ubuf__ T, Dim> *src,
                                  memref_t<__ubuf__ T, Dim> *dst,
                                  memref_t<__ubuf__ T, 1> *tmp) {
  // aligned to aligned, use src and dst.
  // unaligned to unaligned, use src, tmp and dst.
  if constexpr (Dim == 2) {
    brc_first_axis_2d_unknown_align_core(src, dst, tmp);
  } else {
    // reuse 2d core
    memref_t<__ubuf__ T, 2> src_2d;
    memref_t<__ubuf__ T, 2> dst_2d;
    convert_3d_to_2d_args(src, &src_2d);
    convert_3d_to_2d_args(dst, &dst_2d);
    brc_first_axis_2d_unknown_align_core(&src_2d, &dst_2d, tmp);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// Registe brc_first_axis_2d_align_core func
//===-------------------------------------------------------------------===//
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(bfloat16_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(int16_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(uint16_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(int32_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(uint32_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(half);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(float);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(int8_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(uint8_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(int64_t);
REGISTE_BRC_FIRST_AXIS_2D_ALIGN(uint64_t);

//===-------------------------------------------------------------------===//
// brc first axis, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_BRC_FIRST_AXIS(2, bfloat16_t)
REGISTE_BRC_FIRST_AXIS(2, int16_t)
REGISTE_BRC_FIRST_AXIS(2, uint16_t)
REGISTE_BRC_FIRST_AXIS(2, int32_t)
REGISTE_BRC_FIRST_AXIS(2, uint32_t)
REGISTE_BRC_FIRST_AXIS(2, half)
REGISTE_BRC_FIRST_AXIS(2, float)
REGISTE_BRC_FIRST_AXIS(2, int64_t)
REGISTE_BRC_FIRST_AXIS(2, uint64_t)
REGISTE_BRC_FIRST_AXIS_ALIGN(2, bool)
REGISTE_BRC_FIRST_AXIS_ALIGN(2, int8_t)
REGISTE_BRC_FIRST_AXIS_ALIGN(2, uint8_t)

//===-------------------------------------------------------------------===//
// brc first axis, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_BRC_FIRST_AXIS(3, bfloat16_t)
REGISTE_BRC_FIRST_AXIS(3, int16_t)
REGISTE_BRC_FIRST_AXIS(3, uint16_t)
REGISTE_BRC_FIRST_AXIS(3, int32_t)
REGISTE_BRC_FIRST_AXIS(3, uint32_t)
REGISTE_BRC_FIRST_AXIS(3, half)
REGISTE_BRC_FIRST_AXIS(3, float)
REGISTE_BRC_FIRST_AXIS(3, int64_t)
REGISTE_BRC_FIRST_AXIS(3, uint64_t)
REGISTE_BRC_FIRST_AXIS_ALIGN(3, bool)
REGISTE_BRC_FIRST_AXIS_ALIGN(3, int8_t);
REGISTE_BRC_FIRST_AXIS_ALIGN(3, uint8_t);
}
