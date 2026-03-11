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
#include "Vector/Broadcast/BrcUtils.h"

__aiv__ __attribute__((always_inline)) void
set_store_atomic_none(AtomicKind atomic_kind) {
  if (atomic_kind != AtomicKind::None) {
    INTRINSIC_NO_ARGS(set_atomic_none);
    INTRINSIC(pipe_barrier, PIPE_MTE3);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
check_1d_ubuf_stride_align(memref_t<__ubuf__ T, 1> *ub) {
  const int64_t stride0_ub = ub->strides[0];
  return (isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1);
}

/// Rewrite padding for dst correctly in the case of b64 type by applying the
/// padding in intervals with b32 scalars
///
/// Constraints:
/// 1. dim of dst must be 1, dst must be continuous vector
/// 2. dst and pad_value are expected to be int64_t
template <typename T>
__aiv__ __attribute__((always_inline)) void
align_pad_for_load_b64_1d(memref_t<__ubuf__ T, 1> *dst, int64_t pad_value,
                          int64_t left_padding_num) {
  const int64_t factor = sizeof(T) / sizeof(uint32_t);
  int64_t size = dst->sizes[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t size_aligned = CEIL_FACTOR(size, num_per_block);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  // if size and stride is too high, we cant fit the mask so we shift
  int64_t padding_shift_count = size_aligned / num_per_repeat;
  if (size_aligned % num_per_repeat == 0) {
    padding_shift_count = padding_shift_count - 1;
  }
  apply_padding_b64<T, 1>(dst, padding_shift_count * num_per_repeat * factor,
                          pad_value, /*repeat=*/1, padding_shift_count);
}

// core func of load gm <-> ub, 1d
template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_1d_core(memref_t<__gm__ T, 1> *src,
                        memref_t<__ubuf__ T, 1> *dst, PadMode pad_mode,
                        T pad_value, int64_t left_padding_num,
                        AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  // "aicore arch: Ascend910B2"
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((T *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
    pad_value = 0;
  }

  if (!check_1d_ubuf_stride_align(dst)) {
    load_gm_to_ubuf_1d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  // the starting address of dst is not 32byte aligned
  auto dst_ptr = dst->aligned + dst->offset - left_padding_num;
  if (!isAddress32ByteAligned(dst_ptr)) {
    // TODO: use scalar to load first block and use dma to load others
    load_gm_to_ubuf_1d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  const int64_t stride0_gm = src->strides[0];
  const int64_t stride0_ub = dst->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  if (left_padding_num > num_per_block) {
    int padding_remainder = left_padding_num % num_per_block;
    int toBroadcast = left_padding_num - padding_remainder;
    left_padding_num = padding_remainder;
    if (pad_mode == PadMode::Value) {
      __ubuf__ T *dst_ptr = dst->aligned;
      // TODO check whether we need to use barrier here
      brc_scalar_core_1d<T>(pad_value, dst_ptr, toBroadcast);
    }
    dst->offset = dst->offset - padding_remainder;
  }

  if (stride0_gm == 1 && stride0_ub == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_1d_core_with_contiguous_last_dim<T>(src, dst,
                                                        left_padding_num);
    if (sizeof(T) == 8 && pad_mode == PadMode::Value) {
      INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
#ifdef ENABLE_CPU_TRACE_INTRINSIC
      int64_t scalar;
      if constexpr (std::is_same<T, half>::value ||
                    std::is_same<T, bfloat16_t>::value) {
        scalar = static_cast<int64_t>(pad_value.num);
      } else {
        scalar = static_cast<int64_t>(pad_value);
      }
#else
      int64_t scalar = static_cast<int64_t>(pad_value);
#endif
      align_pad_for_load_b64_1d<T>(dst, scalar, left_padding_num);
    }
    return;
  }

  // last dimension of UB is contiguous but GM is not
  // so use scalar
  if ((stride0_ub == 1) && (stride0_gm != 1)) {
    load_gm_to_ubuf_1d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size) with stride [stride] as viewed_src (size, 1) with
  // stride [stride, 1], where last dimension of viewed_src is contiguous
  const int64_t size0 = src->sizes[0];
  memref_t<__gm__ T, 2> gm_2d = {
      src->allocated, src->aligned, src->offset, {size0, 1}, {stride0_gm, 1}};
  memref_t<__ubuf__ T, 2> ub_2d = {
      dst->allocated, dst->aligned, dst->offset, {size0, 1}, {stride0_ub, 1}};

  load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(&gm_2d, &ub_2d,
                                                      left_padding_num);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_1d_core(memref_t<__ubuf__ T, 1> *src,
                         memref_t<__gm__ T, 1> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  if (!check_1d_ubuf_stride_align(src)) {
    if (atomic_kind != AtomicKind::None) {
      cce::printf("Atomic operations are not supported when the stride is not "
                  "block‑aligned");
      return;
    }
    store_ubuf_to_gm_1d_by_scalar<T>(src, dst);
    return;
  }

  // "aicore arch: Ascend910B2"
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((T *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  // arg for atomic op
  if (atomic_kind != AtomicKind::None) {
    INTRINSIC(pipe_barrier, PIPE_MTE3);
    set_atomic_kind<T>(atomic_kind);
  }

  // the starting address of src is not 32byte aligned
  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    if (atomic_kind != AtomicKind::None) {
      cce::printf("Atomic operations are not supported when the ub address is "
                  "not block‑aligned");
      return;
    }
    int64_t size_backup = src->sizes[0];
    auto data_to_copy = store_ubuf_to_gm_initial_misalignment_1d<T>(src, dst);
    if (data_to_copy == size_backup)
      return;
  }

  const int64_t stride0_ub = src->strides[0];
  const int64_t stride0_gm = dst->strides[0];
  if (stride0_gm == 1 && stride0_ub == 1) [[likely]] {
    // last dimension is contiguous
    store_ubuf_to_gm_1d_core_with_contiguous_last_dim<T>(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size) with stride [stride] as viewed_src (size, 1) with
  // stride [stride, 1], where last dimension of viewed_src is contiguous
  const int64_t size0 = src->sizes[0];
  memref_t<__ubuf__ T, 2> ub_2d = {
      src->allocated, src->aligned, src->offset, {size0, 1}, {stride0_ub, 1}};
  memref_t<__gm__ T, 2> gm_2d = {
      dst->allocated, dst->aligned, dst->offset, {size0, 1}, {stride0_gm, 1}};

  store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);

  set_store_atomic_none(atomic_kind);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_copy_ubuf_to_ubuf_1d_core(memref_t<__ubuf__ T, 1> *src,
                                          memref_t<__ubuf__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride0_ub_src = src->strides[0];
  const int64_t stride0_ub_dst = dst->strides[0];
  assert((stride0_ub_src == 1) && "Last dimension of src must be contiguous.");
  assert((stride0_ub_dst == 1) && "Last dimension of dst must be contiguous.");
#endif
}

/// core func of copy ub <-> ub, 1d
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_1d_core(memref_t<__ubuf__ T, 1> *src,
                          memref_t<__ubuf__ T, 1> *dst) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_ubuf_1d_core(src, dst);
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t size0 = src->sizes[0];
  bool is_last_dim_contiguous = (src_stride0 == 1 && dst_stride0 == 1);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  bool is_offset_aligned =
      isAddress32ByteAligned(dst_ptr) && isAddress32ByteAligned(src_ptr);

  // For src and dst which have contiguous last dim and aligned offset, copy by
  // vector
  if (is_last_dim_contiguous && is_offset_aligned) [[likely]] {
    copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(src, dst);
    return;
  }

  // For src and dst which have contiguous last dim but have unaligned offset,
  // try to copy part of src to dst by scalar
  if (is_last_dim_contiguous && !is_offset_aligned) {
    // If src and dst can move offset to aligned with copying some elements by
    // scalar, Use scalar to copy part of src to dst, and others are copied by
    // vector
    if (has_same_alignment_offset_step<T, 1>(src, dst)) {
      int64_t step_to_aligned =
          element_nums_to_move_offset_aligned<T>(src->offset);
      copy_ubuf_to_ubuf_1d_core_by_scalar(src, dst, step_to_aligned);
      // Return if no needs to do copy by vector
      if (size0 <= step_to_aligned) {
        return;
      }

      memref_t<__ubuf__ T, 1> new_src = {src->allocated,
                                         src->aligned,
                                         src->offset + step_to_aligned,
                                         {size0 - step_to_aligned},
                                         {src_stride0}};

      memref_t<__ubuf__ T, 1> new_dst = {dst->allocated,
                                         dst->aligned,
                                         dst->offset + step_to_aligned,
                                         {size0 - step_to_aligned},
                                         {dst_stride0}};

      copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(&new_src, &new_dst);
      return;
    }
  }

  // Else situation, just copy by scalar to ensure accuracy
  copy_ubuf_to_ubuf_1d_core_by_scalar(src, dst, src->sizes[0]);
}

template <>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_1d_core(memref_t<__ubuf__ bool, 1> *src,
                          memref_t<__ubuf__ bool, 1> *dst) {
  // convert bool memref to int8 memref
  memref_t<__ubuf__ int8_t, 1> src_as_int8;
  memref_t<__ubuf__ int8_t, 1> dst_as_int8;
  view_as<bool, int8_t, 1>(src, &src_as_int8);
  view_as<bool, int8_t, 1>(dst, &dst_as_int8);

  copy_ubuf_to_ubuf_1d_core<int8_t>(&src_as_int8, &dst_as_int8);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Load gm to ub, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_LOAD(1, int8_t);
REGISTE_DMA_LOAD(1, uint8_t);
REGISTE_DMA_LOAD(1, int16_t);
REGISTE_DMA_LOAD(1, uint16_t);
REGISTE_DMA_LOAD(1, int32_t);
REGISTE_DMA_LOAD(1, uint32_t);
REGISTE_DMA_LOAD(1, int64_t);
REGISTE_DMA_LOAD(1, uint64_t);
REGISTE_DMA_LOAD(1, half);
REGISTE_DMA_LOAD(1, float);
REGISTE_DMA_LOAD(1, bfloat16_t);

//===-------------------------------------------------------------------===//
// Store ub to gm, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_STORE(1, int8_t);
REGISTE_DMA_STORE(1, uint8_t);
REGISTE_DMA_STORE(1, int16_t);
REGISTE_DMA_STORE(1, uint16_t);
REGISTE_DMA_STORE(1, int32_t);
REGISTE_DMA_STORE(1, uint32_t);
REGISTE_DMA_STORE(1, int64_t);
REGISTE_DMA_STORE(1, uint64_t);
REGISTE_DMA_STORE(1, half);
REGISTE_DMA_STORE(1, float);
REGISTE_DMA_STORE(1, bfloat16_t);

//===-------------------------------------------------------------------===//
// ub to ub, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, bool)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, int8_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, uint8_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, int16_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, uint16_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, int32_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, uint32_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, int64_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, uint64_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, half)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, float)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, bfloat16_t)

//===-------------------------------------------------------------------===//
// ub to ub 1d func
//===-------------------------------------------------------------------===//
REGISTE_COPY_UB_TO_UB_1D_FUNC(int8_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(uint8_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(int16_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(uint16_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(int32_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(uint32_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(int64_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(uint64_t);
REGISTE_COPY_UB_TO_UB_1D_FUNC(half);
REGISTE_COPY_UB_TO_UB_1D_FUNC(float);
REGISTE_COPY_UB_TO_UB_1D_FUNC(bfloat16_t);
}
