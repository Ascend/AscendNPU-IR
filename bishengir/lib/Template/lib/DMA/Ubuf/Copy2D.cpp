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

template <typename T>
__aiv__ __attribute__((always_inline)) bool
check_2d_ubuf_stride_align(memref_t<__ubuf__ T, 2> *ub, int64_t left_padding_num = 0) {
  auto stride0_ub = ub->strides[0];
  auto stride1_ub = ub->strides[1];
  auto size0_ub = ub->sizes[0];
  auto size1_ub = ub->sizes[1] + left_padding_num;
  return (((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1 || size0_ub == 1) &&
           (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1 || size1_ub == 1)));
}

// Constraints: padding_num should be contained in memref.sizes and memref.offset.
template <typename T>
__aiv__ __attribute__((always_inline)) void
padding_value_2d_by_scalar_on_load(T padding_value, memref_t<__ubuf__ T, 2> *dst) {
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  auto padding_start = dst->aligned + dst->offset;
  for (int i = 0; i < dst->sizes[0]; ++i) {
    for (int j = 0; j < dst->sizes[1]; ++j) {
      *(padding_start + i * dst->strides[0] + j * dst->strides[1]) =
          padding_value;
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
}

// Constraints: padding_num should be contained in memref.sizes. 
template <typename T>
__aiv__ __attribute__((always_inline)) void
padding_value_2d_via_brc(T padding_value, memref_t<__ubuf__ T, 2> *dst) {
  if constexpr (sizeof(T) == 1) {
    brc_scalar_2d_by_scalar(padding_value, dst);
  } else {
    broadcast_scalar<T, 2>(padding_value, dst);
  }
}


/// Rewrite padding for dst correctly in the case of b64 type by applying the
/// padding in intervals with b32 scalars
///
/// Constraints:
/// 1. dim of dst must be 2, stride_0 dst must be 1
/// 2. dst and pad_value are expected to be int64_t
template <typename T>
__aiv__ __attribute__((always_inline)) void
align_pad_for_load_b64_2d(memref_t<__ubuf__ T, 2> *dst, int64_t pad_value,
                          int64_t left_padding_num) {
  const int64_t factor = sizeof(T) / sizeof(uint32_t);
  int64_t size = dst->sizes[1];
  int64_t stride_dst = dst->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t size_aligned = CEIL_FACTOR(size, num_per_block);
  // if size and stride is too high, we cant fit the mask so we shift
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  int64_t padding_shift_count = size_aligned / num_per_repeat;
  if (size_aligned % num_per_repeat == 0) {
    padding_shift_count = padding_shift_count - 1;
  }
  // Assume align_pad < num_per_repeat (Maximum mask size)
  int64_t repeat_times = dst->sizes[0];
  int64_t loop_num = repeat_times / INTR_MAX_REPEAT_CNTS;
  for (int64_t i = 0; i < loop_num; ++i) {
    apply_padding_b64<T, 2>(dst,
                            (i * stride_dst * INTR_MAX_REPEAT_CNTS +
                             padding_shift_count * num_per_repeat) *
                                factor,
                            pad_value, INTR_MAX_REPEAT_CNTS,
                            padding_shift_count);
  }
  if (repeat_times % INTR_MAX_REPEAT_CNTS != 0) {
    int64_t remain_repeat_times = repeat_times % INTR_MAX_REPEAT_CNTS;
    apply_padding_b64<T, 2>(dst,
                            (loop_num * stride_dst * INTR_MAX_REPEAT_CNTS +
                             padding_shift_count * num_per_repeat) *
                                factor,
                            pad_value, remain_repeat_times,
                            padding_shift_count);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_2d_core(memref_t<__gm__ T, 2> *src,
                        memref_t<__ubuf__ T, 2> *dst, PadMode pad_mode,
                        T pad_value, int64_t left_padding_num,
                        AtomicKind atomic_kind = AtomicKind::None) {
  /*
    We have following assumptions:
    1. dst->offset = <origi_offset> + left_padding_num
    2. allways use dst->aligned as UB addr, which has 32byte aligned.
  */

  if (is_no_op<2>(src->sizes)) {
    return;
  }

  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((T *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
    pad_value = 0;
  }

  if (!check_2d_ubuf_stride_align(dst, left_padding_num)) {
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  auto dst_ptr = dst->aligned + dst->offset - left_padding_num * dst->strides[1];
  if (!isAddress32ByteAligned(dst_ptr) && dst->strides[1] != 1) {
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst, left_padding_num, pad_value);	 
    return;
  }

  memref_t<__gm__ T, 2> src_memref_aligned = copy_memref<__gm__ T, 2>(src);
  memref_t<__ubuf__ T, 2> dst_memref_aligned = copy_memref<__ubuf__ T, 2>(dst);

  // the starting address of dst is not 32byte aligned
  if (!isAddress32ByteAligned(dst_ptr)) {
    // use scalar to load first block and use dma to load others
    // step 1: fill first unaligned block
    int64_t distance = ((UB_ALIGN_BYTES - (reinterpret_cast<uintptr_t>(dst_ptr) & 0x1F)) % UB_ALIGN_BYTES) / sizeof(T);
    memref_t<__gm__ T, 2> src_memref_unaligned_head = copy_memref<__gm__ T, 2>(src);
    memref_t<__ubuf__ T, 2> dst_memref_unaligned_head = copy_memref<__ubuf__ T, 2>(dst);

    if (distance <= left_padding_num) {
      dst_memref_unaligned_head.sizes[1] = distance;
      dst_memref_unaligned_head.offset -= left_padding_num;
      padding_value_2d_via_brc<T>(pad_value, &dst_memref_unaligned_head);

      // step 2-1: update left_padding_num only.
      left_padding_num -= distance;
    } else {
      int64_t load_num = distance - left_padding_num;
      load_num = (load_num > src->sizes[1]) ? src->sizes[1] : load_num;
      src_memref_unaligned_head.sizes[1] = load_num;
      dst_memref_unaligned_head.sizes[1] = load_num;
      load_gm_to_ubuf_2d_by_scalar<T>(&src_memref_unaligned_head, &dst_memref_unaligned_head, left_padding_num, pad_value);

      // step 2-2: update left_padding_num and src/dst memref.
      src_memref_aligned.offset += load_num;
      src_memref_aligned.sizes[1] -= load_num;

      dst_memref_aligned.offset += load_num;
      dst_memref_aligned.sizes[1] -= load_num;
      left_padding_num = 0;

      if (dst_memref_aligned.sizes[1] <= 0) {
        return;
      }
    }
  }

  // step 3: update memref and continue load data to aligned dst. 
  dst = &dst_memref_aligned;
  src = &src_memref_aligned;

  // do padding for pad more than 32B.
  // step 0: find main block
  int64_t left_padding_num_tail = left_padding_num % (UB_ALIGN_BYTES / sizeof(T));
  int64_t left_padding_num_main = left_padding_num - left_padding_num_tail;
  // step 1: brc paddinng
  if (left_padding_num_main > 0) {
    // reshape dst memref for padding. 
    memref_t<__ubuf__ T, 2> dst_padding_memref = {
      dst->allocated,
      dst->aligned,
      dst->offset - left_padding_num * dst->strides[1], 
      {dst->sizes[0], left_padding_num_main},
      {dst->strides[0], dst->strides[1]}
    };
    if (dst->strides[1] == 1) [[likely]] {
      INTRINSIC(pipe_barrier, PIPE_V);
      padding_value_2d_via_brc<T>(pad_value, &dst_padding_memref);
      
    } else {
      padding_value_2d_by_scalar_on_load<T>(pad_value, &dst_padding_memref);
    }

    int64_t span_1 = (dst->sizes[1] + left_padding_num - 1) * dst->strides[1] + 1;
    if (dst->strides[0] < span_1) {
      // inject sync if padding and load overlap. 
      if (dst->strides[1] == 1) {
        INTRINSIC(set_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
      }
    }
  }
  // step 2: clamp left_padding_num under 32B:
  left_padding_num = left_padding_num_tail;

  // deal copy memref<1x1> to memref<1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    load_gm_to_ubuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T),
                                left_padding_num, 0, 0);
    return;
  }

  int64_t stride1_gm = src->strides[1];
  int64_t stride1_ub = dst->strides[1];
  if (stride1_gm == 1 && stride1_ub == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(src, dst,
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
      //   Use brc padding for last dimension of dst. 
      memref_t<__ubuf__ T, 2> dst_padding_b64_memref_lp = {
        dst->allocated,
        dst->aligned,
        dst->offset - left_padding_num, // findal stride always 1
        {dst->sizes[0], left_padding_num},
        {dst->strides[0], stride1_ub}
      };
      padding_value_2d_via_brc<T>(pad_value, &dst_padding_b64_memref_lp); 

      auto load_end_ptr = (dst->aligned + dst->offset + dst->sizes[1]);
      int64_t right_padding_num = ((UB_ALIGN_BYTES - (reinterpret_cast<uintptr_t>(load_end_ptr) & 0x1F)) % UB_ALIGN_BYTES) / sizeof(T);
      //   Use brc padding for last dimension of dst. 
      memref_t<__ubuf__ T, 2> dst_padding_b64_memref_rp = {
        dst->allocated,
        dst->aligned,
        dst->offset + dst->sizes[1], // final stride always 1
        {dst->sizes[0], right_padding_num},
        {dst->strides[0], stride1_ub}
      };
      padding_value_2d_via_brc<T>(pad_value, &dst_padding_b64_memref_rp);
    }
    return;
  }

  int64_t stride0_gm = src->strides[0];
  int64_t stride0_ub = dst->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if ((stride0_gm < stride1_gm || stride0_ub < stride1_ub) &&
      (stride0_ub % num_per_block != 0 || stride1_ub % num_per_block != 0)) {
    // Implicit transposition scenarios need to be moved through scalar
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  if (stride1_gm > 1 && stride1_ub == 1) {
    // When GM's last dim stride > 1 but UB's last dim is contiguous,
    // we cannot use the dimension-lifting trick because the gap calculation
    // in the underlying intrinsic functions assumes contiguous last dim.
    // Using scalar copy to avoid incorrect gap values.
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1) with stride [stride0, stride1] as viewed_src
  // (size0, size1, 1) with stride [stride0, stride1, 1], where last dimension
  // of viewed_src is contiguous
  // we need deal with padding outside.
  if (left_padding_num > 0) {
    memref_t<__ubuf__ T, 2> dst_memref_non_contiguous_padding = copy_memref<__ubuf__ T, 2>(dst);
    dst_memref_non_contiguous_padding.offset -= left_padding_num * stride1_ub;
    dst_memref_non_contiguous_padding.sizes[1] = left_padding_num;
    padding_value_2d_by_scalar_on_load<T>(pad_value, &dst_memref_non_contiguous_padding);
  }

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  memref_t<__gm__ T, 3> gm_3d = {src->allocated,
                                 src->aligned,
                                 src->offset,
                                 {size0, size1, 1},
                                 {stride0_gm, stride1_gm, 1}};
  memref_t<__ubuf__ T, 3> ub_3d = {dst->allocated,
                                   dst->aligned,
                                   dst->offset,
                                   {size0, size1, 1},
                                   {stride0_ub, stride1_ub, 1}};
  load_gm_to_ubuf_3d_core_with_contiguous_last_dim<T>(&gm_3d, &ub_3d, 0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_2d_core(memref_t<__ubuf__ T, 2> *src,
                         memref_t<__gm__ T, 2> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }


  if (!check_2d_ubuf_stride_align(src)) {
    if (!check_atomic_none(atomic_kind))
      return;
    if ((src->sizes[1] == src->strides[0] && dst->sizes[1] == dst->strides[0] &&
         src->strides[1] == 1 && dst->strides[1] == 1)) {
      store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(src, dst);
      return;
    }
    if (src->sizes[0] == 1 && dst->sizes[0] == 1) {
      memref_t<__ubuf__ T, 1> src_slice{src->allocated,
                                        src->aligned,
                                        src->offset,
                                        {src->sizes[1]},
                                        {1}};
      memref_t<__gm__ T, 1> dst_slice{dst->allocated,
                                      dst->aligned,
                                      dst->offset,
                                      {dst->sizes[1]},
                                      {1}};
      store_ubuf_to_gm_1d_core_with_contiguous_last_dim<T>(&src_slice,
                                                           &dst_slice);
      return;
    }
    store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
    return;
  }

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
  const int64_t stride0_ub = src->strides[0];
  const int64_t stride1_ub = src->strides[1];
  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    if (!check_atomic_none(atomic_kind))
      return;
    if (stride1_ub != 1) {
      store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
      return;
    }
    auto size_backup = src->sizes[1];
    auto data_to_copy =
        store_ubuf_to_gm_initial_misalignment_nd<T, 2>(src, dst);
    if (data_to_copy == size_backup)
      return;
  }

  // deal copy memref<1x1> to memref<1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    store_ubuf_to_gm_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T), 0,
                                 0);
    set_store_atomic_none(atomic_kind);
    return;
  }

  const int64_t stride0_gm = dst->strides[0];
  const int64_t stride1_gm = dst->strides[1];
  if (stride1_gm == 1 && stride1_ub == 1) [[likely]] {
    // last dimension is contiguous
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if ((stride0_ub < stride1_ub || stride0_gm < stride1_gm) &&
      (stride0_ub % num_per_block != 0 || stride1_ub % num_per_block != 0)) {
    // Implicit transposition scenarios need to be moved through scalar
    store_ubuf_to_gm_2d_by_scalar(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }

  // Check whether we need fall back to scalar when stride1_gm>1 and stride1_ub=1

  // last dimension is not contiguous,
  // view the src (size0, size1) with stride [stride0, stride1] as viewed_src
  // (size0, size1, 1) with stride [stride0, stride1, 1], where last dimension
  // of viewed_src is contiguous
  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  memref_t<__gm__ T, 3> gm_3d = {dst->allocated,
                                 dst->aligned,
                                 dst->offset,
                                 {size0, size1, 1},
                                 {stride0_gm, stride1_gm, 1}};
  memref_t<__ubuf__ T, 3> ub_3d = {src->allocated,
                                   src->aligned,
                                   src->offset,
                                   {size0, size1, 1},
                                   {stride0_ub, stride1_ub, 1}};
  store_ubuf_to_gm_3d_core_with_contiguous_last_dim<T>(&ub_3d, &gm_3d);

  set_store_atomic_none(atomic_kind);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_copy_ubuf_to_ubuf_2d_core(memref_t<__ubuf__ T, 2> *src,
                                          memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride1_src = src->strides[1];
  const int64_t stride1_dst = dst->strides[1];
  assert((stride1_src == 1) && "Last dimension of src must be contiguous.");
  assert((stride1_dst == 1) && "Last dimension of dst must be contiguous.");
#endif
}

/// core func of copy ub <-> ub, 2d
/// constraints:
/// 1. src/dst stride1 must be 1
/// 2. src stride0 must be aligned to ub_block_unit
/// 3. dst stride0 must be aligned to ub_block_unit or equal to dst_size[1]
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_core(memref_t<__ubuf__ T, 2> *src,
                          memref_t<__ubuf__ T, 2> *dst) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }
  check_inputs_of_copy_ubuf_to_ubuf_2d_core(src, dst);

  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    // deal copy memref<1x1> to memref<1x1>
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {1}, {1}};
    memref_t<__ubuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {1}, {1}};
    copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
    return;
  }

  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  bool is_last_dim_contiguous = (src_stride1 == 1 && dst_stride1 == 1);
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  bool is_offset_aligned =
      isAddress32ByteAligned(dst_ptr) && isAddress32ByteAligned(src_ptr);
  bool is_stride_aligned =
      is32ByteAligned<T>(src_stride0) && is32ByteAligned<T>(dst_stride0);

  // Copy using 1d contiguous method since the 2d tensor is contiguous in memory
  if (is_offset_aligned && is_all_continuous(src, dst)) {
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {size0 * size1}, {1}};

    memref_t<__ubuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {size0 * size1}, {1}};

    copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(&src_1d, &dst_1d);
    return;
  }

  /// For 2D copy scenarios where dst is a contiguous memory region,
  /// and for src, all strides except the last dimension are 32B-aligned
  /// with offset alignment, vreduceV2 can be used to avoid scalar copy
  /// performance degradation
  /// e.g. src sizes(128, 4) strides(8, 1) dst sizes(128, 4) strides(4, 1)
  /// select PatternMode::ALL_ELEMENTS to copy all of elements of src
  if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    if (is_offset_aligned && is_unalign_collapsible_dims<T, 2>(src, dst)) {
      vreducev2_2d_with_pattern_mode<T, PatternMode::ALL_ELEMENTS>(src, dst);
      return;
    }
  }

  // For src and dst which have aligned high dim stride,
  // contiguous last dim and aligned offset, copy by vector
  if (is_stride_aligned && is_last_dim_contiguous && is_offset_aligned)
      [[likely]] {
    copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(src, dst);
    return;
  }

  // For src and dst which have aligned high dim stride,
  // contiguous last dim but have unaligned offset,
  // try to copy part of src to dst by scalar
  if (is_stride_aligned && is_last_dim_contiguous && !is_offset_aligned) {
    // If src and dst can move offset to aligned with copying some elements by
    // scalar, Use scalar to copy part of src to dst, and others are copied by
    // vector
    if (has_same_alignment_offset_step<T, 2>(src, dst)) {
      int64_t step_to_aligned =
          element_nums_to_move_offset_aligned<T>(src->offset);
      copy_ubuf_to_ubuf_2d_core_by_scalar(src, dst, step_to_aligned);
      // Return if no needs to do copy by vector
      if (size1 <= step_to_aligned) {
        return;
      }

      memref_t<__ubuf__ T, 2> new_src = {src->allocated,
                                         src->aligned,
                                         src->offset + step_to_aligned,
                                         {size0, size1 - step_to_aligned},
                                         {src_stride0, src_stride1}};

      memref_t<__ubuf__ T, 2> new_dst = {dst->allocated,
                                         dst->aligned,
                                         dst->offset + step_to_aligned,
                                         {size0, size1 - step_to_aligned},
                                         {dst_stride0, dst_stride1}};

      copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&new_src, &new_dst);
      return;
    }
  }

  // Else situation, just copy by scalar to ensure accuracy
  copy_ubuf_to_ubuf_2d_core_by_scalar(src, dst, src->sizes[1]);
}

template <>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_core(memref_t<__ubuf__ bool, 2> *src,
                          memref_t<__ubuf__ bool, 2> *dst) {
  // convert bool memref to int8 memref
  memref_t<__ubuf__ int8_t, 2> src_as_int8;
  memref_t<__ubuf__ int8_t, 2> dst_as_int8;
  view_as<bool, int8_t, 2>(src, &src_as_int8);
  view_as<bool, int8_t, 2>(dst, &dst_as_int8);

  copy_ubuf_to_ubuf_2d_core<int8_t>(&src_as_int8, &dst_as_int8);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Load gm to ub, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_LOAD(2, int8_t);
REGISTE_DMA_LOAD(2, uint8_t);
REGISTE_DMA_LOAD(2, int16_t);
REGISTE_DMA_LOAD(2, uint16_t);
REGISTE_DMA_LOAD(2, int32_t);
REGISTE_DMA_LOAD(2, uint32_t);
REGISTE_DMA_LOAD(2, int64_t);
REGISTE_DMA_LOAD(2, uint64_t);
REGISTE_DMA_LOAD(2, half);
REGISTE_DMA_LOAD(2, float);
REGISTE_DMA_LOAD(2, bfloat16_t);

//===-------------------------------------------------------------------===//
// Store ub to gm, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_STORE(2, int8_t);
REGISTE_DMA_STORE(2, uint8_t);
REGISTE_DMA_STORE(2, int16_t);
REGISTE_DMA_STORE(2, uint16_t);
REGISTE_DMA_STORE(2, int32_t);
REGISTE_DMA_STORE(2, uint32_t);
REGISTE_DMA_STORE(2, int64_t);
REGISTE_DMA_STORE(2, uint64_t);
REGISTE_DMA_STORE(2, half);
REGISTE_DMA_STORE(2, float);
REGISTE_DMA_STORE(2, bfloat16_t);

//===-------------------------------------------------------------------===//
// Copy ub to ub, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, bool)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, int8_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, uint8_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, int16_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, uint16_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, int32_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, uint32_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, int64_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, uint64_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, half)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, float)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, bfloat16_t)

//===-------------------------------------------------------------------===//
// ub to ub 2d func
//===-------------------------------------------------------------------===//
REGISTE_COPY_UB_TO_UB_2D_FUNC(int8_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(uint8_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(int16_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(uint16_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(int32_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(uint32_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(int64_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(uint64_t);
REGISTE_COPY_UB_TO_UB_2D_FUNC(half);
REGISTE_COPY_UB_TO_UB_2D_FUNC(float);
REGISTE_COPY_UB_TO_UB_2D_FUNC(bfloat16_t);
}
