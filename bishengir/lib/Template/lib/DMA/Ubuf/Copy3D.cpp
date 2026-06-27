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

template <typename T>
__aiv__ __attribute__((always_inline)) bool
check_3d_ubuf_stride_align(memref_t<__ubuf__ T, 3> *ub) {
  auto stride0_ub = ub->strides[0];
  auto stride1_ub = ub->strides[1];
  auto stride2_ub = ub->strides[2];
  auto size0_ub = ub->sizes[0];
  auto size1_ub = ub->sizes[1];
  auto size2_ub = ub->sizes[2];
  return (((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1 || size0_ub == 1) &&
           (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1 || size1_ub == 1) &&
           (isSizeAlignedToBlock<T>(stride2_ub) || stride2_ub == 1 || size2_ub == 1)));
}

// Constraints: padding_num should be contained in memref.sizes.
template <typename T>
__aiv__ __attribute__((always_inline)) void
padding_value_3d_by_scalar_on_load(T padding_value, memref_t<__ubuf__ T, 3> *dst) {
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_S, LIB_EVENT_ID0);
  auto padding_start = dst->aligned + dst->offset;
  for (int i = 0; i < dst->sizes[0]; ++i) {
    for (int j = 0; j < dst->sizes[1]; ++j) {
      for (int k = 0; k < dst->sizes[2]; ++k) {
        *(padding_start + i * dst->strides[0] + j * dst->strides[1] +
          k * dst->strides[2]) = padding_value;
      }
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
}

// Constraints: padding_num should be contained in memref.sizes. 
template <typename T>
__aiv__ __attribute__((always_inline)) void
padding_value_3d(T padding_value, memref_t<__ubuf__ T, 3> *dst) {
  // we have to do that since brc not support non-contiguous
  if (dst->strides[2] != 1) [[unlikely]] {
    padding_value_3d_by_scalar_on_load<T>(padding_value, dst);
    return;
  }
  for (int64_t i = 0; i < dst->sizes[0]; i++) {
    memref_t<__ubuf__ T, 2> dst_padding_memref = {
        dst->allocated,
        dst->aligned,
        dst->offset + dst->strides[0] * i,
        {dst->sizes[1], dst->sizes[2]},
        {dst->strides[1], dst->strides[2]}
      };
    if constexpr (sizeof(T) == 1) {
        brc_scalar_2d_by_scalar(padding_value, &dst_padding_memref);
    } else {
        broadcast_scalar<T, 2>(padding_value, &dst_padding_memref);
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
load_gm_to_ubuf_3d_core(memref_t<__gm__ T, 3> *src,
                        memref_t<__ubuf__ T, 3> *dst, PadMode pad_mode,
                        T pad_value, int64_t left_padding_num,
                        AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((T *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
    pad_value = 0;
  }

  if (!check_3d_ubuf_stride_align(dst)) {
    load_gm_to_ubuf_3d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  auto dst_ptr = dst->aligned + dst->offset - left_padding_num * dst->strides[2];
  if (!isAddress32ByteAligned(dst_ptr) && dst->strides[2] != 1) {
    load_gm_to_ubuf_3d_by_scalar<T>(src, dst, left_padding_num, pad_value);	 
    return;
  }

  memref_t<__gm__ T, 3> src_memref_aligned = copy_memref<__gm__ T, 3>(src);
  memref_t<__ubuf__ T, 3> dst_memref_aligned = copy_memref<__ubuf__ T, 3>(dst);

  // the starting address of dst is not 32byte aligned
  if (!isAddress32ByteAligned(dst_ptr)) {
    // Use scalar to load first block and use dma to load others
    int64_t distance = ((UB_ALIGN_BYTES - (reinterpret_cast<uintptr_t>(dst_ptr) & 0x1F)) % UB_ALIGN_BYTES) / sizeof(T);
    memref_t<__gm__ T, 3> src_memref_unaligned_head = copy_memref<__gm__ T, 3>(src);
    memref_t<__ubuf__ T, 3> dst_memref_unaligned_head = copy_memref<__ubuf__ T, 3>(dst);

    if (distance <= left_padding_num) {
      dst_memref_unaligned_head.sizes[2] = distance;
      dst_memref_unaligned_head.offset -= left_padding_num;
      padding_value_3d<T>(pad_value, &dst_memref_unaligned_head);

      // step 2-1: update left_padding_num only.
      left_padding_num -= distance;
    } else {
      int64_t load_num = distance - left_padding_num;
      load_num = (load_num > src->sizes[2]) ? src->sizes[2] : load_num;

      src_memref_unaligned_head.sizes[2] = load_num;
      dst_memref_unaligned_head.sizes[2] = load_num;
      load_gm_to_ubuf_3d_by_scalar<T>(&src_memref_unaligned_head, &dst_memref_unaligned_head, left_padding_num, pad_value);

      // step 2-2: update left_padding_num and src/dst memref.
      src_memref_aligned.offset += load_num;
      src_memref_aligned.sizes[2] -= load_num;

      dst_memref_aligned.offset += load_num;
      dst_memref_aligned.sizes[2] -= load_num;
      left_padding_num = 0;

      if (dst_memref_aligned.sizes[2] <= 0) {
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
      memref_t<__ubuf__ T, 3> dst_padding_memref = {
        dst->allocated,
        dst->aligned,
        dst->offset - left_padding_num * dst->strides[2],
        {dst->sizes[0], dst->sizes[1], left_padding_num_main},
        {dst->strides[0], dst->strides[1], dst->strides[2]}
      };
      INTRINSIC(pipe_barrier, PIPE_V);
      padding_value_3d<T>(pad_value, &dst_padding_memref);

      int64_t span_2 = (dst->sizes[2] + left_padding_num - 1) * dst->strides[2] + 1;
      int64_t span_1 = (dst->sizes[1] - 1) * dst->strides[1] +
                       (dst->sizes[2] + left_padding_num - 1) * dst->strides[2] + 1;
      if (
        (dst->strides[0] < span_1) ||
        (dst->strides[1] < span_2)
      ) {
        // inject sync if padding and load overlap. 
        if (dst->strides[2] == 1) {
          INTRINSIC(set_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
          INTRINSIC(wait_flag, PIPE_V, PIPE_MTE2, LIB_EVENT_ID0);
        } else {
          INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
          INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
        }
      }
  }
  
  // step 2: clamp left_padding_num under 32B:
  left_padding_num = left_padding_num_tail;

  // deal copy memref<1x1x1> to memref<1x1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    load_gm_to_ubuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T),
                                left_padding_num, 0, 0);
    return;
  }

  if (src->strides[2] == 1 && dst->strides[2] == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim(src, dst,
                                                     left_padding_num);

    if (sizeof(T) == 8 && pad_mode == PadMode::Value) {
      INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
      memref_t<__ubuf__ T, 3> dst_padding_b64_memref_lp = {
        dst->allocated,
        dst->aligned,
        dst->offset - left_padding_num, // findal stride always 1
        {dst->sizes[0], dst->sizes[1], left_padding_num},
        {dst->strides[0], dst->strides[1], dst->strides[2]}
      };
      padding_value_3d<T>(pad_value, &dst_padding_b64_memref_lp);

      auto load_end_ptr = (dst->aligned + dst->offset + dst->sizes[2]);
      int64_t right_padding_num = ((UB_ALIGN_BYTES - (reinterpret_cast<uintptr_t>(load_end_ptr) & 0x1F)) % UB_ALIGN_BYTES) / sizeof(T);
      // repadding value for B64. 
      memref_t<__ubuf__ T, 3> dst_padding_b64_memref_rp = {
        dst->allocated,
        dst->aligned,
        dst->offset + dst->sizes[2], // final stride always 1
        {dst->sizes[0], dst->sizes[1], right_padding_num},
        {dst->strides[0], dst->strides[1], dst->strides[2]}
      };
      padding_value_3d<T>(pad_value, &dst_padding_b64_memref_rp);
    }
    return;
  }

  if (src->strides[2] > 1 && dst->strides[2] == 1) {
    // When GM's last dim stride > 1 but UB's last dim is contiguous,
    // we cannot use the dimension-lifting trick because the gap calculation
    // in the underlying intrinsic functions assumes contiguous last dim.
    // Using scalar copy to avoid incorrect gap values.
    load_gm_to_ubuf_3d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1, size2) with stride [stride0, stride1, stride2]
  // as viewed_src (size0, size1, size2, 1) with stride [stride0, stride1,
  // stride2, 1], where last dimension of viewed_src is contiguous
  // we need deal with padding outside.
  if (left_padding_num > 0) {
    // do padding via scalar since we do not have brc_3d.
    memref_t<__ubuf__ T, 3> dst_memref_non_contiguous_padding = copy_memref<__ubuf__ T, 3>(dst);
    dst_memref_non_contiguous_padding.offset -= left_padding_num * dst->strides[2];
    dst_memref_non_contiguous_padding.sizes[2] = left_padding_num;
    padding_value_3d_by_scalar_on_load<T>(pad_value, &dst_memref_non_contiguous_padding);
    INTRINSIC(set_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_MTE2, LIB_EVENT_ID0);
  }

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  int64_t stride0_ub = dst->strides[0];
  int64_t stride1_ub = dst->strides[1];
  int64_t stride2_ub = dst->strides[2];
  int64_t stride0_gm = src->strides[0];
  int64_t stride1_gm = src->strides[1];
  int64_t stride2_gm = src->strides[2];

  // choose minimum dimension as dim0 and set the other two dimensions as dim1
  // and dim2
  int64_t min_axis = 0;
  int64_t min_size = size0;
  if (min_size > size1) {
    min_size = size1;
    min_axis = 1;
  }
  if (min_size > size2) {
    min_size = size2;
    min_axis = 2;
  }

  if (min_axis == 1) {
    size0 = src->sizes[1];
    size1 = src->sizes[0];
    stride0_ub = dst->strides[1];
    stride1_ub = dst->strides[0];
    stride0_gm = src->strides[1];
    stride1_gm = src->strides[0];
  } else if (min_axis == 2) {
    size0 = src->sizes[2];
    size1 = src->sizes[0];
    size2 = src->sizes[1];
    stride0_ub = dst->strides[2];
    stride1_ub = dst->strides[0];
    stride2_ub = dst->strides[1];
    stride0_gm = src->strides[2];
    stride1_gm = src->strides[0];
    stride2_gm = src->strides[1];
  }

  // throw dim0 as loop and map (dim1, dim2, 1) to the new 3d pattern to
  // guarantee the last axis is contiguous.
  for (int64_t i = 0; i < size0; i++) {
    int64_t offset_loop_gm = stride0_gm * i;
    int64_t offset_loop_ub = stride0_ub * i;

    memref_t<__gm__ T, 3> gm_3d = {src->allocated,
                                   src->aligned,
                                   src->offset + offset_loop_gm,
                                   {size1, size2, 1},
                                   {stride1_gm, stride2_gm, 1}};
    memref_t<__ubuf__ T, 3> ub_3d = {dst->allocated,
                                     dst->aligned,
                                     dst->offset + offset_loop_ub,
                                     {size1, size2, 1},
                                     {stride1_ub, stride2_ub, 1}};
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim(&gm_3d, &ub_3d, 0);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_3d_core(memref_t<__ubuf__ T, 3> *src,
                         memref_t<__gm__ T, 3> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  if (!check_3d_ubuf_stride_align(src)) {
    if (!check_atomic_none(atomic_kind))
      return;
    store_ubuf_to_gm_3d_by_scalar<T>(src, dst);
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
  int64_t stride0_ub = src->strides[0];
  int64_t stride1_ub = src->strides[1];
  int64_t stride2_ub = src->strides[2];
  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    if (!check_atomic_none(atomic_kind))
      return;
    if (stride2_ub != 1) {
      store_ubuf_to_gm_3d_by_scalar<T>(src, dst);
      return;
    }
    auto size_backup = src->sizes[2];
    auto data_to_copy =
        store_ubuf_to_gm_initial_misalignment_nd<T, 3>(src, dst);
    if (data_to_copy == size_backup)
      return;
  }

  // deal copy memref<1x1x1> to memref<1x1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    store_ubuf_to_gm_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T), 0,
                                 0);
    set_store_atomic_none(atomic_kind);
    return;
  }

  if (dst->strides[2] == 1 && src->strides[2] == 1) [[likely]] {
    // last dimension is contiguous
    store_ubuf_to_gm_3d_core_with_contiguous_last_dim(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }

  // Check whether we need fall back to scalar when stride2_gm>1 and stride2_ub=1

  // last dimension is not contiguous,
  // view the src (size0, size1, size2) with stride [stride0, stride1, stride2]
  // as viewed_src (size0, size1, size2, 1) with stride [stride0, stride1,
  // stride2, 1], where last dimension of viewed_src is contiguous

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  int64_t stride0_gm = dst->strides[0];
  int64_t stride1_gm = dst->strides[1];
  int64_t stride2_gm = dst->strides[2];

  // choose minimum dimension as dim0 and set the other two dimensions as dim1
  // and dim2
  int64_t min_axis = 0;
  int64_t min_size = size0;
  if (min_size > size1) {
    min_size = size1;
    min_axis = 1;
  }
  if (min_size > size2) {
    min_size = size2;
    min_axis = 2;
  }

  if (min_axis == 1) {
    size0 = src->sizes[1];
    size1 = src->sizes[0];
    stride0_ub = src->strides[1];
    stride1_ub = src->strides[0];
    stride0_gm = dst->strides[1];
    stride1_gm = dst->strides[0];
  } else if (min_axis == 2) {
    size0 = src->sizes[2];
    size1 = src->sizes[0];
    size2 = src->sizes[1];
    stride0_ub = src->strides[2];
    stride1_ub = src->strides[0];
    stride2_ub = src->strides[1];
    stride0_gm = dst->strides[2];
    stride1_gm = dst->strides[0];
    stride2_gm = dst->strides[1];
  }

  // throw dim0 as loop and map (dim1, dim2, 1) to the new 3d pattern to
  // guarantee the last axis is contiguous.
  for (int64_t i = 0; i < size0; i++) {
    int64_t offset_loop_gm = stride0_gm * i;
    int64_t offset_loop_ub = stride0_ub * i;

    memref_t<__gm__ T, 3> gm_3d = {dst->allocated,
                                   dst->aligned,
                                   dst->offset + offset_loop_gm,
                                   {size1, size2, 1},
                                   {stride1_gm, stride2_gm, 1}};
    memref_t<__ubuf__ T, 3> ub_3d = {src->allocated,
                                     src->aligned,
                                     src->offset + offset_loop_ub,
                                     {size1, size2, 1},
                                     {stride1_ub, stride2_ub, 1}};
    store_ubuf_to_gm_3d_core_with_contiguous_last_dim(&ub_3d, &gm_3d);
  }

  set_store_atomic_none(atomic_kind);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_copy_ubuf_to_ubuf_3d_core(memref_t<__ubuf__ T, 3> *src,
                                          memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride2_src = src->strides[2];
  const int64_t stride2_dst = dst->strides[2];
  assert((stride2_src == 1) && "Last dimension of src must be contiguous.");
  assert((stride2_dst == 1) && "Last dimension of dst must be contiguous.");
#endif
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_with_last_two_dims_collapsed(memref_t<__ubuf__ T, 3> *src,
                                  memref_t<__ubuf__ T, 3> *dst) {
  int64_t src_size0 = src->sizes[0];
  int64_t src_size1 = src->sizes[1];
  int64_t src_size2 = src->sizes[2];
  int64_t src_stride0 = src->strides[0];
  int64_t src_stride1 = src->strides[1];
  int64_t src_stride2 = src->strides[2];
  int64_t dst_stride0 = dst->strides[0];
  int64_t dst_stride1 = dst->strides[1];
  int64_t dst_stride2 = dst->strides[2];
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert(isSizeAlignedToBlock<T>(dst_stride0) &&
         "The dst strides[0] must be aligned to block for "
         "copy_with_last_two_dims_collapsed.");
#endif

  for (int i = 0; i < src_size0; ++i) {
    memref_t<__ubuf__ T, 2> src_2d{src->allocated,
                                   src->aligned,
                                   src->offset + i * src_stride0,
                                   {src_size1, src_size2},
                                   {src_stride1, src_stride2}};
    memref_t<__ubuf__ T, 2> dst_2d{dst->allocated,
                                   dst->aligned,
                                   dst->offset + i * dst_stride0,
                                   {src_size1, src_size2},
                                   {dst_stride1, dst_stride2}};
    vreducev2_2d_with_pattern_mode<T, PatternMode::ALL_ELEMENTS>(&src_2d,
                                                                 &dst_2d);
  }
}

/// core func of copy ub <-> ub, 3d
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_3d_core(memref_t<__ubuf__ T, 3> *src,
                          memref_t<__ubuf__ T, 3> *dst) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_ubuf_3d_core(src, dst);

  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    // deal copy memref<1x1x1> to memref<1x1x1>
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {1}, {1}};
    memref_t<__ubuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {1}, {1}};
    copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
    return;
  }

  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t src_stride2 = src->strides[2];
  const int64_t dst_stride2 = dst->strides[2];
  const int size0 = src->sizes[0];
  const int size1 = src->sizes[1];
  const int size2 = src->sizes[2];
  bool is_stride_aligned =
      are_outer_strides_32bytes_aligned<T, 3>(src->strides) &&
      are_outer_strides_32bytes_aligned<T, 3>(dst->strides);
  bool is_contiguous = (src_stride2 == 1 && dst_stride2 == 1);
  bool is_offset_aligned =
      isAddress32ByteAligned(dst_ptr) && isAddress32ByteAligned(src_ptr);

  // Copy using 1d contiguous method if the last 2 dim tensor is contiguous in
  // memory
  if (is_offset_aligned && is32ByteAligned<T>(src_stride0) &&
      is32ByteAligned<T>(dst_stride0) &&
      is_all_continuous(src, dst)) {
    for (int i = 0; i < size0; ++i) {
      memref_t<__ubuf__ T, 1> src_1d{src->allocated,
                                     src->aligned,
                                     src->offset + i * src_stride0,
                                     {size1 * size2},
                                     {1}};

      memref_t<__ubuf__ T, 1> dst_1d{dst->allocated,
                                     dst->aligned,
                                     dst->offset + i * dst_stride0,
                                     {size1 * size2},
                                     {1}};

      copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(&src_1d, &dst_1d);
    }
    return;
  }

  /// For 3D copy scenarios where last 2 dims of dst is a contiguous memory region,
  /// and for src, all strides except the last dimension are 32B-aligned with offset
  /// alignment, vreduceV2 can be used to avoid scalar copy performance degradation.
  /// e.g. src sizes(8, 128, 4) strides(1024, 8, 1) dst sizes(8, 128, 4) strides(512, 4, 1)
  /// select PatternMode::ALL_ELEMENTS to copy all of elements of src
  if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    if (is_offset_aligned && is32ByteAligned<T>(src_stride0) &&
      is32ByteAligned<T>(dst_stride0) &&
      is_unalign_collapsible_dims<T, 3>(src, dst)) {
      copy_with_last_two_dims_collapsed<T>(src, dst);
      return;
    }
  }

  // For src and dst which have aligned high dim stride,
  // contiguous last dim and aligned offset, copy by vector
  if (is_stride_aligned && is_contiguous && is_offset_aligned) {
    copy_ubuf_to_ubuf_3d_core_with_contiguous_last_dim<T>(src, dst);
    return;
  }

  // For src and dst which have aligned high dim stride,
  // contiguous last dim but have unaligned offset,
  // try to copy part of src to dst by scalar
  if (is_stride_aligned && is_contiguous && !is_offset_aligned) {
    // If src and dst can move offset to aligned with copying some elements by
    // scalar, Use scalar to copy part of src to dst, and others are copied by
    // vector
    if (has_same_alignment_offset_step<T, 3>(src, dst)) {
      int64_t step_to_aligned =
          element_nums_to_move_offset_aligned<T>(src->offset);
      copy_ubuf_to_ubuf_3d_core_by_scalar(src, dst, step_to_aligned);
      // Return if no needs to do copy by vector
      if (size2 <= step_to_aligned) {
        return;
      }

      memref_t<__ubuf__ T, 3> new_src = {
          src->allocated,
          src->aligned,
          src->offset + step_to_aligned,
          {size0, size1, size2 - step_to_aligned},
          {src_stride0, src_stride1, src_stride2}};

      memref_t<__ubuf__ T, 3> new_dst = {
          dst->allocated,
          dst->aligned,
          dst->offset + step_to_aligned,
          {size0, size1, size2 - step_to_aligned},
          {dst_stride0, dst_stride1, dst_stride2}};
      copy_ubuf_to_ubuf_3d_core_with_contiguous_last_dim<T>(&new_src, &new_dst);

      return;
    }
  }

  // Else situation, just copy by scalar to ensure accuracy
  copy_ubuf_to_ubuf_3d_core_by_scalar(src, dst, size2);
}

template <>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_3d_core(memref_t<__ubuf__ bool, 3> *src,
                          memref_t<__ubuf__ bool, 3> *dst) {
  // convert bool memref to int8 memref
  memref_t<__ubuf__ int8_t, 3> src_as_int8;
  memref_t<__ubuf__ int8_t, 3> dst_as_int8;
  view_as<bool, int8_t, 3>(src, &src_as_int8);
  view_as<bool, int8_t, 3>(dst, &dst_as_int8);

  copy_ubuf_to_ubuf_3d_core<int8_t>(&src_as_int8, &dst_as_int8);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Load gm to ub, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_LOAD(3, int8_t);
REGISTE_DMA_LOAD(3, uint8_t);
REGISTE_DMA_LOAD(3, int16_t);
REGISTE_DMA_LOAD(3, uint16_t);
REGISTE_DMA_LOAD(3, int32_t);
REGISTE_DMA_LOAD(3, uint32_t);
REGISTE_DMA_LOAD(3, int64_t);
REGISTE_DMA_LOAD(3, uint64_t);
REGISTE_DMA_LOAD(3, half);
REGISTE_DMA_LOAD(3, float);
REGISTE_DMA_LOAD(3, bfloat16_t);

//===-------------------------------------------------------------------===//
// Store ub to gm, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_STORE(3, int8_t);
REGISTE_DMA_STORE(3, uint8_t);
REGISTE_DMA_STORE(3, int16_t);
REGISTE_DMA_STORE(3, uint16_t);
REGISTE_DMA_STORE(3, int32_t);
REGISTE_DMA_STORE(3, uint32_t);
REGISTE_DMA_STORE(3, int64_t);
REGISTE_DMA_STORE(3, uint64_t);
REGISTE_DMA_STORE(3, half);
REGISTE_DMA_STORE(3, float);
REGISTE_DMA_STORE(3, bfloat16_t);

//===-------------------------------------------------------------------===//
// ub to ub, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, bool)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, int8_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, uint8_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, int16_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, uint16_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, int32_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, uint32_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, int64_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, uint64_t)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, half)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, float)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, bfloat16_t)
}
