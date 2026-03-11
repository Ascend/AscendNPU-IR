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
  return (((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
           (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1) &&
           (isSizeAlignedToBlock<T>(stride2_ub) || stride2_ub == 1)));
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

  // the starting address of dst is not 32byte aligned
  auto dst_ptr = dst->aligned + dst->offset - left_padding_num;
  if (!isAddress32ByteAligned(dst_ptr)) {
    // TODO: use scalar to load first block and use dma to load others
    load_gm_to_ubuf_3d_by_scalar<T>(src, dst, left_padding_num, pad_value);
    return;
  }

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
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1, size2) with stride [stride0, stride1, stride2]
  // as viewed_src (size0, size1, size2, 1) with stride [stride0, stride1,
  // stride2, 1], where last dimension of viewed_src is contiguous

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
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim(&gm_3d, &ub_3d,
                                                     left_padding_num);
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
    if (atomic_kind != AtomicKind::None) {
      cce::printf("Atomic operations are not supported when the stride is not "
                  "block‑aligned");
      return;
    }
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
  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    if (atomic_kind != AtomicKind::None) {
      cce::printf("Atomic operations are not supported when the ub address is "
                  "not block‑aligned");
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

  // last dimension is not contiguous,
  // view the src (size0, size1, size2) with stride [stride0, stride1, stride2]
  // as viewed_src (size0, size1, size2, 1) with stride [stride0, stride1,
  // stride2, 1], where last dimension of viewed_src is contiguous

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  int64_t stride0_ub = src->strides[0];
  int64_t stride1_ub = src->strides[1];
  int64_t stride2_ub = src->strides[2];
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
  bool is_contiguous = (src_stride0 == 1 && dst_stride0 == 1);
  bool is_offset_aligned =
      isAddress32ByteAligned(dst_ptr) && isAddress32ByteAligned(src_ptr);

  // Copy using 1d contiguous method if the last 2 dim tensor is contiguous in
  // memory
  if (is_offset_aligned && is32ByteAligned<T>(src_stride0) &&
      is32ByteAligned<T>(dst_stride0) &&
      is_unalign_collapsible_dims(src, dst)) {
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