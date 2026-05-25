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

#include "RegBase/DMAUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_load_gm_to_ubuf_3d_core(memref_t<__gm__ T, 3> *src,
                                        memref_t<__ubuf__ T, 3> *dst,
                                        int64_t left_padding_num) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset - left_padding_num;
  auto stride0_ub = dst->strides[0];
  auto stride1_ub = dst->strides[1];
  auto stride2_ub = dst->strides[2];
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
          (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1) &&
          (isSizeAlignedToBlock<T>(stride2_ub) || stride2_ub == 1)) &&
         "The dst strides[0]/strides[1]/strides[2] must be 1 or aligned to"
         "block.");
#endif
}

#if defined(__DAV_C310__)
/// Core func of loading GM -> UB, 3D
/// UB starting address must be 32B aligned; otherwise, it degrades to looped
/// scalar transfer.
/// + `UB&GM stride[2] == 1`: last dimension contiguous
///   - `load_gm_to_ubuf_3d_core_with_contiguous_last_dim`
/// + `UB|GM stride[2] != 1`: last dimension non-contiguous
///   + `UB&GM stride[1] == stride[2] * size[2]`: dimension 1 contiguous
///     + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 contiguous
///       - Convert to 2D: `offset2D = offset3D, size2D = [size3D[0] * size3D[1]
///       * size3D[2], 1], stride2D = [stride3D[2], 1]`
///         `load_gm_to_ubuf_2d_core_with_contiguous_last_dim`
///     + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 non-contiguous
///       - Convert to 3D: `newOffset3D = offset3D, newSize3D = [size3D[0],
///       size3D[1] * size3D[2], 1], newStride3D = [stride3D[0], stride3D[2],
///       1]`
///         `load_gm_to_ubuf_3d_core_with_contiguous_last_dim`, where in the new
///         3D tensor both dimension 0 and 1 are non-contiguous
///   + `UB|GM stride[1] != stride[2] * size[2]`: dimension 1 non-contiguous
///     + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 contiguous
///       - Convert to 3D: `newOffset3D = offset3D, newSize3D = [size3D[0] *
///       size3D[1], size3D[2], 1], newStride3D = [stride3D[1], stride3D[2], 1]`
///         `load_gm_to_ubuf_3d_core_with_contiguous_last_dim`, where in the new
///         3D tensor both dimension 0 and 1 are non-contiguous
///     + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 non-contiguous
///       fall back to looped scalar transfer. Because non-contiguous last
///       dimension restricts burst to transfer only one element, and both
///       higher dimensions being non-contiguous prevents axis merging. One
///       would need to hoist dimension 0 or 1 as an outer for-loop, and also
///       check 32B alignment of these axes. This essentially becomes looped
///       scalar transfer. Although in some corner cases with two nested
///       for-loops where both higher-dimension strides are multiples of 32B,
///       DMA transferring multiple bursts per instruction might yield slightly
///       better performance than pure scalar transfer, this corner case will
///       not be optimized for now.
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_3d_core(
    memref_t<__gm__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy, AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  using PadValueT = typename PadValueType<T>::type;
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((PadValueT *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_ubuf_3d_core(src, dst, left_padding_num);

  auto dst_ptr = dst->aligned + dst->offset;
  if (!isAddress32ByteAligned(dst_ptr)) {
    load_gm_to_ubuf_3d_by_scalar<T>(src, dst);
    return;
  }

  uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);
  const int64_t stride2_gm = src->strides[2];
  const int64_t stride2_ub = dst->strides[2];
  if (stride2_gm == 1 && stride2_ub == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim<T>(
        src, dst, left_padding_num, l2_cache_ctl);
    return;
  }
  // last dimension is not contiguous
  const int64_t stride0_ub = dst->strides[0];
  const int64_t stride1_ub = dst->strides[1];
  const int64_t stride0_gm = src->strides[0];
  const int64_t stride1_gm = src->strides[1];
  if (((stride0_gm < stride1_gm || stride1_gm < stride2_gm) ||
       (stride0_ub < stride1_ub || stride1_ub < stride2_ub))) {
    // Implicit transposition scenarios
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      load_gm_to_ubuf_3d_by_scalar<T>(src, dst);
    } else {
      load_gm_to_ubuf_3d_by_nddma<T>(src, dst);
    }
    return;
  }
  // axis 1 is contiguous
  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  const int64_t size2 = dst->sizes[2];
  if (stride1_ub == stride2_ub * size2 && stride1_gm == stride2_gm * size2) {
    // axis 0 is contiguous
    if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
      memref_t<__gm__ T, 2> gm_2d{src->allocated,
                                  src->aligned,
                                  src->offset,
                                  {size0 * size1 * size2, 1},
                                  {stride2_gm, 1}};
      memref_t<__ubuf__ T, 2> ub_2d{dst->allocated,
                                    dst->aligned,
                                    dst->offset,
                                    {size0 * size1 * size2, 1},
                                    {stride2_ub, 1}};
      load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
          &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
      return;
    }
    // axis 0 is non-contiguous
    memref_t<__gm__ T, 3> gm_3d{src->allocated,
                                src->aligned,
                                src->offset,
                                {size0, size1 * size2, 1},
                                {stride0_gm, stride2_gm, 1}};
    memref_t<__ubuf__ T, 3> ub_3d{dst->allocated,
                                  dst->aligned,
                                  dst->offset,
                                  {size0, size1 * size2, 1},
                                  {stride0_ub, stride2_ub, 1}};
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim<T>(
        &gm_3d, &ub_3d, left_padding_num, l2_cache_ctl);
    return;
  }
  // axis 1 is non-contiguous
  // axis 0 is contiguous
  if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
    memref_t<__gm__ T, 3> gm_3d{src->allocated,
                                src->aligned,
                                src->offset,
                                {size0 * size1, size2, 1},
                                {stride1_gm, stride2_gm, 1}};
    memref_t<__ubuf__ T, 3> ub_3d{dst->allocated,
                                  dst->aligned,
                                  dst->offset,
                                  {size0 * size1, size2, 1},
                                  {stride1_ub, stride2_ub, 1}};
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim<T>(
        &gm_3d, &ub_3d, left_padding_num, l2_cache_ctl);
    return;
  }
  // axis 0 is non-contiguous
  load_gm_to_ubuf_3d_by_scalar<T>(src, dst);
  return;
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_3d_core(
    memref_t<__gm__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy, AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  using PadValueT = typename PadValueType<T>::type;
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((PadValueT *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_ubuf_3d_core(src, dst, left_padding_num);

  uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);
  // deal copy memref<1x1x1> to memref<1x1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    load_gm_to_ubuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T),
                                left_padding_num, 0, 0, l2_cache_ctl);
    return;
  }

  if (src->strides[2] == 1 && dst->strides[2] == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim(src, dst, left_padding_num,
                                                     l2_cache_ctl);
    return;
  }

  int64_t stride0_ub = dst->strides[0];
  int64_t stride1_ub = dst->strides[1];
  int64_t stride2_ub = dst->strides[2];
  int64_t stride0_gm = src->strides[0];
  int64_t stride1_gm = src->strides[1];
  int64_t stride2_gm = src->strides[2];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  if (((stride0_gm < stride1_gm || stride1_gm < stride2_gm) ||
       (stride0_ub < stride1_ub || stride1_ub < stride2_ub)) &&
      (stride0_ub % num_per_block != 0 || stride1_ub % num_per_block != 0 ||
       stride2_ub % num_per_block != 0)) {
    load_gm_to_ubuf_3d_by_scalar(src, dst);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1, size2) with stride [stride0, stride1, stride2]
  // as viewed_src (size0, size1, size2, 1) with stride [stride0, stride1,
  // stride2, 1], where last dimension of viewed_src is contiguous

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];

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
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim(
        &gm_3d, &ub_3d, left_padding_num, l2_cache_ctl);
  }
}
#endif

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_store_ubuf_to_gm_3d_core(memref_t<__ubuf__ T, 3> *src,
                                         memref_t<__gm__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto stride0_ub = src->strides[0];
  auto stride1_ub = src->strides[1];
  auto stride2_ub = src->strides[2];
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
          (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1) &&
          (isSizeAlignedToBlock<T>(stride2_ub) || stride2_ub == 1)) &&
         "The src strides[0]/strides[1]/strides[2] must be 1 or aligned to"
         "block.");
#endif
}

#if defined(__DAV_C310__)
/// Core func of loading UB -> GM, 3D
/// UB starting address must be 32B aligned; otherwise, it degrades to looped
/// scalar transfer.
/// + `UB&GM stride[2] == 1`: last dimension contiguous
///   - `store_ubuf_to_gm_3d_core_with_contiguous_last_dim`
/// + `UB|GM stride[2] != 1`: last dimension non-contiguous
///   + `UB&GM stride[1] == stride[2] * size[2]`: dimension 1 contiguous
///     + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 contiguous
///       - Convert to 2D: `offset2D = offset3D, size2D = [size3D[0] * size3D[1]
///       * size3D[2], 1], stride2D = [stride3D[2], 1]`
///         `store_ubuf_to_gm_2d_core_with_contiguous_last_dim`
///     + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 non-contiguous
///       - Convert to 3D: `newOffset3D = offset3D, newSize3D = [size3D[0],
///       size3D[1] * size3D[2], 1], newStride3D = [stride3D[0], stride3D[2],
///       1]`
///         `store_ubuf_to_gm_3d_core_with_contiguous_last_dim`, where in the
///         new 3D tensor both dimension 0 and 1 are non-contiguous
///   + `UB|GM stride[1] != stride[2] * size[2]`: dimension 1 non-contiguous
///     + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 contiguous
///       - Convert to 3D: `newOffset3D = offset3D, newSize3D = [size3D[0] *
///       size3D[1], size3D[2], 1], newStride3D = [stride3D[1], stride3D[2], 1]`
///         `store_ubuf_to_gm_3d_core_with_contiguous_last_dim`, where in the
///         new 3D tensor both dimension 0 and 1 are non-contiguous
///     + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 non-contiguous
///       fall back to looped scalar transfer. Because non-contiguous last
///       dimension restricts burst to transfer only one element, and both
///       higher dimensions being non-contiguous prevents axis merging. One
///       would need to hoist dimension 0 or 1 as an outer for-loop, and also
///       check 32B alignment of these axes. This essentially becomes looped
///       scalar transfer. Although in some corner cases with two nested
///       for-loops where both higher-dimension strides are multiples of 32B,
///       DMA transferring multiple bursts per instruction might yield slightly
///       better performance than pure scalar transfer, this corner case will
///       not be optimized for now.
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_3d_core(memref_t<__ubuf__ T, 3> *src,
                         memref_t<__gm__ T, 3> *dst, AtomicKind atomic_kind) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_3d_core(src, dst);

  INTRINSIC(set_mov_pad_val, 0);

  // arg for atomic op
  if (atomic_kind != AtomicKind::None) {
    INTRINSIC(pipe_barrier, PIPE_MTE3);
    set_atomic_kind<T>(atomic_kind);
  }

  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    store_ubuf_to_gm_3d_by_scalar<T>(src, dst);
    return;
  }

  const int64_t stride2_ub = src->strides[2];
  const int64_t stride2_gm = dst->strides[2];
  if (stride2_ub == 1 && stride2_gm == 1) [[likely]] {
    // last dimension is contiguous
    store_ubuf_to_gm_3d_core_with_contiguous_last_dim<T>(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }
  // last dimension is not contiguous
  const int64_t stride0_ub = src->strides[0];
  const int64_t stride1_ub = src->strides[1];
  const int64_t stride0_gm = dst->strides[0];
  const int64_t stride1_gm = dst->strides[1];
  if (((stride0_gm < stride1_gm || stride1_gm < stride2_gm) ||
       (stride0_ub < stride1_ub || stride1_ub < stride2_ub))) {
    // Implicit transposition scenarios
    store_ubuf_to_gm_3d_by_scalar<T>(src, dst);
    return;
  }
  // axis 1 is contiguous
  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  const int64_t size2 = dst->sizes[2];
  if (stride1_ub == stride2_ub * size2 && stride1_gm == stride2_gm * size2) {
    // axis 0 is contiguous
    if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
      memref_t<__gm__ T, 2> gm_2d{dst->allocated,
                                  dst->aligned,
                                  dst->offset,
                                  {size0 * size1 * size2, 1},
                                  {stride2_gm, 1}};
      memref_t<__ubuf__ T, 2> ub_2d{src->allocated,
                                    src->aligned,
                                    src->offset,
                                    {size0 * size1 * size2, 1},
                                    {stride2_ub, 1}};
      store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
      set_store_atomic_none(atomic_kind);
      return;
    }
    // axis 0 is non-contiguous
    memref_t<__gm__ T, 3> gm_3d{dst->allocated,
                                dst->aligned,
                                dst->offset,
                                {size0, size1 * size2, 1},
                                {stride0_gm, stride2_gm, 1}};
    memref_t<__ubuf__ T, 3> ub_3d{src->allocated,
                                  src->aligned,
                                  src->offset,
                                  {size0, size1 * size2, 1},
                                  {stride0_ub, stride2_ub, 1}};
    store_ubuf_to_gm_3d_core_with_contiguous_last_dim<T>(&ub_3d, &gm_3d);
    set_store_atomic_none(atomic_kind);
    return;
  }
  // axis 1 is non-contiguous
  // axis 0 is contiguous
  if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
    memref_t<__gm__ T, 3> gm_3d{dst->allocated,
                                dst->aligned,
                                dst->offset,
                                {size0 * size1, size2, 1},
                                {stride1_gm, stride2_gm, 1}};
    memref_t<__ubuf__ T, 3> ub_3d{src->allocated,
                                  src->aligned,
                                  src->offset,
                                  {size0 * size1, size2, 1},
                                  {stride1_ub, stride2_ub, 1}};
    store_ubuf_to_gm_3d_core_with_contiguous_last_dim<T>(&ub_3d, &gm_3d);
    set_store_atomic_none(atomic_kind);
    return;
  }
  // axis 0 is non-contiguous
  store_ubuf_to_gm_3d_by_scalar<T>(src, dst);
  return;
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_3d_core(memref_t<__ubuf__ T, 3> *src,
                         memref_t<__gm__ T, 3> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_3d_core(src, dst);

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

  // deal copy memref<1x1x1> to memref<1x1x1>
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    store_ubuf_to_gm_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T), 0,
                                 0);
    set_store_atomic_none(atomic_kind);
    return;
  }

  if (!isAddress32ByteAligned(src_ptr)) {
    store_ubuf_to_gm_3d_by_scalar(src, dst);
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
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if (src->strides[2] != 1 && src->strides[2] % num_per_block != 0) {
    // TODO: see "DMA/DMAUtils.h" for details.
    store_ubuf_to_gm_3d_by_scalar(src, dst);
    return;
  }

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
#endif

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

// clang-format off
/// core func of copy ub <-> ub, 3d
/// + `src&dst stride[2] == 1`: Last dimension is contiguous, use `copy_ubuf_to_ubuf_3d_core_with_contiguous_last_dim`
/// + `src|dst stride[2] != 1`: Does not satisfy burstLen being a multiple of 32B
///   + `src&dst stride[1] == stride[2] * size[2]`: Dimension 1 is contiguous
///     - `src&dst stride[0] == stride[1] * size[1]`: Dimension 0 is contiguous
///       Fold into 2D
///       ```
///       offset2D = offset3D, size2D = [size3D[0] * size3D[1] * size3D[2], 1], stride2D = [stride3D[2], 1]
///       copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim
///       ```
///     - `src|dst stride[0] != stride[1] * size[1]`: Dimension 0 is non-contiguous
///       ```
///       for i = [0, size3D[0]) {
///         offset2D = offset3D + i * stride3D[0], size2D = [size3D[1] * size3D[2], 1], stride2D = [stride3D[2], 1]
///         copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim
///       }
///       ```
///   + `src|dst stride[1] != stride[2] * size[2]`: Dimension 1 is non-contiguous;
///     since it involves a for-loop anyway, merging dimensions will not accelerate performance
///     ```
///     for i = [0, size3D[0]) {
///       for j = [0, size3D[1]) {
///         offset2D = offset3D + i * stride3D[0] + j * stride3D[1], size2D = [size3D[2], 1], stride2D = [stride3D[2], 1]
///         copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim
///       }
///     }
///     ```
// clang-format on
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_3d_core(memref_t<__ubuf__ T, 3> *src,
                          memref_t<__ubuf__ T, 3> *dst) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_ubuf_3d_core(src, dst);

  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  const int64_t size2 = dst->sizes[2];
  if (size0 == 1 && size1 == 1 && size2 == 1) {
    // deal copy memref<1x1x1> to memref<1x1x1>
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {1}, {1}};
    memref_t<__ubuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {1}, {1}};
    copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
    return;
  }

  const int64_t src_stride2 = src->strides[2];
  const int64_t dst_stride2 = dst->strides[2];
  // axis 2 is contiguous
  if (src_stride2 == 1 && dst_stride2 == 1) {
    copy_ubuf_to_ubuf_3d_core_with_contiguous_last_dim(src, dst);
    return;
  }
  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride1 = dst->strides[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  if (src_stride1 == src_stride2 * size2 &&
      dst_stride1 == dst_stride2 * size2) {
    // axis 1 is contiguous
    if (src_stride0 == src_stride1 * size1 &&
        dst_stride0 == dst_stride1 * size1) {
      // axis 0 is contiguous
      memref_t<__ubuf__ T, 2> src2d{src->allocated,
                                    src->aligned,
                                    src->offset,
                                    {size0 * size1 * size2, 1},
                                    {src_stride2, 1}};
      memref_t<__ubuf__ T, 2> dst2d{dst->allocated,
                                    dst->aligned,
                                    dst->offset,
                                    {size0 * size1 * size2, 1},
                                    {dst_stride2, 1}};
      copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src2d, &dst2d);
      return;
    }
    // axis 0 is non-contiguous
    for (int64_t i = 0; i < size0; ++i) {
      memref_t<__ubuf__ T, 2> src2d{src->allocated,
                                    src->aligned,
                                    src->offset + i * src_stride0,
                                    {size1 * size2, 1},
                                    {src_stride2, 1}};
      memref_t<__ubuf__ T, 2> dst2d{dst->allocated,
                                    dst->aligned,
                                    dst->offset + i * dst_stride0,
                                    {size1 * size2, 1},
                                    {dst_stride2, 1}};
      copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src2d, &dst2d);
    }
    return;
  }
  // axis 1 is non-contiguous
  // even if axis 0 is contiguous, merging loop does not deliver faster impl.
  for (int64_t i = 0; i < size0; ++i) {
    for (int64_t j = 0; j < size1; ++j) {
      memref_t<__ubuf__ T, 2> src2d{src->allocated,
                                    src->aligned,
                                    src->offset + i * src_stride0 +
                                        j * src_stride1,
                                    {size2, 1},
                                    {src_stride2, 1}};
      memref_t<__ubuf__ T, 2> dst2d{dst->allocated,
                                    dst->aligned,
                                    dst->offset + i * dst_stride0 +
                                        j * dst_stride1,
                                    {size2, 1},
                                    {dst_stride2, 1}};
      copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src2d, &dst2d);
    }
  }
  return;
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

#if defined(__DAV_C310__)
template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_copy_ubuf_to_cbuf_3d_core(memref_t<__ubuf__ T, 3> *src,
                                          memref_t<__cbuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride2_src = src->strides[2];
  const int64_t stride2_dst = dst->strides[2];
  assert((stride2_src == 1) && "Last dimension of src must be contiguous.");
  assert((stride2_dst == 1) && "Last dimension of dst must be contiguous.");
#endif
}

/// core func of copy ub -> cbuf, 3d
/// constraints:
/// 1. stride2 must be 1
/// TODO: update for constraints on alignment
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_3d_core(memref_t<__ubuf__ T, 3> *src,
                          memref_t<__cbuf__ T, 3> *dst) {
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_cbuf_3d_core(src, dst);

  if (dst->sizes[0] == 1 && dst->sizes[1] == 1 && dst->sizes[2] == 1) {
    // deal copy memref<1x1x1> to memref<1x1x1>
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {1}, {1}};
    memref_t<__cbuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {1}, {1}};
    copy_ubuf_to_cbuf_1d_core(&src_1d, &dst_1d);
    return;
  }

  copy_ubuf_to_cbuf_3d_core_with_contiguous_last_dim(src, dst);
}
#endif

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
#if defined(__DAV_C310__)
REGISTE_DMA_LOAD_FP8(3, float8_e4m3_t);
REGISTE_DMA_LOAD_FP8(3, float8_e5m2_t);
#endif
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
#if defined(__DAV_C310__)
REGISTE_DMA_STORE(3, float8_e4m3_t);
REGISTE_DMA_STORE(3, float8_e5m2_t);
#endif
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
#if defined(__DAV_C310__)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, float8_e4m3_t);
REGISTE_DMA_UB_COPY(ubuf, ubuf, 3, float8_e5m2_t);
#endif

#if defined(__DAV_C310__)
//===-------------------------------------------------------------------===//
// ub to cbuf, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, int8_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, uint8_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, int16_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, uint16_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, int32_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, uint32_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, float8_e4m3_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, float8_e5m2_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, half)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, float)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 3, bfloat16_t)
#endif
}
