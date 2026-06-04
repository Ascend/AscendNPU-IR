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
check_inputs_of_load_gm_to_ubuf_2d_core(memref_t<__gm__ T, 2> *src,
                                        memref_t<__ubuf__ T, 2> *dst,
                                        int64_t left_padding_num) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset - left_padding_num;
  auto stride0_ub = dst->strides[0];
  auto stride1_ub = dst->strides[1];
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
          (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1)) &&
         "The dst strides[0]/strides[1] must be 1 or aligned to block.");
#endif
}

/// Rewrite padding for dst correctly in the case of b64 type by applying the
/// padding in intervals with b32 scalars
///
/// Constraints:
/// 1. dim of dst must be 2, stride_0 dst must be 1
/// 2. dst and pad_value are expected to be int64_t
template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t> ||
                                                  std::is_same_v<T, uint64_t>>>
__aiv__ __attribute__((always_inline)) void
align_pad_for_load_b64_2d(memref_t<__ubuf__ T, 2> *dst, int64_t pad_value,
                          int64_t left_padding_num) {
  const int64_t factor = sizeof(T) / sizeof(uint32_t);
  int64_t stride_dst = dst->strides[0];
  // if size and stride is too high, we cant fit the mask so we shift
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  int64_t padding_shift_count = stride_dst / num_per_repeat;
  if (stride_dst % num_per_repeat == 0) {
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

#if defined(__DAV_C310__)
/// Core func of loading gm -> ub, 2D
/// UB starting address must be 32B aligned; otherwise, it degrades to looped
/// scalar transfer:
/// + `UB&GM stride[1] == 1`
///   - `load_gm_to_ubuf_2d_core_with_contiguous_last_dim`
/// + `UB|GM stride[1] != 1`
///   + `UB&GM stride[0] < stride[1]`: on-the-fly transpose (load supports
///   NDDMA)
///     - `load_gm_to_ubuf_2d_by_nddma`
///   + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 is contiguous
///     - `newOffset2D = offset2D, newSize2D = [size2D[0] * size2D[1], 1],
///     newStride2D = [stride2D[1], 1]`
///     - `load_gm_to_ubuf_2d_core_with_contiguous_last_dim`
///   + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 is non-contiguous
///     - `strideUB[0] * dtypeBytes % 32 == 0`
///         ```
///         for i = [0, size0) {
///           newOffset2D = offset2D + i * stride2D[0], newSize2D = [size2D[1],
///           1], newStride2D = [stride2D[1], 1]
///           load_gm_to_ubuf_2d_core_with_contiguous_last_dim
///         }
///         ```
///     - `strideUB[0] * dtypeBytes % 32 != 0`
///       looped scalar transfer
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_2d_core(
    memref_t<__gm__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy, AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_ubuf_2d_core(src, dst, left_padding_num);

  using PadValueT = typename PadValueType<T>::type;
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((PadValueT *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  auto dst_ptr = dst->aligned + dst->offset;
  if (!isAddress32ByteAligned(dst_ptr)) {
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst);
    return;
  }

  int64_t stride1_gm = src->strides[1];
  int64_t stride1_ub = dst->strides[1];
  int64_t stride0_gm = src->strides[0];
  int64_t stride0_ub = dst->strides[0];
  bool has_padding = left_padding_num != 0;
  if (stride1_gm < 0 || stride0_gm < 0 || stride1_ub < 0 || stride0_ub < 0)
      [[unlikely]] {
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst);
    return;
  }

  if ((stride0_gm < stride1_gm || stride0_ub < stride1_ub)) {
    // Implicit transposition scenarios need to be moved through scalar
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      load_gm_to_ubuf_2d_by_scalar<T>(src, dst);
    } else {
      load_gm_to_ubuf_2d_by_nddma<T>(src, dst);
    }
    return;
  }

  uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);
  if (stride1_gm == 1 && stride1_ub == 1) [[likely]] {
    // last dimension is contiguous
    if (!has_padding && !isStrideAligned<T>(stride0_ub)) {
      load_gm_to_ubuf_2d_by_scalar<T>(src, dst);
      return;
    }
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        src, dst, left_padding_num, l2_cache_ctl);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      if (pad_mode == PadMode::Value) {
        INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        int64_t scalar = static_cast<int64_t>(pad_value);
        align_pad_for_load_b64_2d<T>(dst, scalar, left_padding_num);
        INTRINSIC(set_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
      }
    }
    return;
  }

  // last dimension is not non-contiguous
  if (stride1_ub != 1 && !isStrideAligned<T>(stride1_ub)) {
    // UB stride1 is not 1 thus compact mode is unavailable.
    // UB stride1 is not 32B aligned thus normal mode is unavailable.
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst);
    return;
  }

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
    // axis 0 is contiguous. thus we can collapse axis 0 and 1
    memref_t<__gm__ T, 2> gm_2d = {src->allocated,
                                   src->aligned,
                                   src->offset,
                                   {size0 * size1, 1},
                                   {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d = {dst->allocated,
                                     dst->aligned,
                                     dst->offset,
                                     {size0 * size1, 1},
                                     {stride1_ub, 1}};
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
    return;
  }

  // axis 0 is non-contiguous.
  if (!isStrideAligned<T>(stride0_ub)) {
    load_gm_to_ubuf_2d_by_scalar<T>(src, dst);
    return;
  }
  // UB stride0 is 32B aligned thus we can DMA to move each row.
  for (int64_t i = 0; i < size0; i++) {
    memref_t<__gm__ T, 2> gm_2d = {src->allocated,
                                   src->aligned,
                                   src->offset + i * stride0_gm,
                                   {size1, 1},
                                   {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d = {dst->allocated,
                                     dst->aligned,
                                     dst->offset + i * stride0_ub,
                                     {size1, 1},
                                     {stride1_ub, 1}};
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
  }
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_2d_core(
    memref_t<__gm__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy, AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_ubuf_2d_core(src, dst, left_padding_num);

  using PadValueT = typename PadValueType<T>::type;
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((PadValueT *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);
  // deal copy memref<1x1> to memref<1x1>
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    auto src_ptr = src->aligned + src->offset;
    auto dst_ptr = dst->aligned + dst->offset;
    load_gm_to_ubuf_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T),
                                left_padding_num, 0, 0, l2_cache_ctl);
    return;
  }

  int64_t stride1_gm = src->strides[1];
  int64_t stride1_ub = dst->strides[1];
  if (stride1_gm == 1 && stride1_ub == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
        src, dst, left_padding_num, l2_cache_ctl);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      if (pad_mode == PadMode::Value) {
        INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        int64_t scalar = static_cast<int64_t>(pad_value);
        align_pad_for_load_b64_2d<T>(dst, scalar, left_padding_num);
      }
    }
    return;
  }

  int64_t stride0_gm = src->strides[0];
  int64_t stride0_ub = dst->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  if ((stride0_gm < stride1_gm || stride0_ub < stride1_ub) &&
      (stride0_ub % num_per_block != 0 || stride1_ub % num_per_block != 0)) {
    // Implicit transposition scenarios need to be moved through scalar
    load_gm_to_ubuf_2d_by_scalar(src, dst);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1) with stride [stride0, stride1] as viewed_src
  // (size0, size1, 1) with stride [stride0, stride1, 1], where last dimension
  // of viewed_src is contiguous
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
  load_gm_to_ubuf_3d_core_with_contiguous_last_dim<T>(
      &gm_3d, &ub_3d, left_padding_num, l2_cache_ctl);
}
#endif

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_store_ubuf_to_gm_2d_core(memref_t<__ubuf__ T, 2> *src,
                                         memref_t<__gm__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto stride0_ub = src->strides[0];
  auto stride1_ub = src->strides[1];
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
          (isSizeAlignedToBlock<T>(stride1_ub) || stride1_ub == 1)) &&
         "The src strides[0]/strides[1] must be 1 or aligned to block.");
#endif
}

#if defined(__DAV_C310__)
/// Core func of loading ub -> gm, 2D
/// UB starting address must be 32B aligned; otherwise, it degrades to looped
/// scalar transfer.
/// + `UB&GM stride[1] == 1`
///   - `store_ubuf_to_gm_2d_core_with_contiguous_last_dim`
/// + `UB|GM stride[1] != 1`
///   + `UB&GM stride[0] < stride[1]`: on-the-fly transpose (store does not
///   supports NDDMA)
///     - `store_ubuf_to_gm_2d_by_scalar`
///   + `UB&GM stride[0] == stride[1] * size[1]`: dimension 0 is contiguous
///     - `newOffset2D = offset2D, newSize2D = [size2D[0] * size2D[1], 1],
///     newStride2D = [stride2D[1], 1]`
///     - `store_ubuf_to_gm_2d_core_with_contiguous_last_dim`
///   + `UB|GM stride[0] != stride[1] * size[1]`: dimension 0 is non-contiguous
///     - `strideUB[0] * dtypeBytes % 32 == 0`
///         ```
///         for i = [0, size0) {
///           newOffset2D = offset2D + i * stride2D[0], newSize2D = [size2D[1],
///           1], newStride2D = [stride2D[1], 1]
///           store_ubuf_to_gm_2d_core_with_contiguous_last_dim
///         }
///         ```
///     - `strideUB[0] * dtypeBytes % 32 != 0`
///       looped scalar transfer
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_2d_core(memref_t<__ubuf__ T, 2> *src,
                         memref_t<__gm__ T, 2> *dst, AtomicKind atomic_kind) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_2d_core(src, dst);

  INTRINSIC(set_mov_pad_val, 0);

  // arg for atomic op
  if (atomic_kind != AtomicKind::None) {
    INTRINSIC(pipe_barrier, PIPE_MTE3);
    set_atomic_kind<T>(atomic_kind);
  }

  // deal copy memref<1x1> to memref<1x1>
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
    return;
  }

  const int64_t stride0_gm = dst->strides[0];
  const int64_t stride0_ub = src->strides[0];
  const int64_t stride1_gm = dst->strides[1];
  const int64_t stride1_ub = src->strides[1];
  if (stride0_ub < stride1_ub || stride0_gm < stride1_gm) {
    // Implicit transposition scenarios need to be moved through scalar
    store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
    return;
  }

  if (stride1_gm == 1 && stride1_ub == 1) [[likely]] {
    // last dimension is contiguous
    // Check if stride0 is 32B aligned, otherwise use scalar path.
    if (!isStrideAligned<T>(stride0_ub)) {
      store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
      set_store_atomic_none(atomic_kind);
      return;
    }
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }

  // last dimension is not non-contiguous
  if (stride1_ub != 1 && !isStrideAligned<T>(stride1_ub)) {
    // UB stride1 is not 1 thus compact mode is unavailable.
    // UB stride1 is not 32B aligned thus normal mode is unavailable.
    store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
    return;
  }

  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  if (stride0_ub == stride1_ub * size1 && stride0_gm == stride1_gm * size1) {
    // axis 0 is contiguous. thus we can collapse axis 0 and 1
    memref_t<__gm__ T, 2> gm_2d = {dst->allocated,
                                   dst->aligned,
                                   dst->offset,
                                   {size0 * size1, 1},
                                   {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d = {src->allocated,
                                     src->aligned,
                                     src->offset,
                                     {size0 * size1, 1},
                                     {stride1_ub, 1}};
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
    set_store_atomic_none(atomic_kind);
    return;
  }

  // axis 0 is non-contiguous.
  if (!isStrideAligned<T>(stride0_ub)) {
    store_ubuf_to_gm_2d_by_scalar<T>(src, dst);
    return;
  }
  // UB stride0 is 32B aligned thus we can DMA to move each row.
  for (int64_t i = 0; i < size0; i++) {
    memref_t<__gm__ T, 2> gm_2d = {dst->allocated,
                                   dst->aligned,
                                   dst->offset + i * stride0_gm,
                                   {size1, 1},
                                   {stride1_gm, 1}};
    memref_t<__ubuf__ T, 2> ub_2d = {src->allocated,
                                     src->aligned,
                                     src->offset + i * stride0_ub,
                                     {size1, 1},
                                     {stride1_ub, 1}};
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);
    set_store_atomic_none(atomic_kind);
  }
}
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_2d_core(memref_t<__ubuf__ T, 2> *src,
                         memref_t<__gm__ T, 2> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_2d_core(src, dst);

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

  // deal copy memref<1x1> to memref<1x1>
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    store_ubuf_to_gm_intrin_core(src_ptr, 0, dst_ptr, 0, 1, 1 * sizeof(T), 0,
                                 0);
    set_store_atomic_none(atomic_kind);
    return;
  }

  if (!isAddress32ByteAligned(src_ptr)) {
    store_ubuf_to_gm_2d_by_scalar(src, dst);
    return;
  }

  const int64_t stride1_gm = dst->strides[1];
  const int64_t stride1_ub = src->strides[1];
  if (stride1_gm == 1 && stride1_ub == 1) [[likely]] {
    // last dimension is contiguous
    store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(src, dst);
    set_store_atomic_none(atomic_kind);
    return;
  }

  if (src->strides[0] < src->strides[1] || dst->strides[0] < dst->strides[1]) {
    // Implicit transposition scenarios need to be moved through scalar
    store_ubuf_to_gm_2d_by_scalar(src, dst);
    return;
  }

  // last dimension is not contiguous,
  // view the src (size0, size1) with stride [stride0, stride1] as viewed_src
  // (size0, size1, 1) with stride [stride0, stride1, 1], where last dimension
  // of viewed_src is contiguous
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if (src->strides[1] != 1 && src->strides[1] % num_per_block != 0) {
    // TODO: see "DMA/DMAUtils.h" for details.
    store_ubuf_to_gm_2d_by_scalar(src, dst);
    return;
  }
  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  const int64_t stride0_gm = dst->strides[0];
  const int64_t stride0_ub = src->strides[0];
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
#endif

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
/// + `src&dst stride[1] == 1`: last dimension contiguous,
/// `copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim`
/// + `src|dst stride[1] != 1`: burstLen not a multiple of 32B
///   - `src&dst stride[0] == stride[1] * size[1]`: dimension 0 contiguous
///     Convert to 2D
///     ```
///     newOffset2D = offset2D, newSize2D = [size2D[0] * size2D[1], 1],
///     newStride2D = [stride2D[1], 1]
///     copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim
///     ```
///   - `src|dst stride[0] != stride[1] * size[1]`: dimension 0 non-contiguous
///     ```
///     for i = [0, size2D[0]) {
///       newOffset2D = offset2D + i * stride2D[0], newSize2D = [size2D[1], 1],
///       newStride2D = [stride2D[1], 1]
///       copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim
///     }
///     ```
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_ubuf_2d_core(memref_t<__ubuf__ T, 2> *src,
                          memref_t<__ubuf__ T, 2> *dst) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_ubuf_2d_core(src, dst);

  const int64_t size0 = dst->sizes[0];
  const int64_t size1 = dst->sizes[1];
  if (size0 == 1 && size1 == 1) {
    // deal copy memref<1x1> to memref<1x1>
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {1}, {1}};
    memref_t<__ubuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {1}, {1}};
    copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
    return;
  }

  const int64_t src_stride1 = src->strides[1];
  const int64_t dst_stride1 = dst->strides[1];
  // axis 1 is contiguous
  if (src_stride1 == 1 && dst_stride1 == 1) {
    copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(src, dst);
    return;
  }
  // axis 1 is non-contiguous
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  // axis 0 is contiguous
  if (src_stride0 == src_stride1 * size1 &&
      dst_stride0 == dst_stride1 * size1) {
    memref_t<__ubuf__ T, 2> src2d{src->allocated,
                                  src->aligned,
                                  src->offset,
                                  {size0 * size1, 1},
                                  {src_stride1, 1}};
    memref_t<__ubuf__ T, 2> dst2d{dst->allocated,
                                  dst->aligned,
                                  dst->offset,
                                  {size0 * size1, 1},
                                  {dst_stride1, 1}};
    copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src2d, &dst2d);
    return;
  }
  // axis 0 is non-contiguous
  for (int64_t i = 0; i < size0; ++i) {
    memref_t<__ubuf__ T, 2> src2d{src->allocated,
                                  src->aligned,
                                  src->offset + i * src_stride0,
                                  {size1, 1},
                                  {src_stride1, 1}};
    memref_t<__ubuf__ T, 2> dst2d{dst->allocated,
                                  dst->aligned,
                                  dst->offset + i * dst_stride0,
                                  {size1, 1},
                                  {dst_stride1, 1}};
    copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src2d, &dst2d);
  }
  return;
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

#if defined(__DAV_C310__)
template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_copy_ubuf_to_cbuf_2d_core(memref_t<__ubuf__ T, 2> *src,
                                          memref_t<__cbuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride1_src = src->strides[1];
  const int64_t stride1_dst = dst->strides[1];
  assert((stride1_src == 1) && "Last dimension of src must be contiguous.");
  assert((stride1_dst == 1) && "Last dimension of dst must be contiguous.");
#endif
}

/// core func of copy ub -> cbuf, 2d
/// constraints:
/// 1. stride1 must be 1
/// TODO: update for constraints on alignment
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_2d_core(memref_t<__ubuf__ T, 2> *src,
                          memref_t<__cbuf__ T, 2> *dst) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_cbuf_2d_core(src, dst);

  if (dst->sizes[0] == 1 && dst->sizes[1] == 1) {
    // deal copy memref<1x1> to memref<1x1>
    memref_t<__ubuf__ T, 1> src_1d{
        src->allocated, src->aligned, src->offset, {1}, {1}};
    memref_t<__cbuf__ T, 1> dst_1d{
        dst->allocated, dst->aligned, dst->offset, {1}, {1}};
    copy_ubuf_to_cbuf_1d_core(&src_1d, &dst_1d);
    return;
  }

  copy_ubuf_to_cbuf_2d_core_with_contiguous_last_dim<T>(src, dst);
}
#endif

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
#if defined(__DAV_C310__)
REGISTE_DMA_LOAD_FP8(2, float8_e4m3_t);
REGISTE_DMA_LOAD_FP8(2, float8_e5m2_t);
#endif
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
#if defined(__DAV_C310__)
REGISTE_DMA_STORE(2, float8_e4m3_t);
REGISTE_DMA_STORE(2, float8_e5m2_t);
#endif
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
#if defined(__DAV_C310__)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, float8_e4m3_t);
REGISTE_DMA_UB_COPY(ubuf, ubuf, 2, float8_e5m2_t);
#endif

#if defined(__DAV_C310__)
//===-------------------------------------------------------------------===//
// ub to cbuf, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, int8_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, uint8_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, int16_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, uint16_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, int32_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, uint32_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, float8_e4m3_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, float8_e5m2_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, half)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, float)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 2, bfloat16_t)
#endif
}
