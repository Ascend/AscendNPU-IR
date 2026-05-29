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

__aiv__ __attribute__((always_inline)) void
set_store_atomic_none(AtomicKind atomic_kind) {
  if (atomic_kind != AtomicKind::None) {
    INTRINSIC_NO_ARGS(set_atomic_none);
    INTRINSIC(pipe_barrier, PIPE_MTE3);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_load_gm_to_ubuf_1d_core(memref_t<__gm__ T, 1> *src,
                                        memref_t<__ubuf__ T, 1> *dst,
                                        int64_t left_padding_num) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset - left_padding_num;
  const int64_t stride0_ub = dst->strides[0];
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
         "The dst strides must be 1 or aligned to block.");
#endif
}

/// Rewrite padding for dst correctly in the case of b64 type by applying the
/// padding in intervals with b32 scalars
///
/// Constraints:
/// 1. dim of dst must be 1, dst must be continuous vector
/// 2. dst and pad_value are expected to be int64_t
template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t> ||
                                                  std::is_same_v<T, uint64_t>>>
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

#if defined(__DAV_C310__)
/// Core func of loading gm -> ub, 1D
/// UB starting address must be 32B aligned; otherwise, it degrades to looped
/// scalar transfer: `load_gm_to_ubuf_1d_by_scalar` As long as the UB starting
/// address is 32B aligned, a single instruction can complete the transfer.
/// - `UB&GM stride == 1`, then `numBurst = 1, burst_len = size[1] * dtypeBytes`
///   `load_gm_to_ubuf_1d_core_with_contiguous_last_dim`
/// - `UB|GM stride != 1`, then `numBurst = size[1], burst_len = 1*dtypeBytes`.
///   Promote to 2D for unified handling of contiguous scenarios: `offset2D =
///   offset1D, size2D = [size1D[0], 1], stride2D = [stride2D[0], 1]`
///   `load_gm_to_ubuf_2d_core_with_contiguous_last_dim`
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_1d_core(
    memref_t<__gm__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy, AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_ubuf_1d_core(src, dst, left_padding_num);

  using PadValueT = typename PadValueType<T>::type;
  // "aicore arch: Ascend910B2"
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((PadValueT *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  auto dst_ptr = dst->aligned + dst->offset;
  if (!isAddress32ByteAligned(dst_ptr)) {
    load_gm_to_ubuf_1d_by_scalar<T>(src, dst);
    return;
  }

  uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);
  const int64_t stride0_gm = src->strides[0];
  const int64_t stride0_ub = dst->strides[0];
  if (stride0_gm == 1 && stride0_ub == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_1d_core_with_contiguous_last_dim<T>(
        src, dst, left_padding_num, l2_cache_ctl);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      if (pad_mode == PadMode::Value) {
        INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        int64_t scalar = static_cast<int64_t>(pad_value);
        align_pad_for_load_b64_1d<T>(dst, scalar, left_padding_num);
        INTRINSIC(set_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
      }
    }
    return;
  }

  // last dimension of UB is contiguous but GM is not
  if ((stride0_ub == 1) && (stride0_gm != 1)) {
    load_gm_to_ubuf_1d_core_with_ubuf_contiguous_last_dim<T>(
        src, dst, left_padding_num, l2_cache_ctl);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      if (pad_mode == PadMode::Value) {
        INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        int64_t scalar = static_cast<int64_t>(pad_value);
        align_pad_for_load_b64_1d<T>(dst, scalar, left_padding_num);
        INTRINSIC(set_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_V, PIPE_MTE3, LIB_EVENT_ID0);
      }
    }
    return;
  }

  // last dimension is not contiguous,
  if (stride0_ub != 1 && !isStrideAligned<T>(stride0_ub)) {
    load_gm_to_ubuf_1d_by_scalar<T>(src, dst);
    return;
  }
  // view the src (size) with stride [stride] as viewed_src (size, 1) with
  // stride [stride, 1], where last dimension of viewed_src is contiguous
  const int64_t size0 = src->sizes[0];
  memref_t<__gm__ T, 2> gm_2d = {
      src->allocated, src->aligned, src->offset, {size0, 1}, {stride0_gm, 1}};
  memref_t<__ubuf__ T, 2> ub_2d = {
      dst->allocated, dst->aligned, dst->offset, {size0, 1}, {stride0_ub, 1}};

  load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
      &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
}
#else
///========================================
/// m300 arch
///========================================
// core func of load gm <-> ub, 1d
template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_1d_core(
    memref_t<__gm__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy, AtomicKind atomic_kind = AtomicKind::None) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_load_gm_to_ubuf_1d_core(src, dst, left_padding_num);

  using PadValueT = typename PadValueType<T>::type;
  // "aicore arch: Ascend910B2"
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((PadValueT *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }

  uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);
  const int64_t stride0_gm = src->strides[0];
  const int64_t stride0_ub = dst->strides[0];
  if (stride0_gm == 1 && stride0_ub == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_ubuf_1d_core_with_contiguous_last_dim<T>(
        src, dst, left_padding_num, l2_cache_ctl);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      if (pad_mode == PadMode::Value) {
        INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, LIB_EVENT_ID0);
        int64_t scalar = static_cast<int64_t>(pad_value);
        align_pad_for_load_b64_1d<T>(dst, scalar, left_padding_num);
      }
    }
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

  load_gm_to_ubuf_2d_core_with_contiguous_last_dim<T>(
      &gm_2d, &ub_2d, left_padding_num, l2_cache_ctl);
}
#endif

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_store_ubuf_to_gm_1d_core(memref_t<__ubuf__ T, 1> *src,
                                         memref_t<__gm__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  const int64_t stride0_ub = src->strides[0];
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(stride0_ub) || stride0_ub == 1) &&
         "The src strides must be 1 or aligned to block.");
#endif
}

#if defined(__DAV_C310__)
/// Core func of storing ub -> gm, 1D
/// UB starting address must be 32B aligned; otherwise, it degrades to looped
/// scalar transfer: `store_ubuf_to_gm_1d_by_scalar` As long as the UB starting
/// address is 32B aligned, a single instruction can complete the transfer.
/// - `UB&GM stride == 1`, then `numBurst = 1, burst_len = size[1] * dtypeBytes`
///   `store_ubuf_to_gm_1d_core_with_contiguous_last_dim`
/// - `UB|GM stride != 1`, then `numBurst = size[1], burst_len = 1*dtypeBytes`.
///   Promote to 2D for unified handling of contiguous scenarios: `offset2D =
///   offset1D, size2D = [size1D[0], 1], stride2D = [stride2D[0], 1]`
///   `store_ubuf_to_gm_2d_core_with_contiguous_last_dim`
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_1d_core(memref_t<__ubuf__ T, 1> *src,
                         memref_t<__gm__ T, 1> *dst, AtomicKind atomic_kind) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_1d_core(src, dst);

  INTRINSIC(set_mov_pad_val, 0);

  // arg for atomic op
  if (atomic_kind != AtomicKind::None) {
    INTRINSIC(pipe_barrier, PIPE_MTE3);
    set_atomic_kind<T>(atomic_kind);
  }

  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    store_ubuf_to_gm_1d_by_scalar<T>(src, dst);
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
  if (stride0_ub != 1 && !isStrideAligned<T>(stride0_ub)) {
    store_ubuf_to_gm_1d_by_scalar<T>(src, dst);
    return;
  }

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
#else
///========================================
/// m300 arch
///========================================
template <typename T>
__aiv__ __attribute__((always_inline)) void
store_ubuf_to_gm_1d_core(memref_t<__ubuf__ T, 1> *src,
                         memref_t<__gm__ T, 1> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_1d_core(src, dst);

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

  auto src_ptr = src->aligned + src->offset;
  if (!isAddress32ByteAligned(src_ptr)) {
    store_ubuf_to_gm_1d_by_scalar(src, dst);
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
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if (src->strides[0] != 1 && src->strides[0] % num_per_block != 0) {
    // TODO: see "DMA/DMAUtils.h" for details.
    store_ubuf_to_gm_1d_by_scalar(src, dst);
    return;
  }
  const int64_t size0 = src->sizes[0];
  memref_t<__ubuf__ T, 2> ub_2d = {
      src->allocated, src->aligned, src->offset, {size0, 1}, {stride0_ub, 1}};
  memref_t<__gm__ T, 2> gm_2d = {
      dst->allocated, dst->aligned, dst->offset, {size0, 1}, {stride0_gm, 1}};

  store_ubuf_to_gm_2d_core_with_contiguous_last_dim<T>(&ub_2d, &gm_2d);

  set_store_atomic_none(atomic_kind);
}
#endif

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
/// - `stride[0] == 1`: Directly call
/// `copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim`
/// - `stride[0] != 1`: Process as 2D memref
/// (`vector_dma_unalign_offset_vv_1d_vf` cannot handle this)
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
  if (src_stride0 == 1 && dst_stride0 == 1) {
    copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim<T>(src, dst);
    return;
  }
  // axis 0 is non-contiguous
  const int64_t size0 = dst->sizes[0];
  memref_t<__ubuf__ T, 2> src2d{
      src->allocated, src->aligned, src->offset, {size0, 1}, {src_stride0, 1}};
  memref_t<__ubuf__ T, 2> dst2d{
      dst->allocated, dst->aligned, dst->offset, {size0, 1}, {dst_stride0, 1}};
  copy_ubuf_to_ubuf_2d_core_with_contiguous_last_dim<T>(&src2d, &dst2d);
  return;
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

#if defined(__DAV_C310__)
template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_copy_ubuf_to_cbuf_1d_core(memref_t<__ubuf__ T, 1> *src,
                                          memref_t<__cbuf__ T, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride0_ub_src = src->strides[0];
  const int64_t stride0_l1_dst = dst->strides[0];
  assert((stride0_ub_src == 1) && "Last dimension of src must be contiguous.");
  assert((stride0_l1_dst == 1) && "Last dimension of dst must be contiguous.");
#endif
}

/// core func of copy ub -> cbuf, 1d
/// constraints:
/// 1. stride0 must be 1
/// TODO: update for constraints on alignment
template <typename T>
__aiv__ __attribute__((always_inline)) void
copy_ubuf_to_cbuf_1d_core(memref_t<__ubuf__ T, 1> *src,
                          memref_t<__cbuf__ T, 1> *dst) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  check_inputs_of_copy_ubuf_to_cbuf_1d_core(src, dst);
  copy_ubuf_to_cbuf_1d_core_with_contiguous_last_dim<T>(src, dst);
}
#endif

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
#if defined(__DAV_C310__)
REGISTE_DMA_LOAD_FP8(1, float8_e4m3_t);
REGISTE_DMA_LOAD_FP8(1, float8_e5m2_t);
#endif
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
#if defined(__DAV_C310__)
REGISTE_DMA_STORE(1, float8_e4m3_t);
REGISTE_DMA_STORE(1, float8_e5m2_t);
#endif
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
#if defined(__DAV_C310__)
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, float8_e4m3_t);
REGISTE_DMA_UB_COPY(ubuf, ubuf, 1, float8_e5m2_t);
#endif

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

#if defined(__DAV_C310__)
//===-------------------------------------------------------------------===//
// ub to cbuf, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, int8_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, uint8_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, int16_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, uint16_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, int32_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, uint32_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, float8_e4m3_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, float8_e5m2_t)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, half)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, float)
REGISTE_DMA_UB_COPY(ubuf, cbuf, 1, bfloat16_t)
#endif
}
