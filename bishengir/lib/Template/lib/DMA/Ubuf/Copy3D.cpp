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

template <typename T>
__aiv__ __attribute__((always_inline)) void load_gm_to_ubuf_3d_core(
    memref_t<__gm__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst, PadMode pad_mode,
    typename PadValueType<T>::type pad_value, int64_t left_padding_num,
    EvictionPolicy eviction_policy,
    AtomicKind atomic_kind = AtomicKind::None) {
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
    load_gm_to_ubuf_3d_core_with_contiguous_last_dim(src, dst,
                                                     left_padding_num, l2_cache_ctl);
    return;
  }

  int64_t stride0_ub = dst->strides[0];
  int64_t stride1_ub = dst->strides[1];
  int64_t stride2_ub = dst->strides[2];
  int64_t stride0_gm = src->strides[0];
  int64_t stride1_gm = src->strides[1];
  int64_t stride2_gm = src->strides[2];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

#if defined(__DAV_C310__)
  if (((stride0_gm < stride1_gm || stride1_gm < stride2_gm) ||
      (stride0_ub < stride1_ub || stride1_ub < stride2_ub))) {
    // Implicit transposition scenarios need to be moved through scalar
    load_gm_to_ubuf_3d_by_nddma<T>(src, dst);
    return;
  }
#else
  if (((stride0_gm < stride1_gm || stride1_gm < stride2_gm) ||
       (stride0_ub < stride1_ub || stride1_ub < stride2_ub)) &&
      (stride0_ub % num_per_block != 0 || stride1_ub % num_per_block != 0 ||
       stride2_ub % num_per_block != 0)) {
    load_gm_to_ubuf_3d_by_scalar(src, dst);
    return;
  }
#endif  

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

template <typename T>
__aiv__ __attribute__((always_inline)) void
#if !defined(__DAV_C310__)
store_ubuf_to_gm_3d_core(memref_t<__ubuf__ T, 3> *src,
                         memref_t<__gm__ T, 3> *dst, AtomicKind atomic_kind,
                         PadMode pad_mode = PadMode::Null,
                         T pad_value = set_pad_value_null<T>()) {
#else
store_ubuf_to_gm_3d_core(memref_t<__ubuf__ T, 3> *src,
                         memref_t<__gm__ T, 3> *dst, AtomicKind atomic_kind) {
#endif
  if (is_no_op<3>(src->sizes)) {
    return;
  }

  // Input parameter constraints assert.
  check_inputs_of_store_ubuf_to_gm_3d_core(src, dst);

#if !defined(__DAV_C310__)
  if (pad_mode == PadMode::Value) {
    INTRINSIC(set_mov_pad_val, *((uint64_t *)((T *)(&pad_value))));
  } else if (pad_mode == PadMode::Null) {
    INTRINSIC(set_mov_pad_val, 0);
  }
#else
  INTRINSIC(set_mov_pad_val, 0);
#endif

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

/// core func of copy ub <-> ub, 3d
/// constraints:
/// 1. stride2 must be 1
/// 2. stride1 must be aligned to ub_block_unit
/// 3. stride0 must be aligned to ub_block_unit
/// 4. size1 must be aligned to ub_block_unit
/// 5. size2 must be aligned to ub_block_unit
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

  copy_ubuf_to_ubuf_3d_core_with_contiguous_last_dim(src, dst);
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
///TODO: update for constraints on alignment 
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