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

#include "Utils.h"
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/VecUtils.h"
#include <cstdint>

template <typename T>
__aiv__ __attribute__((always_inline)) void
vxor_vv_1d(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
           memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *tmp_buf) {
  static_assert(
      (std::is_same<bool, T>::value || std::is_same<int8_t, T>::value ||
       std::is_same<uint8_t, T>::value || std::is_same<int16_t, T>::value ||
       std::is_same<uint16_t, T>::value || std::is_same<int32_t, T>::value ||
       std::is_same<uint32_t, T>::value || std::is_same<int64_t, T>::value ||
       std::is_same<uint64_t, T>::value) &&
      "xor do not support this data type.");

  // A xor B = !(A & B) & (A | B)
  memref_t<__ubuf__ T, 1> tmp_buf_for_xor{tmp_buf->allocated,
                                          tmp_buf->aligned,
                                          tmp_buf->offset,
                                          {dst->sizes[0]},
                                          {1}};
  memref_t<__ubuf__ T, 1> tmp_buf_for_inline_brc{
      tmp_buf->allocated,
      tmp_buf->aligned,
      tmp_buf->offset + dst->sizes[0],
      {tmp_buf->sizes[0] - dst->sizes[0]},
      {1}};
  vector_eltwise_vv_1d<VectorOpTy::VAND, T>(src0, src1, &tmp_buf_for_xor,
                                            &tmp_buf_for_inline_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_1d<VectorOpTy::VNOT, T>(&tmp_buf_for_xor, &tmp_buf_for_xor,
                                           &tmp_buf_for_inline_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_1d<VectorOpTy::VOR, T>(src0, src1, dst,
                                           &tmp_buf_for_inline_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_1d<VectorOpTy::VAND, T>(&tmp_buf_for_xor, dst, dst,
                                            &tmp_buf_for_inline_brc);
}

/// Get the preprocessed mode
/// The following are possible transformations:
/// VV -> VV/VB/BV/BB
/// VS -> VS/VB/BS/BB
/// SV -> BV/BB
/// V  -> V/B
/// B  -> B
/// BS -> BB
/// SB -> BB
/// VB -> VB/BB
/// BV -> BV/BB
/// BB -> BB
template <VectorOpTy OP>
__aiv__ __attribute__((always_inline)) VectorLastAxisMode
get_preprocessed_mode(VectorLastAxisMode mode, bool is_src0_brc_last_dim,
                      bool is_src1_brc_last_dim) {
  if (mode == VectorLastAxisMode::V && is_src0_brc_last_dim) {
    return VectorLastAxisMode::B;
  }
  if ((mode == VectorLastAxisMode::SV && is_src1_brc_last_dim) ||
      (mode == VectorLastAxisMode::VS && is_src0_brc_last_dim &&
       !isHardwareSupportedVS<OP>()) ||
      (mode == VectorLastAxisMode::VV && is_src0_brc_last_dim &&
       is_src1_brc_last_dim) ||
      (mode == VectorLastAxisMode::BS && !isHardwareSupportedVS<OP>()) ||
      (mode == VectorLastAxisMode::SB) ||
      (mode == VectorLastAxisMode::VB && is_src0_brc_last_dim) ||
      (mode == VectorLastAxisMode::BV && is_src1_brc_last_dim)) {
    return VectorLastAxisMode::BB;
  }
  if (mode == VectorLastAxisMode::SV ||
      (mode == VectorLastAxisMode::VV && is_src0_brc_last_dim)) {
    return VectorLastAxisMode::BV;
  }
  if ((mode == VectorLastAxisMode::VS && !isHardwareSupportedVS<OP>()) ||
      (mode == VectorLastAxisMode::VV && is_src1_brc_last_dim)) {
    return VectorLastAxisMode::VB;
  }
  if (mode == VectorLastAxisMode::VS && is_src0_brc_last_dim &&
      isHardwareSupportedVS<OP>()) {
    return VectorLastAxisMode::BS;
  }
  return mode;
}

template <VectorOpTy OP, typename SRC_T, typename DST_T = SRC_T>
__aiv__ __attribute__((always_inline)) void normalize_vector_last_axis_1d(
    memref_t<__ubuf__ SRC_T, 1> *src0, memref_t<__ubuf__ SRC_T, 1> *src1,
    SRC_T scalar, memref_t<__ubuf__ SRC_T, 1> *dst,
    memref_t<__ubuf__ DST_T, 1> *new_src0,
    memref_t<__ubuf__ DST_T, 1> *new_src1, memref_t<__ubuf__ SRC_T, 1> *tmp_buf,
    VectorLastAxisMode *mode) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  constexpr int new_num_per_block = INTR_BYTES_PER_BLOCK / sizeof(DST_T);
  memref_t<__ubuf__ DST_T, 1> tmp_buf_as_dst_t;
  view_as<SRC_T, DST_T, 1>(tmp_buf, &tmp_buf_as_dst_t);

  int64_t base_tmp_buf_offset = tmp_buf->offset;
  // Preprocess src0
  bool is_src0_brc_dim0 = false;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  if (*mode == VectorLastAxisMode::SV) {
    brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
    INTRINSIC(pipe_barrier, PIPE_V);
    *new_src0 = {tmp_buf_as_dst_t.aligned,
                 tmp_buf_as_dst_t.allocated,
                 tmp_buf_as_dst_t.offset,
                 {new_num_per_block},
                 {1}};
    base_tmp_buf_offset = base_tmp_buf_offset + num_per_block;
    tmp_buf_as_dst_t.offset = tmp_buf_as_dst_t.offset + new_num_per_block;
  } else {
    is_src0_brc_dim0 = src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
    if (is_src0_brc_dim0) {
      auto src0_ptr = src0->aligned + src0->offset;
      INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
      SRC_T scalar = *src0_ptr;
      INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
      INTRINSIC(pipe_barrier, PIPE_V);
      *new_src0 = {tmp_buf_as_dst_t.aligned,
                   tmp_buf_as_dst_t.allocated,
                   tmp_buf_as_dst_t.offset,
                   {new_num_per_block},
                   {1}};
      base_tmp_buf_offset = base_tmp_buf_offset + num_per_block;
      tmp_buf_as_dst_t.offset = tmp_buf_as_dst_t.offset + new_num_per_block;
    }
  }

  // Preprocess src1
  tmp_buf_ptr = tmp_buf->aligned + base_tmp_buf_offset;
  bool is_src1_brc_dim0 = false;
  if (*mode != VectorLastAxisMode::V) {
    if (*mode == VectorLastAxisMode::VS) {
      if (!isHardwareSupportedVS<OP>()) {
        brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
        INTRINSIC(pipe_barrier, PIPE_V);
        *new_src1 = {tmp_buf_as_dst_t.aligned,
                     tmp_buf_as_dst_t.allocated,
                     tmp_buf_as_dst_t.offset,
                     {new_num_per_block},
                     {1}};
        tmp_buf_as_dst_t.offset = tmp_buf_as_dst_t.offset + new_num_per_block;
      }
    } else {
      is_src1_brc_dim0 = src1->sizes[0] != dst->sizes[0] && src1->sizes[0] == 1;
      if (is_src1_brc_dim0) {
        auto src1_ptr = src1->aligned + src1->offset;
        INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
        SRC_T scalar = *src1_ptr;
        INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
        INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
        brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
        INTRINSIC(pipe_barrier, PIPE_V);
        *new_src1 = {tmp_buf_as_dst_t.aligned,
                     tmp_buf_as_dst_t.allocated,
                     tmp_buf_as_dst_t.offset,
                     {new_num_per_block},
                     {1}};
        tmp_buf_as_dst_t.offset = tmp_buf_as_dst_t.offset + new_num_per_block;
      }
    }
  }

  // Set the converted mode
  *mode = get_preprocessed_mode<OP>(*mode, is_src0_brc_dim0, is_src1_brc_dim0);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void scalar_eltwise_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, int64_t size, VectorLastAxisMode mode, T scalar) {
  cce::printf("Warning: This implementation uses scalar instructions, which may result in suboptimal performance");
  auto src0_ptr = src0 == nullptr ? nullptr : src0->aligned + src0->offset;
  auto src1_ptr = src1 == nullptr ? nullptr : src1->aligned + src1->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t src0_stride0 = 0;
  int64_t src1_stride0 = 0;
  // In some cases, planmemory may cause dst to reuse the address of src.
  // If dst->offset > src->offset, the result will overwrite src, which causes precision failure.
  // Therefore, reverse calculation is required.
  bool reverse = false;
  if (mode == VectorLastAxisMode::VV){
    // if sizes==1, set stride=0 so that src_operand will always be that num.
    src0_stride0 = src0->sizes[0] == 1 ? 0 : src0->strides[0];
    src1_stride0 = src1->sizes[0] == 1 ? 0 : src1->strides[0];
    reverse = (dst->allocated == src0->allocated && dst->offset > src0->offset) ||
              (dst->allocated == src1->allocated && dst->offset > src1->offset);
  }
  if (mode == VectorLastAxisMode::VS || mode == VectorLastAxisMode::V) {
    src0_stride0 = src0->sizes[0] == 1 ? 0 : src0->strides[0];
    reverse = dst->allocated == src0->allocated && dst->offset > src0->offset;
  }
  if (mode == VectorLastAxisMode::SV) {
    src1_stride0 = src1->sizes[0] == 1 ? 0 : src1->strides[0];
    reverse = dst->allocated == src1->allocated && dst->offset > src1->offset;
  }
  for (int i = 0; i < size; ++i) {
    T src0_oprand = T();
    T src1_oprand = T();
    int index0 = reverse ? size - 1 - i : i;
    if (mode == VectorLastAxisMode::VV || mode == VectorLastAxisMode::VS || mode == VectorLastAxisMode::V) {
      src0_oprand = *(src0_ptr + index0 * src0_stride0);
    } else {
      src0_oprand = scalar;
    }
    if (mode == VectorLastAxisMode::VV || mode == VectorLastAxisMode::SV) {
      src1_oprand = *(src1_ptr + index0 * src1_stride0);
    } else {
      src1_oprand = scalar;
    }
    *(dst_ptr + index0 * dst->strides[0]) = handle_vector_operation<OP, T>(src0_oprand, src1_oprand, mode);
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vv_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 1>(src0, src1, dst);
  memref_t<__ubuf__ T, 1> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 1> aligned_src1 = *src1;
  memref_t<__ubuf__ T, 1> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num > dst->sizes[0]) {
      scalar_eltwise_1d<OP, T>(src0, src1, dst, dst->sizes[0], mode, {0});
      return;
    }
    scalar_eltwise_1d<OP, T>(src0, src1, dst, scalar_num, mode, {0});
    move_memref_to_aligned_1d(&aligned_src0, scalar_num);
    move_memref_to_aligned_1d(&aligned_src1, scalar_num);
    move_memref_to_aligned_1d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[0] <= 0) {
      return;
    }
  }
  vector_eltwise_vv_1d<OP, T>(&aligned_src0, &aligned_src1, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vs_1d(
    memref_t<__ubuf__ T, 1> *src0, T scalar, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 1>(src0, nullptr, dst);
  memref_t<__ubuf__ T, 1> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 1> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num > dst->sizes[0]) {
      scalar_eltwise_1d<OP, T>(src0, nullptr, dst, dst->sizes[0], mode, scalar);
      return;
    }
    scalar_eltwise_1d<OP, T>(src0, nullptr, dst, scalar_num, mode, scalar);
    move_memref_to_aligned_1d(&aligned_src0, scalar_num);
    move_memref_to_aligned_1d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[0] <= 0) {
      return;
    }
  }
  vector_eltwise_vs_1d<OP, T>(&aligned_src0, scalar, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_sv_1d(
    T scalar, memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 1>(nullptr, src1, dst);
  memref_t<__ubuf__ T, 1> aligned_src1 = *src1;
  memref_t<__ubuf__ T, 1> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num > dst->sizes[0]) {
      scalar_eltwise_1d<OP, T>(nullptr, src1, dst, dst->sizes[0], mode, scalar);
      return;
    }
    scalar_eltwise_1d<OP, T>(nullptr, src1, dst, scalar_num, mode, scalar);
    move_memref_to_aligned_1d(&aligned_src1, scalar_num);
    move_memref_to_aligned_1d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[0] <= 0) {
      return;
    }
  }
  vector_eltwise_sv_1d<OP, T>(scalar, &aligned_src1, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_v_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 1>(src0, nullptr, dst);
  memref_t<__ubuf__ T, 1> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 1> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num > dst->sizes[0]) {
      scalar_eltwise_1d<OP, T>(src0, nullptr, dst, dst->sizes[0], mode, {0});
      return;
    }
    scalar_eltwise_1d<OP, T>(src0, nullptr, dst, scalar_num, mode, {0});
    move_memref_to_aligned_1d(&aligned_src0, scalar_num);
    move_memref_to_aligned_1d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[0] <= 0) {
      return;
    }
  }
  vector_eltwise_v_1d<OP, T>(&aligned_src0, &aligned_dst, tmp_buf, mode);
}

/// Elementwise operation between two 1D tensors.
///
/// constraint:
/// 1. src0/src1/dst/tmp need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a,) operation src1:(a,) -> dst:(a,)
///    * scene2: src0:(1,) operation src1:(a,) -> dst:(a,)
///    * scene3: src0:(a,) operation src1:(1,) -> dst:(a,)
///    * scene4: src0:(1,) operation src1:(1,) -> dst:(a,)
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    OP = VXOR:
///        scene1: tmp_buf = a
///        scene2/scene3: tmp_buf = num_per_block
///        scene4: tmp_buf = num_per_block * 2
///    OP = others:
///        scene1: tmp_buf = 0
///        scene2/scene3: tmp_buf = num_per_block
///        scene4: tmp_buf = num_per_block * 2
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vv_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vv_1d(src0, src1, dst);
  if constexpr (OP == VectorOpTy::VXOR) {
    vxor_vv_1d(src0, src1, dst, tmp_buf);
    return;
  }

  // preprocess src tail axis brc inline scene.
  memref_t<__ubuf__ T, 1> new_src0 = *src0;
  memref_t<__ubuf__ T, 1> new_src1 = *src1;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  if constexpr (std::is_same<half, T>::value ||
                std::is_same<bfloat16_t, T>::value) {
    normalize_vector_last_axis_1d<OP, T, T>(src0, src1, {0} /* scalar */, dst,
                                            &new_src0, &new_src1, tmp_buf,
                                            &mode);
  } else {
    normalize_vector_last_axis_1d<OP, T, T>(
        src0, src1, 0 /* scalar */, dst, &new_src0, &new_src1, tmp_buf, &mode);
  }
#else
  normalize_vector_last_axis_1d<OP, T, T>(src0, src1, 0 /* scalar */, dst,
                                          &new_src0, &new_src1, tmp_buf, &mode);
#endif

  // Step2: Perform elementwise operations
  uint16_t src0_block_stride = 1;
  uint16_t src1_block_stride = 1;
  uint16_t src0_repeat_stride = 8;
  uint16_t src1_repeat_stride = 8;
  if (mode == VectorLastAxisMode::BV || mode == VectorLastAxisMode::BB) {
    src0_block_stride = 0;
    src0_repeat_stride = 0;
  }
  if (mode == VectorLastAxisMode::VB || mode == VectorLastAxisMode::BB) {
    src1_block_stride = 0;
    src1_repeat_stride = 0;
  }
  auto new_src0_ptr = new_src0.aligned + new_src0.offset;
  auto new_src1_ptr = new_src1.aligned + new_src1.offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = dst->sizes[0];
  INTRINSIC(set_vector_mask, 0, n);
  vector_eltwise_vv_intrin<OP, T>(
      intrin_args<2, T>{dst_ptr,
                        {new_src0_ptr, new_src1_ptr},
                        0,
                        1,
                        1,
                        {src0_block_stride, src1_block_stride},
                        8,
                        {src0_repeat_stride, src1_repeat_stride}});
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, bool>(memref_t<__ubuf__ bool, 1> *src0,
                                            memref_t<__ubuf__ bool, 1> *src1,
                                            memref_t<__ubuf__ bool, 1> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<bool, int16_t, 1>(src0, &src0_as_int16);
  view_as<bool, int16_t, 1>(src1, &src1_as_int16);
  view_as<bool, int16_t, 1>(dst, &dst_as_int16);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ int32_t, 1> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int32_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<int32_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, int32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, uint32_t>(
    memref_t<__ubuf__ uint32_t, 1> *src0, memref_t<__ubuf__ uint32_t, 1> *src1,
    memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint32_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<uint32_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, uint32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, int64_t>(
    memref_t<__ubuf__ int64_t, 1> *src0, memref_t<__ubuf__ int64_t, 1> *src1,
    memref_t<__ubuf__ int64_t, 1> *dst, memref_t<__ubuf__ int64_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int64_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<int64_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, int64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, uint64_t>(
    memref_t<__ubuf__ uint64_t, 1> *src0, memref_t<__ubuf__ uint64_t, 1> *src1,
    memref_t<__ubuf__ uint64_t, 1> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint64_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<uint64_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, uint64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, int8_t>(
    memref_t<__ubuf__ int8_t, 1> *src0, memref_t<__ubuf__ int8_t, 1> *src1,
    memref_t<__ubuf__ int8_t, 1> *dst, memref_t<__ubuf__ int8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int8_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<int8_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, int8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, uint8_t>(
    memref_t<__ubuf__ uint8_t, 1> *src0, memref_t<__ubuf__ uint8_t, 1> *src1,
    memref_t<__ubuf__ uint8_t, 1> *dst, memref_t<__ubuf__ uint8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint8_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<uint8_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, uint8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, half>(memref_t<__ubuf__ half, 1> *src0,
                                            memref_t<__ubuf__ half, 1> *src1,
                                            memref_t<__ubuf__ half, 1> *dst,
                                            memref_t<__ubuf__ half, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<half, int16_t, 1>(src0, &src0_as_int16);
  view_as<half, int16_t, 1>(src1, &src1_as_int16);
  view_as<half, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_1d<VectorOpTy::VOR, half, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_1d<VectorOpTy::VOR, half, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 1> *src0,
    memref_t<__ubuf__ bfloat16_t, 1> *src1,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<bfloat16_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<bfloat16_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_1d<VectorOpTy::VOR, bfloat16_t, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_1d<VectorOpTy::VOR, bfloat16_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VOR, float>(
    memref_t<__ubuf__ float, 1> *src0, memref_t<__ubuf__ float, 1> *src1,
    memref_t<__ubuf__ float, 1> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<float, int16_t, 1>(src0, &src0_as_int16);
  view_as<float, int16_t, 1>(src1, &src1_as_int16);
  view_as<float, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VOR, float, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, bool>(
    memref_t<__ubuf__ bool, 1> *src0, memref_t<__ubuf__ bool, 1> *src1,
    memref_t<__ubuf__ bool, 1> *dst, memref_t<__ubuf__ bool, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<bool, int16_t, 1>(src0, &src0_as_int16);
  view_as<bool, int16_t, 1>(src1, &src1_as_int16);
  view_as<bool, int16_t, 1>(dst, &dst_as_int16);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ int32_t, 1> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int32_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<int32_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, int32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, uint32_t>(
    memref_t<__ubuf__ uint32_t, 1> *src0, memref_t<__ubuf__ uint32_t, 1> *src1,
    memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint32_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<uint32_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, uint32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, int64_t>(
    memref_t<__ubuf__ int64_t, 1> *src0, memref_t<__ubuf__ int64_t, 1> *src1,
    memref_t<__ubuf__ int64_t, 1> *dst, memref_t<__ubuf__ int64_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int64_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<int64_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, int64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, uint64_t>(
    memref_t<__ubuf__ uint64_t, 1> *src0, memref_t<__ubuf__ uint64_t, 1> *src1,
    memref_t<__ubuf__ uint64_t, 1> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint64_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<uint64_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, uint64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, int8_t>(
    memref_t<__ubuf__ int8_t, 1> *src0, memref_t<__ubuf__ int8_t, 1> *src1,
    memref_t<__ubuf__ int8_t, 1> *dst, memref_t<__ubuf__ int8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int8_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<int8_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, int8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, uint8_t>(
    memref_t<__ubuf__ uint8_t, 1> *src0, memref_t<__ubuf__ uint8_t, 1> *src1,
    memref_t<__ubuf__ uint8_t, 1> *dst, memref_t<__ubuf__ uint8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint8_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<uint8_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, uint8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 1> *src0,
    memref_t<__ubuf__ bfloat16_t, 1> *src1,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<bfloat16_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 1>(src1, &src1_as_int16);
  view_as<bfloat16_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_1d<VectorOpTy::VAND, bfloat16_t, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_1d<VectorOpTy::VAND, bfloat16_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, half>(
    memref_t<__ubuf__ half, 1> *src0, memref_t<__ubuf__ half, 1> *src1,
    memref_t<__ubuf__ half, 1> *dst, memref_t<__ubuf__ half, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<half, int16_t, 1>(src0, &src0_as_int16);
  view_as<half, int16_t, 1>(src1, &src1_as_int16);
  view_as<half, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_1d<VectorOpTy::VAND, half, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_1d<VectorOpTy::VAND, half, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_1d<VectorOpTy::VAND, float>(
    memref_t<__ubuf__ float, 1> *src0, memref_t<__ubuf__ float, 1> *src1,
    memref_t<__ubuf__ float, 1> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> src1_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<float, int16_t, 1>(dst, &dst_as_int16);
  view_as<float, int16_t, 1>(src0, &src0_as_int16);
  view_as<float, int16_t, 1>(src1, &src1_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VAND, float, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_vv_1d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr, mode);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_vs_sv_1d(memref_t<__ubuf__ T, 1> *src,
                                        memref_t<__ubuf__ T, 1> *dst,
                                        memref_t<__ubuf__ T, 1> *tmp_buf) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src->aligned + src->offset;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_buf_ptr) &&
         "The starting address of tmp_buf must be 32byte aligned.");
  assert((src->strides[0] == 1 && dst->strides[0] == 1 &&
          tmp_buf->strides[0] == 1) &&
         "The src/dst/tmp_buf must be continuous vector.");
#endif
}

/// Elementwise operation between scalar and 1D tensors.
///
/// constraint:
/// 1. src1/dst/tmp need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: scalar operation src1:(a,) -> dst:(a,)
///    * scene2: scalar operation src1:(1,) -> dst:(a,)
/// 3. The tmp_buf size is as follows:
///    scene1: tmp_buf = num_per_block
///    scene2: tmp_buf = num_per_block * 2
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_sv_1d(
    T scalar, memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  static_assert((OP == VectorOpTy::VSUB || OP == VectorOpTy::VDIV) &&
                "unsupport op for sv to vv.");

  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_1d(src1, dst, tmp_buf);

  // There is no corresponding vsubs/vdivs instruction, so it can only be
  // implemented by converting it into vsub/vdiv instruction.
  // Step1: preprocess src tail axis brc inline and scalar scene
  memref_t<__ubuf__ T, 1> new_src0;
  memref_t<__ubuf__ T, 1> new_src1 = *src1;
  normalize_vector_last_axis_1d<OP, T, T>(nullptr /* src0 */, src1, scalar, dst,
                                          &new_src0, &new_src1, tmp_buf, &mode);

  // Step2: Perform elementwise operations
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, dst->sizes[0]);
  auto dst_ptr = dst->aligned + dst->offset;
  auto new_src0_ptr = new_src0.aligned + new_src0.offset;
  auto new_src1_ptr = new_src1.aligned + new_src1.offset;
  uint16_t src_block_stride = 1;
  uint16_t src_repeat_stride = 8;
  if (mode == VectorLastAxisMode::BB) {
    src_block_stride = 0;
    src_repeat_stride = 0;
  }
  vector_eltwise_vv_intrin<OP, T>(
      intrin_args<2, T>{dst_ptr,
                        {new_src0_ptr, new_src1_ptr},
                        0,
                        1,
                        1,
                        {0, src_block_stride},
                        8,
                        {0, src_repeat_stride}});

  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// Elementwise operation between 1D tensors and scalar.
///
/// constraint:
/// 1. src0/dst/tmp need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src1:(a,) operation scalar -> dst:(a,)
///    * scene2: src1:(1,) operation scalar -> dst:(a,)
/// 3. The tmp_buf size is as follows:
///    OP = VSUB/VDIV:
///        scene1: tmp_buf = num_per_block
///        scene2: tmp_buf = num_per_block * 2
///    OP = others:
///        scene1: tmp_buf = 0
///        scene2: tmp_buf = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vs_1d(
    memref_t<__ubuf__ T, 1> *src0, T scalar, memref_t<__ubuf__ T, 1> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_1d(src0, dst, tmp_buf);

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  uint16_t src_block_stride = 1;
  uint16_t src_repeat_stride = 8;

  if constexpr (isHardwareSupportedVS<OP>()) {
    // Step1: preprocess src tail axis brc inline scene
    memref_t<__ubuf__ T, 1> new_src0 = *src0;
    normalize_vector_last_axis_1d<OP, T, T>(
        src0, nullptr /* src1 */, scalar, dst, &new_src0,
        nullptr /* new src1 */, tmp_buf, &mode);

    // Step2: Perform elementwise operations
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, dst->sizes[0]);
    __ubuf__ T *new_src0_ptr = new_src0.aligned + new_src0.offset;
    if (mode == VectorLastAxisMode::BS) {
      src_block_stride = 0;
      src_repeat_stride = 0;
    }
    vector_eltwise_vs_intrin<OP, T>(intrin_args<1, T>{dst_ptr,
                                                      {new_src0_ptr},
                                                      scalar,
                                                      1,
                                                      1,
                                                      {src_block_stride},
                                                      8,
                                                      {src_repeat_stride}});
    INTRINSIC_NO_ARGS(set_mask_norm);
    return;
  }

  // There is no corresponding vsubs/vdivs instruction, so it can only be
  // implemented by converting it into vsub/vdiv instruction.
  // Step1: preprocess src tail axis brc inline and scalar scene
  memref_t<__ubuf__ T, 1> new_src0 = *src0;
  memref_t<__ubuf__ T, 1> new_src1;
  normalize_vector_last_axis_1d<OP, T, T>(src0, nullptr /* src1 */, scalar, dst,
                                          &new_src0, &new_src1, tmp_buf, &mode);

  // Step2: Perform elementwise operations
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, dst->sizes[0]);
  __ubuf__ T *new_src0_ptr = new_src0.aligned + new_src0.offset;
  __ubuf__ T *new_src1_ptr = new_src1.aligned + new_src1.offset;
  if (mode == VectorLastAxisMode::BB) {
    src_block_stride = 0;
    src_repeat_stride = 0;
  }
  vector_eltwise_vv_intrin<OP, T>(
      intrin_args<2, T>{dst_ptr,
                        {new_src0_ptr, new_src1_ptr},
                        0,
                        1,
                        1,
                        {src_block_stride, 0},
                        8,
                        {src_repeat_stride, 0}});
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// Unary Elementwise operation 1d
///
/// constraint:
/// 1. src0/dst/tmp need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a,) -> dst:(a,)
///    * scene2: src0:(1,) -> dst:(a,)
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1: tmp_buf = 0
///    scene2: tmp_buf = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_v_1d(src0, dst);

  // Step1: preprocess src tail axis brc inline scene.
  memref_t<__ubuf__ T, 1> new_src0 = *src0;
  if (mode == VectorLastAxisMode::V) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    if constexpr (std::is_same<half, T>::value ||
                  std::is_same<bfloat16_t, T>::value) {
      normalize_vector_last_axis_1d<OP, T, T>(
          src0, nullptr /* src1 */, {0} /* scalar */, dst, &new_src0,
          nullptr /* new src1 */, tmp_buf, &mode);
    } else {
      normalize_vector_last_axis_1d<OP, T, T>(
          src0, nullptr /* src1 */, 0 /* scalar */, dst, &new_src0,
          nullptr /* new src1 */, tmp_buf, &mode);
    }
#else
    normalize_vector_last_axis_1d<OP, T, T>(
        src0, nullptr /* src1 */, 0 /* scalar */, dst, &new_src0,
        nullptr /* new src1 */, tmp_buf, &mode);
#endif
  }

  // Step2: Perform elementwise operations
  uint16_t src_block_stride = 1;
  uint16_t src_repeat_stride = 8;
  if (mode == VectorLastAxisMode::B) {
    src_block_stride = 0;
    src_repeat_stride = 0;
  }
  auto dst_ptr = dst->aligned + dst->offset;
  auto new_src0_ptr = new_src0.aligned + new_src0.offset;
  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = dst->sizes[0];
  INTRINSIC(set_vector_mask, 0, n);
  vector_eltwise_v_intrin<OP, T>(intrin_args<1, T>{dst_ptr,
                                                   {new_src0_ptr},
                                                   0,
                                                   1,
                                                   1,
                                                   {src_block_stride},
                                                   8,
                                                   {src_repeat_stride}});
  INTRINSIC_NO_ARGS(set_mask_norm);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, bool>(memref_t<__ubuf__ bool, 1> *src0,
                                            memref_t<__ubuf__ bool, 1> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<bool, int16_t, 1>(src0, &src0_as_int16);
  view_as<bool, int16_t, 1>(dst, &dst_as_int16);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, int32_t>(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *dst,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int32_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, int32_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, uint32_t>(
    memref_t<__ubuf__ uint32_t, 1> *src0, memref_t<__ubuf__ uint32_t, 1> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint32_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, uint32_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, int64_t>(
    memref_t<__ubuf__ int64_t, 1> *src0, memref_t<__ubuf__ int64_t, 1> *dst,
    memref_t<__ubuf__ int64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int64_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, int64_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, uint64_t>(
    memref_t<__ubuf__ uint64_t, 1> *src0, memref_t<__ubuf__ uint64_t, 1> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint64_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, uint64_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, int8_t>(
    memref_t<__ubuf__ int8_t, 1> *src0, memref_t<__ubuf__ int8_t, 1> *dst,
    memref_t<__ubuf__ int8_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<int8_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, int8_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, uint8_t>(
    memref_t<__ubuf__ uint8_t, 1> *src0, memref_t<__ubuf__ uint8_t, 1> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<uint8_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, uint8_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 1> *src0,
    memref_t<__ubuf__ bfloat16_t, 1> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<bfloat16_t, int16_t, 1>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, bfloat16_t, int16_t>(
      src0, nullptr /* src1 */, {0} /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);
#else
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, bfloat16_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);
#endif

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, half>(memref_t<__ubuf__ half, 1> *src0,
                                            memref_t<__ubuf__ half, 1> *dst,
                                            memref_t<__ubuf__ half, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<half, int16_t, 1>(src0, &src0_as_int16);
  view_as<half, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, half, int16_t>(
      src0, nullptr /* src1 */, {0} /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);
#else
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, half, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);
#endif

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_1d<VectorOpTy::VNOT, float>(
    memref_t<__ubuf__ float, 1> *src0, memref_t<__ubuf__ float, 1> *dst,
    memref_t<__ubuf__ float, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 1> src0_as_int16;
  memref_t<__ubuf__ int16_t, 1> dst_as_int16;
  view_as<float, int16_t, 1>(src0, &src0_as_int16);
  view_as<float, int16_t, 1>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_1d<VectorOpTy::VNOT, float, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new src1 */, tmp_buf, &mode);

  vector_eltwise_v_1d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

extern "C" {
//===-------------------------------------------------------------------===//
// eltwise vv, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, int16_t)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, int32_t)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, half)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 1, float)

REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, int16_t)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, int32_t)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, half)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 1, float)

REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, int16_t)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, int32_t)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, half)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 1, float)

REGISTE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 1, half)
REGISTE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 1, float)

REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, int16_t)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, int32_t)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, half)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 1, float)

REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, int16_t)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, int32_t)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, half)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 1, float)

REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, bool)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int8_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint8_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, half)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, bfloat16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int32_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint32_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, float)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, int64_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 1, uint64_t)

REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, bool)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int8_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint8_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, half)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, bfloat16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int32_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint32_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, float)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, int64_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 1, uint64_t)

REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, bool)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, int8_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, uint8_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, int16_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, uint16_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, int32_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, uint32_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, int64_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 1, uint64_t)

//===-------------------------------------------------------------------===//
// eltwise sv, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, int16_t)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, int32_t)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, half)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 1, float)

REGISTE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 1, half)
REGISTE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 1, float)

//===-------------------------------------------------------------------===//
// eltwise vs, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, int16_t)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, int32_t)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, half)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 1, float)

REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, int16_t)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, int32_t)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, half)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 1, float)

REGISTE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 1, half)
REGISTE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 1, float)

REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, int16_t)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, int32_t)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, half)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 1, float)

REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, int16_t)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, int32_t)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, half)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 1, float)

REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, int16_t)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, int32_t)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, half)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 1, float)

REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, int16_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, uint16_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, int32_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 1, uint32_t)

REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, int16_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, uint16_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, int32_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 1, uint32_t)

//===-------------------------------------------------------------------===//
// eltwise unary, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_V(vabs, VectorOpTy::VABS, 1, half)
REGISTE_ELTWISE_V(vabs, VectorOpTy::VABS, 1, float)

REGISTE_ELTWISE_V(vln, VectorOpTy::VLN, 1, half)
REGISTE_ELTWISE_V(vln, VectorOpTy::VLN, 1, float)

REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 1, half)
REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 1, float)
REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 1, int32_t)

REGISTE_ELTWISE_V(vexp, VectorOpTy::VEXP, 1, half)
REGISTE_ELTWISE_V(vexp, VectorOpTy::VEXP, 1, float)

REGISTE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 1, half)
REGISTE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 1, float)

REGISTE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 1, half)
REGISTE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 1, float)

REGISTE_ELTWISE_V(vrec, VectorOpTy::VREC, 1, half)
REGISTE_ELTWISE_V(vrec, VectorOpTy::VREC, 1, float)

REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, bool)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int8_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint8_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, half)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, bfloat16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int32_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint32_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, float)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, int64_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 1, uint64_t)
}