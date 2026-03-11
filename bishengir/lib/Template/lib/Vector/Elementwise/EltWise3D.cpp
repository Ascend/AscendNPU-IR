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

/// Optimization scenario of elementwise vv is to deal (a, 1, c) $ (a, b, c)
/// operation, and '$' means binary vector operations, such as add, sub.
/// Constraints:
/// 1. src0, src1 and dst must be aligned, which means src0/src1/dst->stride[2]
/// % num_per_block is 0, and num_per_block means element number of UB block
/// unit.
/// 2. 'c' must be num per block.
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d_opt(memref_t<__ubuf__ T, 3> *src0,
                         memref_t<__ubuf__ T, 3> *src1,
                         memref_t<__ubuf__ T, 3> *dst) {
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  const uint64_t main_cnt = dst->sizes[1] / INTR_BLOCKS_PER_REPEAT;
  const uint64_t tail_cnt = dst->sizes[1] % INTR_BLOCKS_PER_REPEAT;
  bool is_mid_brc = src0->sizes[1] != src1->sizes[1];
  bool is_src0_brc = is_mid_brc && (src0->sizes[1] == 1);
  bool is_src1_brc = is_mid_brc && (src1->sizes[1] == 1);

  // intrin args
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  __ubuf__ T *src1_ptr = src1->aligned + src1->offset;
  const uint64_t repeat = dst->sizes[0];
  const uint16_t src0_block_stride = is_src0_brc ? 0 : 1;
  const uint16_t src1_block_stride = is_src1_brc ? 0 : 1;
  const uint16_t dst_repeat_stride = dst->strides[0] / num_per_block;
  const uint16_t src0_repeat_stride = src0->strides[0] / num_per_block;
  const uint16_t src1_repeat_stride = src1->strides[0] / num_per_block;

  if (main_cnt > 0) {
    SetMaskValueByCount(num_per_repeat);
    // set choose repeat axis and compute repeat number
    const uint64_t size1_repeat_cnt = dst->sizes[1] / INTR_BLOCKS_PER_REPEAT;
    bool is_axis0_repeat = dst->sizes[0] >= size1_repeat_cnt;
    uint64_t repeat_num = is_axis0_repeat ? repeat : size1_repeat_cnt;
    int64_t loop_num = is_axis0_repeat ? size1_repeat_cnt : repeat;

    // set loop strides
    uint64_t src0_dim0_repeat_stride =
        is_src0_brc ? num_per_block : dst->strides[0];
    uint64_t src0_dim1_repeat_stride = is_src0_brc ? 0 : num_per_repeat;

    uint64_t src1_dim0_repeat_stride =
        is_src1_brc ? num_per_block : dst->strides[0];
    uint64_t src1_dim1_repeat_stride = is_src1_brc ? 0 : num_per_repeat;

    uint64_t dst_loop_stride =
        is_axis0_repeat ? num_per_repeat : dst->strides[0];
    uint64_t src0_loop_stride =
        is_axis0_repeat ? src0_dim1_repeat_stride : src0_dim0_repeat_stride;
    uint64_t src1_loop_stride =
        is_axis0_repeat ? src1_dim1_repeat_stride : src1_dim0_repeat_stride;
    // set repeat strides
    uint16_t new_dst_repeat_stride =
        is_axis0_repeat ? dst_repeat_stride : INTR_BLOCKS_PER_REPEAT;
    uint16_t new_src0_repeat_stride =
        is_axis0_repeat ? src0_repeat_stride
                        : (is_src0_brc ? 0 : INTR_BLOCKS_PER_REPEAT);
    uint16_t new_src1_repeat_stride =
        is_axis0_repeat ? src1_repeat_stride
                        : (is_src1_brc ? 0 : INTR_BLOCKS_PER_REPEAT);

    for (int64_t loop = 0; loop < loop_num; ++loop) {
      vector_eltwise_2d_intrin<OP, T>({dst_ptr + loop * dst_loop_stride,
                                       {src0_ptr + loop * src0_loop_stride,
                                        src1_ptr + loop * src1_loop_stride},
                                       0,          // scalar
                                       repeat_num, // repeat
                                       1,          // dst_block_stride
                                       src0_block_stride,
                                       src1_block_stride,
                                       new_dst_repeat_stride,
                                       new_src0_repeat_stride,
                                       new_src1_repeat_stride});
    }
  }

  if (tail_cnt > 0) {
    SetMaskValueByCount(tail_cnt * num_per_block);
    int64_t dst_offset = main_cnt * num_per_repeat;
    int64_t src0_offset = is_src0_brc ? 0 : dst_offset;
    int64_t src1_offset = is_src1_brc ? 0 : dst_offset;
    vector_eltwise_2d_intrin<OP, T>(
        {dst_ptr + dst_offset,
         {src0_ptr + src0_offset, src1_ptr + src1_offset},
         0,      // scalar
         repeat, // repeat
         1,      // dst_block_stride
         src0_block_stride,
         src1_block_stride,
         dst_repeat_stride,
         src0_repeat_stride,
         src1_repeat_stride});
  }
}

template <VectorOpTy OP, typename SRC_T, typename DST_T = SRC_T>
__aiv__ __attribute__((always_inline)) void unnormalized_vector_eltwise_vv_3d(
    memref_t<__ubuf__ SRC_T, 3> *src0, memref_t<__ubuf__ SRC_T, 3> *src1,
    memref_t<__ubuf__ SRC_T, 3> *dst, memref_t<__ubuf__ SRC_T, 1> *tmp_buf,
    VectorLastAxisMode *mode) {
  memref_t<__ubuf__ SRC_T, 2> src0_2d;
  memref_t<__ubuf__ SRC_T, 2> src1_2d;
  memref_t<__ubuf__ SRC_T, 2> dst_2d;
  memref_t<__ubuf__ DST_T, 2> new_src0_2d;
  memref_t<__ubuf__ DST_T, 2> new_src1_2d;
  memref_t<__ubuf__ DST_T, 2> new_dst_2d;
  bool is_src0_brc_dim0 =
      src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim0 =
      src1->sizes[0] != dst->sizes[0] && src1->sizes[0] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;

  int64_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  int64_t src1_stride0 = is_src1_brc_dim0 ? 0 : src1->strides[0];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    memref_t<__ubuf__ SRC_T, 2> src0_2d{src0->aligned,
                                        src0->allocated,
                                        src0->offset + i * src0_stride0,
                                        {src0->sizes[1], src0->sizes[2]},
                                        {src0->strides[1], src0->strides[2]}};
    memref_t<__ubuf__ SRC_T, 2> src1_2d{src1->aligned,
                                        src1->allocated,
                                        src1->offset + i * src1_stride0,
                                        {src1->sizes[1], src1->sizes[2]},
                                        {src1->strides[1], src1->strides[2]}};
    memref_t<__ubuf__ SRC_T, 2> dst_2d{dst->aligned,
                                       dst->allocated,
                                       dst->offset + i * dst->strides[0],
                                       {dst->sizes[1], dst->sizes[2]},
                                       {dst->strides[1], dst->strides[2]}};
    view_as<SRC_T, DST_T, 2>(&src0_2d, &new_src0_2d);
    view_as<SRC_T, DST_T, 2>(&src1_2d, &new_src1_2d);
    view_as<SRC_T, DST_T, 2>(&dst_2d, &new_dst_2d);
    int64_t tmp_buf_size = 0;
    if (is_src0_brc_dim2) {
      vector_last_axis_brc_2d<SRC_T, DST_T>(&src0_2d, &new_src0_2d, tmp_buf);
      tmp_buf_size +=
          src0_2d.sizes[0] == 1
              ? num_per_block
              : CEIL_FACTOR(src0_2d.sizes[0], kSrcNumPerRepeatOfVBRCB) *
                    num_per_block;
      *mode = VectorLastAxisMode::BV;
    }
    if (is_src1_brc_dim2) {
      memref_t<__ubuf__ SRC_T, 1> new_tmp_buf{
          tmp_buf->aligned,
          tmp_buf->allocated,
          tmp_buf->offset + tmp_buf_size,
          {tmp_buf->sizes[0] - tmp_buf_size},
          {1}};
      vector_last_axis_brc_2d<SRC_T, DST_T>(&src1_2d, &new_src1_2d,
                                            &new_tmp_buf);
      *mode = (*mode == VectorLastAxisMode::BV) ? VectorLastAxisMode::BB
                                                : VectorLastAxisMode::VB;
    }
    vector_eltwise_vv_2d<OP, DST_T>(&new_src0_2d, &new_src1_2d, &new_dst_2d,
                                    nullptr, *mode);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_vv_3d(memref_t<__ubuf__ T, 3> *src0,
                                     memref_t<__ubuf__ T, 3> *src1,
                                     memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  auto src1_ptr = src1->aligned + src1->offset;
  assert(
      (isAddress32ByteAligned(src0_ptr) && isAddress32ByteAligned(src1_ptr)) &&
      "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src0->strides[0]) &&
          isSizeAlignedToBlock<T>(src0->strides[1]) &&
          isSizeAlignedToBlock<T>(src1->strides[0]) &&
          isSizeAlignedToBlock<T>(src1->strides[1]) &&
          isSizeAlignedToBlock<T>(dst->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[1])) &&
         "The src/dst strides[0]/strides[1] must be aligned to block.");
  assert((src0->strides[2] == 1 && src1->strides[2] == 1 &&
          dst->strides[2] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void scalar_eltwise_3d(
    memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *src1,
    memref_t<__ubuf__ T, 3> *dst, int size0, int size1, int size2,
    VectorLastAxisMode mode, T scalar) {
  cce::printf("Warning: This implementation uses scalar instructions, which may result in suboptimal performance");
  auto src0_ptr = src0 == nullptr ? nullptr : src0->aligned + src0->offset;
  auto src1_ptr = src1 == nullptr ? nullptr : src1->aligned + src1->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  int64_t src0_stride0 = 0;
  int64_t src1_stride0 = 0;
  int64_t src0_stride1 = 0;
  int64_t src1_stride1 = 0;
  int64_t src0_stride2 = 0;
  int64_t src1_stride2 = 0;

  // In some cases, planmemory may cause dst to reuse the address of src. 
  // If dst->offset > src->offset, the result will overwrite src, which causes precision failure.
  // Therefore, reverse calculation is required.
  bool reverse = false;
  if (mode == VectorLastAxisMode::VV){
    src0_stride0 = src0->sizes[0] == 1 ? 0 : src0->strides[0];
    src1_stride0 = src1->sizes[0] == 1 ? 0 : src1->strides[0];
    src0_stride1 = src0->sizes[1] == 1 ? 0 : src0->strides[1];
    src1_stride1 = src1->sizes[1] == 1 ? 0 : src1->strides[1];
    src0_stride2 = src0->sizes[2] == 1 ? 0 : src0->strides[2];
    src1_stride2 = src1->sizes[2] == 1 ? 0 : src1->strides[2];
    reverse = (dst->allocated == src0->allocated && dst->offset > src0->offset) ||
              (dst->allocated == src1->allocated && dst->offset > src1->offset);
  }
  if (mode == VectorLastAxisMode::VS || mode == VectorLastAxisMode::V) {
    src0_stride0 = src0->sizes[0] == 1 ? 0 : src0->strides[0];
    src0_stride1 = src0->sizes[1] == 1 ? 0 : src0->strides[1];
    src0_stride2 = src0->sizes[2] == 1 ? 0 : src0->strides[2];
    reverse = dst->allocated == src0->allocated && dst->offset > src0->offset;
  }
  if (mode == VectorLastAxisMode::SV) {
    src1_stride0 = src1->sizes[0] == 1 ? 0 : src1->strides[0];
    src1_stride1 = src1->sizes[1] == 1 ? 0 : src1->strides[1];
    src1_stride2 = src1->sizes[2] == 1 ? 0 : src1->strides[2];
    reverse = dst->allocated == src1->allocated && dst->offset > src1->offset;
  }

  for (int i = 0; i < size0; ++i) {
    int index0 = reverse ? size0 -1 - i : i;
    for (int j = 0; j < size1; ++j) {
      int index1 = reverse ? size1 -1 - j : j;
      for (int k = 0; k < size2; ++k) {
        int index2 = reverse ? size2 -1 - k : k;
        T src0_oprand;
        T src1_oprand;
        if (mode == VectorLastAxisMode::VV) {
          src0_oprand = *(src0_ptr + index0 * src0_stride0 + index1 * src0_stride1 + index2 * src0_stride2);
          src1_oprand = *(src1_ptr + index0 * src1_stride0 + index1 * src1_stride1 + index2 * src1_stride2);
        }

        if (mode == VectorLastAxisMode::VS) {
          src0_oprand = *(src0_ptr + index0 * src0_stride0 + index1 * src0_stride1 + index2 * src0_stride2);
          src1_oprand = scalar;
        }

        if (mode == VectorLastAxisMode::SV) {
          src0_oprand = scalar;
          src1_oprand = *(src1_ptr + index0 * src1_stride0 + index1 * src1_stride1 + index2 * src1_stride2);
        }

        if (mode == VectorLastAxisMode::V) {
          src0_oprand = *(src0_ptr + index0 * src0_stride0 + index1 * src0_stride1 + index2 * src0_stride2);
          src1_oprand = T();
        }

        *(dst_ptr + index0 * dst->strides[0] + index1 * dst->strides[1] + index2 * dst->strides[2]) =
          handle_vector_operation<OP, T>(src0_oprand, src1_oprand, mode);
      }
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vv_3d(
    memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *src1,
    memref_t<__ubuf__ T, 3> *dst, memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 3>(src0, src1, dst);
  memref_t<__ubuf__ T, 3> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 3> aligned_src1 = *src1;
  memref_t<__ubuf__ T, 3> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[2]) {
      scalar_eltwise_3d<OP, T>(src0, src1, dst, dst->sizes[0], dst->sizes[1], dst->sizes[2], mode, {0});
      return;
    }
    scalar_eltwise_3d<OP, T>(src0, src1, dst, dst->sizes[0], dst->sizes[1], scalar_num, mode, {0});
    move_memref_to_aligned_3d(&aligned_src0, scalar_num);
    move_memref_to_aligned_3d(&aligned_src1, scalar_num);
    move_memref_to_aligned_3d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[2] <= 0) {
      return;
    }
  }
  vector_eltwise_vv_3d<OP, T>(&aligned_src0, &aligned_src1, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vs_3d(
    memref_t<__ubuf__ T, 3> *src0, T scalar, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 3>(src0, nullptr, dst);
  memref_t<__ubuf__ T, 3> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 3> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[2]) {
      scalar_eltwise_3d<OP, T>(src0, nullptr, dst, dst->sizes[0], dst->sizes[1], dst->sizes[2], mode, scalar);
      return;
    }
    scalar_eltwise_3d<OP, T>(src0, nullptr, dst, dst->sizes[0], dst->sizes[1], scalar_num, mode, scalar);
    move_memref_to_aligned_3d(&aligned_src0, scalar_num);
    move_memref_to_aligned_3d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[2] <= 0) {
      return;
    }
  }
  vector_eltwise_vs_3d<OP, T>(&aligned_src0, scalar, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_sv_3d(
    T scalar, memref_t<__ubuf__ T, 3> *src1, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 3>(nullptr, src1, dst);
  memref_t<__ubuf__ T, 3> aligned_src1 = *src1;
  memref_t<__ubuf__ T, 3> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[2]) {
      scalar_eltwise_3d<OP, T>(nullptr, src1, dst, dst->sizes[0], dst->sizes[1], dst->sizes[2], mode, scalar);
      return;
    }
    scalar_eltwise_3d<OP, T>(nullptr, src1, dst, dst->sizes[0], dst->sizes[1], scalar_num, mode, scalar);
    move_memref_to_aligned_3d(&aligned_src1, scalar_num);
    move_memref_to_aligned_3d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[2] <= 0) {
      return;
    }
  }
  vector_eltwise_sv_3d<OP, T>(scalar, &aligned_src1, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_v_3d(
    memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 3>(src0, nullptr, dst);
  memref_t<__ubuf__ T, 3> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 3> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[2]) {
      scalar_eltwise_3d<OP, T>(src0, nullptr, dst, dst->sizes[0], dst->sizes[1], dst->sizes[2], mode, {0});
      return;
    }
    scalar_eltwise_3d<OP, T>(src0, nullptr, dst, dst->sizes[0], dst->sizes[1], scalar_num, mode, {0});
    move_memref_to_aligned_3d(&aligned_src0, scalar_num);
    move_memref_to_aligned_3d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[2] <= 0) {
      return;
    }
  }
  vector_eltwise_v_3d<OP, T>(&aligned_src0, &aligned_dst, tmp_buf, mode);
}
/// Elementwise operation between two 3D tensors.
///
/// constraint:
/// 1. src0/src1/dst strides[0]/strides[1] need to block alignment strides[2]
///    need to be continuous, tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a, b, c) $ src1:(a, b, c) -> dst:(a, b, c)
///    * scene2: src0:(a, b, c) $ src1:(d, e, f) -> dst:(g, h, i)
///        a != g || d != g:
///            g = max(a, d)
///            min(a, d) = 1
///        b != h || e != h:
///            h = max(b, e)
///            min(b, e) = 1
///        c != i || f != i:
///            i = max(c, f)
///            min(c, f) = 1
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1: tmp_buf = 0
///    scene2:
///        (c != 1 && f != 1) || i == 1: no last axis inline brc
///            tmp_buf = 0
///        c == 1 && f != 1: src0 last axis inline brc
///            b == 1:
///                tmp_buf = num_per_block
///            b != 1:
///                tmp_buf = num_per_block * aligned(b, kSrcNumPerRepeatOfVBRCB)
///        c != 1 && f == 1: src1 last axis inline brc
///            e == 1:
///                tmp_buf = num_per_block
///            e != 1:
///                tmp_buf = num_per_block * aligned(e, kSrcNumPerRepeatOfVBRCB)
///        c == 1 && f == 1 && i != 1: src0 and src1 last axis inline brc
///            b == 1:
///                e == 1:
///                    tmp_buf = num_per_block * 2
///                e != 1:
///                    tmp_buf = num_per_block +
///                        num_per_block * aligned(e, kSrcNumPerRepeatOfVBRCB)
///            b != 1:
///                e == 1:
///                    tmp_buf = num_per_block +
///                        num_per_block * aligned(b, kSrcNumPerRepeatOfVBRCB)
///                e != 1:
///                    tmp_buf =
///                        num_per_block * aligned(e, kSrcNumPerRepeatOfVBRCB) +
///                        num_per_block * aligned(b, kSrcNumPerRepeatOfVBRCB)
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vv_3d(
    memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *src1,
    memref_t<__ubuf__ T, 3> *dst, memref_t<__ubuf__ T, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vv_3d(src0, src1, dst);

  INTRINSIC_NO_ARGS(set_mask_norm);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  bool is_src0_brc_dim0 =
      src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim0 =
      src1->sizes[0] != dst->sizes[0] && src1->sizes[0] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;

  bool is_first_brc = is_src0_brc_dim0 || is_src1_brc_dim0;
  bool is_last_brc = is_src0_brc_dim2 || is_src1_brc_dim2;
  if (dst->strides[1] == num_per_block && dst->sizes[2] == num_per_block &&
      dst->sizes[0] != 1 && !is_first_brc && !is_last_brc)
    [[unlikely]] {
      vector_eltwise_vv_3d_opt<OP, T>(src0, src1, dst);
      return;
    }

  if (is_last_brc) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<OP, T>(src0, src1, dst, tmp_buf, &mode);
    return;
  }

  int64_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  int64_t src1_stride0 = is_src1_brc_dim0 ? 0 : src1->strides[0];
  // Deal vv 3d, throw dst_sizes[0] as loop.
  memref_t<__ubuf__ T, 2> src0_2d;
  memref_t<__ubuf__ T, 2> src1_2d;
  memref_t<__ubuf__ T, 2> dst_2d;
  throw_Nd_to_Md_args<T, 3, 2>(src0, &src0_2d);
  throw_Nd_to_Md_args<T, 3, 2>(src1, &src1_2d);
  throw_Nd_to_Md_args<T, 3, 2>(dst, &dst_2d);
  for (int64_t loop_for_throw_axis = 0; loop_for_throw_axis < dst->sizes[0];
       ++loop_for_throw_axis) {
    vector_eltwise_vv_2d<OP, T>(&src0_2d, &src1_2d, &dst_2d, tmp_buf);
    src0_2d.offset = src0_2d.offset + src0_stride0;
    src1_2d.offset = src1_2d.offset + src1_stride0;
    dst_2d.offset = dst_2d.offset + dst->strides[0];
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, bool>(
    memref_t<__ubuf__ bool, 3> *src0, memref_t<__ubuf__ bool, 3> *src1,
    memref_t<__ubuf__ bool, 3> *dst, memref_t<__ubuf__ bool, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // vand instr convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<bool, int16_t, 3>(src0, &src0_as_int16);
  view_as<bool, int16_t, 3>(src1, &src1_as_int16);
  view_as<bool, int16_t, 3>(dst, &dst_as_int16);

  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, int8_t>(
    memref_t<__ubuf__ int8_t, 3> *src0, memref_t<__ubuf__ int8_t, 3> *src1,
    memref_t<__ubuf__ int8_t, 3> *dst, memref_t<__ubuf__ int8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, int8_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<int8_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *src1,
    memref_t<__ubuf__ uint8_t, 3> *dst, memref_t<__ubuf__ uint8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, uint8_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, half>(
    memref_t<__ubuf__ half, 3> *src0, memref_t<__ubuf__ half, 3> *src1,
    memref_t<__ubuf__ half, 3> *dst, memref_t<__ubuf__ half, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, half, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<half, int16_t, 3>(src0, &src0_as_int16);
  view_as<half, int16_t, 3>(src1, &src1_as_int16);
  view_as<half, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 3> *src0,
    memref_t<__ubuf__ bfloat16_t, 3> *src1,
    memref_t<__ubuf__ bfloat16_t, 3> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, bfloat16_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<bfloat16_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<bfloat16_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, float>(
    memref_t<__ubuf__ float, 3> *src0, memref_t<__ubuf__ float, 3> *src1,
    memref_t<__ubuf__ float, 3> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, float, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<float, int16_t, 3>(src0, &src0_as_int16);
  view_as<float, int16_t, 3>(src1, &src1_as_int16);
  view_as<float, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, int32_t>(
    memref_t<__ubuf__ int32_t, 3> *src0, memref_t<__ubuf__ int32_t, 3> *src1,
    memref_t<__ubuf__ int32_t, 3> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, int32_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int32_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<int32_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, uint32_t>(
    memref_t<__ubuf__ uint32_t, 3> *src0, memref_t<__ubuf__ uint32_t, 3> *src1,
    memref_t<__ubuf__ uint32_t, 3> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, uint32_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint32_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<uint32_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, int64_t>(
    memref_t<__ubuf__ int64_t, 3> *src0, memref_t<__ubuf__ int64_t, 3> *src1,
    memref_t<__ubuf__ int64_t, 3> *dst, memref_t<__ubuf__ int64_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, int64_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int64_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<int64_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VAND, uint64_t>(
    memref_t<__ubuf__ uint64_t, 3> *src0, memref_t<__ubuf__ uint64_t, 3> *src1,
    memref_t<__ubuf__ uint64_t, 3> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VAND, uint64_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vand instr convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint64_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<uint64_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VAND, int16_t>(
      &src0_as_int16, &src1_as_int16, &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, bool>(memref_t<__ubuf__ bool, 3> *src0,
                                            memref_t<__ubuf__ bool, 3> *src1,
                                            memref_t<__ubuf__ bool, 3> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // vor instr convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<bool, int16_t, 3>(src0, &src0_as_int16);
  view_as<bool, int16_t, 3>(src1, &src1_as_int16);
  view_as<bool, int16_t, 3>(dst, &dst_as_int16);

  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, int8_t>(
    memref_t<__ubuf__ int8_t, 3> *src0, memref_t<__ubuf__ int8_t, 3> *src1,
    memref_t<__ubuf__ int8_t, 3> *dst, memref_t<__ubuf__ int8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, int8_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<int8_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *src1,
    memref_t<__ubuf__ uint8_t, 3> *dst, memref_t<__ubuf__ uint8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, uint8_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, half>(memref_t<__ubuf__ half, 3> *src0,
                                            memref_t<__ubuf__ half, 3> *src1,
                                            memref_t<__ubuf__ half, 3> *dst,
                                            memref_t<__ubuf__ half, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, half, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<half, int16_t, 3>(src0, &src0_as_int16);
  view_as<half, int16_t, 3>(src1, &src1_as_int16);
  view_as<half, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 3> *src0,
    memref_t<__ubuf__ bfloat16_t, 3> *src1,
    memref_t<__ubuf__ bfloat16_t, 3> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, bfloat16_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<bfloat16_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<bfloat16_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, float>(
    memref_t<__ubuf__ float, 3> *src0, memref_t<__ubuf__ float, 3> *src1,
    memref_t<__ubuf__ float, 3> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, float, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<float, int16_t, 3>(src0, &src0_as_int16);
  view_as<float, int16_t, 3>(src1, &src1_as_int16);
  view_as<float, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, int32_t>(
    memref_t<__ubuf__ int32_t, 3> *src0, memref_t<__ubuf__ int32_t, 3> *src1,
    memref_t<__ubuf__ int32_t, 3> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, int32_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int32_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<int32_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, uint32_t>(
    memref_t<__ubuf__ uint32_t, 3> *src0, memref_t<__ubuf__ uint32_t, 3> *src1,
    memref_t<__ubuf__ uint32_t, 3> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, uint32_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint32_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<uint32_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, int64_t>(
    memref_t<__ubuf__ int64_t, 3> *src0, memref_t<__ubuf__ int64_t, 3> *src1,
    memref_t<__ubuf__ int64_t, 3> *dst, memref_t<__ubuf__ int64_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, int64_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int64_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<int64_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_3d<VectorOpTy::VOR, uint64_t>(
    memref_t<__ubuf__ uint64_t, 3> *src0, memref_t<__ubuf__ uint64_t, 3> *src1,
    memref_t<__ubuf__ uint64_t, 3> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src0_brc_dim2 || is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vv_3d<VectorOpTy::VOR, uint64_t, int16_t>(
        src0, src1, dst, tmp_buf, &mode);
    return;
  }

  // vor instr convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> src1_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint64_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 3>(src1, &src1_as_int16);
  view_as<uint64_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_vv_3d<VectorOpTy::VOR, int16_t>(&src0_as_int16, &src1_as_int16,
                                                 &dst_as_int16, nullptr);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_vs_sv_3d(memref_t<__ubuf__ T, 3> *src0,
                                        memref_t<__ubuf__ T, 3> *dst,
                                        memref_t<__ubuf__ T, 1> *tmp_buf) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isAddress32ByteAligned(tmp_buf_ptr) &&
         "The starting address of tmp_buf must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src0->strides[0]) &&
          isSizeAlignedToBlock<T>(src0->strides[1]) &&
          isSizeAlignedToBlock<T>(dst->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[1])) &&
         "The src/dst strides[0]/strides[1] must be aligned to block.");
  assert((src0->strides[2] == 1 && dst->strides[2] == 1 &&
          tmp_buf->strides[0] == 1) &&
         "The src/dst/tmp_buf last dim must be continuous.");
#endif
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void unnormalized_vector_eltwise_vs_3d(
    memref_t<__ubuf__ T, 3> *src0, T scalar, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode *mode) {
  memref_t<__ubuf__ T, 2> src0_2d;
  memref_t<__ubuf__ T, 2> dst_2d;

  *mode = VectorLastAxisMode::BS;
  bool is_src0_brc_dim0 =
      src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
  int64_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    memref_t<__ubuf__ T, 2> src0_2d{src0->aligned,
                                    src0->allocated,
                                    src0->offset + i * src0_stride0,
                                    {src0->sizes[1], src0->sizes[2]},
                                    {src0->strides[1], src0->strides[2]}};
    memref_t<__ubuf__ T, 2> dst_2d{dst->aligned,
                                   dst->allocated,
                                   dst->offset + i * dst->strides[0],
                                   {dst->sizes[1], dst->sizes[2]},
                                   {dst->strides[1], dst->strides[2]}};
    vector_last_axis_brc_2d<T, T>(&src0_2d, &src0_2d, tmp_buf);
    auto tmp_buf_size =
        src0_2d.sizes[0] == 1
            ? src0_2d.sizes[1]
            : CEIL_FACTOR(src0_2d.sizes[0], kSrcNumPerRepeatOfVBRCB) *
                  src0_2d.sizes[1];
    memref_t<__ubuf__ T, 1> new_tmp_buf{tmp_buf->aligned,
                                        tmp_buf->allocated,
                                        tmp_buf->offset + tmp_buf_size,
                                        {tmp_buf->sizes[0] - tmp_buf_size},
                                        {1}};
    vector_eltwise_vs_2d<OP, T>(&src0_2d, scalar, &dst_2d, &new_tmp_buf, *mode);
  }
}

/// Elementwise operation between 3D tensors and scalar.
///
/// constraint:
/// 1. src0/dst strides[0]/strides[1] need to block alignment strides[1] need to
///    be continuous, tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a, b, c) operation scalar -> dst:(a, b, c)
///    * scene2: src0:(a, b, c) operation scalar -> dst:(d, e, f)
///        a == 1 or a == d
///        b == 1 or b == e
///        c == 1 or c == f
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1: tmp_buf = num_per_block
///    scene2:
///        c == 1 && f != 1: src0 last axis brc inline
///            b == 1:
///                tmp_buf = num_per_block * 2
///            b != 1:
///                tmp_buf = num_per_block +
///                    num_per_block * aligned(b, kSrcNumPerRepeatOfVBRCB)
///        others: tmp_buf = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vs_3d(
    memref_t<__ubuf__ T, 3> *src0, T scalar, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_3d(src0, dst, tmp_buf);

  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_vs_3d<OP, T>(src0, scalar, dst, tmp_buf, &mode);
    return;
  }

  bool is_src0_brc_dim0 =
      src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
  int64_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  memref_t<__ubuf__ T, 2> src0_2d;
  memref_t<__ubuf__ T, 2> dst_2d;
  throw_Nd_to_Md_args<T, 3, 2>(src0, &src0_2d);
  throw_Nd_to_Md_args<T, 3, 2>(dst, &dst_2d);
  for (int64_t loop_for_throw_axis = 0; loop_for_throw_axis < dst->sizes[0];
       ++loop_for_throw_axis) {
    vector_eltwise_vs_2d<OP, T>(&src0_2d, scalar, &dst_2d, tmp_buf);
    src0_2d.offset = src0_2d.offset + src0_stride0;
    dst_2d.offset = dst_2d.offset + dst->strides[0];
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void unnormalized_vector_eltwise_sv_3d(
    T scalar, memref_t<__ubuf__ T, 3> *src1, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode *mode) {
  memref_t<__ubuf__ T, 2> src1_2d;
  memref_t<__ubuf__ T, 2> dst_2d;

  *mode = VectorLastAxisMode::SB;
  bool is_src1_brc_dim0 =
      src1->sizes[0] != dst->sizes[0] && src1->sizes[0] == 1;
  int64_t src1_stride0 = is_src1_brc_dim0 ? 0 : src1->strides[0];
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    memref_t<__ubuf__ T, 2> src1_2d{src1->aligned,
                                    src1->allocated,
                                    src1->offset + i * src1_stride0,
                                    {src1->sizes[1], src1->sizes[2]},
                                    {src1->strides[1], src1->strides[2]}};
    memref_t<__ubuf__ T, 2> dst_2d{dst->aligned,
                                   dst->allocated,
                                   dst->offset + i * dst->strides[0],
                                   {dst->sizes[1], dst->sizes[2]},
                                   {dst->strides[1], dst->strides[2]}};
    vector_last_axis_brc_2d<T, T>(&src1_2d, &src1_2d, tmp_buf);
    auto tmp_buf_size =
        src1_2d.sizes[0] == 1
            ? src1_2d.sizes[1]
            : CEIL_FACTOR(src1_2d.sizes[0], kSrcNumPerRepeatOfVBRCB) *
                  src1_2d.sizes[1];
    memref_t<__ubuf__ T, 1> new_tmp_buf{tmp_buf->aligned,
                                        tmp_buf->allocated,
                                        tmp_buf->offset + tmp_buf_size,
                                        {tmp_buf->sizes[0] - tmp_buf_size},
                                        {1}};
    vector_eltwise_sv_2d<OP, T>(scalar, &src1_2d, &dst_2d, &new_tmp_buf, *mode);
  }
}

/// Elementwise operation between scalar and 3D tensors.
///
/// constraint:
/// 1. src1/dst strides[0]/strides[1] need to block alignment strides[1] need to
///    be continuous, tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: scalar operation src1:(a, b, c) -> dst:(a, b, c)
///    * scene2: scalar operation src1:(a, b, c) -> dst:(d, e, f)
///        a == 1 or a == d
///        b == 1 or b == e
///        c == 1 or c == f
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1: tmp_buf = num_per_block
///    scene2:
///        c == 1 && f != 1: src1 last axis inline brc
///            b == 1:
///                tmp_buf = num_per_block * 2
///            b != 1:
///                tmp_buf = num_per_block +
///                    num_per_block * aligned(b, kSrcNumPerRepeatOfVBRCB)
///        others: tmp_buf = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_sv_3d(
    T scalar, memref_t<__ubuf__ T, 3> *src1, memref_t<__ubuf__ T, 3> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_3d(src1, dst, tmp_buf);

  bool is_src1_brc_dim2 =
      src1->sizes[2] != dst->sizes[2] && src1->sizes[2] == 1;
  if (is_src1_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_sv_3d<OP, T>(scalar, src1, dst, tmp_buf, &mode);
    return;
  }

  bool is_src1_brc_dim0 =
      src1->sizes[0] != dst->sizes[0] && src1->sizes[0] == 1;
  int64_t src1_stride0 = is_src1_brc_dim0 ? 0 : src1->strides[0];
  memref_t<__ubuf__ T, 2> src1_2d;
  memref_t<__ubuf__ T, 2> dst_2d;
  throw_Nd_to_Md_args<T, 3, 2>(src1, &src1_2d);
  throw_Nd_to_Md_args<T, 3, 2>(dst, &dst_2d);
  for (int64_t loop_for_throw_axis = 0; loop_for_throw_axis < dst->sizes[0];
       ++loop_for_throw_axis) {
    vector_eltwise_sv_2d<OP, T>(scalar, &src1_2d, &dst_2d, tmp_buf, mode);
    src1_2d.offset = src1_2d.offset + src1_stride0;
    dst_2d.offset = dst_2d.offset + dst->strides[0];
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_v_3d(memref_t<__ubuf__ T, 3> *src0,
                                    memref_t<__ubuf__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src0_ptr = src0->aligned + src0->offset;
  assert(isAddress32ByteAligned(src0_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src0->strides[0]) &&
          isSizeAlignedToBlock<T>(src0->strides[1]) &&
          isSizeAlignedToBlock<T>(dst->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[1])) &&
         "The src/dst strides[0]/strides[1] must be aligned to block.");
  assert((src0->strides[2] == 1 && dst->strides[2] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

template <VectorOpTy OP, typename SRC_T, typename DST_T = SRC_T>
__aiv__ __attribute__((always_inline)) void unnormalized_vector_eltwise_v_3d(
    memref_t<__ubuf__ SRC_T, 3> *src0, memref_t<__ubuf__ SRC_T, 3> *dst,
    memref_t<__ubuf__ SRC_T, 1> *tmp_buf, VectorLastAxisMode *mode) {
  memref_t<__ubuf__ SRC_T, 2> src0_2d;
  memref_t<__ubuf__ SRC_T, 2> dst_2d;
  memref_t<__ubuf__ DST_T, 2> new_src0_2d;
  memref_t<__ubuf__ DST_T, 2> new_dst_2d;

  *mode = VectorLastAxisMode::B;
  bool is_src0_brc_dim0 =
      src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
  int64_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  for (int64_t i = 0; i < dst->sizes[0]; ++i) {
    memref_t<__ubuf__ SRC_T, 2> src0_2d{src0->aligned,
                                        src0->allocated,
                                        src0->offset + i * src0_stride0,
                                        {src0->sizes[1], src0->sizes[2]},
                                        {src0->strides[1], src0->strides[2]}};
    memref_t<__ubuf__ SRC_T, 2> dst_2d{dst->aligned,
                                       dst->allocated,
                                       dst->offset + i * dst->strides[0],
                                       {dst->sizes[1], dst->sizes[2]},
                                       {dst->strides[1], dst->strides[2]}};
    view_as<SRC_T, DST_T, 2>(&src0_2d, &new_src0_2d);
    view_as<SRC_T, DST_T, 2>(&dst_2d, &new_dst_2d);
    vector_last_axis_brc_2d<SRC_T, DST_T>(&src0_2d, &new_src0_2d, tmp_buf);
    vector_eltwise_v_2d<OP, DST_T>(&new_src0_2d, &new_dst_2d, nullptr, *mode);
  }
}

/// Unary Elementwise operation 3d
///
/// constraint:
/// 1. src0/dst strides[0]/strides[1] need to block alignment strides[2] need to
///    be continuous, tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a, b, c) -> dst:(a, b, c)
///    * scene2: src0:(a, b, c) -> dst:(d, e, f)
///        a == 1 or a == d
///        b == 1 or b == e
///        c == 1 or c == f
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1: tmp_buf = 0
///    scene2:
///        c == 1 && f != 1: src0 last axis inline brc
///            b == 1:
///                tmp_buf = num_per_block
///            b != 1:
///                tmp_buf = num_per_block * aligned(b, kSrcNumPerRepeatOfVBRCB)
///        others: tmp_buf = 0
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d(memref_t<__ubuf__ T, 3> *src0, memref_t<__ubuf__ T, 3> *dst,
                    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_v_3d(src0, dst);

  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<OP, T>(src0, dst, tmp_buf, &mode);
    return;
  }

  bool is_src0_brc_dim0 =
      src0->sizes[0] != dst->sizes[0] && src0->sizes[0] == 1;
  int64_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  memref_t<__ubuf__ T, 2> src0_2d;
  memref_t<__ubuf__ T, 2> dst_2d;
  throw_Nd_to_Md_args<T, 3, 2>(src0, &src0_2d);
  throw_Nd_to_Md_args<T, 3, 2>(dst, &dst_2d);
  for (int64_t loop_for_throw_axis = 0; loop_for_throw_axis < dst->sizes[0];
       ++loop_for_throw_axis) {
    vector_eltwise_v_2d<OP, T>(&src0_2d, &dst_2d);
    src0_2d.offset = src0_2d.offset + src0_stride0;
    dst_2d.offset = dst_2d.offset + dst->strides[0];
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, bool>(memref_t<__ubuf__ bool, 3> *src0,
                                            memref_t<__ubuf__ bool, 3> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // vnot instr convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<bool, int16_t, 3>(src0, &src0_as_int16);
  view_as<bool, int16_t, 3>(dst, &dst_as_int16);

  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, int8_t>(
    memref_t<__ubuf__ int8_t, 3> *src0, memref_t<__ubuf__ int8_t, 3> *dst,
    memref_t<__ubuf__ int8_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, int8_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, uint8_t>(
    memref_t<__ubuf__ uint8_t, 3> *src0, memref_t<__ubuf__ uint8_t, 3> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, uint8_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint8_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, half>(memref_t<__ubuf__ half, 3> *src0,
                                            memref_t<__ubuf__ half, 3> *dst,
                                            memref_t<__ubuf__ half, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, half, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<half, int16_t, 3>(src0, &src0_as_int16);
  view_as<half, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 3> *src0,
    memref_t<__ubuf__ bfloat16_t, 3> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, bfloat16_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<bfloat16_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, float>(
    memref_t<__ubuf__ float, 3> *src0, memref_t<__ubuf__ float, 3> *dst,
    memref_t<__ubuf__ float, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, float, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<float, int16_t, 3>(src0, &src0_as_int16);
  view_as<float, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, int32_t>(
    memref_t<__ubuf__ int32_t, 3> *src0, memref_t<__ubuf__ int32_t, 3> *dst,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, int32_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int32_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, uint32_t>(
    memref_t<__ubuf__ uint32_t, 3> *src0, memref_t<__ubuf__ uint32_t, 3> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, uint32_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<uint32_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, int64_t>(
    memref_t<__ubuf__ int64_t, 3> *src0, memref_t<__ubuf__ int64_t, 3> *dst,
    memref_t<__ubuf__ int64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, int64_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  view_as<int64_t, int16_t, 3>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 3>(dst, &dst_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_3d<VectorOpTy::VNOT, uint64_t>(
    memref_t<__ubuf__ uint64_t, 3> *src0, memref_t<__ubuf__ uint64_t, 3> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  bool is_src0_brc_dim2 =
      src0->sizes[2] != dst->sizes[2] && src0->sizes[2] == 1;
  if (is_src0_brc_dim2) {
    // Special processing src last axis brc inline scene.
    unnormalized_vector_eltwise_v_3d<VectorOpTy::VNOT, uint64_t, int16_t>(
        src0, dst, tmp_buf, &mode);
    return;
  }

  // vnot instr convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 3> dst_as_int16;
  memref_t<__ubuf__ int16_t, 3> src0_as_int16;
  view_as<uint64_t, int16_t, 3>(dst, &dst_as_int16);
  view_as<uint64_t, int16_t, 3>(src0, &src0_as_int16);
  vector_eltwise_v_3d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16,
                                                 nullptr, mode);
}

extern "C" {
//===-------------------------------------------------------------------===//
// eltwise vv, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, int16_t)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, int32_t)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, half)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 3, float)

REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, int16_t)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, int32_t)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, half)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 3, float)

REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, int16_t)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, int32_t)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, half)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 3, float)

REGISTE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 3, half)
REGISTE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 3, float)

REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, int16_t)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, int32_t)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, half)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 3, float)

REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, int16_t)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, int32_t)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, half)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 3, float)

REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, bool)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int8_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint8_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, half)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, bfloat16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, float)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int32_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint32_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, int64_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 3, uint64_t)

REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, bool)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int8_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint8_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, half)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, bfloat16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, float)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int32_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint32_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, int64_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 3, uint64_t)

//===-------------------------------------------------------------------===//
// eltwise sv, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, int16_t)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, int32_t)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, half)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 3, float)

REGISTE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 3, half)
REGISTE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 3, float)

//===-------------------------------------------------------------------===//
// eltwise vs, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, int16_t)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, int32_t)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, half)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 3, float)

REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, int16_t)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, int32_t)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, half)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 3, float)

REGISTE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 3, half)
REGISTE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 3, float)

REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, int16_t)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, int32_t)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, half)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 3, float)

REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, int16_t)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, int32_t)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, half)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 3, float)

REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, int16_t)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, int32_t)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, half)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 3, float)

REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, int16_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, uint16_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, int32_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 3, uint32_t)

REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, int16_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, uint16_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, int32_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 3, uint32_t)

//===-------------------------------------------------------------------===//
// eltwise unary, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_V(vabs, VectorOpTy::VABS, 3, half)
REGISTE_ELTWISE_V(vabs, VectorOpTy::VABS, 3, float)

REGISTE_ELTWISE_V(vln, VectorOpTy::VLN, 3, half)
REGISTE_ELTWISE_V(vln, VectorOpTy::VLN, 3, float)

REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 3, half)
REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 3, float)
REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 3, int32_t)

REGISTE_ELTWISE_V(vexp, VectorOpTy::VEXP, 3, half)
REGISTE_ELTWISE_V(vexp, VectorOpTy::VEXP, 3, float)

REGISTE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 3, half)
REGISTE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 3, float)

REGISTE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 3, half)
REGISTE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 3, float)

REGISTE_ELTWISE_V(vrec, VectorOpTy::VREC, 3, half)
REGISTE_ELTWISE_V(vrec, VectorOpTy::VREC, 3, float)

REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, bool)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int8_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint8_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, half)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, bfloat16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, float)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int32_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint32_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, int64_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 3, uint64_t)
}