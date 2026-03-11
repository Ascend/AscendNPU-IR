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

#include "CPUDebug/CPUDebugUtils.h"
#include "Utils.h"
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/VecUtils.h"
#include <cstdint>
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void
vxor_vv_2d(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
           memref_t<__ubuf__ T, 2> *dst, memref_t<__ubuf__ T, 1> *tmp_buf) {
  static_assert(
      (std::is_same<bool, T>::value || std::is_same<int8_t, T>::value ||
       std::is_same<uint8_t, T>::value || std::is_same<int16_t, T>::value ||
       std::is_same<uint16_t, T>::value || std::is_same<int32_t, T>::value ||
       std::is_same<uint32_t, T>::value || std::is_same<int64_t, T>::value ||
       std::is_same<uint64_t, T>::value) &&
      "xor do not support this data type.");

  // A xor B = !(A & B) & (A | B)
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  auto dst_size0 = dst->sizes[0];
  auto dst_size1 = dst->sizes[1];
  auto dst_size1_align_block = CEIL_FACTOR(dst_size1, num_per_block);
  memref_t<__ubuf__ T, 2> tmp_buf_for_xor{tmp_buf->allocated,
                                          tmp_buf->aligned,
                                          tmp_buf->offset,
                                          {dst_size0, dst_size1},
                                          {dst_size1_align_block, 1}};
  memref_t<__ubuf__ T, 1> tmp_buf_for_inline_brc{
      tmp_buf->allocated,
      tmp_buf->aligned,
      tmp_buf->offset + dst_size0 * dst_size1_align_block,
      {tmp_buf->sizes[0] - dst_size0 * dst_size1_align_block},
      {1}};
  vector_eltwise_vv_2d<VectorOpTy::VAND, T>(src0, src1, &tmp_buf_for_xor,
                                            &tmp_buf_for_inline_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_v_2d<VectorOpTy::VNOT, T>(&tmp_buf_for_xor, &tmp_buf_for_xor,
                                           &tmp_buf_for_inline_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_2d<VectorOpTy::VOR, T>(src0, src1, dst,
                                           &tmp_buf_for_inline_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  vector_eltwise_vv_2d<VectorOpTy::VAND, T>(&tmp_buf_for_xor, dst, dst,
                                            &tmp_buf_for_inline_brc);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_2d_intrin_main(intrin_args<2, T> args) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  int64_t loop_num = args.repeat / INTR_MAX_REPEAT_CNTS;
  constexpr uint64_t repeat_num = INTR_MAX_REPEAT_CNTS;
  const uint16_t dst_block_stride = args.dst_block_stride;
  const uint16_t src0_block_stride = args.src_block_stride[0];
  const uint16_t dst_repeat_stride = args.dst_repeat_stride;
  const uint16_t src0_repeat_stride = args.src_repeat_stride[0];
  for (int64_t i = 0; i < loop_num; i++) {
    __ubuf__ T *dst_addr = args.dst + i * INTR_MAX_REPEAT_CNTS *
                                          args.dst_repeat_stride *
                                          num_per_block;
    __ubuf__ T *src0_addr = args.src[0] + i * INTR_MAX_REPEAT_CNTS *
                                              args.src_repeat_stride[0] *
                                              num_per_block;

    if constexpr (OP > VectorOpTy::BINARY_VV_BEGIN &&
                  OP < VectorOpTy::BINARY_VV_END) {
      __ubuf__ T *src1_addr = args.src[1] + i * INTR_MAX_REPEAT_CNTS *
                                                args.src_repeat_stride[1] *
                                                num_per_block;
      const uint16_t src1_block_stride = args.src_block_stride[1];
      const uint16_t src1_repeat_stride = args.src_repeat_stride[1];
      vector_eltwise_vv_intrin<OP, T>(
          intrin_args<2, T>{dst_addr,
                            {src0_addr, src1_addr},
                            0, /* scalar */
                            repeat_num,
                            dst_block_stride,
                            {src0_block_stride, src1_block_stride},
                            dst_repeat_stride,
                            {src0_repeat_stride, src1_repeat_stride}});
    } else if constexpr (OP > VectorOpTy::BINARY_VS_BEGIN &&
                         OP < VectorOpTy::BINARY_VS_END) {
      vector_eltwise_vs_intrin<OP, T>(intrin_args<1, T>{dst_addr,
                                                        {src0_addr},
                                                        args.scalar,
                                                        repeat_num,
                                                        dst_block_stride,
                                                        {src0_block_stride},
                                                        dst_repeat_stride,
                                                        {src0_repeat_stride}});
    } else if constexpr (OP > VectorOpTy::UNARY_BEGIN &&
                         OP < VectorOpTy::UNARY_END) {
      vector_eltwise_v_intrin<OP, T>(intrin_args<1, T>{dst_addr,
                                                       {src0_addr},
                                                       0, /* scalar */
                                                       repeat_num,
                                                       dst_block_stride,
                                                       {src0_block_stride},
                                                       dst_repeat_stride,
                                                       {src0_repeat_stride}});
    }
  }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_2d_intrin(intrin_args<2, T> args) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  if (args.repeat >= INTR_MAX_REPEAT_CNTS)
    [[unlikely]] { vector_eltwise_2d_intrin_main<OP, T>(args); }

  if (args.repeat % INTR_MAX_REPEAT_CNTS != 0)
    [[likely]] {
      __ubuf__ T *dst_addr =
          args.dst + FLOOR_FACTOR(args.repeat, INTR_MAX_REPEAT_CNTS) *
                         args.dst_repeat_stride * num_per_block;
      __ubuf__ T *src0_addr =
          args.src[0] + FLOOR_FACTOR(args.repeat, INTR_MAX_REPEAT_CNTS) *
                            args.src_repeat_stride[0] * num_per_block;
      const uint64_t repeat_num = args.repeat % INTR_MAX_REPEAT_CNTS;
      const uint16_t dst_block_stride = args.dst_block_stride;
      const uint16_t src0_block_stride = args.src_block_stride[0];
      const uint16_t dst_repeat_stride = args.dst_repeat_stride;
      const uint16_t src0_repeat_stride = args.src_repeat_stride[0];

      if constexpr (OP > VectorOpTy::BINARY_VV_BEGIN &&
                    OP < VectorOpTy::BINARY_VV_END) {
        __ubuf__ T *src1_addr =
            args.src[1] + FLOOR_FACTOR(args.repeat, INTR_MAX_REPEAT_CNTS) *
                              args.src_repeat_stride[1] * num_per_block;
        const uint16_t src1_block_stride = args.src_block_stride[1];
        const uint16_t src1_repeat_stride = args.src_repeat_stride[1];
        vector_eltwise_vv_intrin<OP, T>(
            intrin_args<2, T>{dst_addr,
                              {src0_addr, src1_addr},
                              0, /* scalar */
                              repeat_num,
                              dst_block_stride,
                              {src0_block_stride, src1_block_stride},
                              dst_repeat_stride,
                              {src0_repeat_stride, src1_repeat_stride}});
      } else if constexpr (OP > VectorOpTy::BINARY_VS_BEGIN &&
                           OP < VectorOpTy::BINARY_VS_END) {
        vector_eltwise_vs_intrin<OP, T>(
            intrin_args<1, T>{dst_addr,
                              {src0_addr},
                              args.scalar,
                              repeat_num,
                              dst_block_stride,
                              {src0_block_stride},
                              dst_repeat_stride,
                              {src0_repeat_stride}});
      } else if constexpr (OP > VectorOpTy::UNARY_BEGIN &&
                           OP < VectorOpTy::UNARY_END) {
        vector_eltwise_v_intrin<OP, T>(intrin_args<1, T>{dst_addr,
                                                         {src0_addr},
                                                         0, /* scalar */
                                                         repeat_num,
                                                         dst_block_stride,
                                                         {src0_block_stride},
                                                         dst_repeat_stride,
                                                         {src0_repeat_stride}});
      }
    }
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void scalar_eltwise_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
    memref_t<__ubuf__ T, 2> *dst, int64_t size0, int64_t size1,
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
  // In some cases, planmemory may cause dst to reuse the address of src. 
  // If dst->offset > src->offset, the result will overwrite src, which causes precision failure.
  // Therefore, reverse calculation is required.
  bool reverse = false;
  if (mode == VectorLastAxisMode::VV){
    src0_stride0 = src0->sizes[0] == 1 ? 0 : src0->strides[0];
    src1_stride0 = src1->sizes[0] == 1 ? 0 : src1->strides[0];
    src0_stride1 = src0->sizes[1] == 1 ? 0 : src0->strides[1];
    src1_stride1 = src1->sizes[1] == 1 ? 0 : src1->strides[1];
    reverse = (dst->allocated == src0->allocated && dst->offset > src0->offset) ||
              (dst->allocated == src1->allocated && dst->offset > src1->offset);
  }
  if (mode == VectorLastAxisMode::VS || mode == VectorLastAxisMode::V) {
    src0_stride0 = src0->sizes[0] == 1 ? 0 : src0->strides[0];
    src0_stride1 = src0->sizes[1] == 1 ? 0 : src0->strides[1];
    reverse = dst->allocated == src0->allocated && dst->offset > src0->offset;
  }
  if (mode == VectorLastAxisMode::SV) {
    src1_stride0 = src1->sizes[0] == 1 ? 0 : src1->strides[0];
    src1_stride1 = src1->sizes[1] == 1 ? 0 : src1->strides[1];
    reverse = dst->allocated == src1->allocated && dst->offset > src1->offset;
  }

  for (int i = 0; i < size0; ++i) {
    int index0 = reverse ? size0 -1 - i : i;
    for (int j = 0; j < size1; ++j) {
      int index1 = reverse ? size1 -1 - j : j;
      T src0_oprand = T();
      T src1_oprand = T();
      if (mode == VectorLastAxisMode::VV || mode == VectorLastAxisMode::VS || mode == VectorLastAxisMode::V) {
        src0_oprand = *(src0_ptr + index0 * src0_stride0 + index1 * src0_stride1);
      } else {
        src0_oprand = scalar;
      }
      if (mode == VectorLastAxisMode::VV || mode == VectorLastAxisMode::SV) {
        src1_oprand = *(src1_ptr + index0 * src1_stride0 + index1 * src1_stride1);
      } else {
        src1_oprand = scalar;
      }
      *(dst_ptr + index0 * dst->strides[0] + index1 * dst->strides[1]) =
        handle_vector_operation<OP, T>(src0_oprand, src1_oprand, mode);
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vv_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
    memref_t<__ubuf__ T, 2> *dst, memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 2>(src0, src1, dst);
  memref_t<__ubuf__ T, 2> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 2> aligned_src1 = *src1;
  memref_t<__ubuf__ T, 2> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[1]) {
      scalar_eltwise_2d<OP, T>(src0, src1, dst, dst->sizes[0], dst->sizes[1], mode, {0});
      return;
    }
    scalar_eltwise_2d<OP, T>(src0, src1, dst, dst->sizes[0], scalar_num, mode, {0});
    move_memref_to_aligned_2d(&aligned_src0, scalar_num);
    move_memref_to_aligned_2d(&aligned_src1, scalar_num);
    move_memref_to_aligned_2d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[1] <= 0) {
      return;
    }
  }
  vector_eltwise_vv_2d<OP, T>(&aligned_src0, &aligned_src1, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_vs_2d(
    memref_t<__ubuf__ T, 2> *src0, T scalar, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 2>(src0, nullptr, dst);
  memref_t<__ubuf__ T, 2> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 2> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[1]) {
      scalar_eltwise_2d<OP, T>(src0, nullptr, dst, dst->sizes[0], dst->sizes[1], mode, scalar);
      return;
    }
    scalar_eltwise_2d<OP, T>(src0, nullptr, dst, dst->sizes[0], scalar_num, mode, scalar);
    move_memref_to_aligned_2d(&aligned_src0, scalar_num);
    move_memref_to_aligned_2d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[1] <= 0) {
      return;
    }
  }
  vector_eltwise_vs_2d<OP, T>(&aligned_src0, scalar, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_sv_2d(
    T scalar, memref_t<__ubuf__ T, 2> *src1,
    memref_t<__ubuf__ T, 2> *dst, memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 2>(nullptr, src1, dst);
  memref_t<__ubuf__ T, 2> aligned_src1 = *src1;
  memref_t<__ubuf__ T, 2> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[1]) {
      scalar_eltwise_2d<OP, T>(nullptr, src1, dst, dst->sizes[0], dst->sizes[1], mode, scalar);
      return;
    }
    scalar_eltwise_2d<OP, T>(nullptr, src1, dst, dst->sizes[0], scalar_num, mode, scalar);
    move_memref_to_aligned_2d(&aligned_src1, scalar_num);
    move_memref_to_aligned_2d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[1] <= 0) {
      return;
    }
  }
  vector_eltwise_sv_2d<OP, T>(scalar, &aligned_src1, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void eltwise_v_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  auto scalar_num = eltwise_get_element_nums_on_scalar<T, 2>(src0, nullptr, dst);
  memref_t<__ubuf__ T, 2> aligned_src0 = *src0;
  memref_t<__ubuf__ T, 2> aligned_dst = *dst;
  if (scalar_num != 0)[[unlikely]] {
    if (scalar_num >= dst->sizes[1]) {
      scalar_eltwise_2d<OP, T>(src0, nullptr, dst, dst->sizes[0], dst->sizes[1], mode, {0});
      return;
    }
    scalar_eltwise_2d<OP, T>(src0, nullptr, dst, dst->sizes[0], scalar_num, mode, {0});
    move_memref_to_aligned_2d(&aligned_src0, scalar_num);
    move_memref_to_aligned_2d(&aligned_dst, scalar_num);
    if(aligned_dst.sizes[1] <= 0) {
      return;
    }
  }
  vector_eltwise_v_2d<OP, T>(&aligned_src0, &aligned_dst, tmp_buf, mode);
}

template <VectorOpTy OP, typename T, size_t OPERANUM>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_2d(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
                  T scalar, memref_t<__ubuf__ T, 2> *dst,
                  VectorLastAxisMode mode) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert((mode != VectorLastAxisMode::SV && mode != VectorLastAxisMode::SB) &&
         "unsupport VectorLastAxisMode for vector_eltwise_2d.");
#endif

  const int64_t dst_size0 = dst->sizes[0];
  const int64_t dst_size1 = dst->sizes[1];
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  constexpr int num_per_repeat = INTR_BYTES_PER_REPEAT / sizeof(T);
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  auto dst_stride0 = dst->strides[0];

  bool is_src0_brc_dim0 = src0->sizes[0] != dst_size0 && src0->sizes[0] == 1;
  bool is_src1_brc_dim0 = false;
  __ubuf__ T *src0_ptr = src0->aligned + src0->offset;
  uint16_t src0_stride0 = is_src0_brc_dim0 ? 0 : src0->strides[0];
  uint16_t src0_block_stride = 1;
  __ubuf__ T *src1_ptr = nullptr;
  uint16_t src1_stride0 = 0;
  uint16_t src1_block_stride = 0;
  if constexpr (OPERANUM == 2) {
    is_src1_brc_dim0 = src1->sizes[0] != dst_size0 && src1->sizes[0] == 1;
    src1_ptr = src1->aligned + src1->offset;
    src1_stride0 = is_src1_brc_dim0 ? 0 : src1->strides[0];
    src1_block_stride = 1;
  }

  auto src0_tail_offset = FLOOR_FACTOR(dst_size1, num_per_repeat);
  auto src1_tail_offset = FLOOR_FACTOR(dst_size1, num_per_repeat);
  auto src0_default_stride = INTR_BLOCKS_PER_REPEAT;
  auto src1_default_stride = INTR_BLOCKS_PER_REPEAT;
  if (mode == VectorLastAxisMode::BV || mode == VectorLastAxisMode::BB ||
      mode == VectorLastAxisMode::BS || mode == VectorLastAxisMode::B) {
    src0_block_stride = 0;
    src0_tail_offset = 0;
    src0_default_stride = 0;
  }
  if (mode == VectorLastAxisMode::VB || mode == VectorLastAxisMode::BB ||
      mode == VectorLastAxisMode::VS || mode == VectorLastAxisMode::BS) {
    src1_block_stride = 0;
    src1_tail_offset = 0;
    src1_default_stride = 0;
  }

  // pick the bigger axis as the repeat axis between size0 and
  // size1/num_per_repeat
  if (dst_size1 >= num_per_repeat) {
    bool repeat_size0 = dst_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                        src0_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                        src1_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                        dst_size0 >= dst_size1 / num_per_repeat;

    uint64_t repeat_num = repeat_size0 ? dst_size0 : dst_size1 / num_per_repeat;
    int64_t loop_num = repeat_size0 ? dst_size1 / num_per_repeat : dst_size0;

    uint16_t dst_repeat_stride =
        repeat_size0 ? dst_stride0 / num_per_block : INTR_BLOCKS_PER_REPEAT;
    uint16_t dst_loop_stride =
        repeat_size0 ? INTR_BLOCKS_PER_REPEAT : dst_stride0 / num_per_block;
    // Calculate the main block
    // src0_repeat_stride/src1_repeat_stride/src0_loop_stride/src1_loop_stride.
    uint16_t src0_repeat_stride =
        repeat_size0 ? src0_stride0 / num_per_block : src0_default_stride;
    uint16_t src0_loop_stride =
        repeat_size0 ? src0_default_stride : src0_stride0 / num_per_block;
    uint16_t src1_repeat_stride = 0;
    uint16_t src1_loop_stride = 0;
    if constexpr (OPERANUM == 2) {
      src1_repeat_stride =
          repeat_size0 ? src1_stride0 / num_per_block : src1_default_stride;
      src1_loop_stride =
          repeat_size0 ? src1_default_stride : src1_stride0 / num_per_block;
    }

    INTRINSIC_NO_ARGS(set_mask_norm);
    SetMaskValueByCount(num_per_repeat);
    for (int64_t i = 0; i < loop_num; i++) {
      vector_eltwise_2d_intrin<OP, T>(
          intrin_args<2, T>{dst_ptr + i * dst_loop_stride * num_per_block,
                            {src0_ptr + i * src0_loop_stride * num_per_block,
                             src1_ptr + i * src1_loop_stride * num_per_block},
                            scalar,
                            repeat_num,
                            1, // dst_block_stride
                            {src0_block_stride, src1_block_stride},
                            dst_repeat_stride,
                            {src0_repeat_stride, src1_repeat_stride}});
    }
  }

  // process tail repeat
  if (dst_size1 % num_per_repeat != 0) {
    bool repeat_size0 = dst_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                        src0_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                        src1_stride0 <= INTR_MAX_BLOCK_CNTS * num_per_block &&
                        dst_size0 >= 1;
    uint64_t repeat_num = repeat_size0 ? dst_size0 : 1;
    uint64_t loop_num = repeat_size0 ? 1 : dst_size0;

    uint16_t dst_repeat_stride = repeat_size0 ? dst_stride0 / num_per_block : 0;
    uint16_t dst_loop_stride = repeat_size0 ? 0 : dst_stride0 / num_per_block;
    // Calculate the tail block
    // src0_repeat_stride/src1_repeat_stride/src0_loop_stride/src1_loop_stride.
    uint16_t src0_repeat_stride =
        repeat_size0 ? src0_stride0 / num_per_block : 0;
    uint16_t src0_loop_stride = repeat_size0 ? 0 : src0_stride0 / num_per_block;
    uint16_t src1_repeat_stride = 0;
    uint16_t src1_loop_stride = 0;
    if constexpr (OPERANUM == 2) {
      src1_repeat_stride = repeat_size0 ? src1_stride0 / num_per_block : 0;
      src1_loop_stride = repeat_size0 ? 0 : src1_stride0 / num_per_block;
    }

    INTRINSIC_NO_ARGS(set_mask_norm);
    SetMaskValueByCount(dst_size1 % num_per_repeat);
    for (int64_t i = 0; i < loop_num; i++) {
      vector_eltwise_2d_intrin<OP, T>(intrin_args<2, T>{
          dst_ptr + FLOOR_FACTOR(dst_size1, num_per_repeat) +
              i * dst_loop_stride * num_per_block,
          {src0_ptr + src0_tail_offset + i * src0_loop_stride * num_per_block,
           src1_ptr + src1_tail_offset + i * src1_loop_stride * num_per_block},
          scalar,
          repeat_num,
          1, // dst_block_stride
          {src0_block_stride, src1_block_stride},
          dst_repeat_stride,
          {src0_repeat_stride, src1_repeat_stride}});
    }
  }
}

/// Deal vector elementwise 2d last axis brc inline.
/// src:(a, 1) -> dst:(a, num_per_block)
template <typename SRC_T, typename DST_T>
__aiv__ __attribute__((always_inline)) void
vector_last_axis_brc_2d(memref_t<__ubuf__ SRC_T, 2> *src,
                        memref_t<__ubuf__ DST_T, 2> *dst,
                        memref_t<__ubuf__ SRC_T, 1> *tmp_buf) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  memref_t<__ubuf__ SRC_T, 2> tmp_buf_for_brc{tmp_buf->aligned,
                                              tmp_buf->allocated,
                                              tmp_buf->offset,
                                              {src->sizes[0], num_per_block},
                                              {num_per_block, 1}};
  if (src->sizes[0] == 1) {
    auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
    auto src_ptr = src->aligned + src->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    SRC_T scalar = *src_ptr;
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
    INTRINSIC(pipe_barrier, PIPE_V);
    view_as<SRC_T, DST_T, 2>(&tmp_buf_for_brc, dst);
    return;
  }

  if (src->strides[0] != 1) {
    for (int64_t i = 0; i < src->sizes[0]; ++i) {
      memref_t<__ubuf__ SRC_T, 2> tmp_buf_for_vdup{tmp_buf->aligned,
                                                   tmp_buf->allocated,
                                                   tmp_buf->offset +
                                                       i * num_per_block,
                                                   {1, num_per_block},
                                                   {num_per_block, 1}};
      auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset + i * num_per_block;
      auto src_ptr = src->aligned + src->offset + i * src->strides[0];
      INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
      SRC_T scalar = *src_ptr;
      INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
      brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
    }
    view_as<SRC_T, DST_T, 2>(&tmp_buf_for_brc, dst);
    INTRINSIC(pipe_barrier, PIPE_V);
    return;
  }

  INTRINSIC_NO_ARGS(set_mask_count);
  brc_last_axis_2d_vbrcb<SRC_T>(src, &tmp_buf_for_brc);
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC_NO_ARGS(set_mask_norm);
  view_as<SRC_T, DST_T, 2>(&tmp_buf_for_brc, dst);
}

template <VectorOpTy OP, typename SRC_T, typename DST_T = SRC_T>
__aiv__ __attribute__((always_inline)) void normalize_vector_last_axis_2d(
    memref_t<__ubuf__ SRC_T, 2> *src0, memref_t<__ubuf__ SRC_T, 2> *src1,
    SRC_T scalar, memref_t<__ubuf__ SRC_T, 2> *dst,
    memref_t<__ubuf__ DST_T, 2> *new_src0,
    memref_t<__ubuf__ DST_T, 2> *new_src1, memref_t<__ubuf__ SRC_T, 1> *tmp_buf,
    VectorLastAxisMode *mode) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(SRC_T);
  constexpr int new_num_per_block = INTR_BYTES_PER_BLOCK / sizeof(DST_T);
  memref_t<__ubuf__ DST_T, 1> tmp_buf_as_dst_t;
  view_as<SRC_T, DST_T, 1>(tmp_buf, &tmp_buf_as_dst_t);

  int64_t base_tmp_buf_offset = tmp_buf->offset;
  // Preprocess src0
  auto tmp_buf_ptr = tmp_buf->aligned + tmp_buf->offset;
  bool is_src0_brc_dim1 = false;
  if (*mode == VectorLastAxisMode::SV || *mode == VectorLastAxisMode::SB) {
    brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
    INTRINSIC(pipe_barrier, PIPE_V);
    *new_src0 = {tmp_buf_as_dst_t.aligned,
                 tmp_buf_as_dst_t.allocated,
                 tmp_buf_as_dst_t.offset,
                 {1, new_num_per_block},
                 {new_num_per_block, 1}};
    base_tmp_buf_offset = base_tmp_buf_offset + num_per_block;
    tmp_buf_as_dst_t.offset = tmp_buf_as_dst_t.offset + num_per_block;
  } else {
    is_src0_brc_dim1 = src0->sizes[1] != dst->sizes[1] && src0->sizes[1] == 1;
    if (is_src0_brc_dim1) {
      vector_last_axis_brc_2d<SRC_T, DST_T>(src0, new_src0, tmp_buf);
      base_tmp_buf_offset =
          base_tmp_buf_offset + src0->sizes[0] == 1
              ? num_per_block
              : num_per_block *
                    CEIL_FACTOR(src0->sizes[0], kSrcNumPerRepeatOfVBRCB);
      tmp_buf_as_dst_t.offset =
          tmp_buf_as_dst_t.offset + src0->sizes[0] == 1
              ? new_num_per_block
              : new_num_per_block *
                    CEIL_FACTOR(src0->sizes[0], kSrcNumPerRepeatOfVBRCB);
    }
  }

  // Preprocess src1
  tmp_buf_ptr = tmp_buf->aligned + base_tmp_buf_offset;
  bool is_src1_brc_dim1 = false;
  if (*mode != VectorLastAxisMode::V && *mode != VectorLastAxisMode::B) {
    if ((*mode == VectorLastAxisMode::VS || *mode == VectorLastAxisMode::BS)) {
      if (!isHardwareSupportedVS<OP>()) {
        brc_scalar_core_1d(scalar, tmp_buf_ptr, num_per_block);
        INTRINSIC(pipe_barrier, PIPE_V);
        *new_src1 = {tmp_buf_as_dst_t.aligned,
                     tmp_buf_as_dst_t.allocated,
                     tmp_buf_as_dst_t.offset,
                     {1, new_num_per_block},
                     {new_num_per_block, 1}};
        tmp_buf_as_dst_t.offset = tmp_buf_as_dst_t.offset + num_per_block;
      }
    } else {
      is_src1_brc_dim1 = src1->sizes[1] != dst->sizes[1] && src1->sizes[1] == 1;
      if (is_src1_brc_dim1) {
        memref_t<__ubuf__ SRC_T, 1> new_tmp_buf{
            tmp_buf->aligned,
            tmp_buf->allocated,
            tmp_buf->offset + base_tmp_buf_offset,
            {tmp_buf->sizes[0] - base_tmp_buf_offset},
            {1}};
        vector_last_axis_brc_2d<SRC_T, DST_T>(src1, new_src1, &new_tmp_buf);
        tmp_buf_as_dst_t.offset =
            tmp_buf_as_dst_t.offset + src1->sizes[0] == 1
                ? new_num_per_block
                : new_num_per_block *
                      CEIL_FACTOR(src1->sizes[0], kSrcNumPerRepeatOfVBRCB);
      }
    }
  }

  // Set the converted mode
  *mode = get_preprocessed_mode<OP>(*mode, is_src0_brc_dim1, is_src1_brc_dim1);
}

/// Elementwise operation between two 2D tensors.
///
/// constraint:
/// 1. src0/src1/dst strides[0] need to block alignment strides[1] need to be
///    continuous, tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a, b) operation src1:(a, b) -> dst:(a, b)
///    * scene2: src0:(a, b) operation src1:(c, d) -> dst:(e, f)
///        a != c:
///            e = max(a, c)
///            min(a, c) = 1
///        b != d:
///            f = max(b, d)
///            min(b, d) = 1
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    OP != VXOR:
///        scene1: tmp_buf = 0
///        scene2:
///            (b != 1 && d != 1) || f == 1: no last axis inline brc
///                tmp_buf = 0
///            b == 1 && d != 1: src0 last axis inline brc
///                a == 1:
///                    tmp_buf = num_per_block
///                a != 1:
///                    tmp_buf =
///                        aligned(a, kSrcNumPerRepeatOfVBRCB) * num_per_block
///            b != 1 && d == 1: src1 last axis inline brc
///                c == 1:
///                    tmp_buf = num_per_block
///                c != 1:
///                    tmp_buf =
///                        aligned(c, kSrcNumPerRepeatOfVBRCB) * num_per_block
///            b == 1 && d == 1 && f != 1: src0 and src1 last axis inline brc
///                a == 1:
///                    c == 1:
///                        tmp_buf = num_per_block * 2
///                    c != 1:
///                        tmp_buf = num_per_block +
///                            aligned(c, kSrcNumPerRepeatOfVBRCB) *
///                            num_per_block
///                a != 1:
///                    c == 1:
///                        tmp_buf = num_per_block +
///                            aligned(a, kSrcNumPerRepeatOfVBRCB) *
///                            num_per_block
///                    c != 1:
///                        tmp_buf =
///                            aligned(c, kSrcNumPerRepeatOfVBRCB) *
///                            num_per_block + aligned(a,
///                            kSrcNumPerRepeatOfVBRCB) * num_per_block
///    OP = VXOR:
///        On the basis of OP != VXOR, An additional tmp_buf of size * align(f,
///        num_per_block) is required
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vv_2d(
    memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *src1,
    memref_t<__ubuf__ T, 2> *dst, memref_t<__ubuf__ T, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  if constexpr (OP == VectorOpTy::VXOR) {
    vxor_vv_2d(src0, src1, dst, tmp_buf);
    return;
  }

  memref_t<__ubuf__ T, 2> new_src0 = *src0;
  memref_t<__ubuf__ T, 2> new_src1 = *src1;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  if constexpr (std::is_same<half, T>::value ||
                std::is_same<bfloat16_t, T>::value) {
    normalize_vector_last_axis_2d<OP, T>(src0, src1, {0} /* scalar */, dst,
                                         &new_src0, &new_src1, tmp_buf, &mode);

    vector_eltwise_2d<OP, T, 2>(&new_src0, &new_src1, {0} /* scalar */, dst,
                                mode);
  } else {
    normalize_vector_last_axis_2d<OP, T>(src0, src1, 0 /* scalar */, dst,
                                         &new_src0, &new_src1, tmp_buf, &mode);

    vector_eltwise_2d<OP, T, 2>(&new_src0, &new_src1, 0 /* scalar */, dst,
                                mode);
  }
#else
  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<OP, T>(src0, src1, 0 /* scalar */, dst,
                                       &new_src0, &new_src1, tmp_buf, &mode);

  // Step2: Perform elementwise operations
  vector_eltwise_2d<OP, T, 2>(&new_src0, &new_src1, 0 /* scalar */, dst, mode);
#endif
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, bool>(memref_t<__ubuf__ bool, 2> *src0,
                                            memref_t<__ubuf__ bool, 2> *src1,
                                            memref_t<__ubuf__ bool, 2> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<bool, int16_t, 2>(src0, &src0_as_int16);
  view_as<bool, int16_t, 2>(src1, &src1_as_int16);
  view_as<bool, int16_t, 2>(dst, &dst_as_int16);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, int32_t>(
    memref_t<__ubuf__ int32_t, 2> *src0, memref_t<__ubuf__ int32_t, 2> *src1,
    memref_t<__ubuf__ int32_t, 2> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<int32_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, int32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, uint32_t>(
    memref_t<__ubuf__ uint32_t, 2> *src0, memref_t<__ubuf__ uint32_t, 2> *src1,
    memref_t<__ubuf__ uint32_t, 2> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<uint32_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, uint32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, int64_t>(
    memref_t<__ubuf__ int64_t, 2> *src0, memref_t<__ubuf__ int64_t, 2> *src1,
    memref_t<__ubuf__ int64_t, 2> *dst, memref_t<__ubuf__ int64_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<int64_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, int64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, uint64_t>(
    memref_t<__ubuf__ uint64_t, 2> *src0, memref_t<__ubuf__ uint64_t, 2> *src1,
    memref_t<__ubuf__ uint64_t, 2> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<uint64_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, uint64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, int8_t>(
    memref_t<__ubuf__ int8_t, 2> *src0, memref_t<__ubuf__ int8_t, 2> *src1,
    memref_t<__ubuf__ int8_t, 2> *dst, memref_t<__ubuf__ int8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<int8_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, int8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, uint8_t>(
    memref_t<__ubuf__ uint8_t, 2> *src0, memref_t<__ubuf__ uint8_t, 2> *src1,
    memref_t<__ubuf__ uint8_t, 2> *dst, memref_t<__ubuf__ uint8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<uint8_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, uint8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, half>(memref_t<__ubuf__ half, 2> *src0,
                                            memref_t<__ubuf__ half, 2> *src1,
                                            memref_t<__ubuf__ half, 2> *dst,
                                            memref_t<__ubuf__ half, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<half, int16_t, 2>(src0, &src0_as_int16);
  view_as<half, int16_t, 2>(src1, &src1_as_int16);
  view_as<half, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_2d<VectorOpTy::VOR, half, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_2d<VectorOpTy::VOR, half, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 2> *src0,
    memref_t<__ubuf__ bfloat16_t, 2> *src1,
    memref_t<__ubuf__ bfloat16_t, 2> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<bfloat16_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<bfloat16_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_2d<VectorOpTy::VOR, bfloat16_t, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_2d<VectorOpTy::VOR, bfloat16_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VOR, float>(
    memref_t<__ubuf__ float, 2> *src0, memref_t<__ubuf__ float, 2> *src1,
    memref_t<__ubuf__ float, 2> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<float, int16_t, 2>(src0, &src0_as_int16);
  view_as<float, int16_t, 2>(src1, &src1_as_int16);
  view_as<float, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VOR, float, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VOR, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, bool>(
    memref_t<__ubuf__ bool, 2> *src0, memref_t<__ubuf__ bool, 2> *src1,
    memref_t<__ubuf__ bool, 2> *dst, memref_t<__ubuf__ bool, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<bool, int16_t, 2>(src0, &src0_as_int16);
  view_as<bool, int16_t, 2>(src1, &src1_as_int16);
  view_as<bool, int16_t, 2>(dst, &dst_as_int16);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, int32_t>(
    memref_t<__ubuf__ int32_t, 2> *src0, memref_t<__ubuf__ int32_t, 2> *src1,
    memref_t<__ubuf__ int32_t, 2> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<int32_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, int32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, uint32_t>(
    memref_t<__ubuf__ uint32_t, 2> *src0, memref_t<__ubuf__ uint32_t, 2> *src1,
    memref_t<__ubuf__ uint32_t, 2> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<uint32_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, uint32_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, int64_t>(
    memref_t<__ubuf__ int64_t, 2> *src0, memref_t<__ubuf__ int64_t, 2> *src1,
    memref_t<__ubuf__ int64_t, 2> *dst, memref_t<__ubuf__ int64_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<int64_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, int64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, uint64_t>(
    memref_t<__ubuf__ uint64_t, 2> *src0, memref_t<__ubuf__ uint64_t, 2> *src1,
    memref_t<__ubuf__ uint64_t, 2> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<uint64_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, uint64_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, int8_t>(
    memref_t<__ubuf__ int8_t, 2> *src0, memref_t<__ubuf__ int8_t, 2> *src1,
    memref_t<__ubuf__ int8_t, 2> *dst, memref_t<__ubuf__ int8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<int8_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, int8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, uint8_t>(
    memref_t<__ubuf__ uint8_t, 2> *src0, memref_t<__ubuf__ uint8_t, 2> *src1,
    memref_t<__ubuf__ uint8_t, 2> *dst, memref_t<__ubuf__ uint8_t, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<uint8_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, uint8_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, half>(
    memref_t<__ubuf__ half, 2> *src0, memref_t<__ubuf__ half, 2> *src1,
    memref_t<__ubuf__ half, 2> *dst, memref_t<__ubuf__ half, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<half, int16_t, 2>(src0, &src0_as_int16);
  view_as<half, int16_t, 2>(src1, &src1_as_int16);
  view_as<half, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_2d<VectorOpTy::VAND, half, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_2d<VectorOpTy::VAND, half, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 2> *src0,
    memref_t<__ubuf__ bfloat16_t, 2> *src1,
    memref_t<__ubuf__ bfloat16_t, 2> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<bfloat16_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 2>(src1, &src1_as_int16);
  view_as<bfloat16_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_2d<VectorOpTy::VAND, bfloat16_t, int16_t>(
      src0, src1, {0} /* scalar */, dst, &src0_as_int16, &src1_as_int16,
      tmp_buf, &mode);
#else
  normalize_vector_last_axis_2d<VectorOpTy::VAND, bfloat16_t, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);
#endif

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vv_2d<VectorOpTy::VAND, float>(
    memref_t<__ubuf__ float, 2> *src0, memref_t<__ubuf__ float, 2> *src1,
    memref_t<__ubuf__ float, 2> *dst, memref_t<__ubuf__ float, 1> *tmp_buf,
    VectorLastAxisMode mode) {
  // convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> src1_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<float, int16_t, 2>(src0, &src0_as_int16);
  view_as<float, int16_t, 2>(src1, &src1_as_int16);
  view_as<float, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VAND, float, int16_t>(
      src0, src1, 0 /* scalar */, dst, &src0_as_int16, &src1_as_int16, tmp_buf,
      &mode);

  vector_eltwise_2d<VectorOpTy::VAND, int16_t, 2>(
      &src0_as_int16, &src1_as_int16, 0 /* scalar */, &dst_as_int16, mode);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vector_eltwise_vs_sv_2d(memref_t<__ubuf__ T, 2> *src,
                                        memref_t<__ubuf__ T, 2> *dst,
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
  assert((isSizeAlignedToBlock<T>(src->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[0])) &&
         "The src/dst strides[0]/strides[1] must be aligned to block.");
  assert((src->strides[1] == 1 && dst->strides[1] == 1 &&
          tmp_buf->strides[0] == 1) &&
         "The src/dst/tmp_buf last dim must be continuous.");
#endif
}

/// Elementwise operation between scalar and 2D tensors.
///
/// constraint:
/// 1. src1/dst strides[0] need to block alignment strides[1] need to be
///    continuous. tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: scalar operation src1:(a, b) -> dst:(a, b)
///    * scene2: scalar operation src1:(1, b) -> dst:(a, b)
///    * scene3: scalar operation src1:(a, 1) -> dst:(a, b)
///    * scene4: scalar operation src1:(1, 1) -> dst:(a, b)
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1/scene2: tmp_buf = num_per_block
///    scene3:        tmp_buf = num_per_block +
///                       aligned(a, kSrcNumPerRepeatOfVBRCB) * num_per_block
///    scene4         tmp_buf = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_sv_2d(
    T scalar, memref_t<__ubuf__ T, 2> *src1, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_2d(src1, dst, tmp_buf);

  // There is no corresponding vsubs/vdivs instruction, so it can only be
  // implemented by converting it into vsub/vdiv instruction.
  // Step1: preprocess src tail axis brc inline and scalar scene
  memref_t<__ubuf__ T, 2> new_src0;
  memref_t<__ubuf__ T, 2> new_src1 = *src1;
  normalize_vector_last_axis_2d<OP, T>(nullptr /* src0 */, src1, scalar, dst,
                                       &new_src0, &new_src1, tmp_buf, &mode);

  // Step2: Perform elementwise operations
  vector_eltwise_2d<OP, T, 2>(&new_src0, &new_src1, scalar, dst, mode);
}

/// Elementwise operation between 2D tensors and scalar.
///
/// constraint:
/// 1. src0/dst strides[0] need to block alignment strides[1] need to be
///    continuous. tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a, b) operation scalar -> dst:(a, b)
///    * scene2: src0:(a, 1) operation scalar -> dst:(a, b)
///    * scene3: src0:(1, b) operation scalar -> dst:(a, b)
///    * scene4: src0:(1, 1) operation scalar -> dst:(a, b)
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1/scene2: tmp_buf = num_per_block
///    scene3:        tmp_buf = num_per_block +
///                       aligned(a, kSrcNumPerRepeatOfVBRCB) * num_per_block
///    scene4:        tmp_buf = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void vector_eltwise_vs_2d(
    memref_t<__ubuf__ T, 2> *src0, T scalar, memref_t<__ubuf__ T, 2> *dst,
    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_2d(src0, dst, tmp_buf);

  if constexpr (isHardwareSupportedVS<OP>()) {
    // Step1: preprocess src tail axis brc inline scene
    memref_t<__ubuf__ T, 2> new_src0 = *src0;
    normalize_vector_last_axis_2d<OP, T, T>(
        src0, nullptr /* src1 */, scalar, dst, &new_src0,
        nullptr /* new src1 */, tmp_buf, &mode);

    // Step2: Perform elementwise operations
    vector_eltwise_2d<OP, T, 1>(&new_src0, nullptr, scalar, dst, mode);
    return;
  }

  // There is no corresponding vsubs/vdivs instruction, so it can only be
  // implemented by converting it into vsub/vdiv instruction.
  // Step1: preprocess src tail axis brc inline and scalar scene
  memref_t<__ubuf__ T, 2> new_src0 = *src0;
  memref_t<__ubuf__ T, 2> new_src1;
  normalize_vector_last_axis_2d<OP, T>(src0, nullptr /* src1 */, scalar, dst,
                                       &new_src0, &new_src1, tmp_buf, &mode);

  // Step2: Perform elementwise operations
  vector_eltwise_2d<OP, T, 2>(&new_src0, &new_src1, scalar, dst, mode);
}

/// Unary Elementwise operation 2d
///
/// constraint:
/// 1. src0/dst strides[0] need to block alignment strides[1] need to be
///    continuous. tmp_buf need to be continuous.
/// 2. Currently supports the following scenarios:
///    * scene1: src0:(a, b) -> dst:(a, b)
///    * scene2: src0:(1, b) -> dst:(a, b)
///    * scene3: src0:(a, 1) -> dst:(a, b)
///    * scene4: src0:(1, 1) -> dst:(a, b)
///    Except dype = bool unsupport inline brc.
/// 3. The tmp_buf size is as follows:
///    scene1/scene2: tmp_buf = 0
///    scene3 = aligned(a, kSrcNumPerRepeatOfVBRCB) * num_per_block
///    scene4 = num_per_block
template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d(memref_t<__ubuf__ T, 2> *src0, memref_t<__ubuf__ T, 2> *dst,
                    memref_t<__ubuf__ T, 1> *tmp_buf, VectorLastAxisMode mode) {
  // Input parameter constraints assert.
  check_inputs_of_vector_eltwise_vs_sv_2d(src0, dst, tmp_buf);

  memref_t<__ubuf__ T, 2> new_src0 = *src0;
  if (mode == VectorLastAxisMode::V) {
    // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
    if constexpr (std::is_same<half, T>::value ||
                  std::is_same<bfloat16_t, T>::value) {
      normalize_vector_last_axis_2d<OP, T, T>(
          src0, nullptr /* src1 */, {0} /* scalar */, dst, &new_src0,
          nullptr /* new_src1 */, tmp_buf, &mode);
    } else {
      normalize_vector_last_axis_2d<OP, T, T>(
          src0, nullptr /* src1 */, 0 /* scalar */, dst, &new_src0,
          nullptr /* new_src1 */, tmp_buf, &mode);
    }
#else
    normalize_vector_last_axis_2d<OP, T, T>(
        src0, nullptr /* src1 */, 0 /* scalar */, dst, &new_src0,
        nullptr /* new_src1 */, tmp_buf, &mode);
#endif
  }

#ifdef ENABLE_CPU_TRACE_INTRINSIC
  if constexpr (std::is_same<half, T>::value ||
                std::is_same<bfloat16_t, T>::value) {
    vector_eltwise_2d<OP, T, 1>(&new_src0, nullptr /* src1 */, {0} /* scalar */,
                                dst, mode);
  } else {
    vector_eltwise_2d<OP, T, 1>(&new_src0, nullptr /* src1 */, 0 /* scalar */,
                                dst, mode);
  }
#else
  vector_eltwise_2d<OP, T, 1>(&new_src0, nullptr /* src1 */, 0 /* scalar */,
                              dst, mode);
#endif
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, bool>(memref_t<__ubuf__ bool, 2> *src0,
                                            memref_t<__ubuf__ bool, 2> *dst,
                                            memref_t<__ubuf__ bool, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert bool memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<bool, int16_t, 2>(src0, &src0_as_int16);
  view_as<bool, int16_t, 2>(dst, &dst_as_int16);

  vector_eltwise_v_2d<VectorOpTy::VNOT, int16_t>(&src0_as_int16, &dst_as_int16);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, int32_t>(
    memref_t<__ubuf__ int32_t, 2> *src0, memref_t<__ubuf__ int32_t, 2> *dst,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int32_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, int32_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, uint32_t>(
    memref_t<__ubuf__ uint32_t, 2> *src0, memref_t<__ubuf__ uint32_t, 2> *dst,
    memref_t<__ubuf__ uint32_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint32_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint32_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint32_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, uint32_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, int64_t>(
    memref_t<__ubuf__ int64_t, 2> *src0, memref_t<__ubuf__ int64_t, 2> *dst,
    memref_t<__ubuf__ int64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int64_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, int64_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, uint64_t>(
    memref_t<__ubuf__ uint64_t, 2> *src0, memref_t<__ubuf__ uint64_t, 2> *dst,
    memref_t<__ubuf__ uint64_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint64_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint64_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint64_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, uint64_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, int8_t>(
    memref_t<__ubuf__ int8_t, 2> *src0, memref_t<__ubuf__ int8_t, 2> *dst,
    memref_t<__ubuf__ int8_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert int8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<int8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<int8_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, int8_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, uint8_t>(
    memref_t<__ubuf__ uint8_t, 2> *src0, memref_t<__ubuf__ uint8_t, 2> *dst,
    memref_t<__ubuf__ uint8_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert uint8_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<uint8_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<uint8_t, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, uint8_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, half>(memref_t<__ubuf__ half, 2> *src0,
                                            memref_t<__ubuf__ half, 2> *dst,
                                            memref_t<__ubuf__ half, 1> *tmp_buf,
                                            VectorLastAxisMode mode) {
  // convert half memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<half, int16_t, 2>(src0, &src0_as_int16);
  view_as<half, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, half, int16_t>(
      src0, nullptr /* src1 */, {0} /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);
#else
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, half, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);
#endif

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, bfloat16_t>(
    memref_t<__ubuf__ bfloat16_t, 2> *src0,
    memref_t<__ubuf__ bfloat16_t, 2> *dst,
    memref_t<__ubuf__ bfloat16_t, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert bfloat16_t memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<bfloat16_t, int16_t, 2>(src0, &src0_as_int16);
  view_as<bfloat16_t, int16_t, 2>(dst, &dst_as_int16);

// preprocess src tail axis brc inline scene.
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, bfloat16_t, int16_t>(
      src0, nullptr /* src1 */, {0} /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);
#else
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, bfloat16_t, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);
#endif

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_v_2d<VectorOpTy::VNOT, float>(
    memref_t<__ubuf__ float, 2> *src0, memref_t<__ubuf__ float, 2> *dst,
    memref_t<__ubuf__ float, 1> *tmp_buf, VectorLastAxisMode mode) {
  // convert float memref to int16 memref
  memref_t<__ubuf__ int16_t, 2> src0_as_int16;
  memref_t<__ubuf__ int16_t, 2> dst_as_int16;
  view_as<float, int16_t, 2>(src0, &src0_as_int16);
  view_as<float, int16_t, 2>(dst, &dst_as_int16);

  // preprocess src tail axis brc inline scene.
  normalize_vector_last_axis_2d<VectorOpTy::VNOT, float, int16_t>(
      src0, nullptr /* src1 */, 0 /* scalar */, dst, &src0_as_int16,
      nullptr /* new_src1 */, tmp_buf, &mode);

  vector_eltwise_2d<VectorOpTy::VNOT, int16_t, 1>(
      &src0_as_int16, nullptr /* src1 */, 0 /* scalar */, &dst_as_int16, mode);
}

extern "C" {
//===-------------------------------------------------------------------===//
// eltwise vv, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, int16_t)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, int32_t)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, half)
REGISTE_ELTWISE_VV(vadd, VectorOpTy::VADD, 2, float)

REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, int16_t)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, int32_t)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, half)
REGISTE_ELTWISE_VV(vsub, VectorOpTy::VSUB, 2, float)

REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, int16_t)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, int32_t)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, half)
REGISTE_ELTWISE_VV(vmul, VectorOpTy::VMUL, 2, float)

REGISTE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 2, half)
REGISTE_ELTWISE_VV(vdiv, VectorOpTy::VDIV, 2, float)

REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, int16_t)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, int32_t)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, half)
REGISTE_ELTWISE_VV(vmin, VectorOpTy::VMIN, 2, float)

REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, int16_t)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, int32_t)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, half)
REGISTE_ELTWISE_VV(vmax, VectorOpTy::VMAX, 2, float)

REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, bool)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int8_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, half)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, bfloat16_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int32_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint32_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, float)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, int64_t)
REGISTE_ELTWISE_VV(vand, VectorOpTy::VAND, 2, uint64_t)

REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, bool)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int8_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, half)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, bfloat16_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int32_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint32_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, float)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, int64_t)
REGISTE_ELTWISE_VV(vor, VectorOpTy::VOR, 2, uint64_t)

REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, bool)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, int8_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, uint8_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, int16_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, uint16_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, int32_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, uint32_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, int64_t)
REGISTE_ELTWISE_VV(vxor, VectorOpTy::VXOR, 2, uint64_t)

//===-------------------------------------------------------------------===//
// eltwise sv, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, int16_t)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, int32_t)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, half)
REGISTE_ELTWISE_SV(vsub, VectorOpTy::VSUB, 2, float)

REGISTE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 2, half)
REGISTE_ELTWISE_SV(vdiv, VectorOpTy::VDIV, 2, float)

//===-------------------------------------------------------------------===//
// eltwise vs, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, int16_t)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, int32_t)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, half)
REGISTE_ELTWISE_VS(vadds, VectorOpTy::VADDS, 2, float)

REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, int16_t)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, int32_t)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, half)
REGISTE_ELTWISE_VS(vsub, VectorOpTy::VSUB, 2, float)

REGISTE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 2, half)
REGISTE_ELTWISE_VS(vdiv, VectorOpTy::VDIV, 2, float)

REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, int16_t)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, int32_t)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, half)
REGISTE_ELTWISE_VS(vmuls, VectorOpTy::VMULS, 2, float)

REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, int16_t)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, int32_t)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, half)
REGISTE_ELTWISE_VS(vmins, VectorOpTy::VMINS, 2, float)

REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, int16_t)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, int32_t)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, half)
REGISTE_ELTWISE_VS(vmaxs, VectorOpTy::VMAXS, 2, float)

REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, int16_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, uint16_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, int32_t)
REGISTE_ELTWISE_VS(vshls, VectorOpTy::VSHL, 2, uint32_t)

REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, int16_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, uint16_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, int32_t)
REGISTE_ELTWISE_VS(vshrs, VectorOpTy::VSHR, 2, uint32_t)

//===-------------------------------------------------------------------===//
// eltwise unary, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_V(vabs, VectorOpTy::VABS, 2, half)
REGISTE_ELTWISE_V(vabs, VectorOpTy::VABS, 2, float)

REGISTE_ELTWISE_V(vln, VectorOpTy::VLN, 2, half)
REGISTE_ELTWISE_V(vln, VectorOpTy::VLN, 2, float)

REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 2, half)
REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 2, float)
REGISTE_ELTWISE_V(vrelu, VectorOpTy::VRELU, 2, int32_t)

REGISTE_ELTWISE_V(vexp, VectorOpTy::VEXP, 2, half)
REGISTE_ELTWISE_V(vexp, VectorOpTy::VEXP, 2, float)

REGISTE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 2, half)
REGISTE_ELTWISE_V(vrsqrt, VectorOpTy::VRSQRT, 2, float)

REGISTE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 2, half)
REGISTE_ELTWISE_V(vsqrt, VectorOpTy::VSQRT, 2, float)

REGISTE_ELTWISE_V(vrec, VectorOpTy::VREC, 2, half)
REGISTE_ELTWISE_V(vrec, VectorOpTy::VREC, 2, float)

REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, bool)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int8_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, half)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, bfloat16_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int32_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint32_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, float)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, int64_t)
REGISTE_ELTWISE_V(vnot, VectorOpTy::VNOT, 2, uint64_t)

//===-------------------------------------------------------------------===//
// Registe vector_eltwise_2d_intrin fun, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VADD, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VADD, int32_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VADD, half);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VADD, float);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VSUB, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VSUB, int32_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VSUB, half);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VSUB, float);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMUL, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMUL, int32_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMUL, half);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMUL, float);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VDIV, half);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VDIV, float);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMIN, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMIN, int32_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMIN, half);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMIN, float);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMAX, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMAX, int32_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMAX, half);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VMAX, float);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VAND, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VAND, uint16_t);

REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VOR, int16_t);
REGISTE_ELTWISE_2D_INTRIN(VectorOpTy::VOR, uint16_t);
}
