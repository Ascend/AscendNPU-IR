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

#include "Fixpipe/FixpipeUtils.h"

__aicore__ __attribute__((always_inline)) void
set_nd_para(uint64_t nd_num, uint64_t src_nd_stride, uint64_t dst_nd_stride) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(set_nd_para,
            (uint64_t)(nd_num | (src_nd_stride << 16) | (dst_nd_stride << 32)));
#endif
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_normal_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                          memref_t<__gm__ DST_TYPE, 2> *gm,
                                          int64_t pre_quant, int64_t pre_relu,
                                          bool channel_split,
                                          uint8_t unit_flag) {
  __gm__ DST_TYPE *gm_ptr = gm->aligned + gm->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m = gm->sizes[0];
  uint16_t n = gm->sizes[1];
  uint16_t src_stride = l0c->sizes[1] / FRACTAL_BLOCK_NUM;

  // burst count
  uint16_t n_size = l0c->sizes[0];
  // burst length
  uint16_t m_size = l0c->sizes[1] * sizeof(SRC_TYPE);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_gm_intrin(
      copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE>{
          gm_ptr, l0c_ptr,
          0, // sid
          n_size, m_size,
          src_stride,                     // dstStride_dst_D
          src_stride,                     // srcStride
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split,
          false // NZ2ND_EN
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__gm__ DST_TYPE, 2> *gm,
                                         int64_t pre_quant, int64_t pre_relu,
                                         bool channel_split,
                                         uint8_t unit_flag) {
  __gm__ DST_TYPE *gm_ptr = gm->aligned + gm->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2];
  uint16_t m_size = gm->sizes[0];
  uint16_t n_size = gm->sizes[1];
  uint32_t dst_D = gm->strides[0];

  set_nd_para(1, 1, 1);
  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_gm_intrin(
      copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE>{
          gm_ptr, l0c_ptr,
          0, // sid
          n_size, m_size,
          dst_D,                          // dstStride_dst_D
          m_tile_ceil,                    // srcStride
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split,
          true // NZ2ND_EN
      });
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                   memref_t<__gm__ DST_TYPE, 2> *gm,
                                   int64_t pre_quant, int64_t pre_relu,
                                   bool channel_split, uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NZ_2_ND) {
    copy_matrix_cc_to_gm_nz2nd_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, gm, pre_quant, pre_relu, channel_split, unit_flag);
    return;
  }

  static_assert("fixpipe 4d unsupports this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                   memref_t<__gm__ DST_TYPE, 2> *gm,
                                   int64_t pre_quant, int64_t pre_relu,
                                   bool channel_split, uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_gm_normal_2d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, gm, pre_quant, pre_relu, channel_split, unit_flag);
    return;
  }

  static_assert("fixpipe 2d unsupports this transform mode");
}

extern "C" {
//===-------------------------------------------------------------------===//
// fixpipe, 2 dim to 2 dim, normal
//===-------------------------------------------------------------------===//
REGISTE_FIXPIPE(cc, gm, 2, 2, float, half, normal, TransformMode::NORMAL)
REGISTE_FIXPIPE(cc, gm, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL)

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2nd
//===-------------------------------------------------------------------===//
REGISTE_FIXPIPE(cc, gm, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int16_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND)
}