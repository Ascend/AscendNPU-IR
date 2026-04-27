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
set_pre_quant_scale(float32_t quant_scale) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  // it will only consider the first 19 bits for TensorFloat32
  INTRINSIC(set_quant_pre,
            static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&quant_scale)));
#endif
}

__aicore__ __attribute__((always_inline)) void
set_fpc(uint64_t relu_pre_addr, uint64_t quant_pre_addr,
        uint64_t relu_post_addr, uint64_t quant_post_addr,
        uint64_t antiq_para_addr, uint64_t l0c_unit_flag) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(set_fpc,
            (uint64_t)(relu_pre_addr | (quant_pre_addr << 8) |
                       (relu_post_addr << 16) | (quant_post_addr << 24) |
                       (antiq_para_addr << 32) | (l0c_unit_flag << 63)));
#endif
}

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
                                          int64_t pre_quant,
                                          float32_t quant_scale,
                                          int64_t pre_relu, bool channel_split,
                                          uint8_t unit_flag) {
  __gm__ DST_TYPE *gm_ptr = gm->aligned + gm->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t src_stride = l0c->sizes[1] / FRACTAL_BLOCK_NUM;

  // burst count
  uint16_t n_size = l0c->sizes[0];
  // burst length
  uint16_t m_size = l0c->sizes[1] * sizeof(SRC_TYPE);

  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_gm_intrin(
      copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE>{
          gm_ptr, l0c_ptr,
          0, // sid
          n_size, m_size,
          src_stride,                     // dstStride_dst_D
          src_stride,                     // srcStride
          FIXPIPE_ARGS_XT1_VALUES_TO_GM
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split,
          false // NZ2ND_EN
          FIXPIPE_ARGS_XT2_VALUES(false)  // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_normal_4d_to_4d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                          memref_t<__gm__ DST_TYPE, 4> *gm,
                                          int64_t pre_quant,
                                          float32_t quant_scale,
                                          int64_t pre_relu, bool channel_split,
                                          uint8_t unit_flag) {
  __gm__ DST_TYPE *gm_ptr = gm->aligned + gm->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  // For (M,N) : nz format is n1,m1,m0,n0 with m = m1*m0, n = n1*n0
  uint16_t src_m_size = l0c->sizes[1] * l0c->sizes[2];
  uint16_t dst_m_size = gm->sizes[1] * gm->sizes[2];

  uint16_t src_n_size = l0c->sizes[0] * l0c->sizes[3];
  // We assume fully contiguous and use m,n to compute strides
  uint16_t src_stride = src_m_size; //in unit of C0
  uint16_t dst_stride = dst_m_size * FRACTAL_BLOCK_NUM; // in unit of element

  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_gm_intrin(
      copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE>{
          gm_ptr, l0c_ptr,
          0, // sid
          src_n_size, src_m_size,
          dst_stride,                     // dstStride_dst_D
          src_stride,                     // srcStride
          FIXPIPE_ARGS_XT1_VALUES_TO_GM
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split,
          false // NZ2ND_EN
          FIXPIPE_ARGS_XT2_VALUES(false)  // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_normal_2d_to_2d_core(
    memref_t<__cc__ SRC_TYPE, 2> *l0c, memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag, uint8_t dual_dst) {
  __ubuf__ DST_TYPE *ubuf_ptr = ubuf->aligned + ubuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t src_stride = l0c->sizes[1] / FRACTAL_BLOCK_NUM;

  // burst count
  uint16_t n_size = l0c->sizes[0];
  // burst length
  uint16_t m_size = l0c->sizes[1] * sizeof(SRC_TYPE);

  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_ubuf_intrin(
      copy_matrix_cc_to_ubuf_intrin_args<SRC_TYPE, DST_TYPE> {
        ubuf_ptr, l0c_ptr,
            0, // sid
            n_size, m_size,
            src_stride, // dstStride_dst_D
            src_stride, // srcStride
            FIXPIPE_ARGS_XT1_VALUES_TO_UB(dual_dst)
            unit_flag,                      // UnitFlagMode
            quant_mode,                     // QuantPRE
            static_cast<uint8_t>(pre_relu), // ReLUPRE
            channel_split,
            false  // NZ2ND_EN
            FIXPIPE_ARGS_XT2_VALUES(false) // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_normal_4d_to_4d_core(
    memref_t<__cc__ SRC_TYPE, 4> *l0c, memref_t<__ubuf__ DST_TYPE, 4> *ubuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag, uint8_t dual_dst) {
  __ubuf__ DST_TYPE *ubuf_ptr = ubuf->aligned + ubuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  // For (M,N) : nz format is n1,m1,m0,n0 with m = m1*m0, n = n1*n0
  uint16_t src_m_size = l0c->sizes[1] * l0c->sizes[2];
  uint16_t dst_m_size = ubuf->sizes[1] * ubuf->sizes[2];

  uint16_t src_n_size = l0c->sizes[0] * l0c->sizes[3];
  // We assume fully contiguous and use m,n to compute strides
  uint16_t src_stride = src_m_size; //in unit of C0
  uint16_t dst_stride = dst_m_size * FRACTAL_BLOCK_NUM; // in unit of element

  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_ubuf_intrin(
      copy_matrix_cc_to_ubuf_intrin_args<SRC_TYPE, DST_TYPE> {
        ubuf_ptr, l0c_ptr,
            0, // sid
            src_n_size, src_m_size,
            dst_stride, // dstStride_dst_D
            src_stride, // srcStride
            FIXPIPE_ARGS_XT1_VALUES_TO_UB(dual_dst)
            unit_flag,                      // UnitFlagMode
            quant_mode,                     // QuantPRE
            static_cast<uint8_t>(pre_relu), // ReLUPRE
            channel_split,
            false  // NZ2ND_EN
            FIXPIPE_ARGS_XT2_VALUES(false) // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_normal_2d_to_2d_core(
    memref_t<__cc__ SRC_TYPE, 2> *l0c, memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag) {
  __cbuf__ DST_TYPE *cbuf_ptr = cbuf->aligned + cbuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t src_stride = l0c->sizes[1] / FRACTAL_BLOCK_NUM;

  // burst count
  uint16_t n_size = l0c->sizes[0];
  // burst length
  uint16_t m_size = l0c->sizes[1] * sizeof(SRC_TYPE);

  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_cbuf_intrin(
      copy_matrix_cc_to_cbuf_intrin_args<SRC_TYPE, DST_TYPE>{
          cbuf_ptr, l0c_ptr,
          0, // sid
          n_size, m_size, src_stride,
          src_stride,                              // srcStride
          FIXPIPE_ARGS_XT1_VALUES_TO_L1 unit_flag, // UnitFlagMode
          quant_mode,                              // QuantPRE
          static_cast<uint8_t>(pre_relu),          // ReLUPRE
          channel_split,
          false                          // NZ2ND_EN
          FIXPIPE_ARGS_XT2_VALUES(false) // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_normal_4d_to_4d_core(
    memref_t<__cc__ SRC_TYPE, 4> *l0c, memref_t<__cbuf__ DST_TYPE, 4> *cbuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag) {
  __cbuf__ DST_TYPE *cbuf_ptr = cbuf->aligned + cbuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  // For (M,N) : nz format is n1,m1,m0,n0 with m = m1*m0, n = n1*n0
  uint16_t src_m_size = l0c->sizes[1] * l0c->sizes[2];
  uint16_t dst_m_size = cbuf->sizes[1] * cbuf->sizes[2];

  uint16_t src_n_size = l0c->sizes[0] * l0c->sizes[3];
  // We assume fully contiguous and use m,n to compute strides
  uint16_t src_stride = src_m_size; //in unit of C0
  uint16_t dst_stride = dst_m_size * FRACTAL_BLOCK_NUM; // in unit of element

  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_cbuf_intrin(
    copy_matrix_cc_to_cbuf_intrin_args<SRC_TYPE, DST_TYPE> {
      cbuf_ptr, l0c_ptr,
      0, //sid
      src_n_size, src_m_size,
      dst_stride,
      src_stride,                     // srcStride
      FIXPIPE_ARGS_XT1_VALUES_TO_L1 
      unit_flag,                      // UnitFlagMode
      quant_mode,                     // QuantPRE
      static_cast<uint8_t>(pre_relu), // ReLUPRE
      channel_split,
      false   // NZ2ND_EN
      FIXPIPE_ARGS_XT2_VALUES(false)  // with NZ2DN_EN control
    });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__gm__ DST_TYPE, 2> *gm,
                                         int64_t pre_quant,
                                         float32_t quant_scale,
                                         int64_t pre_relu, bool channel_split,
                                         uint8_t unit_flag) {
  __gm__ DST_TYPE *gm_ptr = gm->aligned + gm->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2];
  uint16_t m_size = gm->sizes[0];
  uint16_t n_size = gm->sizes[1];
  uint32_t dst_D = gm->strides[0];

  set_nd_para(1, 1, 1);
  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_gm_intrin(
      copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE>{
          gm_ptr, l0c_ptr,
          0, // sid
          n_size, m_size,
          dst_D,                          // dstStride_dst_D
          m_tile_ceil,                    // srcStride
          FIXPIPE_ARGS_XT1_VALUES_TO_GM
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split,
          true  // NZ2ND_EN
          FIXPIPE_ARGS_XT2_VALUES(false)  // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_nz2nd_4d_to_2d_core(
    memref_t<__cc__ SRC_TYPE, 4> *l0c, memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag, uint8_t dual_dst) {
  __ubuf__ DST_TYPE *ubuf_ptr = ubuf->aligned + ubuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2];
  uint16_t m_size = (dual_dst == DualDstMode::ROW_SPLIT) ? ubuf->sizes[0] * 2
                                                         : ubuf->sizes[0];
  uint16_t n_size = (dual_dst == DualDstMode::COLUMN_SPLIT) ? ubuf->sizes[1] * 2
                                                            : ubuf->sizes[1];
  uint32_t dst_D = ubuf->strides[0];

  set_nd_para(1,1,1);
  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_ubuf_intrin(
      copy_matrix_cc_to_ubuf_intrin_args<SRC_TYPE, DST_TYPE> {
        ubuf_ptr, l0c_ptr,
            0, // sid
            n_size, m_size,
            dst_D,       // dstStride_dst_D
            m_tile_ceil, // srcStride
            FIXPIPE_ARGS_XT1_VALUES_TO_UB(dual_dst)
            unit_flag,                      // UnitFlagMode
            quant_mode,                     // QuantPRE
            static_cast<uint8_t>(pre_relu), // ReLUPRE
            channel_split,
            true  // NZ2ND_EN
            FIXPIPE_ARGS_XT2_VALUES(false)  // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
                                         int64_t pre_quant,
                                         float32_t quant_scale,
                                         int64_t pre_relu, bool channel_split,
                                         uint8_t unit_flag) {
  __cbuf__ DST_TYPE *cbuf_ptr = cbuf->aligned + cbuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2];
  uint16_t m_size = cbuf->sizes[0];
  uint16_t n_size = cbuf->sizes[1];
  uint32_t dst_D = cbuf->strides[0];

  set_nd_para(1, 1, 1);
  set_pre_quant_scale(quant_scale);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_cbuf_intrin(
      copy_matrix_cc_to_cbuf_intrin_args<SRC_TYPE, DST_TYPE>{
          cbuf_ptr, l0c_ptr,
          0, // sid
          n_size, m_size,
          dst_D,                          // dstStride_dst_D
          m_tile_ceil,                    // srcStride
          FIXPIPE_ARGS_XT1_VALUES_TO_L1
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split,
          true  // NZ2ND_EN
          FIXPIPE_ARGS_XT2_VALUES(false)  // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_nz2dn_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__gm__ DST_TYPE, 2> *gm,
                                         int64_t pre_quant,
                                         float32_t quant_scale,
                                         int64_t pre_relu, bool channel_split,
                                         uint8_t unit_flag) {
  __gm__ DST_TYPE *gm_ptr = gm->aligned + gm->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2]; // srcStride
  uint16_t n_size = gm->sizes[0];
  uint16_t m_size = gm->sizes[1];
  uint32_t dst_D = gm->strides[0];                         // dstStride_dst_D

  set_nd_para(1,1,1);
  set_pre_quant_scale(quant_scale);

  uint64_t src_dn_stride = 1;
  uint64_t channel_para = (src_dn_stride & 0xffff) << 48;
  set_channel_para(channel_para);

  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_gm_intrin(
      copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE>{
          gm_ptr, l0c_ptr,
          0, // sid
          n_size, m_size,
          dst_D,                          // dstStride_dst_D
          m_tile_ceil,                    // srcStride
          FIXPIPE_ARGS_XT1_VALUES_TO_GM
          unit_flag,                      // UnitFlagMode
          quant_mode,                     // QuantPRE
          static_cast<uint8_t>(pre_relu), // ReLUPRE
          channel_split, 
          false   //  NZ2ND_EN
          FIXPIPE_ARGS_XT2_VALUES(true)   // with NZ2DN_EN control
    });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_nz2dn_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                           memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
                                           int64_t pre_quant,
                                           float32_t quant_scale,
                                           int64_t pre_relu, bool channel_split,
                                           uint8_t unit_flag) {
  __ubuf__ DST_TYPE *ubuf_ptr = ubuf->aligned + ubuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2];
  uint16_t n_size = ubuf->sizes[0];
  uint16_t m_size = ubuf->sizes[1];
  uint32_t dst_D = ubuf->strides[0];
  
  set_nd_para(1,1,1);
  set_pre_quant_scale(quant_scale);

  uint64_t src_dn_stride = 1;
  uint64_t channel_para = (src_dn_stride & 0xffff) << 48;
  set_channel_para(channel_para);
  
  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_ubuf_intrin(
      copy_matrix_cc_to_ubuf_intrin_args<SRC_TYPE, DST_TYPE> {
         ubuf_ptr, l0c_ptr,
             0,  // sid
             n_size, m_size, 
             dst_D,        // dstStride_dst_D
             m_tile_ceil,  // srcStride
             FIXPIPE_ARGS_XT1_VALUES_TO_UB(0)
             unit_flag,                      // unitFlagMode
             quant_mode,                     // QuantPRE
             static_cast<uint8_t>(pre_relu), // ReLUPRE
             channel_split, 
             false   // NZ2ND_EN
             FIXPIPE_ARGS_XT2_VALUES(true)   // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_nz2dn_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                           memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
                                           int64_t pre_quant,
                                           float32_t quant_scale,
                                           int64_t pre_relu, bool channel_split,
                                           uint8_t unit_flag) {
  __cbuf__ DST_TYPE *cbuf_ptr = cbuf->aligned + cbuf->offset;
  __cc__ SRC_TYPE *l0c_ptr = l0c->aligned + l0c->offset;

  uint16_t m_tile_ceil = l0c->strides[0] / l0c->strides[2];
  uint16_t n_size = cbuf->sizes[0];
  uint16_t m_size = cbuf->sizes[1];
  uint32_t dst_D = cbuf->strides[0];
  
  set_nd_para(1,1,1);
  set_pre_quant_scale(quant_scale);

  uint64_t src_dn_stride = 1;    
  uint64_t channel_para = (src_dn_stride & 0xffff) << 48;
  set_channel_para(channel_para);
  
  QuantMode_t quant_mode = get_quant_mode(pre_quant);
  copy_matrix_cc_to_cbuf_intrin(
      copy_matrix_cc_to_cbuf_intrin_args<SRC_TYPE, DST_TYPE> {
         cbuf_ptr, l0c_ptr,
             0,  // sid
             n_size, m_size, 
             dst_D,        // dstStride_dst_D
             m_tile_ceil,  // srcStride
             FIXPIPE_ARGS_XT1_VALUES_TO_L1
             unit_flag,                      // unitFlagMode
             quant_mode,                     // QuantPRE
             static_cast<uint8_t>(pre_relu), // ReLUPRE
             channel_split, 
             false   // NZ2ND_EN
             FIXPIPE_ARGS_XT2_VALUES(true)   // with NZ2DN_EN control
      });
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                   memref_t<__gm__ DST_TYPE, 2> *gm,
                                   int64_t pre_quant, float32_t quant_scale,
                                   int64_t pre_relu, bool channel_split,
                                   uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NZ_2_ND) {
    copy_matrix_cc_to_gm_nz2nd_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, gm, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }
  
  if constexpr (MODE == TransformMode::NZ_2_DN) {
    copy_matrix_cc_to_gm_nz2dn_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, gm, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }

  static_assert("fixpipe 4d unsupports this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                     memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
                                     int64_t pre_quant, float32_t quant_scale,
                                     int64_t pre_relu, bool channel_split,
                                     uint8_t unit_flag, uint8_t dual_dst = 0) {
  if constexpr (MODE == TransformMode::NZ_2_ND) {
    copy_matrix_cc_to_ubuf_nz2nd_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, ubuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag,
        dual_dst);
    return;
  }

  if constexpr (MODE == TransformMode::NZ_2_DN) {
    copy_matrix_cc_to_ubuf_nz2dn_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, ubuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
  }

  static_assert("fixpipe 4d unsupports this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                     memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
                                     int64_t pre_quant, float32_t quant_scale,
                                     int64_t pre_relu, bool channel_split,
                                     uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NZ_2_ND) {
    copy_matrix_cc_to_cbuf_nz2nd_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, cbuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }

  if constexpr (MODE == TransformMode::NZ_2_DN) {
    copy_matrix_cc_to_cbuf_nz2dn_4d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, cbuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
  }

  static_assert("fixpipe 4d unsupports this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                   memref_t<__gm__ DST_TYPE, 2> *gm,
                                   int64_t pre_quant, float32_t quant_scale,
                                   int64_t pre_relu, bool channel_split,
                                   uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_gm_normal_2d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, gm, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }

  static_assert("fixpipe 2d unsupports this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                     memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
                                     int64_t pre_quant, float32_t quant_scale,
                                     int64_t pre_relu, bool channel_split,
                                     uint8_t unit_flag, uint8_t dual_dst = 0) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_ubuf_normal_2d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, ubuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag,
        dual_dst);
    return;
  }

  static_assert("fixpipe 2d unsupports this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                     memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
                                     int64_t pre_quant, float32_t quant_scale,
                                     int64_t pre_relu, bool channel_split,
                                     uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_cbuf_normal_2d_to_2d_core<SRC_TYPE, DST_TYPE>(
        l0c, cbuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }
    
    static_assert("fixpipe 2d unsupport this transform mode");
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_4d_to_4d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                   memref_t<__gm__ DST_TYPE, 4> *gm,
                                   int64_t pre_quant, float32_t quant_scale,
                                   int64_t pre_relu, bool channel_split,
                                   uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_gm_normal_4d_to_4d_core<SRC_TYPE, DST_TYPE>(
        l0c, gm, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_4d_to_4d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                     memref_t<__ubuf__ DST_TYPE, 4> *ubuf,
                                     int64_t pre_quant, float32_t quant_scale,
                                     int64_t pre_relu, bool channel_split,
                                     uint8_t unit_flag, uint8_t dual_dst = 0) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_ubuf_normal_4d_to_4d_core<SRC_TYPE, DST_TYPE>(
        l0c, ubuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag,
        dual_dst);
    return;
  }
}

template <typename SRC_TYPE, typename DST_TYPE, TransformMode MODE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_4d_to_4d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                     memref_t<__cbuf__ DST_TYPE, 4> *cbuf,
                                     int64_t pre_quant, float32_t quant_scale,
                                     int64_t pre_relu, bool channel_split,
                                     uint8_t unit_flag) {
  if constexpr (MODE == TransformMode::NORMAL) {
    copy_matrix_cc_to_cbuf_normal_4d_to_4d_core<SRC_TYPE, DST_TYPE>(
        l0c, cbuf, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);
    return;
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// fixpipe, 2 dim to 2 dim, normal
//===-------------------------------------------------------------------===//
REGISTE_FIXPIPE(cc, gm, 2, 2, float, half, normal, TransformMode::NORMAL)
REGISTE_FIXPIPE(cc, gm, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 2, 2, float, half, normal, TransformMode::NORMAL)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL)

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2nd
//===-------------------------------------------------------------------===//
REGISTE_FIXPIPE(cc, gm, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND)

REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND)

#if !defined(__DAV_M300__)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND)

REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND)

#if !defined(__DAV_C310__)
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int16_t, nz2nd, TransformMode::NZ_2_ND)
REGISTE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, int16_t, nz2nd, TransformMode::NZ_2_ND)
#endif // !defined(__DAV_C310__)
#endif // !defined(__DAV_M300__)
}
