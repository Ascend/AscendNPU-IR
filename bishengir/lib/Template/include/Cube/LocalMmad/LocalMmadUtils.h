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

#ifndef HIVM_MLIR_TEMPLATE_MMA_TILE_UTILS_H
#define HIVM_MLIR_TEMPLATE_MMA_TILE_UTILS_H
#include "Utils.h"

constexpr uint32_t HF32_CTRL_REGISTER_BIT = 46;

enum class Load2DTransposeMode : uint8_t {
  ADDADDR = 0,
  SUBADDR = 1,
};

template <typename SRC_TYPE, typename DST_TYPE>
struct mmad_intrin_args {
  __cc__ DST_TYPE *dst_ptr;
  __ca__ SRC_TYPE *src0_ptr;
  __cb__ SRC_TYPE *src1_ptr;
  uint16_t m;
  uint16_t k;
  uint16_t n;
  uint8_t unitFlag;
  bool kDirectionAlign;
  bool cmatrixSource;
  uint8_t cmatrixInitVal;
};

template <typename T, typename DST_QUALIFER>
struct img2colv2_intrin_args {
  DST_QUALIFER *dst_ptr;
  __cbuf__ T *src_ptr;
  uint16_t stepK;
  uint16_t stepM;
  uint16_t posK;
  uint16_t posM;
  uint8_t strideW;
  uint8_t strideH;
  uint8_t Wk;
  uint8_t Hk;
  uint8_t dilationW;
  uint8_t dilationH;
  bool filterW;
  bool filterH;
  bool transpose;
  bool fmatrixCtrl;
  uint16_t sizeChannel;
};

template <typename T, typename DST_QUALIFER>
struct load2d_transpose_intrin_args {
  DST_QUALIFER *dst_ptr;
  __cbuf__ T *src_ptr;
  uint16_t indexID;
  uint8_t repeat;
  uint16_t srcStride;
  uint16_t dstStride;
  Load2DTransposeMode addrmode;
  uint16_t dstFracStride;
};

template <typename T, typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
img2colv2_cbuf_to_ca_intrin_core(img2colv2_intrin_args<T, DST_QUALIFER> args) {
  INTRINSIC(img2colv2_cbuf_to_ca, args.dst_ptr, args.src_ptr, args.stepK,
            args.stepM, args.posK, args.posM, args.strideW, args.strideH,
            args.Wk, args.Hk, args.dilationW, args.dilationH, args.filterW,
            args.filterH, args.transpose, args.fmatrixCtrl, args.sizeChannel);
}

template <typename T, typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
img2colv2_cbuf_to_cb_intrin_core(img2colv2_intrin_args<T, DST_QUALIFER> args) {
  INTRINSIC(img2colv2_cbuf_to_cb, args.dst_ptr, args.src_ptr, args.stepK,
            args.stepM, args.posK, args.posM, args.strideW, args.strideH,
            args.Wk, args.Hk, args.dilationW, args.dilationH, args.filterW,
            args.filterH, args.transpose, args.fmatrixCtrl, args.sizeChannel);
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
mad_intrin_core(mmad_intrin_args<SRC_TYPE, DST_TYPE> args) {
  INTRINSIC(mad, args.dst_ptr, args.src0_ptr, args.src1_ptr, args.m, args.k,
            args.n, args.unitFlag, args.kDirectionAlign, args.cmatrixSource,
            args.cmatrixInitVal);
}

template <typename T, typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
load2d_transpose_cbuf_to_cb_intrin_core(
    load2d_transpose_intrin_args<T, DST_QUALIFER> args) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  INTRINSIC(load_cbuf_to_cb_transpose, args.dst_ptr, args.src_ptr, args.indexID,
            args.repeat, args.srcStride, args.dstStride, uint8_t(args.addrmode),
            args.dstFracStride);
#else
  INTRINSIC(load_cbuf_to_cb_transpose, args.dst_ptr, args.src_ptr, args.indexID,
            args.repeat, args.srcStride, args.dstStride, args.addrmode,
            args.dstFracStride);
#endif
}

#if !defined(__DAV_C310__)
template <typename T, typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
load2d_transpose_cbuf_to_ca_intrin_core(
    load2d_transpose_intrin_args<T, DST_QUALIFER> args) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  INTRINSIC(load_cbuf_to_ca_transpose, args.dst_ptr, args.src_ptr, args.indexID,
            args.repeat, args.srcStride, args.dstStride, uint8_t(args.addrmode),
            args.dstFracStride);
#else
  INTRINSIC(load_cbuf_to_ca_transpose, args.dst_ptr, args.src_ptr, args.indexID,
            args.repeat, args.srcStride, args.dstStride, args.addrmode,
            args.dstFracStride);
#endif
}
#endif

#define DECLARE_MMA_TILE(src_scope, dst_scope, dim, src_type, dst_type,        \
                         bias_type)                                            \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type(                        \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE(src_scope, dst_scope, dim, src_type, dst_type,       \
                          bias_type)                                           \
  DECLARE_MMA_TILE(src_scope, dst_scope, dim, src_type, dst_type, bias_type) { \
    mma_tile_core<src_type, dst_type, bias_type, false, false, false>(         \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_BIAS(src_scope, dst_scope, dim, src_type, dst_type,    \
                              bias_type)                                        \
  __aicore__ __attribute__((always_inline)) void                                \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                      \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,           \
          int64_t m, int64_t k, int64_t n,                                      \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                         \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,       \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,       \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,       \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS(src_scope, dst_scope, dim, src_type, dst_type,  \
                               bias_type)                                      \
  DECLARE_MMA_TILE_BIAS(src_scope, dst_scope, dim, src_type, dst_type,         \
                        bias_type) {                                           \
    mma_tile_bias<src_type, dst_type, bias_type, false, false, false>(         \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,          \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA(src_scope, dst_scope, dim, src_type, dst_type,     \
                            bias_type)                                         \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta(                   \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA(src_scope, dst_scope, dim, src_type, dst_type,    \
                             bias_type)                                        \
  DECLARE_MMA_TILE_TA(src_scope, dst_scope, dim, src_type, dst_type,           \
                      bias_type) {                                             \
    mma_tile_core<src_type, dst_type, bias_type, true, false, false>(          \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TB(src_scope, dst_scope, dim, src_type, dst_type,     \
                            bias_type)                                         \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_tb(                   \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TB(src_scope, dst_scope, dim, src_type, dst_type,    \
                             bias_type)                                        \
  DECLARE_MMA_TILE_TB(src_scope, dst_scope, dim, src_type, dst_type,           \
                      bias_type) {                                             \
    mma_tile_core<src_type, dst_type, bias_type, false, true, false>(          \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA_TB(src_scope, dst_scope, dim, src_type, dst_type,  \
                               bias_type)                                      \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta_tb(                \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA_TB(src_scope, dst_scope, dim, src_type, dst_type, \
                                bias_type)                                     \
  DECLARE_MMA_TILE_TA_TB(src_scope, dst_scope, dim, src_type, dst_type,        \
                         bias_type) {                                          \
    mma_tile_core<src_type, dst_type, bias_type, true, true, false>(           \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_HF32(src_scope, dst_scope, dim, src_type, dst_type,   \
                              bias_type)                                       \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_hf32(                 \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_HF32(src_scope, dst_scope, dim, src_type, dst_type,  \
                               bias_type)                                      \
  DECLARE_MMA_TILE_HF32(src_scope, dst_scope, dim, src_type, dst_type,         \
                        bias_type) {                                           \
    mma_tile_core<src_type, dst_type, bias_type, false, false, true>(          \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA_HF32(src_scope, dst_scope, dim, src_type,          \
                                 dst_type, bias_type)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta_hf32(              \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA_HF32(src_scope, dst_scope, dim, src_type,         \
                                  dst_type, bias_type)                         \
  DECLARE_MMA_TILE_TA_HF32(src_scope, dst_scope, dim, src_type, dst_type,      \
                           bias_type) {                                        \
    mma_tile_core<src_type, dst_type, bias_type, true, false, true>(           \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TB_HF32(src_scope, dst_scope, dim, src_type,          \
                                 dst_type, bias_type)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_tb_hf32(              \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TB_HF32(src_scope, dst_scope, dim, src_type,         \
                                  dst_type, bias_type)                         \
  DECLARE_MMA_TILE_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type,      \
                           bias_type) {                                        \
    mma_tile_core<src_type, dst_type, bias_type, false, true, true>(           \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA_TB_HF32(src_scope, dst_scope, dim, src_type,       \
                                    dst_type, bias_type)                       \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta_tb_hf32(           \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA_TB_HF32(src_scope, dst_scope, dim, src_type,      \
                                     dst_type, bias_type)                      \
  DECLARE_MMA_TILE_TA_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type,   \
                              bias_type) {                                     \
    mma_tile_core<src_type, dst_type, bias_type, true, true, true>(            \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_MX(src_type, dst_type, bias_type)                          \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_mmadmxL1_##src_type##_to_##dst_type(                            \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,  \
      memref_t<__cbuf__ src_type, 4> *l1B,                                     \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                               \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                    \
      uint32_t m, uint32_t k, uint32_t n,                                      \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_BIAS(src_type, dst_type, bias_type)                     \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_mmadmxL1_with_##bias_type##_bias_##src_type##_to_##dst_type(    \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,  \
      memref_t<__cbuf__ src_type, 4> *l1B,                                     \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                               \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                    \
      uint32_t m, uint32_t k, uint32_t n,                                      \
      memref_t<__cbuf__ bias_type, 4> *bias,                                   \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_TRANS(src_type, dst_type, bias_type, suffix)            \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_mmadmxL1_##src_type##_to_##dst_type##suffix(                    \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,  \
      memref_t<__cbuf__ src_type, 4> *l1B,                                     \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                               \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                    \
      uint32_t m, uint32_t k, uint32_t n,                                      \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_BIAS_TRANS(src_type, dst_type, bias_type, suffix)       \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_mmadmxL1_with_##bias_type##_bias_##src_type##_to_##dst_type##suffix( \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,  \
      memref_t<__cbuf__ src_type, 4> *l1B,                                     \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                               \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                    \
      uint32_t m, uint32_t k, uint32_t n,                                      \
      memref_t<__cbuf__ bias_type, 4> *bias,                                   \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_FORMAT(src_type, dst_type, bias_type, a_format,                              \
                              b_format)                                                             \
  __aicore__ __attribute__((always_inline)) void                                                    \
  _mlir_ciface_mmadmxL1_##src_type##_to_##dst_type##_lhs_format_##a_format##_rhs_format_##b_format( \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,                       \
      memref_t<__cbuf__ src_type, 4> *l1B,                                                          \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                                                    \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                                         \
      uint32_t m, uint32_t k, uint32_t n,                                                           \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                                     \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_BIAS_FORMAT(src_type, dst_type, bias_type, a_format,                         \
                                   b_format)                                                         \
  __aicore__ __attribute__((always_inline)) void                                                    \
  _mlir_ciface_mmadmxL1_with_##bias_type##_bias_##src_type##_to_##dst_type##_lhs_format_##a_format##_rhs_format_##b_format( \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,                       \
      memref_t<__cbuf__ src_type, 4> *l1B,                                                          \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                                                    \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                                         \
      uint32_t m, uint32_t k, uint32_t n,                                                           \
      memref_t<__cbuf__ bias_type, 4> *bias,                                                        \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                                     \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_FORMAT_TRANS(src_type, dst_type, bias_type, a_format,                                \
                                    b_format, suffix)                                                       \
  __aicore__ __attribute__((always_inline)) void                                                            \
  _mlir_ciface_mmadmxL1_##src_type##_to_##dst_type##suffix##_lhs_format_##a_format##_rhs_format_##b_format( \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,                               \
      memref_t<__cbuf__ src_type, 4> *l1B,                                                                  \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                                                            \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                                                 \
      uint32_t m, uint32_t k, uint32_t n,                                                                   \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                                             \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define DECLARE_MMA_MX_BIAS_FORMAT_TRANS(src_type, dst_type, bias_type, a_format,                           \
                                         b_format, suffix)                                                  \
  __aicore__ __attribute__((always_inline)) void                                                            \
  _mlir_ciface_mmadmxL1_with_##bias_type##_bias_##src_type##_to_##dst_type##suffix##_lhs_format_##a_format##_rhs_format_##b_format( \
      memref_t<__cc__ dst_type, 4> *l0C, memref_t<__cbuf__ src_type, 4> *l1A,                               \
      memref_t<__cbuf__ src_type, 4> *l1B,                                                                  \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleA,                                                            \
      memref_t<__cbuf__ uint8_t, 1> *l1MxScaleB, bool init,                                                 \
      uint32_t m, uint32_t k, uint32_t n,                                                                   \
      memref_t<__cbuf__ bias_type, 4> *bias,                                                                \
      uint32_t l1AMTE2MTE1EventId, uint32_t l1AMTE1MTE2EventId,                                             \
      uint32_t l1BMTE2MTE1EventId, uint32_t l1BMTE1MTE2EventId)

#define REGISTER_MMA_MX(src_type, dst_type, bias_type)                         \
  DECLARE_MMA_MX(src_type, dst_type, bias_type) {                              \
    mmamx_tile_core<src_type, dst_type, bias_type>(                            \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n,                                                         \
        l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,                                \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId);                               \
  }

#define REGISTER_MMA_MX_BIAS(src_type, dst_type, bias_type)                    \
  DECLARE_MMA_MX_BIAS(src_type, dst_type, bias_type) {                         \
    mmamx_tile_bias<src_type, dst_type, bias_type>(                            \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n, bias,                                                   \
        l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,                                \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId);                               \
  }

#define REGISTER_MMA_MX_TRANS(src_type, dst_type, bias_type, suffix, ta, tb)   \
  DECLARE_MMA_MX_TRANS(src_type, dst_type, bias_type, suffix) {                \
    mmamx_tile_core<src_type, dst_type, bias_type, ta, tb>(                    \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n,                                                         \
        l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,                                \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId);                               \
  }

#define REGISTER_MMA_MX_BIAS_TRANS(src_type, dst_type, bias_type, suffix, ta,  \
                                   tb)                                         \
  DECLARE_MMA_MX_BIAS_TRANS(src_type, dst_type, bias_type, suffix) {           \
    mmamx_tile_bias<src_type, dst_type, bias_type, ta, tb>(                    \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n, bias,                                                   \
        l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,                                \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId);                               \
  }

#define REGISTER_MMA_MX_FORMAT(src_type, dst_type, bias_type, a_format,        \
                               b_format, lhs_format, rhs_format)               \
  DECLARE_MMA_MX_FORMAT(src_type, dst_type, bias_type, a_format, b_format) {   \
    mmamx_tile_core<src_type, dst_type, bias_type>(                            \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n, l1AMTE2MTE1EventId, l1AMTE1MTE2EventId, l1BMTE2MTE1EventId,   \
        l1BMTE1MTE2EventId, lhs_format, rhs_format);                           \
  }

#define REGISTER_MMA_MX_BIAS_FORMAT(src_type, dst_type, bias_type, a_format,   \
                                    b_format, lhs_format, rhs_format)          \
  DECLARE_MMA_MX_BIAS_FORMAT(src_type, dst_type, bias_type, a_format,          \
                             b_format) {                                       \
    mmamx_tile_bias<src_type, dst_type, bias_type>(                            \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n, bias, l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,           \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId, lhs_format, rhs_format);      \
  }

#define REGISTER_MMA_MX_FORMAT_TRANS(src_type, dst_type, bias_type, a_format,  \
                                     b_format, suffix, ta, tb, lhs_format,     \
                                     rhs_format)                               \
  DECLARE_MMA_MX_FORMAT_TRANS(src_type, dst_type, bias_type, a_format,         \
                              b_format, suffix) {                              \
    mmamx_tile_core<src_type, dst_type, bias_type, ta, tb>(                    \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n, l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,                      \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId, lhs_format, rhs_format);       \
  }

#define REGISTER_MMA_MX_BIAS_FORMAT_TRANS(src_type, dst_type, bias_type,       \
                                          a_format, b_format, suffix, ta, tb,  \
                                          lhs_format, rhs_format)              \
  DECLARE_MMA_MX_BIAS_FORMAT_TRANS(src_type, dst_type, bias_type, a_format,    \
                                   b_format, suffix) {                         \
    mmamx_tile_bias<src_type, dst_type, bias_type, ta, tb>(                    \
        l0C, l1A, l1B,                                                         \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleA, 1> *>(l1MxScaleA), \
        reinterpret_cast<memref_t<__cbuf__ ElementMxScaleB, 1> *>(l1MxScaleB), \
        init, m, k, n, bias, l1AMTE2MTE1EventId, l1AMTE1MTE2EventId,           \
        l1BMTE2MTE1EventId, l1BMTE1MTE2EventId, lhs_format, rhs_format);      \
  }

#define REGISTER_MMA_MX_FP4(src_type, dst_type, bias_type, a_format, b_format) \
  REGISTER_MMA_MX_FORMAT(src_type, dst_type, bias_type, a_format, b_format,    \
                         Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T,      \
                         Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T)

#define REGISTER_MMA_MX_FP4_TRANS(src_type, dst_type, bias_type, a_format,     \
                                  b_format, suffix, ta, tb)                   \
  REGISTER_MMA_MX_FORMAT_TRANS(src_type, dst_type, bias_type, a_format,        \
                               b_format, suffix, ta, tb,                      \
                               Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T, \
                               Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T)

extern "C" {
DECLARE_MMA_TILE(cbuf, cc, 4, half, float, float);
DECLARE_MMA_TILE(cbuf, cc, 4, bfloat16_t, float, float);
DECLARE_MMA_TILE(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE(cbuf, cc, 4, int8_t, int32_t, int32_t);
DECLARE_MMA_TILE_HF32(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_BIAS(cbuf, cc, 4, half, float, float);
DECLARE_MMA_TILE_BIAS(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_BIAS(cbuf, cc, 4, bfloat16_t, float, float);
DECLARE_MMA_TILE_BIAS(cbuf, cc, 4, int8_t, int32_t, int32_t);
DECLARE_MMA_TILE_TA(cbuf, cc, 4, half, float, float);
DECLARE_MMA_TILE_TA(cbuf, cc, 4, bfloat16_t, float, float);
DECLARE_MMA_TILE_TA(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_TA_HF32(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_TA(cbuf, cc, 4, int8_t, int32_t, int32_t);
DECLARE_MMA_TILE_TB(cbuf, cc, 4, half, float, float);
DECLARE_MMA_TILE_TB(cbuf, cc, 4, bfloat16_t, float, float);
DECLARE_MMA_TILE_TB(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
DECLARE_MMA_TILE_TB_HF32(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_TA_TB(cbuf, cc, 4, half, float, float);
DECLARE_MMA_TILE_TA_TB(cbuf, cc, 4, bfloat16_t, float, float);
DECLARE_MMA_TILE_TA_TB(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_TA_TB_HF32(cbuf, cc, 4, float, float, float);
DECLARE_MMA_TILE_TA_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
DECLARE_MMA_TILE_BIAS(cbuf, cc, 4, half, float, half);
DECLARE_MMA_TILE_BIAS(cbuf, cc, 4, float, float, half);
#if defined(__DAV_C310__)
DECLARE_MMA_MX(float8_e5m2_t, float, float);
DECLARE_MMA_MX(float8_e4m3_t, float, float);
DECLARE_MMA_MX_BIAS(float8_e5m2_t, float, float);
DECLARE_MMA_MX_BIAS(float8_e4m3_t, float, float);
DECLARE_MMA_MX_TRANS(float8_e5m2_t, float, float, _ta);
DECLARE_MMA_MX_TRANS(float8_e5m2_t, float, float, _tb);
DECLARE_MMA_MX_TRANS(float8_e5m2_t, float, float, _ta_tb);
DECLARE_MMA_MX_TRANS(float8_e4m3_t, float, float, _ta);
DECLARE_MMA_MX_TRANS(float8_e4m3_t, float, float, _tb);
DECLARE_MMA_MX_TRANS(float8_e4m3_t, float, float, _ta_tb);
DECLARE_MMA_MX_BIAS_TRANS(float8_e5m2_t, float, float, _ta);
DECLARE_MMA_MX_BIAS_TRANS(float8_e5m2_t, float, float, _tb);
DECLARE_MMA_MX_BIAS_TRANS(float8_e5m2_t, float, float, _ta_tb);
DECLARE_MMA_MX_BIAS_TRANS(float8_e4m3_t, float, float, _ta);
DECLARE_MMA_MX_BIAS_TRANS(float8_e4m3_t, float, float, _tb);
DECLARE_MMA_MX_BIAS_TRANS(float8_e4m3_t, float, float, _ta_tb);
DECLARE_MMA_MX_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t, _ta);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t, _tb);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                            _ta_tb);
DECLARE_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                                 _ta);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                                 _tb);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                                 _ta_tb);
DECLARE_MMA_MX_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t, _ta);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t, _tb);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                            _ta_tb);
DECLARE_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                                 _ta);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                                 _tb);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                                 _ta_tb);
DECLARE_MMA_MX_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t, _ta);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t, _tb);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                            _ta_tb);
DECLARE_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                                 _ta);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                                 _tb);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                                 _ta_tb);
DECLARE_MMA_MX_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t, _ta);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t, _tb);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                            _ta_tb);
DECLARE_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                                 _ta);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                                 _tb);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                                 _ta_tb);
DECLARE_MMA_MX_FORMAT(int8_t, float, float, fp4x2_e2m1_t, fp4x2_e2m1_t);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                            fp4x2_e2m1_t, _ta);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                            fp4x2_e2m1_t, _tb);
DECLARE_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                            fp4x2_e2m1_t, _ta_tb);
DECLARE_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp4x2_e2m1_t, fp4x2_e2m1_t);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                                 fp4x2_e2m1_t, _ta);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                                 fp4x2_e2m1_t, _tb);
DECLARE_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                                 fp4x2_e2m1_t, _ta_tb);
#endif
}
#endif // HIVM_MLIR_TEMPLATE_MMA_TILE_UTILS_H
