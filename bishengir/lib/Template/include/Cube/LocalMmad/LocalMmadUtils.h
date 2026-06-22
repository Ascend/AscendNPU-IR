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
#include "../Utils.h"

constexpr uint32_t HF32_CTRL_REGISTER_BIT = 49;

enum class Load2DTransposeMode : uint8_t {
  ADDADDR = 0,
  SUBADDR = 1,
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
img2colv2_cbuf_to_cb_intrin_core(img2colv2_intrin_args<T, DST_QUALIFER> args) {
  INTRINSIC(img2colv2_cbuf_to_cb, args.dst_ptr, args.src_ptr, args.stepK,
            args.stepM, args.posK, args.posM, args.strideW, args.strideH,
            args.Wk, args.Hk, args.dilationW, args.dilationH, args.filterW,
            args.filterH, args.transpose, args.fmatrixCtrl, args.sizeChannel);
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

template<typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
load2d_transpose_cbuf_to_cb_intrin_core_s4(
    load2d_transpose_intrin_args<void, DST_QUALIFER> args) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  INTRINSIC(load_cbuf_to_cb_transpose_s4, args.dst_ptr, args.src_ptr, args.indexID,
            args.repeat, args.srcStride, args.dstStride, uint8_t(args.addrmode),
            args.dstFracStride);
#else
  INTRINSIC(load_cbuf_to_cb_transpose_s4, args.dst_ptr, args.src_ptr, args.indexID,
            args.repeat, args.srcStride, args.dstStride, args.addrmode,
            args.dstFracStride);
#endif
}

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
    mma_tile_core<src_type, dst_type, bias_type, false, false, false, false>(  \
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
    mma_tile_bias<src_type, dst_type, bias_type, false, false, false, false>(  \
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
    mma_tile_core<src_type, dst_type, bias_type, true, false, false, false>(   \
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
    mma_tile_core<src_type, dst_type, bias_type, false, true, false, false>(   \
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
    mma_tile_core<src_type, dst_type, bias_type, true, true, false, false>(    \
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
    mma_tile_core<src_type, dst_type, bias_type, false, false, true, false>(   \
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
    mma_tile_core<src_type, dst_type, bias_type, true, false, true, false>(    \
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
    mma_tile_core<src_type, dst_type, bias_type, false, true, true, false>(    \
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
    mma_tile_core<src_type, dst_type, bias_type, true, true, true, false>(     \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_I4(src_scope, dst_scope, dim, src_type,               \
                            dst_type, bias_type)                               \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_i4(                   \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_I4(src_scope, dst_scope, dim, src_type,              \
                             dst_type, bias_type)                              \
  DECLARE_MMA_TILE_I4(src_scope, dst_scope, dim, src_type, dst_type,           \
                      bias_type) {                                             \
    mma_tile_core<src_type, dst_type, bias_type, false, false, false, true>(   \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }


#define DECLARE_MMA_TILE_TB_I4(src_scope, dst_scope, dim, src_type,            \
                               dst_type, bias_type)                            \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_tb_i4(                \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TB_I4(src_scope, dst_scope, dim, src_type,           \
                                dst_type, bias_type)                           \
  DECLARE_MMA_TILE_TB_I4(src_scope, dst_scope, dim, src_type, dst_type,        \
                         bias_type) {                                          \
    mma_tile_core<src_type, dst_type, bias_type, false, true, false, true>(    \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_HF32_I4(src_scope, dst_scope, dim, src_type,          \
                                 dst_type, bias_type)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_hf32_i4(              \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_HF32_I4(src_scope, dst_scope, dim, src_type,         \
                                     dst_type, bias_type)                      \
  DECLARE_MMA_TILE_HF32_I4(src_scope, dst_scope, dim, src_type, dst_type,      \
                           bias_type) {                                        \
    mma_tile_core<src_type, dst_type, bias_type, false, false, true, true>(    \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TB_HF32_I4(src_scope, dst_scope, dim, src_type,       \
                                    dst_type, bias_type)                       \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_tb_hf32_i4(           \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TB_HF32_I4(src_scope, dst_scope, dim, src_type,      \
                                     dst_type, bias_type)                      \
  DECLARE_MMA_TILE_TB_HF32_I4(src_scope, dst_scope, dim, src_type, dst_type,   \
                              bias_type) {                                     \
    mma_tile_core<src_type, dst_type, bias_type, false, true, true, true>(     \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

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
DECLARE_MMA_TILE_I4(cbuf, cc, 4, int8_t, int32_t, float);
DECLARE_MMA_TILE_TB_I4(cbuf, cc, 4, int8_t, int32_t, float);
DECLARE_MMA_TILE_HF32_I4(cbuf, cc, 4, int8_t, int32_t, float);
DECLARE_MMA_TILE_TB_HF32_I4(cbuf, cc, 4, int8_t, int32_t, float);
}
#endif // HIVM_MLIR_TEMPLATE_MMA_TILE_UTILS_H