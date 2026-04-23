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

#ifndef HIVM_MLIR_TEMPLATE_CONV2D_GROUP_UTILS_H
#define HIVM_MLIR_TEMPLATE_CONV2D_GROUP_UTILS_H
#include "Utils.h"

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
  bool cmatrixInitVal;
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
__aicore__ __attribute__((always_inline)) void
img2colv2_cbuf_to_ca_intrin_core(img2colv2_intrin_args<T, DST_QUALIFER> args) {
  INTRINSIC(img2colv2_cbuf_to_ca, args.dst_ptr, args.src_ptr, args.stepK,
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

#define DECLARE_CONV2D_GROUP(src_scope, dst_scope, dim, src_type, dst_type)    \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_conv2d_group_##src_type##_to_##dst_type(                    \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          memref_t<__##dst_scope##__ dst_type, 4> *dst, int64_t G,             \
          int64_t padT, int64_t padB, int64_t padL, int64_t padR,              \
          int64_t strideH, int64_t strideW, int64_t dilationH,                 \
          int64_t dilationW, int64_t conv_l1_wait_l1a_event,                   \
          int64_t conv_l1_wait_l1b_event, int64_t l1a_wait_conv_l1_event,      \
          int64_t l1b_wait_conv_l1_event,                                      \
          int64_t back_pipe_m_pipe_mte1_db_event0,                             \
          int64_t back_pipe_m_pipe_mte1_db_event1)

#define REGISTER_CONV2D_GROUP(src_scope, dst_scope, dim, src_type, dst_type)   \
  DECLARE_CONV2D_GROUP(src_scope, dst_scope, dim, src_type, dst_type) {        \
    conv2d_group<src_type, dst_type>(                                          \
        src0, src1, init, dst, G, padT, padB, padL, padR, strideH, strideW,    \
        dilationH, dilationW, conv_l1_wait_l1a_event, conv_l1_wait_l1b_event,  \
        l1a_wait_conv_l1_event, l1b_wait_conv_l1_event,                        \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1);     \
  }

extern "C" {
DECLARE_CONV2D_GROUP(cbuf, cc, 5, float, float);
DECLARE_CONV2D_GROUP(cbuf, cc, 5, half, float);
DECLARE_CONV2D_GROUP(cbuf, cc, 5, bfloat16_t, float);
}
#endif // HIVM_MLIR_TEMPLATE_CONV2D_GROUP_UTILS_H
