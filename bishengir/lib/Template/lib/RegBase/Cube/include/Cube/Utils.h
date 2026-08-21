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

#ifndef HIVM_MLIR_TEMPLATE_REGBASE_CUBE_UTILS_H
#define HIVM_MLIR_TEMPLATE_REGBASE_CUBE_UTILS_H

#include "../Utils.h"

template <typename SRC_TYPE, typename DST_TYPE>
struct mmad_intrin_args {
  __cc__ DST_TYPE *dst_ptr;
  __ca__ SRC_TYPE *src0_ptr;
  __cb__ SRC_TYPE *src1_ptr;
  uint16_t m;
  uint16_t k;
  uint16_t n;
  uint8_t unitFlag;
  bool disableGemv;
  bool cmatrixSource;
  bool cmatrixInitVal;
};

template <typename T, typename DST_QUALIFIER>
struct img2colv2_intrin_args {
  DST_QUALIFIER *dst_ptr;
  __cbuf__ T *src_ptr;
  uint16_t stepK;
  uint16_t stepM;
  uint16_t posK;
  uint16_t posM;
  uint8_t strideW;
  uint8_t strideH;
  uint8_t filterW;
  uint8_t filterH;
  uint8_t dilationW;
  uint8_t dilationH;
  bool fetchFilterW;
  bool fetchFilterH;
  bool transpose;
  bool fmatrixCtrl;
  uint16_t channelSize;
};

template <typename T, typename DST_QUALIFIER>
__aicore__ __attribute__((always_inline)) void
img2colv2_cbuf_to_ca_intrin_core(
    img2colv2_intrin_args<T, DST_QUALIFIER> args) {
  img2colv2_cbuf_to_ca(
      args.dst_ptr, args.src_ptr, args.stepK, args.stepM, args.posK, args.posM,
      args.strideW, args.strideH, args.filterW, args.filterH, args.dilationW,
      args.dilationH, args.fetchFilterW, args.fetchFilterH, args.transpose,
      args.fmatrixCtrl, args.channelSize);
}

template <typename T, bool TRANSPOSE>
__aicore__ __attribute__((always_inline)) void load_cbuf_to_cb_intrin_core(
    __cb__ T *dst, __cbuf__ T *src, uint32_t mStart, uint32_t kStart,
    uint16_t mStep, uint16_t kStep, int32_t srcStride,
    uint16_t dstStride) {
  load_cbuf_to_cb(dst, src, mStart, kStart, mStep, kStep, srcStride,
                  dstStride, TRANSPOSE);
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
mad_intrin_core(mmad_intrin_args<SRC_TYPE, DST_TYPE> args) {
  mad(args.dst_ptr, args.src0_ptr, args.src1_ptr, args.m, args.k, args.n,
      args.unitFlag, args.disableGemv, args.cmatrixSource,
      args.cmatrixInitVal);
}

#endif // HIVM_MLIR_TEMPLATE_REGBASE_CUBE_UTILS_H
