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

#ifndef HIVM_MLIR_TEMPLATE_CUBE_UTILS_H
#define HIVM_MLIR_TEMPLATE_CUBE_UTILS_H
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

#endif // HIVM_MLIR_TEMPLATE_CUBE_UTILS_H
