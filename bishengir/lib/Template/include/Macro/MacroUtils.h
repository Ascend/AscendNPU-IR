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

#ifndef HIVM_MLIR_TEMPLATE_MACRO_UTILS_H
#define HIVM_MLIR_TEMPLATE_MACRO_UTILS_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#include "Utils.h"

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_XtransposeA_XtransposeB(memref_t<__gm__ ElementA, 2> *A,
                                              memref_t<__gm__ ElementB, 2> *B,
                                              memref_t<__gm__ ElementC, 2> *C);

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_transposeA_XtransposeB(memref_t<__gm__ ElementA, 2> *A,
                                             memref_t<__gm__ ElementB, 2> *B,
                                             memref_t<__gm__ ElementC, 2> *C);

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_XtransposeA_transposeB(memref_t<__gm__ ElementA, 2> *A,
                                             memref_t<__gm__ ElementB, 2> *B,
                                             memref_t<__gm__ ElementC, 2> *C);

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_transposeA_transposeB(memref_t<__gm__ ElementA, 2> *A,
                                            memref_t<__gm__ ElementB, 2> *B,
                                            memref_t<__gm__ ElementC, 2> *C);

#define DECLARE_MATMUL_XB_XD(transpose_a, transpose_b, a_type, b_type, c_type)                                 \
  __aicore__ __attribute__((always_inline)) void                                                               \
      _mlir_ciface_matmul_Xbias_Xdescale_##transpose_a##_##transpose_b##_TA##a_type##_TB##b_type##_TC##c_type( \
          memref_t<__gm__ a_type, 2> *A, memref_t<__gm__ b_type, 2> *B,                                        \
          memref_t<__gm__ c_type, 2> *C)

#define REGISTE_MATMUL_XB_XD(transpose_a, transpose_b, a_type, b_type, c_type) \
  DECLARE_MATMUL_XB_XD(transpose_a, transpose_b, a_type, b_type, c_type) {     \
    matmul_Xbias_Xdescale_##transpose_a##_##transpose_b<a_type, b_type,        \
                                                        c_type>(A, B, C);      \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// matmul, Xbias, Xdescale
//===-------------------------------------------------------------------===//
DECLARE_MATMUL_XB_XD(XtransposeA, XtransposeB, half, half, half);
DECLARE_MATMUL_XB_XD(transposeA, XtransposeB, half, half, half);
DECLARE_MATMUL_XB_XD(XtransposeA, transposeB, half, half, half);
DECLARE_MATMUL_XB_XD(transposeA, transposeB, half, half, half);
DECLARE_MATMUL_XB_XD(XtransposeA, XtransposeB, int8_t, int8_t, half);
DECLARE_MATMUL_XB_XD(transposeA, XtransposeB, int8_t, int8_t, half);
DECLARE_MATMUL_XB_XD(XtransposeA, transposeB, int8_t, int8_t, half);
DECLARE_MATMUL_XB_XD(transposeA, transposeB, int8_t, int8_t, half);
}

#endif // HIVM_MLIR_TEMPLATE_MACRO_UTILS_H