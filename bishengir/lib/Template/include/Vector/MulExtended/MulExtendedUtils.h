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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_MUL_EXTENDED_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_MUL_EXTENDED_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void vector_mul_extended_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ int16_t, 1> *dst0, memref_t<__ubuf__ int16_t, 1> *dst1,
    memref_t<__ubuf__ int32_t, 1> *tmp_buf);

#define DECLARE_MULEXTENDED(dim, dtype)                                        \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_mul_extended_##dim##d_##dtype(                              \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *src1,                                 \
          memref_t<__ubuf__ int16_t, dim> *dst0,                               \
          memref_t<__ubuf__ int16_t, dim> *dst1,                               \
          memref_t<__ubuf__ int32_t, 1> *tmp_buf)

#define REGISTE_MULEXTENDED(dim, dtype)                                        \
  DECLARE_MULEXTENDED(dim, dtype) {                                            \
    vector_mul_extended_##dim##d<dtype>(src0, src1, dst0, dst1, tmp_buf);      \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// mul_extended, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_MULEXTENDED(1, int16_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_MUL_EXTENDED_UTILS_H