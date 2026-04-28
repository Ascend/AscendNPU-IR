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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_INTERLEAVE_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_INTERLEAVE_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void
interleave_vsel_1d(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
                   memref_t<__ubuf__ T, 1> *dst,
                   memref_t<__ubuf__ T, 1> *tmp_buf);

template <typename T>
__aiv__ __attribute__((always_inline)) void vector_interleave_1d(
    memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
    memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *temp);

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_interleave_1d_scalar(memref_t<__ubuf__ T, 1> *src0,
                            memref_t<__ubuf__ T, 1> *src1,
                            memref_t<__ubuf__ T, 1> *dst);

#define DECLARE_INTERLEAVE(dim, dtype)                                         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_interleave_##dim##d_##dtype(                                \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *src1,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *temp)

#define REGISTE_INTERLEAVE(dim, dtype)                                         \
  DECLARE_INTERLEAVE(dim, dtype) {                                             \
    vector_interleave_##dim##d<dtype>(src0, src1, dst, temp);                  \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// interleave, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_INTERLEAVE(1, half);
DECLARE_INTERLEAVE(1, bfloat16_t);
DECLARE_INTERLEAVE(1, float);
DECLARE_INTERLEAVE(1, int16_t);
DECLARE_INTERLEAVE(1, int32_t);
DECLARE_INTERLEAVE(1, uint16_t);
DECLARE_INTERLEAVE(1, uint32_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_INTERLEAVE_UTILS_H