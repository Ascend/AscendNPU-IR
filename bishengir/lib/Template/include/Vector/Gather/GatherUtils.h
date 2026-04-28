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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_GATHER_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_GATHER_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

#define DECLARE_FLIP(dim, dtype)                                               \
  __aiv__                                                                      \
      __attribute__((always_inline)) void _mlir_ciface_flip_##dim##d_##dtype(  \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ dtype, dim> *dst)

#define REGISTE_FLIP(dim, dtype)                                               \
  DECLARE_FLIP(dim, dtype) { flip_##dim##d<dtype>(src, dst); }

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *indices,
          memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ int32_t, 1> *tmp_buf);

#define DECLARE_GATHER(dim, dtype)                                             \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_gather_##dim##d_##dtype(                                    \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ int32_t, dim> *indices,                            \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ int32_t, dim> *tmp_buf)

#define REGISTE_GATHER(dim, dtype)                                             \
  DECLARE_GATHER(dim, dtype) {                                                 \
    gather_##dim##d<dtype>(src, indices, dst, tmp_buf);                        \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// flip, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_FLIP(1, uint16_t);
DECLARE_FLIP(1, uint32_t);
DECLARE_FLIP(1, int16_t);
DECLARE_FLIP(1, int32_t);
DECLARE_FLIP(1, int8_t);
DECLARE_FLIP(1, uint8_t);
DECLARE_FLIP(1, int64_t);
DECLARE_FLIP(1, uint64_t);
DECLARE_FLIP(1, half);
DECLARE_FLIP(1, float);
DECLARE_FLIP(1, bfloat16_t);

//===-------------------------------------------------------------------===//
// gather, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_GATHER(1, uint16_t);
DECLARE_GATHER(1, uint32_t);
DECLARE_GATHER(1, int16_t);
DECLARE_GATHER(1, int32_t);
DECLARE_GATHER(1, half);
DECLARE_GATHER(1, float);
DECLARE_GATHER(1, bfloat16_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_GATHER_UTILS_H
