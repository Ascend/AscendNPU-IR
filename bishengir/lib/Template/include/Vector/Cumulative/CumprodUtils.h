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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst);

#define DECLARE_CUMPROD(dim, dtype)                                            \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_cumprod_ra_##dtype( \
      memref_t<__ubuf__ dtype, dim> *src, memref_t<__ubuf__ dtype, dim> *dst)

#define REGISTE_CUMPROD(dim, dtype)                                            \
  DECLARE_CUMPROD(dim, dtype) { vector_cumprod_ra<dtype>(src, dst); }

extern "C" {
//===-------------------------------------------------------------------===//
// cumprod ra, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_CUMPROD(2, int16_t);
DECLARE_CUMPROD(2, int32_t);
DECLARE_CUMPROD(2, half);
DECLARE_CUMPROD(2, float);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H