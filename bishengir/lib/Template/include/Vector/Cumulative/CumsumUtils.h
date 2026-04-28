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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMSUM_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMSUM_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst);

#define DECLARE_CUMSUM(dim, dtype)                                             \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_cumsum_ra_##dtype(  \
      memref_t<__ubuf__ dtype, dim> *src, memref_t<__ubuf__ dtype, dim> *dst)

#define REGISTE_CUMSUM(dim, dtype)                                             \
  DECLARE_CUMSUM(dim, dtype) { vector_cumsum_ra<dtype>(src, dst); }

extern "C" {
//===-------------------------------------------------------------------===//
// cumsum ra, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_CUMSUM(2, int16_t);
DECLARE_CUMSUM(2, int32_t);
DECLARE_CUMSUM(2, half);
DECLARE_CUMSUM(2, float);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMSUM_UTILS_H