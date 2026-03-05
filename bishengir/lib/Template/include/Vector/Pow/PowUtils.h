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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_POW_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_POW_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_pow_1d(memref_t<__ubuf__ T, 1> *base, memref_t<__ubuf__ T, 1> *exp,
              memref_t<__ubuf__ T, 1> *res, memref_t<__ubuf__ T, 1> *tmp_buf);

#define DECLARE_POW(dim, dtype)                                                \
  __aiv__                                                                      \
      __attribute__((always_inline)) void _mlir_ciface_vpow_##dim##d_##dtype(  \
          memref_t<__ubuf__ dtype, dim> *base,                                 \
          memref_t<__ubuf__ dtype, dim> *exp,                                  \
          memref_t<__ubuf__ dtype, dim> *res,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp_buf)

#define REGISTE_POW(dim, dtype)                                                \
  DECLARE_POW(dim, dtype) {                                                    \
    vector_pow_##dim##d<dtype>(base, exp, res, tmp_buf);                       \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// pow, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_POW(1, int32_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_POW_UTILS_H