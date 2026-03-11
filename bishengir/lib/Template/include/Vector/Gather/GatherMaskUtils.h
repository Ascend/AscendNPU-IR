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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_GATHER_MASK_BASIC_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_GATHER_MASK_BASIC_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_mask_1d(memref_t<__ubuf__ T, 1> *src,
               memref_t<__ubuf__ bool, 1> *mask,
               memref_t<__ubuf__ T, 1> *dst,
               memref_t<__ubuf__ int64_t, 1> *dst_size);  

#define DECLARE_GATHER_MASK(dim, dtype)                                        \
  __aiv__ __attribute__((always_inline)) void                                  \
    _mlir_ciface_vgathermask_##dim##d_##dtype(                                 \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ bool, dim> *mask,                                  \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ int64_t, dim> *dst_size)  


#define REGISTE_GATHER_MASK(dim, dtype)                                        \
  DECLARE_GATHER_MASK(dim, dtype) {                                            \
    gather_mask_1d<dtype>(src, mask, dst, dst_size);                           \
  }

extern "C" {
DECLARE_GATHER_MASK(1, uint16_t);
DECLARE_GATHER_MASK(1, uint32_t);
DECLARE_GATHER_MASK(1, int16_t);
DECLARE_GATHER_MASK(1, int32_t);
DECLARE_GATHER_MASK(1, half);
DECLARE_GATHER_MASK(1, float);
DECLARE_GATHER_MASK(1, bfloat16_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_GATHER_MASK_BASIC_UTILS_H