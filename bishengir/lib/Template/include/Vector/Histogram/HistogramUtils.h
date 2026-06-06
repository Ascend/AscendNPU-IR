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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_HISTOGRAM_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_HISTOGRAM_UTILS_H

#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d(memref_t<__ubuf__ T, 1> *src,
             memref_t<__ubuf__ int32_t, 1> *dst,
             int64_t num_bins);

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d_masked(memref_t<__ubuf__ T, 1> *src,
                    memref_t<__ubuf__ int32_t, 1> *dst,
                    memref_t<__ubuf__ bool, 1> *mask,
                    int64_t num_bins);

#define DECLARE_HISTOGRAM(dim, dtype)                                         \
  __aiv__ __attribute__((always_inline)) void                                 \
    _mlir_ciface_histogram_##dim##d_##dtype(                                  \
      memref_t<__ubuf__ dtype, dim> *src,                                     \
      int64_t num_bins,                                                       \
      memref_t<__ubuf__ int32_t, dim> *dst)

#define REGISTE_HISTOGRAM(dim, dtype)                                         \
  DECLARE_HISTOGRAM(dim, dtype) {                                             \
    histogram_##dim##d<dtype>(src, dst, num_bins);                            \
  }

#define DECLARE_HISTOGRAM_MASKED(dim, dtype)                                  \
  __aiv__ __attribute__((always_inline)) void                                 \
    _mlir_ciface_histogram_##dim##d_masked_##dtype(                           \
      memref_t<__ubuf__ dtype, dim> *src,                                     \
      int64_t num_bins,                                                       \
      memref_t<__ubuf__ bool, dim> *mask,                                     \
      memref_t<__ubuf__ int32_t, dim> *dst)

#define REGISTE_HISTOGRAM_MASKED(dim, dtype)                                  \
  DECLARE_HISTOGRAM_MASKED(dim, dtype) {                                      \
    histogram_##dim##d_masked<dtype>(src, dst, mask, num_bins);               \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// histogram, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_HISTOGRAM(1, uint8_t);
DECLARE_HISTOGRAM(1, int8_t);
DECLARE_HISTOGRAM(1, uint16_t);
DECLARE_HISTOGRAM(1, int16_t);
DECLARE_HISTOGRAM(1, uint32_t);
DECLARE_HISTOGRAM(1, int32_t);
DECLARE_HISTOGRAM(1, uint64_t);
DECLARE_HISTOGRAM(1, int64_t);
DECLARE_HISTOGRAM_MASKED(1, uint8_t);
DECLARE_HISTOGRAM_MASKED(1, int8_t);
DECLARE_HISTOGRAM_MASKED(1, uint16_t);
DECLARE_HISTOGRAM_MASKED(1, int16_t);
DECLARE_HISTOGRAM_MASKED(1, uint32_t);
DECLARE_HISTOGRAM_MASKED(1, int32_t);
DECLARE_HISTOGRAM_MASKED(1, uint64_t);
DECLARE_HISTOGRAM_MASKED(1, int64_t);
}
#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_HISTOGRAM_UTILS_H
