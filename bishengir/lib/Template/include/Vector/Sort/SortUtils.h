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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_SORT_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_SORT_UTILS_H

#include "Vector/VecUtils.h"
#include <limits>

constexpr uint32_t BIT_SORT_NUM_PER_REPEAT = 32;
constexpr float FLOAT_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr float FLOAT_NEG_INF = -std::numeric_limits<float>::infinity();
constexpr int S32_MIN_VALUE = 0x80000000;
constexpr int16_t S16_MIN_VALUE = 0x8000;
constexpr int PROPOSALS_BYTES = 8;
constexpr int MERGE_SORT_MAX_PROPOSALS_LIST = 4;

// Xt[11:8]: to indicate which of the 4 input list is valid
constexpr int WAY2_CONFIG_MODE = 0x300; // 0011 0000 0000
constexpr int WAY3_CONFIG_MODE = 0x700; // 0111 0000 0000
constexpr int WAY4_CONFIG_MODE = 0xF00; // 1111 0000 0000

// Record the tail block information left over from the last merge
struct merge_remainder_info {
  // Whether there is a tail block left in the previous round of merge
  bool merge_remainder_flag;
  // The number of proposals to be sorted in the tail block left over from the
  // previous merge round
  int64_t merge_remainder_num;
};

#define DECLARE_SORT(dim, dtype)                                               \
  __aiv__                                                                      \
      __attribute__((always_inline)) void _mlir_ciface_sort_##dim##d_##dtype(  \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp, bool descending)

#define REGISTE_SORT(dim, dtype)                                               \
  DECLARE_SORT(dim, dtype) { sort_##dim##d<dtype>(src, dst, tmp, descending); }

#define DECLARE_SORT_WITH_INDEX(dim, dtype)                                    \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_sort_with_index_##dim##d_##dtype(                           \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ int32_t, dim> *index,                              \
          memref_t<__ubuf__ dtype, 1> *tmp, bool descending)

#define REGISTE_SORT_WITH_INDEX(dim, dtype)                                    \
  DECLARE_SORT_WITH_INDEX(dim, dtype) {                                        \
    sort_##dim##d_with_index<dtype>(src, dst, index, tmp, descending);         \
  }

extern "C" {
//===----------------------------------------------------------------------===//
// sort, 1 dim
//===----------------------------------------------------------------------===//
DECLARE_SORT(1, half);
DECLARE_SORT(1, float);
DECLARE_SORT_WITH_INDEX(1, half);
DECLARE_SORT_WITH_INDEX(1, float);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_SORT_UTILS_H
