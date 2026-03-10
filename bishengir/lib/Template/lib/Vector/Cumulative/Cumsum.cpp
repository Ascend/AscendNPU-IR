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

#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Cumulative/CumOpUtils.h"
#include "Vector/Cumulative/CumsumUtils.h"
#include "Vector/VecUtils.h"

/// cumsum ra op description:
/// Returns the cumulative sum of elements of input in the first axis.
///
/// \param src (type: memref<a x b x T>)
/// \param dst (type: memref<a x b x T>)
///
/// Constraints:
/// 1. cumsum ra op supports int16_t, int32_t, float16 and float32 types.
/// 2. cumsum ra op only accepts 2d type memrefs as src and dst.
/// 3. r axis should be aligned to ub_block_unit.
/// 4. a axis should be continuous.
/// 5. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
template <typename T, bool reverse>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  static_assert_supported_type<T>();
  check_inputs_of_cumop<T, 2>(src, dst);
  vector_cum_op_ra<VectorOpTy::VADD, T, reverse>(src, dst);
}

template <typename T, bool reverse>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ara(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
  static_assert_supported_type<T>();
  check_inputs_of_cumop<T, 3>(src, dst);
  vector_cum_op_ara<VectorOpTy::VADD, T, reverse>(src, dst);
}

extern "C" {
//===-------------------------------------------------------------------===//
// cumsum ra, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_CUMSUM(ra, 2, int16_t)
REGISTE_CUMSUM(ra, 2, int32_t)
REGISTE_CUMSUM(ra, 2, half)
REGISTE_CUMSUM(ra, 2, float)

REGISTE_REVERSE_CUMSUM(ra, 2, int16_t)
REGISTE_REVERSE_CUMSUM(ra, 2, int32_t)
REGISTE_REVERSE_CUMSUM(ra, 2, half)
REGISTE_REVERSE_CUMSUM(ra, 2, float)
//===-------------------------------------------------------------------===//
// cumsum ara, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_CUMSUM(ara, 3, int16_t)
REGISTE_CUMSUM(ara, 3, int32_t)
REGISTE_CUMSUM(ara, 3, half)
REGISTE_CUMSUM(ara, 3, float)

REGISTE_REVERSE_CUMSUM(ara, 3, int16_t)
REGISTE_REVERSE_CUMSUM(ara, 3, int32_t)
REGISTE_REVERSE_CUMSUM(ara, 3, half)
REGISTE_REVERSE_CUMSUM(ara, 3, float)
}
