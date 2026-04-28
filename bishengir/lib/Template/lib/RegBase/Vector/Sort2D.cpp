/**
  copy_ubuf_to_ubuf_1d_core(src, dst_value);
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
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/Sort/SortUtils.h"

/**
 * @brief  Inline 2D sorting function that operates on UB (Unified Buffer)
 * memory for Bisheng NPU. This function sorts each row of the 2D memref
 * independently by converting 2D rows to 1D memrefs and invoking the
 * corresponding 1D sorting implementation.
 *
 * @tparam T  Data type of the elements to be sorted, restricted to half (f16)
 *          or float (f32) by static assertion.
 * @param  src       Pointer to the input 2D UB memref (source data to be
 *          sorted).
 * @param  dst       Pointer to the output 2D UB memref (stores the sorted
 *          result).
 * @param  tmp_buf   Pointer to the temporary 2D UB memref (used as auxiliary
 *          buffer for 1D sorting internal operations).
 * @param  descending  Boolean flag to control sorting order: true for
 *          descending sort (from large to small), false for ascending sort
 *          (from small to large).
 *
 * @note   Make sure the sum of all bufs is less than UB_MAX.
 */
template <typename T>
__aiv__ __attribute__((always_inline)) void
sort_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
        memref_t<__ubuf__ T, 1> *tmp_buf, bool descending) {
  static_assert(
      (std::is_same<T, half>::value || std::is_same<T, float>::value ||
       std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value) &&
      "Sort unsupport this type");
  for (int i = 0; i < src->sizes[0]; i++) {
    memref_t<__ubuf__ T, 1> src_row{src->aligned,
                                    src->allocated,
                                    src->offset + i * src->strides[0],
                                    {src->sizes[1]},
                                    {1}};
    memref_t<__ubuf__ T, 1> dst_row{dst->aligned,
                                    dst->allocated,
                                    dst->offset + i * dst->strides[0],
                                    {dst->sizes[1]},
                                    {1}};
    if constexpr (std::is_same<T, half>::value)
      _mlir_ciface_sort_1d_half(&src_row, &dst_row, tmp_buf, descending);
    else if constexpr (std::is_same<T, float>::value)
      _mlir_ciface_sort_1d_float(&src_row, &dst_row, tmp_buf, descending);
    else if constexpr (std::is_same<T, int32_t>::value)
      _mlir_ciface_sort_1d_int32_t(&src_row, &dst_row, tmp_buf, descending);
    else if constexpr (std::is_same<T, int64_t>::value)
      _mlir_ciface_sort_1d_int64_t(&src_row, &dst_row, tmp_buf, descending);
  }
}

extern "C" {
//===----------------------------------------------------------------------===//
// sort, 2 dim
//===----------------------------------------------------------------------===//
REGISTE_SORT(2, half);
REGISTE_SORT(2, float);
REGISTE_SORT(2, int32_t);
REGISTE_SORT(2, int64_t);
}
