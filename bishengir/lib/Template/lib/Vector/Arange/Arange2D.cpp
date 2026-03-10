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

#include "Utils.h"
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/VecUtils.h"
#include <cstdint>
#include <type_traits>

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_2d_core(__ubuf__ T *dst_ptr, int64_t size0, int64_t size1,
               int64_t offset, int64_t stride0, int64_t stride1,
               int64_t dst_strides0) {
  static_assert((sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
                "Arange_2d_core do not support this data type");

  // scalar write to UB
  // TODO: use vadds() when the last dimension is aligned
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
      if constexpr (std::is_same<T, half>::value) {
        dst_ptr[i * dst_strides0 + j] = {
            static_cast<unsigned short>(offset + stride1 * j + stride0 * i)};
      } else {
        dst_ptr[i * dst_strides0 + j] = offset + stride1 * j + stride0 * i;
      }
#else
      dst_ptr[i * dst_strides0 + j] = offset + stride1 * j + stride0 * i;
#endif
    }
  }
}

template <typename T,
          typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 ||
                                             sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
arange_2d(memref_t<__ubuf__ T, 2> *dst, int64_t offset, int64_t stride0,
          int64_t stride1) {

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  int64_t size1 = dst->sizes[1];
  int64_t size0 = dst->sizes[0];
  // if continuous, view as 1D
  if (size1 == stride0 / stride1) {
    arange_1d_core<T>(dst_ptr, size1 * size0, offset, stride1);
  }

  arange_2d_core<T>(dst_ptr, size0, size1, offset, stride0, stride1,
                    dst->strides[0]);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Vector arange 1d
//===-------------------------------------------------------------------===//
REGISTE_ARANGE_2D(int16_t)
REGISTE_ARANGE_2D(int32_t)
REGISTE_ARANGE_2D(half)
REGISTE_ARANGE_2D(float)
REGISTE_ARANGE_2D(int64_t)
}