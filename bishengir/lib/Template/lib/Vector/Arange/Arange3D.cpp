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
arange_3d_core(__ubuf__ T *dst_ptr, int64_t size0, int64_t size1, int64_t size2,
               int64_t offset, int64_t stride0, int64_t stride1,
               int64_t stride2, int64_t dst_stride0, int64_t dst_stride1) {
  static_assert((sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
                "Arange_3d_core do not support this data type");

  // scalar write to UB
  // TODO: use vadds() when the last dimension is aligned
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
        if constexpr (std::is_same<T, half>::value) {
          dst_ptr[i * dst_stride0 + j * dst_stride1 + k] = {
              static_cast<unsigned short>(offset + stride2 * k + stride1 * j +
                                          stride0 * i)};
        } else {
          dst_ptr[i * dst_stride0 + j * dst_stride1 + k] =
              offset + stride2 * k + stride1 * j + stride0 * i;
        }
#else
        dst_ptr[i * dst_stride0 + j * dst_stride1 + k] =
            offset + stride2 * k + stride1 * j + stride0 * i;
#endif
      }
    }
  }
}

template <typename T,
          typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 ||
                                             sizeof(T) == 8>::type>
__aiv__ __attribute__((always_inline)) void
arange_3d(memref_t<__ubuf__ T, 3> *dst, int64_t offset, int64_t stride0,
          int64_t stride1, int64_t stride2) {

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;

  int64_t size2 = dst->sizes[2];
  int64_t size1 = dst->sizes[1];
  int64_t size0 = dst->sizes[0];
  // if continuous, view as 1D
  if ((size1 == stride1 / stride2) && (size1 * size2 == stride0 / stride2)) {
    arange_1d_core<T>(dst_ptr, size2 * size1 * size0, offset, stride2);
  }

  arange_3d_core<T>(dst_ptr, size0, size1, size2, offset, stride0, stride1,
                    stride2, dst->strides[0], dst->strides[1]);
}

extern "C" {
//===-------------------------------------------------------------------===//
// Vector arange 3d
//===-------------------------------------------------------------------===//
REGISTE_ARANGE_3D(int16_t)
REGISTE_ARANGE_3D(int32_t)
REGISTE_ARANGE_3D(half)
REGISTE_ARANGE_3D(float)
REGISTE_ARANGE_3D(int64_t)
}