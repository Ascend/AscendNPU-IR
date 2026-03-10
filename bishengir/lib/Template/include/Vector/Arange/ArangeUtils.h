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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_ARANGE_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_ARANGE_UTILS_H

#include <cstdint>

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_1d(memref_t<__ubuf__ T, 1> *dst, int64_t offset, int64_t stride);

#define DECLARE_ARANGE_1D(dtype)                                               \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_arange_1d_##dtype(  \
      memref_t<__ubuf__ dtype, 1> *dst, int64_t offset, int64_t stride)

#define REGISTE_ARANGE_1D(dtype)                                               \
  DECLARE_ARANGE_1D(dtype) { arange_1d<dtype>(dst, offset, stride); }

#define DECLARE_ARANGE_2D(dtype)                                               \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_arange_2d_##dtype(  \
      memref_t<__ubuf__ dtype, 2> *dst, int64_t offset, int64_t stride0,       \
      int64_t stride1)

#define REGISTE_ARANGE_2D(dtype)                                               \
  DECLARE_ARANGE_2D(dtype) { arange_2d<dtype>(dst, offset, stride0, stride1); }

#define DECLARE_ARANGE_3D(dtype)                                               \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_arange_3d_##dtype(  \
      memref_t<__ubuf__ dtype, 3> *dst, int64_t offset, int64_t stride0,       \
      int64_t stride1, int64_t stride2)

#define REGISTE_ARANGE_3D(dtype)                                               \
  DECLARE_ARANGE_3D(dtype) {                                                   \
    arange_3d<dtype>(dst, offset, stride0, stride1, stride2);                  \
  }

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_1d_core(__ubuf__ T *dst_ptr, int64_t n, int64_t offset, int64_t stride);

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_2d_core(__ubuf__ T *dst_ptr, int64_t n, int64_t offset, int64_t stride0,
               int64_t stride1);

template <typename T>
__aiv__ __attribute__((always_inline)) void
arange_3d_core(__ubuf__ T *dst_ptr, int64_t n, int64_t offset, int64_t stride0,
               int64_t stride1, int64_t stride2);

extern "C" {
//===-------------------------------------------------------------------===//
// arange 1d
//===-------------------------------------------------------------------===//
DECLARE_ARANGE_1D(int16_t);
DECLARE_ARANGE_1D(int32_t);
DECLARE_ARANGE_1D(half);
DECLARE_ARANGE_1D(float);
DECLARE_ARANGE_1D(int64_t);

DECLARE_ARANGE_2D(int16_t);
DECLARE_ARANGE_2D(int32_t);
DECLARE_ARANGE_2D(half);
DECLARE_ARANGE_2D(float);
DECLARE_ARANGE_2D(int64_t);

DECLARE_ARANGE_3D(int16_t);
DECLARE_ARANGE_3D(int32_t);
DECLARE_ARANGE_3D(half);
DECLARE_ARANGE_3D(float);
DECLARE_ARANGE_3D(int64_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_ARANGE_UTILS_H