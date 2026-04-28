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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H

#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst, bool reverse);

extern "C" {
#define DECLARE_CUMPROD(dtype)                                                 \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_cumprod_ra_##dtype( \
      memref_t<__ubuf__ dtype, 2> *src, memref_t<__ubuf__ dtype, 2> *dst, bool reverse = false)

#define REGISTER_CUMPROD(dtype)                                                  \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_cumprod_ra_##dtype(  \
      memref_t<__ubuf__ dtype, 2> *src, memref_t<__ubuf__ dtype, 2> *dst, bool reverse)
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst, bool reverse);

extern "C" {
#define DECLARE_CUMSUM(dtype)                                                  \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_cumsum_ra_##dtype(  \
      memref_t<__ubuf__ dtype, 2> *src, memref_t<__ubuf__ dtype, 2> *dst, bool reverse = false)

#define REGISTER_CUMSUM(dtype)                                                  \
  __aiv__ __attribute__((always_inline)) void _mlir_ciface_cumsum_ra_##dtype(  \
      memref_t<__ubuf__ dtype, 2> *src, memref_t<__ubuf__ dtype, 2> *dst, bool reverse)
}

template <typename SRC_T, typename DST_T = SRC_T>
struct cumulative_args {
  memref_t<__ubuf__ DST_T, 1> *dst;
  memref_t<__ubuf__ SRC_T, 1> *src[2];
  int64_t dst_stride;
  int64_t src_stride[2];
  int64_t src_size;
  bool reverse;
};

template <typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
cumprod_2d(cumulative_args<SRC_TYPE, DST_TYPE> args);

template <>
__aiv__ __attribute__((always_inline)) void
cumprod_2d(cumulative_args<uint8_t, uint8_t> args);

template <>
__aiv__ __attribute__((always_inline)) void
cumprod_2d(cumulative_args<int8_t, int8_t> args);

template <typename SRC_TYPE, typename DST_TYPE>
__aiv__ __attribute__((always_inline)) void
cumsum_2d(cumulative_args<SRC_TYPE, DST_TYPE> args);

extern "C" {
//===-------------------------------------------------------------------===//
// cumprod ra, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_CUMPROD(int8_t);
DECLARE_CUMPROD(uint8_t);
DECLARE_CUMPROD(int16_t);
DECLARE_CUMPROD(uint16_t);
DECLARE_CUMPROD(int32_t);
DECLARE_CUMPROD(uint32_t);
DECLARE_CUMPROD(half);
DECLARE_CUMPROD(float);
DECLARE_CUMPROD(bfloat16_t);

DECLARE_CUMSUM(int8_t);
DECLARE_CUMSUM(uint8_t);
DECLARE_CUMSUM(int16_t);
DECLARE_CUMSUM(uint16_t);
DECLARE_CUMSUM(int32_t);
DECLARE_CUMSUM(uint32_t);
DECLARE_CUMSUM(half);
DECLARE_CUMSUM(float);
DECLARE_CUMSUM(bfloat16_t);
}
#endif // defined(__DAV_C310__)
#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CUMPROD_UTILS_H