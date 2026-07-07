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

#if defined(__DAV_C310__)

#include "__clang_cce_simt_intrinsics.h"
#include "Vector/Histogram/HistogramUtils.h"

constexpr unsigned int MAX_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_histogram_1d(__ubuf__ T *inputs, __ubuf__ int32_t *bins,
                  int64_t input_size, int64_t input_stride,
                  int64_t bins_stride, int64_t num_bins) {
  using U = typename std::make_unsigned<T>::type;
  for (int64_t i = threadIdx.x; i < input_size; i += blockDim.x) {
    uint64_t value = static_cast<uint64_t>(static_cast<U>(inputs[i * input_stride]));
    if (value >= num_bins) {
      continue;
    }
    atomicAdd(bins + value * bins_stride, static_cast<int32_t>(1));
  }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_histogram_1d_masked(__ubuf__ T *inputs, __ubuf__ int32_t *bins, __ubuf__ bool *mask,
                         int64_t input_size, int64_t input_stride,
                         int64_t bins_stride,
                         int64_t mask_stride,
                         int64_t num_bins) {
  using U = typename std::make_unsigned<T>::type;
  for (uint32_t i = threadIdx.x; i < input_size; i += blockDim.x) {
    if (!mask[i * mask_stride]) {
      continue;
    }
    uint64_t value = static_cast<uint64_t>(static_cast<U>(inputs[i * input_stride]));
    if (value >= num_bins) {
      continue;
    }
    atomicAdd(bins + value * bins_stride, static_cast<int32_t>(1));
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *dst,
            int64_t num_bins) {
  cce::async_invoke<simt_histogram_1d<T>>(
    cce::dim3{MAX_THREAD_NUM},
    reinterpret_cast<__ubuf__ T *>(src->aligned + src->offset),
    reinterpret_cast<__ubuf__ int32_t *>(dst->aligned + dst->offset),
    src->sizes[0], src->strides[0], dst->strides[0], num_bins);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d_masked(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *dst,
                    memref_t<__ubuf__ bool, 1> *mask, int64_t num_bins) {
  cce::async_invoke<simt_histogram_1d_masked<T>>(
    cce::dim3{MAX_THREAD_NUM},
    reinterpret_cast<__ubuf__ T *>(src->aligned + src->offset),
    reinterpret_cast<__ubuf__ int32_t *>(dst->aligned + dst->offset),
    reinterpret_cast<__ubuf__ bool *>(mask->aligned + mask->offset),
    src->sizes[0], src->strides[0], dst->strides[0], mask->strides[0],
    num_bins);
}

extern "C" {
//===-------------------------------------------------------------------===//
// histogram, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_HISTOGRAM(1, uint8_t)
REGISTE_HISTOGRAM(1, int8_t)
REGISTE_HISTOGRAM(1, uint16_t)
REGISTE_HISTOGRAM(1, int16_t)
REGISTE_HISTOGRAM(1, uint32_t)
REGISTE_HISTOGRAM(1, int32_t)
REGISTE_HISTOGRAM(1, uint64_t)
REGISTE_HISTOGRAM(1, int64_t)
REGISTE_HISTOGRAM_MASKED(1, uint8_t)
REGISTE_HISTOGRAM_MASKED(1, int8_t)
REGISTE_HISTOGRAM_MASKED(1, uint16_t)
REGISTE_HISTOGRAM_MASKED(1, int16_t)
REGISTE_HISTOGRAM_MASKED(1, uint32_t)
REGISTE_HISTOGRAM_MASKED(1, int32_t)
REGISTE_HISTOGRAM_MASKED(1, uint64_t)
REGISTE_HISTOGRAM_MASKED(1, int64_t)
}
#endif
