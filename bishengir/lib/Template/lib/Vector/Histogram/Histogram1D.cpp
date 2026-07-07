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

#include "Vector/Histogram/HistogramUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d(memref_t<__ubuf__ T, 1> *src,
             memref_t<__ubuf__ int32_t, 1> *dst,
             int64_t num_bins) {
  const int64_t src_size = src->sizes[0];
  const int64_t src_stride = src->strides[0];
  const int64_t bins_stride = dst->strides[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ int32_t *bins_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  using U = typename std::make_unsigned<T>::type;
  for (int64_t i = 0; i < src_size; ++i) {
    uint64_t value = static_cast<uint64_t>(static_cast<U>(src_ptr[i * src_stride]));
    if (value >= num_bins) {
        continue;
    }
    *(bins_ptr + value * bins_stride) += static_cast<int32_t>(1);
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d_masked(memref_t<__ubuf__ T, 1> *src,
                    memref_t<__ubuf__ int32_t, 1> *dst,
                    memref_t<__ubuf__ bool, 1> *mask,
                    int64_t num_bins) {
  const int64_t src_size = src->sizes[0];
  const int64_t src_stride = src->strides[0];
  const int64_t bins_stride = dst->strides[0];
  const int64_t mask_stride = mask->strides[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ int32_t *bins_ptr = dst->aligned + dst->offset;
  __ubuf__ bool *mask_ptr = mask->aligned + mask->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  using U = typename std::make_unsigned<T>::type;
  for (int64_t i = 0; i < src_size; ++i) {
    if (!mask_ptr[i * mask_stride]) {
      continue;
    }
    uint64_t value = static_cast<uint64_t>(static_cast<U>(src_ptr[i * src_stride]));
    if (value >= num_bins) {
      continue;
    }
    *(bins_ptr + value * bins_stride) += static_cast<int32_t>(1);
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
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
