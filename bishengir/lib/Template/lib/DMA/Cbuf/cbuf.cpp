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

/// gm -> cbuf, direct load 1d
template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_1d_core(memref_t<__gm__ T, 1> *src,
                        memref_t<__cbuf__ T, 1> *dst, pad_t pad_mode) {
  if (is_no_op<1>(src->sizes)) {
    return;
  }

  const int64_t stride0_gm = src->strides[0];
  const int64_t stride0_cbuf = dst->strides[0];

  if (stride0_gm == 1 && stride0_cbuf == 1) [[likely]] {
    // last dimension is contiguous
    load_gm_to_cbuf_1d_core_with_contiguous_last_dim<T>(src, dst, pad_mode);
    return;
  }

  static_assert("not implemented non-contiguous last dimension.");
}

