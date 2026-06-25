/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef HIVM_MLIR_TEMPLATE_SIMT_GATHER_UB_TO_UB_H
#define HIVM_MLIR_TEMPLATE_SIMT_GATHER_UB_TO_UB_H

#include "Utils.h"
#if defined(__DAV_C310__)

// UB-to-UB SIMT gather. Differences from OutToUB:
//   1) src is an N-dim __ubuf__ memref with its own sizes/strides;
//   2) No external startOffset/endOffset (UB tensors are naturally bounded
//      by src->sizes[] / dst->sizes[] / indices->sizes[]);
//   3) indexBoundary = src->sizes[dimension], injected by the dispatcher;
//   4) Negative indices are folded (indexVal += indexBoundary); out-of-range
//      after folding are skipped via `continue`.
#define REGISTER_GATHER_UB_TO_UB(dim, dtype, itype)                            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_gather_simt_##dim##d_##dtype##_##itype(                     \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ itype, dim> *indices,                              \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          const int32_t dimension) {                                           \
    GatherUBToUB<dim, dtype, itype>(src, indices, dst, dimension);             \
  }
#endif
#endif // HIVM_MLIR_TEMPLATE_SIMT_GATHER_UB_TO_UB_H
