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

#ifndef HIVM_MLIR_TEMPLATE_SIMT_GATHER_OUT_TO_UB_H
#define HIVM_MLIR_TEMPLATE_SIMT_GATHER_OUT_TO_UB_H

#include "Utils.h"
#if defined(__DAV_C310__)

#define DECLARE_GATHER_OUT_TO_UB(dim, dtype, itype, ...)                       \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_gather_out_to_ub_##dim##d_##dtype##_##itype(                \
          memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ itype, dim> *idx,  \
          memref_t<__ubuf__ dtype, dim> *dst, const int64_t indexBoundary,     \
          const int32_t dimension, __VA_ARGS__)

#define DECLARE_GATHER_OUT_TO_UB_1D(dim, dtype, itype)                         \
  DECLARE_GATHER_OUT_TO_UB(dim, dtype, itype, const int64_t srcStride0,        \
                           const int32_t endOffset0,                           \
                           const int32_t startOffset0)

#define REGISTER_GATHER_OUT_TO_UB_1D(dim, dtype, itype)                        \
  DECLARE_GATHER_OUT_TO_UB_1D(dim, dtype, itype) {                             \
    int64_t srcStride[1] = {srcStride0};                                       \
    int32_t endOffset[1] = {endOffset0};                                       \
    int32_t startOffset[1] = {startOffset0};                                   \
    GatherOutToUB<dim, dtype, itype>(src, idx, dst, srcStride, endOffset,      \
                                     startOffset, dimension, indexBoundary);   \
  }

#define DECLARE_GATHER_OUT_TO_UB_2D(dim, dtype, itype)                         \
  DECLARE_GATHER_OUT_TO_UB(                                                    \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t startOffset0, const int32_t startOffset1)

#define REGISTER_GATHER_OUT_TO_UB_2D(dim, dtype, itype)                        \
  DECLARE_GATHER_OUT_TO_UB_2D(dim, dtype, itype) {                             \
    int64_t srcStride[2] = {srcStride0, srcStride1};                           \
    int32_t endOffset[2] = {endOffset0, endOffset1};                           \
    int32_t startOffset[2] = {startOffset0, startOffset1};                     \
    GatherOutToUB<dim, dtype, itype>(src, idx, dst, srcStride, endOffset,      \
                                     startOffset, dimension, indexBoundary);   \
  }

#define DECLARE_GATHER_OUT_TO_UB_3D(dim, dtype, itype)                         \
  DECLARE_GATHER_OUT_TO_UB(                                                    \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int64_t srcStride2, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2)

#define REGISTER_GATHER_OUT_TO_UB_3D(dim, dtype, itype)                        \
  DECLARE_GATHER_OUT_TO_UB_3D(dim, dtype, itype) {                             \
    int64_t srcStride[3] = {srcStride0, srcStride1, srcStride2};               \
    int32_t endOffset[3] = {endOffset0, endOffset1, endOffset2};               \
    int32_t startOffset[3] = {startOffset0, startOffset1, startOffset2};       \
    GatherOutToUB<dim, dtype, itype>(src, idx, dst, srcStride, endOffset,      \
                                     startOffset, dimension, indexBoundary);   \
  }

#define DECLARE_GATHER_OUT_TO_UB_4D(dim, dtype, itype)                         \
  DECLARE_GATHER_OUT_TO_UB(                                                    \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int64_t srcStride2, const int64_t srcStride3,                      \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t endOffset2, const int32_t endOffset3,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3)

#define REGISTER_GATHER_OUT_TO_UB_4D(dim, dtype, itype)                        \
  DECLARE_GATHER_OUT_TO_UB_4D(dim, dtype, itype) {                             \
    int64_t srcStride[4] = {srcStride0, srcStride1, srcStride2, srcStride3};   \
    int32_t endOffset[4] = {endOffset0, endOffset1, endOffset2, endOffset3};   \
    int32_t startOffset[4] = {startOffset0, startOffset1, startOffset2,        \
                              startOffset3};                                   \
    GatherOutToUB<dim, dtype, itype>(src, idx, dst, srcStride, endOffset,      \
                                     startOffset, dimension, indexBoundary);   \
  }

#define DECLARE_GATHER_OUT_TO_UB_5D(dim, dtype, itype)                         \
  DECLARE_GATHER_OUT_TO_UB(                                                    \
      dim, dtype, itype, const int64_t srcStride0, const int64_t srcStride1,   \
      const int64_t srcStride2, const int64_t srcStride3,                      \
      const int64_t srcStride4, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t endOffset3, const int32_t endOffset4,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3,                  \
      const int32_t startOffset4)

#define REGISTER_GATHER_OUT_TO_UB_5D(dim, dtype, itype)                        \
  DECLARE_GATHER_OUT_TO_UB_5D(dim, dtype, itype) {                             \
    int64_t srcStride[5] = {srcStride0, srcStride1, srcStride2, srcStride3,    \
                            srcStride4};                                       \
    int32_t endOffset[5] = {endOffset0, endOffset1, endOffset2, endOffset3,    \
                            endOffset4};                                       \
    int32_t startOffset[5] = {startOffset0, startOffset1, startOffset2,        \
                              startOffset3, startOffset4};                     \
    GatherOutToUB<dim, dtype, itype>(src, idx, dst, srcStride, endOffset,      \
                                     startOffset, dimension, indexBoundary);   \
  }
#endif
#endif