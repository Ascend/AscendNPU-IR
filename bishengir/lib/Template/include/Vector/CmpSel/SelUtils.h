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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_SEL_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_SEL_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

//===----------------------------------------------------------------------===//
// vsel for vector-vector Inputs
//===----------------------------------------------------------------------===//
template <typename T, typename COND_T = bool>
__aiv__ __attribute__((always_inline)) void
vector_select_vv_1d(memref_t<__ubuf__ COND_T, 1> *condition,
                    memref_t<__ubuf__ T, 1> *src0,
                    memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf);

#define DECLARE_VSEL_VV_WITHOUT_TMP(dim, condtype, dtype)                      \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_vv_##dim##d_##condtype##_##dtype(                      \
          memref_t<__ubuf__ condtype, dim> *condition,                         \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *src1,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf = nullptr)

#define DECLARE_VSEL_VV(dim, condtype, dtype)                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_vv_##dim##d_##condtype##_##dtype(                      \
          memref_t<__ubuf__ condtype, dim> *condition,                         \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *src1,                                 \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf)

#define REGISTE_VSEL_VV(dim, condtype, dtype)                                  \
  DECLARE_VSEL_VV(dim, condtype, dtype) {                                      \
    vector_select_vv_##dim##d<dtype, condtype>(condition, src0, src1, dst,     \
                                               condition_addr_buf);            \
  }

//===----------------------------------------------------------------------===//
// vsel for scalar-scalar Inputs
//===----------------------------------------------------------------------===//
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_ss_1d(memref_t<__ubuf__ bool, 1> *condition, T src0, T src1,
                    memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf);

#define DECLARE_VSEL_SS_WITHOUT_TMP(dim, dtype)                                \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_ss_##dim##d_##dtype(                                   \
          memref_t<__ubuf__ bool, dim> *condition, dtype src0, dtype src1,     \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf = nullptr)

#define DECLARE_VSEL_SS(dim, dtype)                                            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_ss_##dim##d_##dtype(                                   \
          memref_t<__ubuf__ bool, dim> *condition, dtype src0, dtype src1,     \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf)

#define REGISTE_VSEL_SS(dim, dtype)                                            \
  DECLARE_VSEL_SS(dim, dtype) {                                                \
    vector_select_ss_##dim##d<dtype>(condition, src0, src1, dst,               \
                                     condition_addr_buf);                      \
  }

//===----------------------------------------------------------------------===//
// vsel for vector-scalar Inputs
//===----------------------------------------------------------------------===//
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_vs_1d(memref_t<__ubuf__ bool, 1> *condition,
                    memref_t<__ubuf__ T, 1> *src0, T src1,
                    memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf);

#define DECLARE_VSEL_VS_WITHOUT_TMP(dim, dtype)                                \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_vs_##dim##d_##dtype(                                   \
          memref_t<__ubuf__ bool, dim> *condition,                             \
          memref_t<__ubuf__ dtype, 1> *src0, dtype src1,                       \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf = nullptr)

#define DECLARE_VSEL_VS(dim, dtype)                                            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_vs_##dim##d_##dtype(                                   \
          memref_t<__ubuf__ bool, dim> *condition,                             \
          memref_t<__ubuf__ dtype, 1> *src0, dtype src1,                       \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf)

#define REGISTE_VSEL_VS(dim, dtype)                                            \
  DECLARE_VSEL_VS(dim, dtype) {                                                \
    vector_select_vs_##dim##d<dtype>(condition, src0, src1, dst,               \
                                     condition_addr_buf);                      \
  }

//===----------------------------------------------------------------------===//
// vsel for scalar-vector Inputs
//===----------------------------------------------------------------------===//
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_select_sv_1d(memref_t<__ubuf__ bool, 1> *condition, T src0,
                    memref_t<__ubuf__ T, 1> *src1, memref_t<__ubuf__ T, 1> *dst,
                    memref_t<__ubuf__ T, 1> *condition_addr_buf);

#define DECLARE_VSEL_SV_WITHOUT_TMP(dim, dtype)                                \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_sv_##dim##d_##dtype(                                   \
          memref_t<__ubuf__ bool, dim> *condition, dtype src0,                 \
          memref_t<__ubuf__ dtype, 1> *src1,                                   \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf = nullptr)

#define DECLARE_VSEL_SV(dim, dtype)                                            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vsel_sv_##dim##d_##dtype(                                   \
          memref_t<__ubuf__ bool, dim> *condition, dtype src0,                 \
          memref_t<__ubuf__ dtype, 1> *src1,                                   \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, dim> *condition_addr_buf)

#define REGISTE_VSEL_SV(dim, dtype)                                            \
  DECLARE_VSEL_SV(dim, dtype) {                                                \
    vector_select_sv_##dim##d<dtype>(condition, src0, src1, dst,               \
                                     condition_addr_buf);                      \
  }

extern "C" {
//===----------------------------------------------------------------------===//
// vsel, 1 dim for vector-vector Inputs
//===----------------------------------------------------------------------===//
DECLARE_VSEL_VV(1, bool, half);
DECLARE_VSEL_VV(1, bool, float);
DECLARE_VSEL_VV(1, bool, int16_t);
DECLARE_VSEL_VV(1, bool, int32_t);
DECLARE_VSEL_VV(1, bool, uint16_t);
DECLARE_VSEL_VV(1, bool, uint32_t);
DECLARE_VSEL_VV(1, bool, bfloat16_t);
DECLARE_VSEL_VV(1, int8_t, half);
DECLARE_VSEL_VV(1, int8_t, float);
DECLARE_VSEL_VV(1, int8_t, int16_t);
DECLARE_VSEL_VV(1, int8_t, int32_t);
DECLARE_VSEL_VV(1, int8_t, uint16_t);
DECLARE_VSEL_VV(1, int8_t, uint32_t);
DECLARE_VSEL_VV(1, int8_t, bfloat16_t);
DECLARE_VSEL_VV(1, int8_t, int64_t);
DECLARE_VSEL_VV_WITHOUT_TMP(1, bool, int64_t);

//===----------------------------------------------------------------------===//
// vsel, 1 dim for scalar-scalar Inputs
//===----------------------------------------------------------------------===//
DECLARE_VSEL_SS(1, half);
DECLARE_VSEL_SS(1, float);
DECLARE_VSEL_SS(1, int16_t);
DECLARE_VSEL_SS(1, int32_t);
DECLARE_VSEL_SS(1, uint16_t);
DECLARE_VSEL_SS(1, uint32_t);
DECLARE_VSEL_SS(1, bfloat16_t);
DECLARE_VSEL_SS_WITHOUT_TMP(1, int64_t);

//===----------------------------------------------------------------------===//
// vsel, 1 dim for vector-scalar Inputs
//===----------------------------------------------------------------------===//
DECLARE_VSEL_VS(1, half);
DECLARE_VSEL_VS(1, float);
DECLARE_VSEL_VS(1, int16_t);
DECLARE_VSEL_VS(1, int32_t);
DECLARE_VSEL_VS(1, uint16_t);
DECLARE_VSEL_VS(1, uint32_t);
DECLARE_VSEL_VS(1, bfloat16_t);
DECLARE_VSEL_VS_WITHOUT_TMP(1, int64_t);

//===----------------------------------------------------------------------===//
// vsel, 1 dim for scalar-vector Inputs
//===----------------------------------------------------------------------===//
DECLARE_VSEL_SV(1, half);
DECLARE_VSEL_SV(1, float);
DECLARE_VSEL_SV(1, int16_t);
DECLARE_VSEL_SV(1, int32_t);
DECLARE_VSEL_SV(1, uint16_t);
DECLARE_VSEL_SV(1, uint32_t);
DECLARE_VSEL_SV(1, bfloat16_t);
DECLARE_VSEL_SV_WITHOUT_TMP(1, int64_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_SEL_UTILS_H
