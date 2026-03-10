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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CMP_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CMP_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                     memref_t<__ubuf__ T, 1> *src1,
                     memref_t<__ubuf__ bool, 1> *dst);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                     memref_t<__ubuf__ bool, 1> *dst);

#define DECLARE_CMP_VV(op_name, op_type, dim, dtype)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vcmp_##op_name##_##dim##d_##dtype(                          \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *src1,                                 \
          memref_t<__ubuf__ bool, dim> *dst)

#define DECLARE_CMP_VS(op_name, op_type, dim, dtype)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_vcmps_##op_name##_##dim##d_##dtype(                         \
          memref_t<__ubuf__ dtype, dim> *src0, dtype scalar,                   \
          memref_t<__ubuf__ bool, dim> *dst)

#define REGISTE_CMP_VV(op_name, op_type, dim, dtype)                           \
  DECLARE_CMP_VV(op_name, op_type, dim, dtype) {                               \
    vector_compare_vv_##dim##d<op_type, dtype>(src0, src1, dst);               \
  }

#define REGISTE_CMP_VS(op_name, op_type, dim, dtype)                           \
  DECLARE_CMP_VS(op_name, op_type, dim, dtype) {                               \
    vector_compare_vs_##dim##d<op_type, dtype>(src0, scalar, dst);             \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// vcmpv, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, half);
DECLARE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, float);
DECLARE_CMP_VV(eq, VectorOpTy::VCMP_EQ, 1, int32_t);

DECLARE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, half);
DECLARE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, float);
DECLARE_CMP_VV(ne, VectorOpTy::VCMP_NE, 1, int32_t);

DECLARE_CMP_VV(lt, VectorOpTy::VCMP_LT, 1, half);
DECLARE_CMP_VV(lt, VectorOpTy::VCMP_LT, 1, float);

DECLARE_CMP_VV(gt, VectorOpTy::VCMP_GT, 1, half);
DECLARE_CMP_VV(gt, VectorOpTy::VCMP_GT, 1, float);

DECLARE_CMP_VV(ge, VectorOpTy::VCMP_GE, 1, half);
DECLARE_CMP_VV(ge, VectorOpTy::VCMP_GE, 1, float);

DECLARE_CMP_VV(le, VectorOpTy::VCMP_LE, 1, half);
DECLARE_CMP_VV(le, VectorOpTy::VCMP_LE, 1, float);

//===-------------------------------------------------------------------===//
// vcmpvs, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, half);
DECLARE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, float);
DECLARE_CMP_VS(eq, VectorOpTy::VCMPS_EQ, 1, int32_t);

DECLARE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, half);
DECLARE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, float);
DECLARE_CMP_VS(ne, VectorOpTy::VCMPS_NE, 1, int32_t);

DECLARE_CMP_VS(lt, VectorOpTy::VCMPS_LT, 1, half);
DECLARE_CMP_VS(lt, VectorOpTy::VCMPS_LT, 1, float);

DECLARE_CMP_VS(gt, VectorOpTy::VCMPS_GT, 1, half);
DECLARE_CMP_VS(gt, VectorOpTy::VCMPS_GT, 1, float);

DECLARE_CMP_VS(ge, VectorOpTy::VCMPS_GE, 1, half);
DECLARE_CMP_VS(ge, VectorOpTy::VCMPS_GE, 1, float);

DECLARE_CMP_VS(le, VectorOpTy::VCMPS_LE, 1, half);
DECLARE_CMP_VS(le, VectorOpTy::VCMPS_LE, 1, float);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CMP_UTILS_H