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

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_memref_aligned_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                                memref_t<__ubuf__ T, 1> *src1,
                                memref_t<__ubuf__ bool, 1> *dst);

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_memref_aligned_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0,
                                memref_t<__ubuf__ bool, 1> *dst);

// Check whether vector access on the last dimension stays within UB.
// Assumes strides[Dim - 1] == 1 (vector path precondition). For multi-dim
// buffers, the farthest slice start is derived from offset and strides; the
// last dimension span is repeat(256B)-aligned.
template <typename T, int Dim>
__aiv__ __attribute__((always_inline)) bool
is_ub_repeat_space_enough(memref_t<__ubuf__ T, Dim> *buf) {
  if (buf == nullptr) {
    return false;
  }

  int64_t last_slice_offset = buf->offset;
  for (int i = 0; i < Dim - 1; ++i) {
    last_slice_offset += (buf->sizes[i] - 1) * buf->strides[i];
  }

  int64_t last_dim_access_bytes =
      CEIL_FACTOR(buf->sizes[Dim - 1] * sizeof(T), INTR_BYTES_PER_REPEAT);
  auto max_address =
      reinterpret_cast<uintptr_t>(buf->aligned + last_slice_offset) +
      static_cast<uintptr_t>(last_dim_access_bytes);
  if (max_address > UB_LIMIT_BYTES) {
    return false;
  }
  return true;
}

template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_ub_repeat_space_enough_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                                        memref_t<__ubuf__ T, 1> *src1);

// Check ub size remained for vector-scalar compare: src0 vector access span
// (repeat-aligned) must not exceed UB limit.
template <typename T>
__aiv__ __attribute__((always_inline)) bool
is_ub_repeat_space_enough_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                     memref_t<__ubuf__ T, 1> *src1,
                     memref_t<__ubuf__ bool, 1> *dst);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
scalar_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                     memref_t<__ubuf__ bool, 1> *dst);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_scalar_compare_vv_1d(memref_t<__ubuf__ T, 1> *src0,
                            memref_t<__ubuf__ T, 1> *src1,
                            memref_t<__ubuf__ bool, 1> *dst);

template <VectorOpTy OP, typename T>
__aiv__ __attribute__((always_inline)) void
vector_scalar_compare_vs_1d(memref_t<__ubuf__ T, 1> *src0, T scalar,
                            memref_t<__ubuf__ bool, 1> *dst);

#define DECLARE_CMP_VV(op_name, op_type, dim, dtype)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_vcmp_##op_name##_##dim##d_##dtype(                              \
      memref_t<__ubuf__ dtype, dim> *src0,                                     \
      memref_t<__ubuf__ dtype, dim> *src1, memref_t<__ubuf__ bool, dim> *dst)

#define DECLARE_CMP_VS(op_name, op_type, dim, dtype)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_vcmps_##op_name##_##dim##d_##dtype(                             \
      memref_t<__ubuf__ dtype, dim> *src0, dtype scalar,                       \
      memref_t<__ubuf__ bool, dim> *dst)

#define REGISTE_CMP_VV(op_name, op_type, dim, dtype)                           \
  DECLARE_CMP_VV(op_name, op_type, dim, dtype) {                               \
    if (!is_memref_aligned_compare_vv_##dim##d<dtype>(src0, src1, dst))        \
        [[unlikely]] {                                                         \
      scalar_compare_vv_##dim##d<op_type, dtype>(src0, src1, dst);             \
    } else if (!is_ub_repeat_space_enough_compare_vv_##dim##d<dtype>(          \
                   src0, src1)) [[unlikely]] {                                 \
      vector_scalar_compare_vv_##dim##d<op_type, dtype>(src0, src1, dst);      \
    } else {                                                                   \
      vector_compare_vv_##dim##d<op_type, dtype>(src0, src1, dst);             \
    }                                                                          \
  }

#define REGISTE_CMP_VS(op_name, op_type, dim, dtype)                           \
  DECLARE_CMP_VS(op_name, op_type, dim, dtype) {                               \
    if (!is_memref_aligned_compare_vs_##dim##d<dtype>(src0, dst))              \
        [[unlikely]] {                                                         \
      scalar_compare_vs_##dim##d<op_type, dtype>(src0, scalar, dst);           \
    } else if (!is_ub_repeat_space_enough_compare_vs_##dim##d<dtype>(src0))    \
        [[unlikely]] {                                                         \
      vector_scalar_compare_vs_##dim##d<op_type, dtype>(src0, scalar, dst);    \
    } else {                                                                   \
      vector_compare_vs_##dim##d<op_type, dtype>(src0, scalar, dst);           \
    }                                                                          \
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