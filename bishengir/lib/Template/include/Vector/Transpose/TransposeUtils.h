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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_TRANSPOSE_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_TRANSPOSE_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"

// the row or col number that vnchwconv intr can process per repeat is fixed
// number
constexpr int TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT = 16;
enum MODE { TRANSPOSE_MODE1, TRANSPOSE_MODE2, DEFAULT };

template <typename T>
__aiv__ __attribute__((always_inline)) void
gen_transpose_mode(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                   MODE &mode) {
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  // For b32, the transpose intr can process (16, 8) or (8, 16) data per repeat,
  // therefore there are two modes to do transpose.
  // mode1 : transpose (16, 8) to (8, 16) per repeat.
  // mode2 : transpose (8, 16) to (16, 8) per repeat.
  // Here we will choose mode according to the input shape, mode1 is with higher
  // priority if both mode is suitable.
  bool is_mode1_suitable =
      (src->sizes[0] % TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT == 0) &&
      (src->sizes[1] % num_per_block == 0);
  bool is_mode2_suitable =
      (src->sizes[0] % num_per_block == 0) &&
      (src->sizes[1] % TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT == 0);

  if (!is_mode1_suitable && !is_mode2_suitable) {
    // select mode2, if both modes are not suitable
    src->sizes[1] =
          CEIL_FACTOR(src->sizes[1], TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT);
    dst->sizes[0] =
        CEIL_FACTOR(dst->sizes[0], TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT);
    is_mode2_suitable = true;
  }

#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert((src->sizes[0] <= (dst->strides[0] / dst->strides[1])) &&
         "src size0 should be equal or less than stride");
  assert((src->sizes[1] <= src->strides[0]) &&
         "src size1 should be equal or less than stride");
  assert((dst->sizes[0] <= (src->strides[0] / src->strides[1])) &&
         "dst size0 should be equal or less than stride");
  assert((dst->sizes[1] <= dst->strides[0]) &&
         "dst size1 should be equal or less than stride");
#endif
  mode = is_mode1_suitable ? TRANSPOSE_MODE1 : TRANSPOSE_MODE2;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vnchwconv_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst);

template <typename T>
__aiv__ __attribute__((always_inline)) void
vnchwconv_2d_with_tmp(memref_t<__ubuf__ T, 2> *src,
                      memref_t<__ubuf__ T, 2> *dst,
                      memref_t<__ubuf__ T, 1> *tmp);

template <typename T>
__aiv__ __attribute__((always_inline)) void
gen_transpose_b64_core(memref_t<__ubuf__ T, 1> *src,
                       memref_t<__ubuf__ T, 1> *dst);

#define REGISTE_VNCHWCONV_2D(type)                                             \
  template __aiv__ __attribute__((always_inline)) void vnchwconv_2d<type>(     \
      memref_t<__ubuf__ type, 2> * src, memref_t<__ubuf__ type, 2> * dst)

#define DECLARE_TRANSPOSE_WITH_LAST_AXIS(dim, dtype)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_transpose_##dim##d_with_last_axis_##dtype(                  \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ dtype, dim> *dst)

#define REGISTE_TRANSPOSE_WITH_LAST_AXIS(dim, dtype)                           \
  DECLARE_TRANSPOSE_WITH_LAST_AXIS(dim, dtype) {                               \
    vnchwconv_##dim##d<dtype>(src, dst);                                       \
  }

#define DECLARE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(dim, dtype)                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_transpose_##dim##d_with_last_axis_##dtype(                  \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ dtype, dim> *dst,                                  \
          memref_t<__ubuf__ dtype, 1> *tmp)

#define REGISTE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(dim, dtype)                  \
  DECLARE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(dim, dtype) {                      \
    vnchwconv_##dim##d_with_tmp<dtype>(src, dst, tmp);                         \
  }

// transpose src (a, b, c) with stride [n1, n2, 1] to dst (b, a, c) with
// stride [m1, m2, 1].
#define DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(dtype)                            \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_transpose_3d_without_last_axis_##dtype(                     \
          memref_t<__ubuf__ dtype, 3> *src, memref_t<__ubuf__ dtype, 3> *dst)

#define REGISTE_TRANSPOSE_3D_WITH_NLAST_AXIS(dtype)                            \
  DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(dtype) {                                \
    transpose_nlast_3d<dtype>(src, dst);                                       \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// Transpose 2d last axis
//===-------------------------------------------------------------------===//
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, uint8_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, int8_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, uint16_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, int16_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, half);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, bfloat16_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, uint32_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, int32_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(2, float);
DECLARE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(2, uint64_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(2, int64_t);

//===-------------------------------------------------------------------===//
// Transpose 3d last axis
//===-------------------------------------------------------------------===//
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, uint8_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, int8_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, uint16_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, int16_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, half);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, bfloat16_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, uint32_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, int32_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS(3, float);
DECLARE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(3, uint64_t);
DECLARE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(3, int64_t);

//===-------------------------------------------------------------------===//
// Transpose 3d without last axis
//===-------------------------------------------------------------------===//
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint8_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(int8_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint16_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(int16_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(half);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(bfloat16_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint32_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(int32_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(float);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(uint64_t);
DECLARE_TRANSPOSE_3D_WITH_NLAST_AXIS(int64_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_TRANSPOSE_UTILS_H
