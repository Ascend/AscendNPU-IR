/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
#include "Macro/MacroUtils.h"
#include "Macro/common.h"
#include "Macro/core/gemm_core.h"
#include "Macro/tiling.h"

template <ArchType ArchTag, typename ElementA, CubeFormat LayoutA,
          typename ElementB, CubeFormat LayoutB, typename ElementC,
          CubeFormat LayoutC, typename ElementAccumulator, typename ElementBias,
          typename ElementScale, bool IsTransPoseA, bool IsTransPoseB,
          bool WithBias, bool WithScale>
__aicore__ __attribute__((always_inline)) void
GemmKernel(__gm__ uint8_t *__restrict__ gm_a, __gm__ uint8_t *__restrict__ gm_b,
           __gm__ uint8_t *__restrict__ gm_c,
           __gm__ uint8_t *__restrict__ gm_bias,
           __gm__ uint8_t *__restrict__ gm_scale, TilingParams &gmTilingData) {
  GemmCore<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
           ElementAccumulator, ElementBias, ElementScale, IsTransPoseA,
           IsTransPoseB, WithBias, WithScale>
      gemmCore;
  gemmCore.Initialize(gm_a, gm_b, gm_c, gm_bias, gm_scale, gmTilingData);
  gemmCore.Run();
}

__aicore__ __attribute__((always_inline)) void
calcTiling(int64_t m, int64_t k, int64_t n, TilingParams &gmTilingData) {
  gmTilingData.m = (uint32_t)m;
  gmTilingData.k = (uint32_t)k;
  gmTilingData.n = (uint32_t)n;
  gmTilingData.mTile = 256;
  gmTilingData.kTile = 256;
  gmTilingData.nTile = 128;
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_XtransposeA_XtransposeB(memref_t<__gm__ ElementA, 2> *A,
                                              memref_t<__gm__ ElementB, 2> *B,
                                              memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[0];
  int64_t k = A->sizes[1];
  int64_t n = B->sizes[1];

  TilingParams tiling_data;
  calcTiling(m, k, n, tiling_data);

  constexpr int32_t bytes = sizeof(ElementA);
  if constexpr (bytes == BYTES_B8) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, int32_t, int32_t,
               uint64_t, false, false, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  } else if constexpr (bytes == BYTES_B16) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, float, float, uint64_t,
               false, false, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  }
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_transposeA_XtransposeB(memref_t<__gm__ ElementA, 2> *A,
                                             memref_t<__gm__ ElementB, 2> *B,
                                             memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[1];
  int64_t k = A->sizes[0];
  int64_t n = B->sizes[1];

  TilingParams tiling_data;
  calcTiling(m, k, n, tiling_data);

  constexpr int32_t bytes = sizeof(ElementA);
  if constexpr (bytes == BYTES_B8) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, int32_t, int32_t,
               uint64_t, true, false, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  } else if constexpr (bytes == BYTES_B16) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, float, float, uint64_t,
               true, false, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  }
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_XtransposeA_transposeB(memref_t<__gm__ ElementA, 2> *A,
                                             memref_t<__gm__ ElementB, 2> *B,
                                             memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[0];
  int64_t k = A->sizes[1];
  int64_t n = B->sizes[0];

  TilingParams tiling_data;
  calcTiling(m, k, n, tiling_data);

  constexpr int32_t bytes = sizeof(ElementA);
  if constexpr (bytes == BYTES_B8) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, int32_t, int32_t,
               uint64_t, false, true, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  } else if constexpr (bytes == BYTES_B16) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, float, float, uint64_t,
               false, true, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  }
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_transposeA_transposeB(memref_t<__gm__ ElementA, 2> *A,
                                            memref_t<__gm__ ElementB, 2> *B,
                                            memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[1];
  int64_t k = A->sizes[0];
  int64_t n = B->sizes[0];

  TilingParams tiling_data;
  calcTiling(m, k, n, tiling_data);

  constexpr int32_t bytes = sizeof(ElementA);
  if constexpr (bytes == BYTES_B8) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, int32_t, int32_t,
               uint64_t, true, true, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  } else if constexpr (bytes == BYTES_B16) {
    GemmKernel<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, ElementB,
               CubeFormat::ND, ElementC, CubeFormat::ND, float, float, uint64_t,
               true, true, false, false>(
        (__gm__ uint8_t *)a_ptr, (__gm__ uint8_t *)b_ptr,
        (__gm__ uint8_t *)c_ptr, nullptr, nullptr, tiling_data);
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// matmul, Xbias, Xdescale
//===-------------------------------------------------------------------===//
REGISTE_MATMUL_XB_XD(XtransposeA, XtransposeB, half, half, half);
REGISTE_MATMUL_XB_XD(transposeA, XtransposeB, half, half, half);
REGISTE_MATMUL_XB_XD(XtransposeA, transposeB, half, half, half);
REGISTE_MATMUL_XB_XD(transposeA, transposeB, half, half, half);
REGISTE_MATMUL_XB_XD(XtransposeA, XtransposeB, int8_t, int8_t, half);
REGISTE_MATMUL_XB_XD(transposeA, XtransposeB, int8_t, int8_t, half);
REGISTE_MATMUL_XB_XD(XtransposeA, transposeB, int8_t, int8_t, half);
REGISTE_MATMUL_XB_XD(transposeA, transposeB, int8_t, int8_t, half);
}
#endif