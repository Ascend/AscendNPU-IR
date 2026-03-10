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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_HOST_SINGLE_CUBE_HOST_TILING_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_HOST_SINGLE_CUBE_HOST_TILING_H

#include <cstddef>
#include <cstdint>

template <typename T, size_t Dim> struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

struct TilingParamsForMatmul {
  int64_t key{0};
  int64_t mTile{0};
  int64_t nTile{0};
  int64_t kTile{0};
  int64_t processM{0};
  int64_t processN{0};
  int64_t processK{0};
  int64_t splitKSlices{1}; // Indicates the number of partition alone K
                           // dimension while enabling split-k.
  int64_t swizzleDir{0};
  int64_t swizzleCnt{0};
  int64_t shuffleKType{0};
  int64_t pTiles{0};
};

struct TilingParamsForMixMatmul {
  int64_t key{0};
  int64_t mTile{0};
  int64_t nTile{0};
  int64_t kTile{0};
  int64_t processM{0};
  int64_t processN{0};
  int64_t processK{0};
  int64_t splitKSlices{1}; // Indicates the number of partition alone K
                           // dimension while enabling split-k.
  int64_t swizzleDir{0};
  int64_t swizzleCnt{0};
  int64_t shuffleKType{0};
  int64_t workspaceReuse{0};
  int64_t workspaceBufferNum{0};
  int64_t pTiles{0};
  int64_t ubMaxBitSize{0};
};

// TODO: obtain ub size from platform config
static constexpr int64_t kUBMaxSizeInBits = 192 * 1024 * 8;
static constexpr int64_t kUBAlignSizeInBytes = 32;
static constexpr int64_t kNumBitsInByte = 8;

// matmul host tiling func
#define DECLARE_MATMUL_TILING_FUNC(transpose_a, transpose_b, a_type, b_type,                                               \
                                   c_type, tiling_type)                                                                    \
  __attribute__((always_inline)) void                                                                                      \
      _mlir_ciface_matmul_host_tiling_func_##transpose_a##_##transpose_b##_##a_type##_##b_type##_##c_type##_##tiling_type( \
          memref_t<a_type, 2> *A, memref_t<b_type, 2> *B,                                                                  \
          memref_t<c_type, 2> *C, memref_t<tiling_type, 1> *tiling)

#define REGISTE_MATMUL_TILING_FUNC(transpose_a, transpose_b, a_type, b_type,   \
                                   c_type, tiling_type)                        \
  DECLARE_MATMUL_TILING_FUNC(transpose_a, transpose_b, a_type, b_type, c_type, \
                             tiling_type) {                                    \
    matmul_host_tiling_func_##transpose_a##_##transpose_b<                     \
        a_type, b_type, c_type, tiling_type>(A, B, C, tiling);                 \
  }

// mix matmul host tiling func
#define DECLARE_MIX_MATMUL_TILING_FUNC(transpose_a, transpose_b, a_type,                                                                             \
                                       b_type, c_type, tiling_type,                                                                                  \
                                       workspace_reuse)                                                                                              \
  __attribute__((always_inline)) void                                                                                                                \
      _mlir_ciface_mix_matmul_host_tiling_func_##transpose_a##_##transpose_b##_##a_type##_##b_type##_##c_type##_##tiling_type##_WR##workspace_reuse( \
          memref_t<a_type, 2> *A, memref_t<b_type, 2> *B,                                                                                            \
          memref_t<c_type, 2> *C, memref_t<tiling_type, 1> *tiling)

#define REGISTE_MIX_MATMUL_TILING_FUNC(transpose_a, transpose_b, a_type,       \
                                       b_type, c_type, tiling_type,            \
                                       workspace_reuse)                        \
  DECLARE_MIX_MATMUL_TILING_FUNC(transpose_a, transpose_b, a_type, b_type,     \
                                 c_type, tiling_type, workspace_reuse) {       \
    mix_matmul_host_tiling_func_##transpose_a##_##transpose_b<                 \
        a_type, b_type, c_type, tiling_type, workspace_reuse>(A, B, C,         \
                                                              tiling);         \
  }

#endif // HACC_MLIR_HOST_TILING_UTILS_H