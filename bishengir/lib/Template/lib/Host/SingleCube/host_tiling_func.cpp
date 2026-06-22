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

#include "Host/SingleCube/host_tiling.h"
#include <cstdint>

void balanceWorkload_formatmul(uint32_t m, uint32_t k, uint32_t n, uint32_t &m0,
                               uint32_t k0, uint32_t &n0) {
  uint32_t m0t = 384, n0t = 0, bestm0 = 0, bestn0 = 0;
  uint64_t minData = UINT64_MAX;
  while (m0t > 0) {
    n0t = 128 * 256 / m0t / 16 * 16;
    while (n0t > 0) {
      if (n0t <= 384) {
        uint64_t data = (uint64_t)m * (uint64_t)k * ((n + n0t - 1) / n0t) +
                        (uint64_t)k * (uint64_t)n * ((m + m0t - 1) / m0t);
        if (data <= minData) {
          minData = data;
          bestm0 = m0t;
          bestn0 = n0t;
        }
      }
      n0t -= 16;
    }
    m0t -= 16;
  }
  uint32_t tilesRound =
      ((m + bestm0 - 1) / bestm0) * ((n + bestn0 - 1) / bestn0);
  tilesRound = (tilesRound + 19) / 20 * 20;
  while (((m + bestm0 - 1) / bestm0) *
                 ((n + (bestn0 - 16) - 1) / (bestn0 - 16)) <=
             tilesRound &&
         (bestn0 - 16) > 0) {
    bestn0 -= 16;
  }
  m0 = bestm0;
  n0 = bestn0;
}

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementTiling>
void matmul_host_tiling_func_XtransposeA_XtransposeB(
    memref_t<ElementA, 2> *A, memref_t<ElementB, 2> *B,
    memref_t<ElementC, 2> *C, memref_t<ElementTiling, 1> *Tiling) {
  int64_t length_m = A->sizes[0];
  int64_t length_k = A->sizes[1];
  int64_t length_n = B->sizes[1];

  auto tiling_data = Tiling->aligned + Tiling->offset;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

extern "C" void
_mlir_ciface_matmul_host_tiling_func_XtransposeA_XtransposeB___fp16___fp16___fp16_int64_t(
    void *AAllocated, void *AAligned, int64_t offsetA, int64_t sizeA0,
    int64_t sizeA1, int64_t strideA0, int64_t strideA1, void *BAllocated,
    void *BAligned, int64_t offsetB, int64_t sizeB0, int64_t sizeB1,
    int64_t strideB0, int64_t strideB1, void *CAllocated, void *CAligned,
    int64_t offsetC, int64_t sizeC0, int64_t sizeC1, int64_t strideC0,
    int64_t strideC1, int64_t *TilingAllocated, int64_t *TilingAligned,
    int64_t offsetTiling, int64_t sizeTiling, int64_t strideTiling) {
  int64_t length_m = sizeA0;
  int64_t length_k = sizeA1;
  int64_t length_n = sizeB1;

  auto tiling_data = TilingAligned + offsetTiling;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementTiling>
void matmul_host_tiling_func_transposeA_XtransposeB(
    memref_t<ElementA, 2> *A, memref_t<ElementB, 2> *B,
    memref_t<ElementC, 2> *C, memref_t<ElementTiling, 1> *Tiling) {
  int64_t length_m = A->sizes[1];
  int64_t length_k = A->sizes[0];
  int64_t length_n = B->sizes[1];

  auto tiling_data = Tiling->aligned + Tiling->offset;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

extern "C" void
_mlir_ciface_matmul_host_tiling_func_transposeA_XtransposeB___fp16___fp16___fp16_int64_t(
    void *AAllocated, void *AAligned, int64_t offsetA, int64_t sizeA0,
    int64_t sizeA1, int64_t strideA0, int64_t strideA1, void *BAllocated,
    void *BAligned, int64_t offsetB, int64_t sizeB0, int64_t sizeB1,
    int64_t strideB0, int64_t strideB1, void *CAllocated, void *CAligned,
    int64_t offsetC, int64_t sizeC0, int64_t sizeC1, int64_t strideC0,
    int64_t strideC1, int64_t *TilingAllocated, int64_t *TilingAligned,
    int64_t offsetTiling, int64_t sizeTiling, int64_t strideTiling) {
  int64_t length_m = sizeA1;
  int64_t length_k = sizeA0;
  int64_t length_n = sizeB1;

  auto tiling_data = TilingAligned + offsetTiling;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementTiling>
void matmul_host_tiling_func_XtransposeA_transposeB(
    memref_t<ElementA, 2> *A, memref_t<ElementB, 2> *B,
    memref_t<ElementC, 2> *C, memref_t<ElementTiling, 1> *Tiling) {
  int64_t length_m = A->sizes[0];
  int64_t length_k = A->sizes[1];
  int64_t length_n = B->sizes[0];

  auto tiling_data = Tiling->aligned + Tiling->offset;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

extern "C" void
_mlir_ciface_matmul_host_tiling_func_XtransposeA_transposeB___fp16___fp16___fp16_int64_t(
    void *AAllocated, void *AAligned, int64_t offsetA, int64_t sizeA0,
    int64_t sizeA1, int64_t strideA0, int64_t strideA1, void *BAllocated,
    void *BAligned, int64_t offsetB, int64_t sizeB0, int64_t sizeB1,
    int64_t strideB0, int64_t strideB1, void *CAllocated, void *CAligned,
    int64_t offsetC, int64_t sizeC0, int64_t sizeC1, int64_t strideC0,
    int64_t strideC1, int64_t *TilingAllocated, int64_t *TilingAligned,
    int64_t offsetTiling, int64_t sizeTiling, int64_t strideTiling) {
  int64_t length_m = sizeA0;
  int64_t length_k = sizeA1;
  int64_t length_n = sizeB0;

  auto tiling_data = TilingAligned + offsetTiling;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementTiling>
void matmul_host_tiling_func_transposeA_transposeB(
    memref_t<ElementA, 2> *A, memref_t<ElementB, 2> *B,
    memref_t<ElementC, 2> *C, memref_t<ElementTiling, 1> *Tiling) {
  int64_t length_m = A->sizes[1];
  int64_t length_k = A->sizes[0];
  int64_t length_n = B->sizes[0];

  auto tiling_data = Tiling->aligned + Tiling->offset;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

extern "C" void
_mlir_ciface_matmul_host_tiling_func_transposeA_transposeB___fp16___fp16___fp16_int64_t(
    void *AAllocated, void *AAligned, int64_t offsetA, int64_t sizeA0,
    int64_t sizeA1, int64_t strideA0, int64_t strideA1, void *BAllocated,
    void *BAligned, int64_t offsetB, int64_t sizeB0, int64_t sizeB1,
    int64_t strideB0, int64_t strideB1, void *CAllocated, void *CAligned,
    int64_t offsetC, int64_t sizeC0, int64_t sizeC1, int64_t strideC0,
    int64_t strideC1, int64_t *TilingAllocated, int64_t *TilingAligned,
    int64_t offsetTiling, int64_t sizeTiling, int64_t strideTiling) {
  int64_t length_m = sizeA1;
  int64_t length_k = sizeA0;
  int64_t length_n = sizeB0;

  auto tiling_data = TilingAligned + offsetTiling;
  uint8_t *tiling_ins = (uint8_t *)(tiling_data);
  auto tilingData = reinterpret_cast<TilingParamsForMatmul *>(tiling_ins);
  tilingData->mTile = 128;
  tilingData->nTile = 256;
  tilingData->kTile = 256;
  uint32_t m0 = 128, n0 = 256, k0 = 256;
  if (length_m <= 256) {
    balanceWorkload_formatmul(length_m, length_k, length_n, m0, k0, n0);
  } else if (length_m <= 384) {
    m0 = (length_m + 15) / 16 * 16;
    n0 = 128 * 256 / m0 / 16 * 16;
    uint32_t tilesRound = ((length_n + n0 - 1) / n0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (n0 > 16 && (length_n + (n0 - 16) - 1) / (n0 - 16) <= tilesRound) {
      n0 -= 16;
    }
  } else {
    if (length_m < length_n) {
      m0 = 256;
      n0 = 128;
    }
  }
  if (length_m == 256) {
    m0 = 256;
    n0 = 128;
  }
  tilingData->mTile = m0;
  tilingData->nTile = n0;
  tilingData->kTile = k0;
  tilingData->processM = 128;
  tilingData->processN = 256;
  tilingData->processK = 64;
  tilingData->swizzleDir = 0;
  if (length_m < length_n) {
    tilingData->swizzleDir = 1;
  }
  tilingData->swizzleCnt = 3;
  tilingData->splitKSlices = 1;
  tilingData->shuffleKType = 0;
  tilingData->pTiles = 1;
}

extern "C" {
// REGISTE_MATMUL_TILING_FUNC(XtransposeA, XtransposeB, __fp16, __fp16, __fp16,
//                            int64_t);
// REGISTE_MATMUL_TILING_FUNC(XtransposeA, transposeB, __fp16, __fp16, __fp16,
//                            int64_t);
// REGISTE_MATMUL_TILING_FUNC(transposeA, XtransposeB, __fp16, __fp16, __fp16,
//                            int64_t);
// REGISTE_MATMUL_TILING_FUNC(transposeA, transposeB, __fp16, __fp16, __fp16,
//                            int64_t);
}
