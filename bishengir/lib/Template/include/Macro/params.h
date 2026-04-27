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
#pragma once

#include "Macro/coord.h"
#include "Macro/functional.h"

template <ArchType ArchTag, typename ElementA, typename ElementB,
          bool IsTransPoseA, bool IsTransPoseB>
struct DefaultTileParams {
  //
  // Data members
  //

  uint32_t coreIdx;
  uint32_t kIdxTile;
  Coord<3> tileSizeActual;
  Coord<2> tileASizeRound;
  Coord<2> tileBSizeRound;
  Coord<2> refASize;
  Coord<2> refBSize;
  Coord<3> problemSize;
  Coord<3> tileSizeTiling;

  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t SUB_BLOCK_A_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(ElementA);
  static constexpr uint32_t SUB_BLOCK_B_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(ElementB);
  static constexpr uint32_t SUB_BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  //
  // Methods
  //

  /// Default ctor
  __aicore__ __attribute__((always_inline)) DefaultTileParams(){};

  /// Constructs an DefaultTileParams structure
  __aicore__ __attribute__((always_inline))
  DefaultTileParams(uint32_t coreIdx_, uint32_t kIdx, Coord<3> tileSizeActual_,
                    Coord<2> refASize_, Coord<2> refBSize_,
                    Coord<3> problemSize_, Coord<3> tileSizeTiling_)
      : coreIdx(coreIdx_), kIdxTile(kIdx), tileSizeActual(tileSizeActual_),
        refASize(refASize_), refBSize(refBSize_), problemSize(problemSize_),
        tileSizeTiling(tileSizeTiling_) {
    uint32_t mTileRound = 0;
    if constexpr (IsTransPoseA) {
      mTileRound = RoundUp<SUB_BLOCK_A_SIZE>(tileSizeActual[0]);
    } else {
      mTileRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(tileSizeActual[0]);
    }

    uint32_t nTileRound = 0;
    if constexpr (IsTransPoseB) {
      nTileRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(tileSizeActual[2]);
    } else {
      nTileRound = RoundUp<SUB_BLOCK_B_SIZE>(tileSizeActual[2]);
    }

    uint32_t kTileARound = 0;
    kTileARound = RoundUp<SUB_BLOCK_A_SIZE>(tileSizeActual[1]);

    uint32_t kTileBRound = 0;
    kTileBRound = RoundUp<SUB_BLOCK_B_SIZE>(tileSizeActual[1]);

    tileASizeRound = Coord<2>({mTileRound, kTileARound});
    tileBSizeRound = Coord<2>({kTileBRound, nTileRound});
  }

  __aicore__ __attribute__((always_inline)) void
  update(uint32_t kIdx, Coord<3> tileSizeActual_) {
    kIdxTile = kIdx;
    tileSizeActual = tileSizeActual_;

    uint32_t mTileRound = 0;
    if constexpr (IsTransPoseA) {
      mTileRound = RoundUp<SUB_BLOCK_A_SIZE>(tileSizeActual[0]);
    } else {
      mTileRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(tileSizeActual[0]);
    }

    uint32_t nTileRound = 0;
    if constexpr (IsTransPoseB) {
      nTileRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(tileSizeActual[2]);
    } else {
      nTileRound = RoundUp<SUB_BLOCK_B_SIZE>(tileSizeActual[2]);
    }

    uint32_t kTileARound = 0;
    kTileARound = RoundUp<SUB_BLOCK_A_SIZE>(tileSizeActual[1]);

    uint32_t kTileBRound = 0;
    kTileBRound = RoundUp<SUB_BLOCK_B_SIZE>(tileSizeActual[1]);

    tileASizeRound = Coord<2>({mTileRound, kTileARound});
    tileBSizeRound = Coord<2>({kTileBRound, nTileRound});
  }

  __aicore__ __attribute__((always_inline)) void Clear() {
    tileSizeActual = Coord<3>({0, 0, 0});
  }

  __aicore__ __attribute__((always_inline)) bool IsTileEmpty() {
    for (int i = 0; i < 3; i++) {
      if (!tileSizeActual[i]) {
        return true;
      }
    }
    return false;
  }
};

template <ArchType ArchTag, typename ElementA, typename ElementB,
          bool IsTransPoseA, bool IsTransPoseB>
struct DefaultBlockParams {
  //
  // Data members
  //

  uint32_t coreIdx;
  Coord<3> blockSizeActual;
  Coord<2> blockASizeRound;
  Coord<2> blockBSizeRound;
  Coord<2> refASize;
  Coord<2> refBSize;
  Coord<3> problemSize;
  Coord<3> tileSizeTiling;
  Coord<3> blockSizeTiling;

  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t SUB_BLOCK_A_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(ElementA);
  static constexpr uint32_t SUB_BLOCK_B_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(ElementB);
  static constexpr uint32_t SUB_BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  //
  // Methods
  //

  /// Default ctor
  __aicore__ __attribute__((always_inline)) DefaultBlockParams(){};

  /// Constructs an DefaultBlockParams structure
  __aicore__ __attribute__((always_inline))
  DefaultBlockParams(uint32_t coreIdx_, Coord<3> blockSizeActual_,
                     Coord<2> refASize_, Coord<2> refBSize_,
                     Coord<3> problemSize_, Coord<3> tileSizeTiling_,
                     Coord<3> blockSizeTiling_)
      : coreIdx(coreIdx_), blockSizeActual({blockSizeActual_}),
        refASize(refASize_), refBSize(refBSize_), problemSize(problemSize_),
        tileSizeTiling(tileSizeTiling_), blockSizeTiling(blockSizeTiling_) {
    uint32_t mBlockRound = 0;
    if constexpr (IsTransPoseA) {
      mBlockRound = RoundUp<SUB_BLOCK_A_SIZE>(blockSizeActual[0]);
    } else {
      mBlockRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(blockSizeActual[0]);
    }

    uint32_t nBlockRound = 0;
    if constexpr (IsTransPoseB) {
      nBlockRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(blockSizeActual[2]);
    } else {
      nBlockRound = RoundUp<SUB_BLOCK_B_SIZE>(blockSizeActual[2]);
    }

    uint32_t kBlockARound = 0;
    kBlockARound = RoundUp<SUB_BLOCK_A_SIZE>(blockSizeActual[1]);

    uint32_t kBlockBRound = 0;
    kBlockBRound = RoundUp<SUB_BLOCK_B_SIZE>(blockSizeActual[1]);

    blockASizeRound = Coord<2>({mBlockRound, kBlockARound});
    blockBSizeRound = Coord<2>({kBlockBRound, nBlockRound});
  }

  __aicore__ __attribute__((always_inline)) void
  update(Coord<3> blockSizeActual_) {
    blockSizeActual = blockSizeActual_;

    uint32_t mBlockRound = 0;
    if constexpr (IsTransPoseA) {
      mBlockRound = RoundUp<SUB_BLOCK_A_SIZE>(blockSizeActual[0]);
    } else {
      mBlockRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(blockSizeActual[0]);
    }

    uint32_t nBlockRound = 0;
    if constexpr (IsTransPoseB) {
      nBlockRound = RoundUp<SUB_BLOCK_NUM_PER_FRACTAL>(blockSizeActual[2]);
    } else {
      nBlockRound = RoundUp<SUB_BLOCK_B_SIZE>(blockSizeActual[2]);
    }

    uint32_t kBlockARound = 0;
    kBlockARound = RoundUp<SUB_BLOCK_A_SIZE>(blockSizeActual[1]);

    uint32_t kBlockBRound = 0;
    kBlockBRound = RoundUp<SUB_BLOCK_B_SIZE>(blockSizeActual[1]);

    blockASizeRound = Coord<2>({mBlockRound, kBlockARound});
    blockBSizeRound = Coord<2>({kBlockBRound, nBlockRound});
  }

  __aicore__ __attribute__((always_inline)) void clear() {
    blockSizeActual = Coord<3>({0, 0, 0});
  }

  __aicore__ __attribute__((always_inline)) bool is_block_empty() {
    for (int i = 0; i < 3; i++) {
      if (!blockSizeActual[i]) {
        return true;
      }
    }
    return false;
  }
};
