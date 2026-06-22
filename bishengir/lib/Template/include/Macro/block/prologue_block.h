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

#include "Macro/common.h"
#include "Macro/params.h"

#include "Macro/tile/load_bias_tile.h"
#include "Macro/tile/load_scale_tile.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <ArchType Archtag, typename ElementA, CubeFormat LayoutA,
          typename ElementB, CubeFormat LayoutB, typename ElementBias,
          typename ElementScale, bool IsTransPoseA, bool IsTransPoseB,
          bool WithBias>
struct DefaultPrologueBlock;

// Partial specialization for V220 Architecture
template <typename ElementA, CubeFormat LayoutA, typename ElementB,
          CubeFormat LayoutB, typename ElementBias, typename ElementScale,
          bool IsTransPoseA, bool IsTransPoseB, bool WithBias>
struct DefaultPrologueBlock<ArchType::ASCEND_V220, ElementA, LayoutA, ElementB,
                            LayoutB, ElementBias, ElementScale, IsTransPoseA,
                            IsTransPoseB, WithBias> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  using TileParams = DefaultTileParams<ArchTag, ElementA, ElementB,
                                       IsTransPoseA, IsTransPoseB>;
  using BlockParams = DefaultBlockParams<ArchTag, ElementA, ElementB,
                                         IsTransPoseA, IsTransPoseB>;
  // Define iterators over tiles from the Bias operand
  using IteratorBias = IteratorTileBias<ArchTag, ElementBias, TileParams>;
  // Define iterators over tiles from the Scale operand
  using IteratorScale = IteratorTileScale<ArchTag, ElementA, ElementB,
                                          ElementScale, WithBias, TileParams>;

  /// Constructor
  __aicore__ __attribute__((always_inline))
  DefaultPrologueBlock(__gm__ ElementBias *refBias,
                       __gm__ ElementScale *refScale, BlockParams &bp) {
    Coord<3> tileSize({bp.blockSizeActual[0],
                       min(bp.blockSizeActual[1], bp.tileSizeTiling[1]),
                       bp.blockSizeActual[2]});
    TileParams tileParams =
        TileParams(bp.coreIdx, 0, tileSize, bp.refASize, bp.refBSize,
                   bp.problemSize, bp.tileSizeTiling);
    if constexpr (WithBias) {
      __cbuf__ ElementBias *l1Bias = (__cbuf__ ElementBias *)get_imm(0);
      uint64_t btBias{0};
      IteratorBias(l1Bias, btBias, refBias, tileParams);
    }
    if constexpr (std::is_same<ElementA, int8_t>::value) {
      __cbuf__ ElementScale *l1Scale = (__cbuf__ ElementScale *)get_imm(0);
      __fbuf__ ElementScale *fbScale = (__fbuf__ ElementScale *)get_imm(0);
      IteratorScale(l1Scale, fbScale, refScale, tileParams);
    }
  }
};
