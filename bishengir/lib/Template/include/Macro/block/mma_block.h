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

#include "Macro/buf_offset.h"
#include "Macro/common.h"
#include "Macro/params.h"

#include "Macro/tile/load_a_tile.h"
#include "Macro/tile/load_b_tile.h"
#include "Macro/tile/mma_tile.h"

template <ArchType ArchTag, typename ElementA, CubeFormat LayoutA,
          typename ElementB, CubeFormat LayoutB, typename ElementAccumulator,
          typename ElementBias, typename ElementScale, bool IsTransPoseA,
          bool IsTransPoseB, bool WithBias>
struct DefaultMma;

#ifdef __DAV_C220_CUBE__

/// Partial specialization for v220
template <typename ElementA, CubeFormat LayoutA, typename ElementB,
          CubeFormat LayoutB, typename ElementAccumulator, typename ElementBias,
          typename ElementScale, bool IsTransPoseA, bool IsTransPoseB,
          bool WithBias>
struct DefaultMma<ArchType::ASCEND_V220, ElementA, LayoutA, ElementB, LayoutB,
                  ElementAccumulator, ElementBias, ElementScale, IsTransPoseA,
                  IsTransPoseB, WithBias> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t ELEMENT_SIZE_A = sizeof(ElementA);
  static constexpr uint32_t ELEMENT_SIZE_B = sizeof(ElementB);
  static constexpr uint32_t FRACTAL_SIZE_A =
      HardwareParams::fractalSize / ELEMENT_SIZE_A;
  static constexpr uint32_t FRACTAL_SIZE_B =
      HardwareParams::fractalSize / ELEMENT_SIZE_B;
  using TileParams = DefaultTileParams<ArchTag, ElementA, ElementB,
                                       IsTransPoseA, IsTransPoseB>;
  using BlockParams = DefaultBlockParams<ArchTag, ElementA, ElementB,
                                         IsTransPoseA, IsTransPoseB>;
  // Define iterators over tiles from the A operand
  using IteratorA = IteratorTileA<ArchTag, ElementA, LayoutA, CubeFormat::ZN,
                                  IsTransPoseA, TileParams>;
  // Define iterators over tiles from the B operand
  using IteratorB = IteratorTileB<ArchTag, ElementB, LayoutB, CubeFormat::ZN,
                                  IsTransPoseB, TileParams>;
  // Define the tile-scoped pipelined matrix multiply
  using MmaTile = DefaultMmaTile<ArchTag, ElementA, CubeFormat::ZN, ElementB,
                                 CubeFormat::ZN, ElementAccumulator,
                                 IsTransPoseA, IsTransPoseB, WithBias>;
  // Define buffer offset
  using L1BufOffset = L1BufferOffset<ArchTag, ElementA, ElementB, WithBias>;

  /// Constructor
  __aicore__ __attribute__((always_inline))
  DefaultMma(__gm__ ElementA *refA, __gm__ ElementB *refB, BlockParams &bp,
             uint32_t &pingPongFlag) {
    uint32_t kBlockActual = bp.blockSizeActual[1];
    uint32_t l1ASize =
        RoundUp<FRACTAL_SIZE_A>(bp.tileSizeTiling[0] * bp.tileSizeTiling[1]) *
        ELEMENT_SIZE_A;
    uint32_t l1BSize =
        RoundUp<FRACTAL_SIZE_B>(bp.tileSizeTiling[2] * bp.tileSizeTiling[1]) *
        ELEMENT_SIZE_B;
    uint32_t l1PingPongBufferSize = l1ASize + l1BSize;

    Coord<3> tileSize({bp.blockSizeActual[0],
                       min(kBlockActual, bp.tileSizeTiling[1]),
                       bp.blockSizeActual[2]});
    tileParamsNext = TileParams(bp.coreIdx, 0, tileSize, bp.refASize,
                                bp.refBSize, bp.problemSize, bp.tileSizeTiling);

    tileLoops = CeilDiv(kBlockActual, bp.tileSizeTiling[1]);
    for (uint32_t loopIdx = 0; loopIdx < tileLoops; loopIdx++) {
      tileParams = tileParamsNext;
      auto eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;

      if (pingPongFlag) {
        l1AAddr = (__cbuf__ ElementA *)(l1BaseAddr);
        l1BAddr = (__cbuf__ ElementB *)(l1BaseAddr + l1ASize);
      } else {
        l1AAddr = (__cbuf__ ElementA *)(l1BaseAddr + l1PingPongBufferSize);
        l1BAddr =
            (__cbuf__ ElementB *)(l1BaseAddr + l1PingPongBufferSize + l1ASize);
      }
      if (loopIdx == 0) {
        IteratorA(l1AAddr, refA, tileParams, eventId);
        IteratorB(l1BAddr, refB, tileParams, eventId + 2);
      }

      if (loopIdx < tileLoops - 1) {
        auto eventIdNext = (1 - pingPongFlag) ? EVENT_ID0 : EVENT_ID1;

        if (1 - pingPongFlag) {
          l1AAddrNext = (__cbuf__ ElementA *)(l1BaseAddr);
          l1BAddrNext = (__cbuf__ ElementB *)(l1BaseAddr + l1ASize);
        } else {
          l1AAddrNext =
              (__cbuf__ ElementA *)(l1BaseAddr + l1PingPongBufferSize);
          l1BAddrNext = (__cbuf__ ElementB *)(l1BaseAddr +
                                              l1PingPongBufferSize + l1ASize);
        }

        uint32_t loopIdxNext = loopIdx + 1;
        Coord<3> tileSizeNext(
            {bp.blockSizeActual[0],
             (loopIdxNext == tileLoops - 1)
                 ? kBlockActual - loopIdxNext * bp.tileSizeTiling[1]
                 : bp.tileSizeTiling[1],
             bp.blockSizeActual[2]});
        tileParamsNext.update(loopIdxNext, tileSizeNext);
        IteratorA(l1AAddrNext, refA, tileParamsNext, eventIdNext);
        IteratorB(l1BAddrNext, refB, tileParamsNext, eventIdNext + 2);
      }

      MmaTile(l1AAddr, l1BAddr, tileParams, loopIdx, eventId);
      pingPongFlag = 1 - pingPongFlag;
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  };

private:
  uint32_t tileLoops;
  TileParams tileParams, tileParamsNext;
  __cbuf__ uint8_t *l1BaseAddr =
      (__cbuf__ uint8_t *)get_imm(L1BufOffset::L1_A_OFFSET);
  __cbuf__ ElementA *l1AAddr;
  __cbuf__ ElementA *l1AAddrNext;
  __cbuf__ ElementB *l1BAddr;
  __cbuf__ ElementB *l1BAddrNext;
};

#endif // __DAV_C220_CUBE__