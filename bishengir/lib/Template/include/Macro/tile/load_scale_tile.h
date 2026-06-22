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

#include "Macro/operators/iterator.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <ArchType ArchTag, typename ElementA, typename ElementB,
          typename ElementScale, bool WithBias, typename TileParams>
struct IteratorTileScale;

template <bool WithBias, typename TileParams>
struct IteratorTileScale<ArchType::ASCEND_V220, int8_t, int8_t, uint64_t,
                         WithBias, TileParams> {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementScale = uint64_t;
  static const CubeFormat LayoutSrc = CubeFormat::ND;
  static const CubeFormat LayoutDst = CubeFormat::ND;
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;

  // Define the copy gm to l1 operator
  using TileGm2L1 = gm_to_l1<ArchTag, ElementScale, LayoutSrc, LayoutDst>;
  // Define the copy l1 to fb operator
  using TileL12Fb = l1_to_fb<ArchType::ASCEND_V220, ElementScale>;

  // Define buffer offset
  using L1BufOffset = L1BufferOffset<ArchTag, ElementA, ElementB, WithBias>;

  __aicore__ __attribute__((always_inline))
  IteratorTileScale(__cbuf__ ElementScale *bufScale,
                    __fbuf__ ElementScale *fbScale,
                    __gm__ ElementScale *refScale, const TileParams &tp) {
    uint32_t l1ScaleOffset =
        L1BufOffset::L1_SCALE_OFFSET / sizeof(ElementScale);
    auto l1ScaleAddr = bufScale + l1ScaleOffset;
    // uint64_t offsetScale;

    // offsetScale = (uint64_t)tp.batchIdx * bp.n + tp.nIdx * bp.nTile;

    pipe_barrier(PIPE_MTE2);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    TileGm2L1(l1ScaleAddr, refScale, 1, 1, 1, tp.tileSizeActual[2],
              tp.tileBSizeRound[1], tp.problemSize[2]);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);

    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    TileL12Fb(fbScale, l1ScaleAddr, tp.tileSizeActual[2]);
    // when move scalar form L1 to fifpipe end, can move A/B from gm to L1
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  }
};