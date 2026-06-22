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

template <ArchType ArchTag, typename ElementBias, typename TileParams>
struct IteratorTileBias;

template <typename ElementBias, typename TileParams>
struct IteratorTileBias<ArchType::ASCEND_V220, ElementBias, TileParams> {
  static const CubeFormat LayoutSrc = CubeFormat::ND;
  static const CubeFormat LayoutDst = CubeFormat::ND;
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;

  // Define the copy gm to l1 operator
  using TileGm2L1 = gm_to_l1<ArchTag, ElementBias, LayoutSrc, LayoutDst>;

  // Define buffser offset
  using BTOffset = BiasTableOffset<ArchTag>;

  __aicore__ __attribute__((always_inline))
  IteratorTileBias(__cbuf__ ElementBias *bufBias, uint64_t buf_BiasTable,
                   __gm__ ElementBias *refBias, const TileParams &tp) {
    pipe_barrier(PIPE_MTE2);
    TileGm2L1(bufBias, refBias, 1, 1, 1, tp.tileSizeActual[2],
              tp.tileBSizeRound[1], tp.problemSize[2]);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    auto biasTable = buf_BiasTable + BTOffset::BIAS_BT_OFFSET;

    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    l1_to_bt<ArchType::ASCEND_V220, ElementBias>(biasTable, bufBias,
                                                 tp.tileSizeActual[2]);
    set_flag(PIPE_MTE1, PIPE_MTE2,
             EVENT_ID7); // bias ready, mte2 can begin move A/B or scale
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7); // bias ready, mmad can begin
    wait_flag(PIPE_MTE1, PIPE_MTE2,
              EVENT_ID7); // A/B or scale wait moving bias from L1 to BT
  }
};