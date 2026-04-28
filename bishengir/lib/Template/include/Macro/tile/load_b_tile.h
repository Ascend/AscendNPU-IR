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

#include "Macro/operators/iterator.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <ArchType ArchTag, typename ElementB, CubeFormat LayoutSrc,
          CubeFormat LayoutDst, bool IsTransPoseB, typename TileParams>
struct IteratorTileB;

template <typename ElementB, CubeFormat LayoutDst, bool IsTransPoseB,
          typename TileParams>
struct IteratorTileB<ArchType::ASCEND_V220, ElementB, CubeFormat::ND, LayoutDst,
                     IsTransPoseB, TileParams> {
  static const CubeFormat LayoutSrc = CubeFormat::ND;
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;

  // Define the copy gm to l1 operator
  using TileGm2L1 = gm_to_l1<ArchTag, ElementB, LayoutSrc, LayoutDst>;

  __aicore__ __attribute__((always_inline))
  IteratorTileB(__cbuf__ ElementB *l1BAddr, __gm__ ElementB *refB,
                const TileParams &tp, uint32_t eventId) {
    uint64_t offset;
    wait_flag(PIPE_MTE1, PIPE_MTE2, eventId);
    if constexpr (IsTransPoseB) {
      offset = tp.kIdxTile * tp.tileSizeTiling[1];
      TileGm2L1(l1BAddr, refB + offset, tp.tileSizeActual[2],
                tp.tileBSizeRound[1], tp.refBSize[1], tp.tileSizeActual[1],
                tp.tileBSizeRound[0], tp.refBSize[0]);
    } else {
      offset = tp.kIdxTile * tp.tileSizeTiling[1] * tp.refBSize[1];
      TileGm2L1(l1BAddr, refB + offset, tp.tileSizeActual[1],
                tp.tileBSizeRound[0], tp.refBSize[0], tp.tileSizeActual[2],
                tp.tileBSizeRound[1], tp.refBSize[1]);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, eventId);
  }
};