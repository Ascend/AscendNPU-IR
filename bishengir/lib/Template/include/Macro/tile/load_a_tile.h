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

template <ArchType ArchTag, typename ElementA, CubeFormat LayoutSrc,
          CubeFormat LayoutDst, bool IsTransPoseA, typename TileParams>
struct IteratorTileA;

template <typename ElementA, CubeFormat LayoutDst, bool IsTransPoseA,
          typename TileParams>
struct IteratorTileA<ArchType::ASCEND_V220, ElementA, CubeFormat::ND, LayoutDst,
                     IsTransPoseA, TileParams> {
  static const CubeFormat LayoutSrc = CubeFormat::ND;
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;

  // Define the copy gm to l1 operator
  using TileGm2L1 = gm_to_l1<ArchTag, ElementA, LayoutSrc, LayoutDst>;

  __aicore__ __attribute__((always_inline))
  IteratorTileA(__cbuf__ ElementA *l1AAddr, __gm__ ElementA *refA,
                const TileParams &tp, uint32_t eventId) {
    uint64_t offset;
    wait_flag(PIPE_MTE1, PIPE_MTE2, eventId);
    if constexpr (IsTransPoseA) {
      offset = tp.kIdxTile * tp.tileSizeTiling[1] * tp.refASize[0];
      TileGm2L1(l1AAddr, refA + offset, tp.tileSizeActual[1],
                tp.tileASizeRound[1], tp.refASize[1], tp.tileSizeActual[0],
                tp.tileASizeRound[0], tp.refASize[0]);
    } else {
      offset = tp.kIdxTile * tp.tileSizeTiling[1];
      TileGm2L1(l1AAddr, refA + offset, tp.tileSizeActual[0],
                tp.tileASizeRound[0], tp.refASize[0], tp.tileSizeActual[1],
                tp.tileASizeRound[1], tp.refASize[1]);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, eventId);
  }
};