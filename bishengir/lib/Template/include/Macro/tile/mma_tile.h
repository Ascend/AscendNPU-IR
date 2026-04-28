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
#include "Macro/operators/mma.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <ArchType ArchTag, typename ElementA, CubeFormat LayoutA,
          typename ElementB, CubeFormat LayoutB, typename ElementAccululator,
          bool IsTransPoseA, bool IsTransPoseB, bool WithBias>
struct DefaultMmaTile;

template <typename ElementA, typename ElementB, typename ElementAccumulator,
          bool IsTransPoseA, bool IsTransPoseB, bool WithBias>
struct DefaultMmaTile<ArchType::ASCEND_V220, ElementA, CubeFormat::ZN, ElementB,
                      CubeFormat::ZN, ElementAccumulator, IsTransPoseA,
                      IsTransPoseB, WithBias> {
  static const CubeFormat LayoutASrc = CubeFormat::ZN;
  static const CubeFormat LayoutADst = CubeFormat::ZZ;
  static const CubeFormat LayoutBSrc = CubeFormat::ZN;
  static const CubeFormat LayoutBDst = CubeFormat::NZ;
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;

  // Define iterators over A parts form L1 to L0
  using IteratorPartA =
      l1_to_l0_a<ArchTag, ElementA, IsTransPoseA, LayoutASrc, LayoutADst>;
  // Define iterators over B parts form L1 to L0
  using IteratorPartB =
      l1_to_l0_b<ArchTag, ElementB, IsTransPoseB, LayoutBSrc, LayoutBDst>;
  // Define the part-scoped pipelined matrix multiply
  using MmaPart =
      mmad<ArchTag, ElementA, ElementB, ElementAccumulator, IsTransPoseA>;
  // Define params
  using TileParams = DefaultTileParams<ArchTag, ElementA, ElementB,
                                       IsTransPoseA, IsTransPoseB>;

  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t L0A_PINGPONG_BUF_LEN =
      (HardwareParams::l0ASize / 2) / sizeof(ElementA);
  static constexpr uint32_t L0B_PINGPONG_BUF_LEN =
      (HardwareParams::l0BSize / 2) / sizeof(ElementB);
  static constexpr uint32_t SUB_BLOCK_A_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(ElementA);
  static constexpr uint32_t SUB_BLOCK_B_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(ElementB);
  static constexpr uint32_t SUB_BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  /// Constructor
  __aicore__ __attribute__((always_inline))
  DefaultMmaTile(__cbuf__ ElementA *l1AAddr, __cbuf__ ElementB *l1BAddr,
                 const TileParams &tp, uint32_t tileId, uint32_t eventId) {
    __ca__ ElementA *l0AAddr = (__ca__ ElementA *)get_imm(0);
    __cb__ ElementB *l0BAddr = (__cb__ ElementB *)get_imm(0);
    __cc__ ElementAccumulator *l0CAddr =
        (__cc__ ElementAccumulator *)get_imm(0);

    uint32_t kPartLenMax = min(L0A_PINGPONG_BUF_LEN / tp.tileASizeRound[0] /
                                   SUB_BLOCK_A_SIZE * SUB_BLOCK_A_SIZE,
                               L0B_PINGPONG_BUF_LEN / tp.tileBSizeRound[1] /
                                   SUB_BLOCK_B_SIZE * SUB_BLOCK_B_SIZE);
    uint32_t kPartLoop = (tp.tileSizeActual[1] + kPartLenMax - 1) / kPartLenMax;
    for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
      uint32_t kPartActual =
          (kPartIdx < kPartLoop - 1)
              ? kPartLenMax
              : tp.tileSizeActual[1] - kPartIdx * kPartLenMax;
      uint32_t kPartRoundA =
          (kPartIdx < kPartLoop - 1)
              ? kPartLenMax
              : tp.tileASizeRound[1] - kPartIdx * kPartLenMax;
      uint32_t kPartRoundB =
          (kPartIdx < kPartLoop - 1)
              ? kPartLenMax
              : tp.tileBSizeRound[0] - kPartIdx * kPartLenMax;

      auto mte1MadPingFlag = 1 - kPartIdx % 2;
      auto mte1MadEventId = mte1MadPingFlag ? EVENT_ID0 : EVENT_ID1;
      auto l0BufAAddr = l0AAddr + (kPartIdx % 2) * L0A_PINGPONG_BUF_LEN;
      auto l0BufBAddr = l0BAddr + (kPartIdx % 2) * L0B_PINGPONG_BUF_LEN;

      // *** load matrix A from L1 to L0A
      if (kPartIdx == 0) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, eventId);
      }
      wait_flag(PIPE_M, PIPE_MTE1, mte1MadEventId);
      if constexpr (IsTransPoseA) {
        IteratorPartA(l0BufAAddr,
                      l1AAddr + kPartIdx * kPartLenMax * SUB_BLOCK_A_SIZE,
                      tp.tileASizeRound[0], kPartRoundA,
                      tp.tileASizeRound[1] / SUB_BLOCK_NUM_PER_FRACTAL, 1,
                      kPartRoundA / SUB_BLOCK_NUM_PER_FRACTAL, 1);
      } else {
        IteratorPartA(l0BufAAddr,
                      l1AAddr + kPartIdx * kPartLenMax * tp.tileASizeRound[0],
                      tp.tileASizeRound[0], kPartRoundA, 1,
                      tp.tileASizeRound[0] / SUB_BLOCK_NUM_PER_FRACTAL,
                      kPartRoundA / SUB_BLOCK_A_SIZE, 1);
      }
      if (kPartIdx == kPartLoop - 1) {
        set_flag(PIPE_MTE1, PIPE_MTE2, eventId);
      }

      // *** load matrix B from L1 to L0B
      if (kPartIdx == 0) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, eventId + CONST_2);
      }
      if constexpr (IsTransPoseB) {
        IteratorPartB(l0BufBAddr,
                      l1BAddr + tp.tileBSizeRound[1] * kPartLenMax * kPartIdx,
                      tp.tileBSizeRound[1], kPartRoundB, 1,
                      tp.tileBSizeRound[1] / SUB_BLOCK_NUM_PER_FRACTAL, 1,
                      tp.tileBSizeRound[1] / SUB_BLOCK_NUM_PER_FRACTAL);
      } else {
        IteratorPartB(l0BufBAddr,
                      l1BAddr + kPartIdx * kPartLenMax * SUB_BLOCK_B_SIZE,
                      tp.tileBSizeRound[1], kPartRoundB,
                      tp.tileBSizeRound[0] / SUB_BLOCK_NUM_PER_FRACTAL, 1, 1,
                      tp.tileBSizeRound[1] / SUB_BLOCK_B_SIZE);
      }
      if (kPartIdx == kPartLoop - 1) {
        set_flag(PIPE_MTE1, PIPE_MTE2, eventId + CONST_2);
      }

      set_flag(PIPE_MTE1, PIPE_M, mte1MadEventId);
      wait_flag(PIPE_MTE1, PIPE_M, mte1MadEventId);
      bool initC = (tileId == 0 && kPartIdx == 0);
      if (initC) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      }

      if (initC) {
        if constexpr (WithBias) {
          wait_flag(PIPE_MTE1, PIPE_M,
                    EVENT_ID7); // wait move bias fron L1 to BT
          MmaPart(l0CAddr, l0BufAAddr, l0BufBAddr, 0, tp.tileASizeRound[0],
                  tp.tileBSizeRound[1], kPartActual, initC);
        } else {
          MmaPart(l0CAddr, l0BufAAddr, l0BufBAddr, tp.tileASizeRound[0],
                  tp.tileBSizeRound[1], kPartActual, initC);
        }
      } else {
        MmaPart(l0CAddr, l0BufAAddr, l0BufBAddr, tp.tileASizeRound[0],
                tp.tileBSizeRound[1], kPartActual, initC);
      }

      pipe_barrier(PIPE_M);
      set_flag(PIPE_M, PIPE_MTE1, mte1MadEventId);
    }
  }
};