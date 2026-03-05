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
#include "Macro/functional.h"
#include "Macro/params.h"
#include "Macro/tiling.h"

#include "Macro/block/mma_block.h"
#include "Macro/swizzle.h"

#include "Macro/block/epilogue_block_store.h"
#include "Macro/block/prologue_block.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <ArchType ArchTag, typename ElementA, CubeFormat LayoutA,
          typename ElementB, CubeFormat LayoutB, typename ElementC,
          CubeFormat LayoutC, typename ElementAccumulator, typename ElementBias,
          typename ElementScale, bool IsTransPoseA, bool IsTransPoseB,
          bool WithBias, bool WitScale>
class GemmCore {
public:
  // Define params
  using BlockParams = DefaultBlockParams<ArchTag, ElementA, ElementB,
                                         IsTransPoseA, IsTransPoseB>;

  using Prologue = DefaultPrologueBlock<ArchTag, ElementA, LayoutA, ElementB,
                                        LayoutB, ElementBias, ElementScale,
                                        IsTransPoseA, IsTransPoseB, WithBias>;

  using Mma = DefaultMma<ArchTag, ElementA, LayoutA, ElementB, LayoutB,
                         ElementAccumulator, ElementBias, ElementScale,
                         IsTransPoseA, IsTransPoseB, WithBias>;

  using EpilogueStore =
      EpilogueBlockStore<ArchTag, ElementC, LayoutC, ElementAccumulator,
                         ElementScale, BlockParams, false>;

  __aicore__ __attribute__((always_inline)) GemmCore() {}

  __aicore__ __attribute__((always_inline)) void Initialize(
      __gm__ uint8_t *__restrict__ gm_a, __gm__ uint8_t *__restrict__ gm_b,
      __gm__ uint8_t *__restrict__ gm_c, __gm__ uint8_t *__restrict__ gm_bias,
      __gm__ uint8_t *__restrict__ gm_scale, TilingParams &gmTilingData) {
    set_mask_norm();
    set_atomic_none();
    set_padding((uint64_t)0x0);
    set_nd_para((uint64_t)0x1);

    coreIdx = get_block_idx();
    blockDim = get_block_num();

    /// parser argument
    m = gmTilingData.m;
    k = gmTilingData.k;
    n = gmTilingData.n;
    mTile = gmTilingData.mTile;
    kTile = gmTilingData.kTile;
    nTile = gmTilingData.nTile;
    splitKSilces = 1;

    pingPongFlag = 1;
    refA = (__gm__ ElementA *)gm_a;
    refB = (__gm__ ElementB *)gm_b;
    refC = (__gm__ ElementC *)gm_c;
    refBias = (__gm__ ElementBias *)gm_bias;
    refScale = (__gm__ ElementScale *)gm_scale;
    gemmSwizzle = GemmSwizzle(m, k, n, mTile, nTile, blockDim, splitKSilces);
  }

  __aicore__ __attribute__((always_inline)) void Run() {
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);

    blockParamsCur = BlockParams(
        coreIdx, Coord<3>({mBlockActual, kBlockActual, nBlockActual}),
        Coord<2>({m, k}), Coord<2>({k, n}), Coord<3>({m, k, n}),
        Coord<3>({mTile, kTile, nTile}),
        Coord<3>({mTile, CeilDiv(k, splitKSilces), nTile}));
    blockParamsCur.clear();

    coreLoops = gemmSwizzle.mLoops * gemmSwizzle.nLoops;
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockDim) {
      gemmSwizzle.get_block_offset(loopIdx, bIdx, mIdx, kIdx, nIdx);
      gemmSwizzle.get_block_size(mIdx, kIdx, nIdx, mBlockActual, kBlockActual,
                                 nBlockActual);
      blockParamsCur.update(
          Coord<3>({mBlockActual, kBlockActual, nBlockActual}));

      /// the block-scoped prolologue
      offset = bIdx * blockParamsCur.problemSize[2] +
               nIdx * blockParamsCur.blockSizeTiling[2];
      __gm__ ElementBias *refBiasBlock = refBias + offset;
      __gm__ ElementScale *refScaleBlock = refScale + offset;
      Prologue(refBiasBlock, refScaleBlock, blockParamsCur);

      /// the block-scoped mma
      if constexpr (IsTransPoseA) {
        offset =
            bIdx * blockParamsCur.refASize[0] * blockParamsCur.refASize[1] +
            mIdx * blockParamsCur.tileSizeTiling[0];
      } else {
        offset =
            bIdx * blockParamsCur.refASize[0] * blockParamsCur.refASize[1] +
            mIdx * blockParamsCur.tileSizeTiling[0] *
                blockParamsCur.refASize[1];
      }
      __gm__ ElementA *refABlock = refA + offset;
      if constexpr (IsTransPoseB) {
        offset =
            bIdx * blockParamsCur.refBSize[0] * blockParamsCur.refBSize[1] +
            nIdx * blockParamsCur.tileSizeTiling[2] *
                blockParamsCur.refBSize[0];
      } else {
        offset =
            bIdx * blockParamsCur.refBSize[0] * blockParamsCur.refBSize[1] +
            nIdx * blockParamsCur.tileSizeTiling[2];
      }
      __gm__ ElementB *refBBlock = refB + offset;
      Mma(refABlock, refBBlock, blockParamsCur, pingPongFlag);

      /// the block-scoped epilogue
      offset =
          bIdx * blockParamsCur.problemSize[0] * blockParamsCur.problemSize[2] +
          mIdx * blockParamsCur.blockSizeTiling[0] *
              blockParamsCur.problemSize[2] +
          nIdx * blockParamsCur.blockSizeTiling[2];
      __gm__ ElementC *refCBlock = refC + offset;
      EpilogueStore(refCBlock, blockParamsCur);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
  }

private:
  uint32_t pingPongFlag;

  uint32_t m, k, n;
  uint32_t mTile, kTile, nTile;
  uint32_t bIdx, mIdx, kIdx, nIdx;
  uint32_t splitKSilces, enShuffleK;
  uint32_t coreIdx, blockDim, coreLoops;
  uint32_t mBlockActual, kBlockActual, nBlockActual;
  uint64_t offset;

  GemmSwizzle gemmSwizzle;
  BlockParams blockParamsCur, blockParamsNext;

  __gm__ ElementA *refA;
  __gm__ ElementB *refB;
  __gm__ ElementC *refC;
  __gm__ ElementBias *refBias;
  __gm__ ElementScale *refScale;
};
