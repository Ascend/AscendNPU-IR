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

struct GemmSwizzle {
  //
  // Data members
  //

  uint32_t swizzlCount;
  uint32_t swizzlDirect;
  uint32_t m, k, n;
  uint32_t mTile, nTile;
  uint32_t mLoops, nLoops;
  uint32_t blockDim;
  uint32_t splitKSilces = 1;

  //
  // Methods
  //

  /// Default ctor
  __aicore__ __attribute__((always_inline)) GemmSwizzle(){};

  __aicore__ __attribute__((always_inline))
  GemmSwizzle(uint32_t m_, uint32_t k_, uint32_t n_, uint32_t mTile_,
              uint32_t nTile_, uint32_t blockDim_, uint32_t splitKSilces_ = 1)
      : m(m_), k(k_), n(n_), mTile(mTile_), nTile(nTile_), blockDim(blockDim_),
        splitKSilces(splitKSilces_) {
    mLoops = CeilDiv(m, mTile);
    nLoops = CeilDiv(n, nTile);

    /// calculate swizzleCount & siwzzleDirect
    swizzlCount = 1;
    swizzlDirect = 0;

    uint32_t costCur = 0;
    uint32_t costMin = m * k + k * n;
    for (uint32_t i = 1; i <= blockDim; ++i) {
      uint32_t c = (blockDim + i - 1) / i;
      if (i * nTile + m < mTile * c + n) {
        swizzlDirect = 1; // Nz
        costCur = nTile * i + mTile * c;
        if (costCur <= costMin) {
          costMin = costCur;
          swizzlCount = i;
        }
      } else {
        swizzlDirect = 0; // Zn
        costCur = mTile * i + nTile * c;
        if (costCur < costMin) {
          costMin = costCur;
          swizzlCount = i;
        }
      }
    }
  }

  __aicore__ __attribute__((always_inline)) void
  get_block_offset(uint32_t taskIdx, uint32_t &bIdx, uint32_t &mIdx,
                   uint32_t &kIdx, uint32_t &nIdx) {
    bIdx = taskIdx / (mLoops * nLoops * splitKSilces);
    kIdx = taskIdx % (mLoops * nLoops * splitKSilces) / (mLoops * nLoops);
    uint32_t innerIdx = taskIdx % (mLoops * nLoops);
    if (swizzlDirect == 0) { // Zn
      uint32_t tileBlockLoop = (mLoops + swizzlCount - 1) / swizzlCount;
      uint32_t tileBlockIdx = innerIdx / (swizzlCount * nLoops);
      uint32_t inTileBlockIdx = innerIdx % (swizzlCount * nLoops);

      uint32_t n_row = swizzlCount;
      if (tileBlockIdx == tileBlockLoop - 1) {
        n_row = mLoops - swizzlCount * tileBlockIdx;
      }
      mIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % n_row;
      nIdx = inTileBlockIdx / n_row;
      if (tileBlockIdx % 2 == 1) {
        nIdx = nLoops - nIdx - 1;
      }
    } else if (swizzlDirect == 1) { // Nz
      uint32_t tileBlockLoop = (nLoops + swizzlCount - 1) / swizzlCount;
      uint32_t tileBlockIdx = innerIdx / (swizzlCount * mLoops);
      uint32_t inTileBlockIdx = innerIdx % (swizzlCount * mLoops);

      uint32_t n_col = swizzlCount;
      if (tileBlockIdx == tileBlockLoop - 1) {
        n_col = nLoops - swizzlCount * tileBlockIdx;
      }
      mIdx = inTileBlockIdx / n_col;
      nIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % n_col;
      if (tileBlockIdx % 2 == 1) {
        mIdx = mLoops - mIdx - 1;
      }
    }
  }

  __aicore__ __attribute__((always_inline)) void
  get_block_size(uint32_t mIdx, uint32_t kIdx, uint32_t nIdx,
                 uint32_t &mBlockActual, uint32_t &kBlockActual,
                 uint32_t &nBlockActual) {
    uint32_t splitKLen = k / splitKSilces;
    mBlockActual = (mIdx == (mLoops - 1)) ? (m - mIdx * mTile) : mTile;
    nBlockActual = (nIdx == (nLoops - 1)) ? (n - nIdx * nTile) : nTile;
    kBlockActual =
        (kIdx == (splitKSilces - 1)) ? (k - splitKLen * kIdx) : splitKLen;
  }
};
