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

template <ArchType ArchTag, typename ElementC, CubeFormat LayoutC,
          typename ElementAccumulator, typename ElementScale,
          typename BlockParams, bool WithScale>
struct EpilogueBlockStore;

/// Partial specialization for ASCEND_V220, with scale.
template <typename ElementC, CubeFormat LayoutC, typename ElementAccumulator,
          typename ElementScale, typename BlockParams>
struct EpilogueBlockStore<ArchType::ASCEND_V220, ElementC, LayoutC,
                          ElementAccumulator, ElementScale, BlockParams, true> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  static constexpr uint32_t USELESS_BIT_NUM = 7; // quant_pre_addr中以128B为单位
  static constexpr uint32_t QUANT_PRE_ADDR_MASK = 0xffff;
  static constexpr uint32_t QUANT_PRE_BIT_POS_IN_FPC = 8;

  using HardwareParams = HardwareInfo<ArchTag>;
  using BlockL0C2GM = l0c_to_gm<ArchTag, LayoutC, ElementC, ElementAccumulator>;

  __aicore__ __attribute__((always_inline))
  EpilogueBlockStore(__gm__ ElementC *refC, const BlockParams &bp) {
    __cc__ ElementAccumulator *l0cAddr =
        (__cc__ ElementAccumulator *)get_imm(0);
    __fbuf__ ElementScale *fbScaleAddr = (__fbuf__ ElementScale *)get_imm(0);

    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);

    uint64_t scaleAddr = (uint64_t)(__fbuf__ uint64_t *)fbScaleAddr;
    set_fpc((int64_t)0 |
            ((int64_t)(((scaleAddr & (uint64_t)QUANT_PRE_ADDR_MASK) >>
                        (uint64_t)USELESS_BIT_NUM)
                       << (uint64_t)QUANT_PRE_BIT_POS_IN_FPC)));
    BlockL0C2GM(refC, l0cAddr, bp.blockSizeActual[0], bp.blockSizeActual[2],
                bp.blockASizeRound[0], bp.problemSize[2]);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
};

/// Partial specialization for ASCEND_V220, no scale.
template <typename ElementC, CubeFormat LayoutC, typename ElementAccumulator,
          typename ElementScale, typename BlockParams>
struct EpilogueBlockStore<ArchType::ASCEND_V220, ElementC, LayoutC,
                          ElementAccumulator, ElementScale, BlockParams,
                          false> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;

  using HardwareParams = HardwareInfo<ArchTag>;
  using BlockL0C2GM = l0c_to_gm<ArchTag, LayoutC, ElementC, ElementAccumulator>;

  __aicore__ __attribute__((always_inline))
  EpilogueBlockStore(__gm__ ElementC *refC, const BlockParams &bp) {
    __cc__ ElementAccumulator *l0cAddr =
        (__cc__ ElementAccumulator *)get_imm(0);

    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    BlockL0C2GM(refC, l0cAddr, bp.blockSizeActual[0], bp.blockSizeActual[2],
                bp.blockASizeRound[0], bp.problemSize[2]);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
};

////////////////////////////////////////////////////////////////////////////////