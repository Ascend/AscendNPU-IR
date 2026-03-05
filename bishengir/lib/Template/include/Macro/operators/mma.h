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

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <ArchType ArchTag, typename ElementA, typename ElementB,
          typename AccDTypeC, bool IsTransposeA>
struct mmad {
  __aicore__ __attribute__((always_inline))
  mmad(__cc__ AccDTypeC *l0cAddr, __ca__ ElementA *l0aAddr,
       __cb__ ElementB *l0bAddr, uint32_t mTileActual, uint32_t nTileActual,
       uint32_t kPartActual, bool initC){};

  __aicore__ __attribute__((always_inline))
  mmad(__cc__ AccDTypeC *l0cAddr, __ca__ ElementA *l0aAddr,
       __cb__ ElementB *l0bAddr, uint64_t biasBt, uint32_t mTileActual,
       uint32_t nTileActual, uint32_t kPartActual, bool initC){};
};

template <typename ElementA, typename ElementB, typename AccDTypeC,
          bool IsTransposeA>
struct mmad<ArchType::ASCEND_V220, ElementA, ElementB, AccDTypeC,
            IsTransposeA> {
  __aicore__ __attribute__((always_inline))
  mmad(__cc__ AccDTypeC *l0cAddr, __ca__ ElementA *l0aAddr,
       __cb__ ElementB *l0bAddr, uint32_t mTileActual, uint32_t nTileActual,
       uint32_t kPartActual, bool initC) {
    mad(l0cAddr, l0aAddr, l0bAddr,
        mTileActual, // m
        kPartActual, // k
        nTileActual, // n
        0,           // unitFlag
        0,           // kDirectionAlign
        0,           // cmatrixSource
        initC        // cmatrixInitVal
    );
  };

  __aicore__ __attribute__((always_inline))
  mmad(__cc__ AccDTypeC *l0cAddr, __ca__ ElementA *l0aAddr,
       __cb__ ElementB *l0bAddr, uint64_t biasBt, uint32_t mTileActual,
       uint32_t nTileActual, uint32_t kPartActual, bool initC) {
    mad(l0cAddr, l0aAddr, l0bAddr, biasBt,
        mTileActual, // m
        kPartActual, // k
        nTileActual, // n
        0,           // unitFlag
        0,           // kDirectionAlign
        1,           // cmatrixSource
        0            // cmatrixInitVal
    );
  };
};