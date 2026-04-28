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

/////////////////////////////////////////////////////
// l1_to_fb
/////////////////////////////////////////////////////

// Partial specialization for V220
template <typename DataType> struct l1_to_fb<ArchType::ASCEND_V220, DataType> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::fbBlockSize / sizeof(DataType);

  __aicore__ __attribute__((always_inline))
  l1_to_fb(__fbuf__ DataType *fbAddr, __cbuf__ DataType *l1Addr,
           uint32_t ntileActual) {
    copy_cbuf_to_fbuf(fbAddr, l1Addr, 1, CeilDiv<BLOCK_SIZE>(ntileActual), 0,
                      0);
  };
};