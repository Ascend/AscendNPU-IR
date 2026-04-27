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

/////////////////////////////////////////////////////
// gm_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, CubeFormat FormatInGM,
          CubeFormat FormatInL1>
struct gm_to_l1 {
  __aicore__ __attribute__((always_inline))
  gm_to_l1(__cbuf__ DataType *l1Addr, __gm__ DataType *gmAddr,
           uint32_t nTileActual, uint32_t nTileCeil, uint32_t nVal,
           uint32_t dTileActual, uint32_t dTileCeil, uint32_t dVal){};
};

/////////////////////////////////////////////////////
// l1_to_l0_a
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, bool IsTransPose,
          CubeFormat DFmtIn, CubeFormat DFmtOut>
struct l1_to_l0_a {
  __aicore__ __attribute__((always_inline))
  l1_to_l0_a(__ca__ DataType *l0aAddr, __cbuf__ DataType *l1Addr,
             uint32_t mTileCeil, uint32_t kPartCeil, uint32_t mSrcStride,
             uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride){};
};

/////////////////////////////////////////////////////
// l1_to_l0_b
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, bool IsTransPose,
          CubeFormat DFmtIn, CubeFormat DFmtOut>
struct l1_to_l0_b {
  __aicore__ __attribute__((always_inline))
  l1_to_l0_b(__cb__ DataType *l0bAddr, __cbuf__ DataType *l1Addr,
             uint32_t nTileCeil, uint32_t kPartCeil, uint32_t nSrcStride,
             uint32_t kSrcStride, uint32_t nDstStride, uint32_t kDstStride){};
};

/////////////////////////////////////////////////////
// l0c_to_gm
/////////////////////////////////////////////////////
template <ArchType ArchTag, CubeFormat OutFormatType, typename OutDataType,
          typename L0CDataType>
struct l0c_to_gm {
  __aicore__ __attribute__((always_inline))
  l0c_to_gm(__gm__ OutDataType *gmAddr, __cc__ L0CDataType *l0cAddr,
            uint32_t mTileActual, uint32_t nTileActual, uint32_t mTileCeil,
            uint32_t nActual){};
};

/////////////////////////////////////////////////////
// l1_to_bt
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType> struct l1_to_bt {
  __aicore__ __attribute__((always_inline))
  l1_to_bt(uint64_t biasTableAddr, __cbuf__ DataType *biasL1Addr,
           uint32_t ntileActual){};
};

/////////////////////////////////////////////////////
// l1_to_fb
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType> struct l1_to_fb {
  __aicore__ __attribute__((always_inline))
  l1_to_fb(__fbuf__ DataType *fbAddr, __cbuf__ DataType *l1Addr,
           uint32_t ntileActual){};
};

#include "iterators/gm_to_l1_iterator.h"
#include "iterators/l0c_to_gm_iterator.h"
#include "iterators/l1_to_bt_iterator.h"
#include "iterators/l1_to_fb_iterator.h"
#include "iterators/l1_to_l0_iterator.h"