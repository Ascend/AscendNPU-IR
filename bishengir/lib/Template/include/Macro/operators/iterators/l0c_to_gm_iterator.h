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
// l0c_to_gm
/////////////////////////////////////////////////////

// Partial specialization for V220, ND, half
template <typename L0CDataType>
struct l0c_to_gm<ArchType::ASCEND_V220, CubeFormat::ND, half, L0CDataType> {
  __aicore__ __attribute__((always_inline))
  l0c_to_gm(__gm__ half *gmAddr, __cc__ L0CDataType *l0cAddr,
            uint32_t mTileActual, uint32_t nTileActual, uint32_t mTileCeil,
            uint32_t nActual) {
    copy_matrix_cc_to_gm(gmAddr, l0cAddr,
                         0,           // sid
                         nTileActual, // NSize
                         mTileActual, // MSize
                         nActual,     // dstStride_dst_D
                         mTileCeil,   // srcStride
                         0,           // UnitFlagMode
                         F322F16,     // QuantPRE
                         0,           // ReLUPRE
                         false,       // channelSplit
                         true         // NZ2ND_EN
    );
  };
};

// Partial specialization for V220, ND, half, int32_t
template <>
struct l0c_to_gm<ArchType::ASCEND_V220, CubeFormat::ND, half, int32_t> {
  __aicore__ __attribute__((always_inline))
  l0c_to_gm(__gm__ half *gmAddr, __cc__ int32_t *l0cAddr, uint32_t mTileActual,
            uint32_t nTileActual, uint32_t mTileCeil, uint32_t nActual) {
    copy_matrix_cc_to_gm(gmAddr, l0cAddr,
                         0,           // sid
                         nTileActual, // NSize
                         mTileActual, // MSize
                         nActual,     // dstStride_dst_D
                         mTileCeil,   // srcStride
                         0,           // UnitFlagMode
                         VDEQF16,     // QuantPRE
                         0,           // ReLUPRE
                         false,       // channelSplit
                         true         // NZ2ND_EN
    );
  };
};

// Partial specialization for V220, ND, __bf16
template <typename L0CDataType>
struct l0c_to_gm<ArchType::ASCEND_V220, CubeFormat::ND, __bf16, L0CDataType> {
  __aicore__ __attribute__((always_inline))
  l0c_to_gm(__gm__ __bf16 *gmAddr, __cc__ L0CDataType *l0cAddr,
            uint32_t mTileActual, uint32_t nTileActual, uint32_t mTileCeil,
            uint32_t nActual) {
    copy_matrix_cc_to_gm(gmAddr, l0cAddr,
                         0,           // sid
                         nTileActual, // NSize
                         mTileActual, // MSize
                         nActual,     // dstStride_dst_D
                         mTileCeil,   // srcStride
                         0,           // UnitFlagMode
                         F322BF16,    // QuantPRE
                         0,           // ReLUPRE
                         false,       // channelSplit
                         true         // NZ2ND_EN
    );
  };
};

// Partial specialization for V220, ND, float
template <typename L0CDataType>
struct l0c_to_gm<ArchType::ASCEND_V220, CubeFormat::ND, float, L0CDataType> {
  __aicore__ __attribute__((always_inline))
  l0c_to_gm(__gm__ float *gmAddr, __cc__ L0CDataType *l0cAddr,
            uint32_t mTileActual, uint32_t nTileActual, uint32_t mTileCeil,
            uint32_t nActual) {
    copy_matrix_cc_to_gm(gmAddr, l0cAddr,
                         0,           // sid
                         nTileActual, // NSize
                         mTileActual, // MSize
                         nActual,     // dstStride_dst_D
                         mTileCeil,   // srcStride
                         0,           // UnitFlagMode
                         NoQuant,     // QuantPRE
                         0,           // ReLUPRE
                         false,       // channelSplit
                         true         // NZ2ND_EN
    );
  };
};

// Partial specialization for V220, NZ, half
template <typename L0CDataType>
struct l0c_to_gm<ArchType::ASCEND_V220, CubeFormat::NZ, half, L0CDataType> {
  __aicore__ __attribute__((always_inline))
  l0c_to_gm(__gm__ half *gmAddr, __cc__ L0CDataType *l0cAddr,
            uint32_t mTileActual, uint32_t nTileActual, uint32_t mTileCeil,
            uint32_t nActual) {
    copy_matrix_cc_to_gm(gmAddr, l0cAddr,
                         0,           // sid
                         nTileActual, // NSize
                         mTileActual, // MSize
                         nActual,     // dstStride_dst_D
                         mTileCeil,   // srcStride
                         0,           // UnitFlagMode
                         F322F16,     // QuantPRE
                         0,           // ReLUPRE
                         false,       // channelSplit
                         false        // NZ2ND_EN
    );
  };
};