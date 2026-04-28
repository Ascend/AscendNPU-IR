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

constexpr int STRIDE_LIMIT = 65536;

// Partial specialization for V220, ND_in, ND_out
template <typename DataType>
struct gm_to_l1<ArchType::ASCEND_V220, DataType, CubeFormat::ND,
                CubeFormat::ND> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);

  __aicore__ __attribute__((always_inline))
  gm_to_l1(__cbuf__ DataType *l1Addr, __gm__ DataType *gmAddr,
           uint32_t nTileActual, uint32_t nTileCeil, uint32_t nVal,
           uint32_t dTileActual, uint32_t dTileCeil, uint32_t dVal) {
    copy_gm_to_cbuf(l1Addr, gmAddr,
                    0,                                              // sid
                    1,                                              // nBurst
                    CeilDiv<BLOCK_SIZE>(nTileActual * dTileActual), // lenBurst
                    0,                                              // srcGap
                    0,                                              // dstGap
                    PAD_NONE                                        // padMode
    );
  };
};

// Partial specialization for V220, ND_in, NZ_out
template <>
struct gm_to_l1<ArchType::ASCEND_V220, int8_t, CubeFormat::ND, CubeFormat::ZN> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using DataType = int8_t;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);

  __aicore__ __attribute__((always_inline))
  gm_to_l1(__cbuf__ int8_t *l1Addr, __gm__ int8_t *gmAddr, uint32_t nTileActual,
           uint32_t nTileCeil, uint32_t nVal, uint32_t dTileActual,
           uint32_t dTileCeil, uint32_t dVal) {
    if (dVal < STRIDE_LIMIT) {
      copy_gm_to_cbuf_multi_nd2nz_b8(l1Addr, gmAddr,
                                     0,           // sid
                                     1,           // ndNum
                                     nTileActual, // nValue
                                     dTileActual, // dValue
                                     0,           // srcNdMatrixStride, unused
                                     dVal,        // srcDValue
                                     nTileCeil,   // dstNzC0Stride
                                     1,           // dstNzNStride
                                     0            // dstNzMatrixStride, unused
      );
    } else {
      for (int i = 0; i < nTileActual; i++) {
        copy_gm_to_cbuf_multi_nd2nz_b8(l1Addr + i * BLOCK_SIZE,
                                       gmAddr + i * dVal,
                                       0,           // sid
                                       1,           // ndNum
                                       1,           // nValue
                                       dTileActual, // dValue
                                       0,           // srcNdMatrixStride, unused
                                       0,           // srcDValue, unused
                                       nTileCeil,   // dstNzC0Stride
                                       0,           // dstNzNStride, unused
                                       0            // dstNzMatrixStride, unused
        );
      }
    }
  };
};

// Partial specialization for V220, ND_in, NZ_out
template <>
struct gm_to_l1<ArchType::ASCEND_V220, half, CubeFormat::ND, CubeFormat::ZN> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using DataType = half;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);

  __aicore__ __attribute__((always_inline))
  gm_to_l1(__cbuf__ half *l1Addr, __gm__ half *gmAddr, uint32_t nTileActual,
           uint32_t nTileCeil, uint32_t nVal, uint32_t dTileActual,
           uint32_t dTileCeil, uint32_t dVal) {
    if (dVal < STRIDE_LIMIT) {
      copy_gm_to_cbuf_multi_nd2nz_b16(l1Addr, gmAddr,
                                      0,           // sid
                                      1,           // ndNum
                                      nTileActual, // nValue
                                      dTileActual, // dValue
                                      0,           // srcNdMatrixStride, unused
                                      dVal,        // srcDValue
                                      nTileCeil,   // dstNzC0Stride
                                      1,           // dstNzNStride
                                      0            // dstNzMatrixStride, unused
      );
    } else {
      for (int i = 0; i < nTileActual; i++) {
        copy_gm_to_cbuf_multi_nd2nz_b16(l1Addr + i * BLOCK_SIZE,
                                        gmAddr + i * dVal,
                                        0,           // sid
                                        1,           // ndNum
                                        1,           // nValue
                                        dTileActual, // dValue
                                        0,         // srcNdMatrixStride, unused
                                        0,         // srcDValue, unused
                                        nTileCeil, // dstNzC0Stride
                                        0,         // dstNzNStride, unused
                                        0          // dstNzMatrixStride, unused
        );
      }
    }
  };
};
