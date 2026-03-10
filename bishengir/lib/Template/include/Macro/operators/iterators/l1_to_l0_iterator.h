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
// l1_to_l0_a
/////////////////////////////////////////////////////

// Partial specialization for V220, vector
template <typename DataType, bool IsTransPose>
struct l1_to_l0_a<ArchType::ASCEND_V220, DataType, IsTransPose,
                  CubeFormat::VECTOR, CubeFormat::VECTOR> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);

  __aicore__ __attribute__((always_inline))
  l1_to_l0_a(__ca__ DataType *l0aAddr, __cbuf__ DataType *l1Addr,
             uint32_t mTileCeil, uint32_t kPartCeil, uint32_t mSrcStride,
             uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride) {
    load_cbuf_to_ca(l0aAddr, l1Addr,
                    0,                                // baseIdx
                    CeilDiv<FRACTAL_SIZE>(kPartCeil), // repeat
                    kSrcStride,                       // srcStride
                    kDstStride - 1,                   // dstStride
                    0,                                // sid
                    false,                            // transpose
                    inc                               // addr_cal_mode_t
    );
  };
};

// Partial specialization for V220, no transpose, not vector
template <typename DataType>
struct l1_to_l0_a<ArchType::ASCEND_V220, DataType, false, CubeFormat::ZN,
                  CubeFormat::ZZ> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);
  static constexpr uint32_t BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  __aicore__ __attribute__((always_inline))
  l1_to_l0_a(__ca__ DataType *l0aAddr, __cbuf__ DataType *l1Addr,
             uint32_t mTileCeil, uint32_t kPartCeil, uint32_t mSrcStride,
             uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride) {
    for (uint32_t i = 0; i < mTileCeil / BLOCK_NUM_PER_FRACTAL; i++) {
      load_cbuf_to_ca(l0aAddr + i * mDstStride * FRACTAL_SIZE,
                      l1Addr + i * mSrcStride * FRACTAL_SIZE,
                      0,                      // baseIdx
                      kPartCeil / BLOCK_SIZE, // repeat
                      kSrcStride,             // srcStride
                      kDstStride - 1,         // dstStride
                      0,                      // sid
                      false,                  // transpose
                      inc                     // addr_cal_mode_t
      );
    }
  };
};

// Partial specialization for V220, transpose, not vector
template <typename DataType>
struct l1_to_l0_a<ArchType::ASCEND_V220, DataType, true, CubeFormat::ZN,
                  CubeFormat::ZZ> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);
  static constexpr uint32_t BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  __aicore__ __attribute__((always_inline))
  l1_to_l0_a(__ca__ DataType *l0aAddr, __cbuf__ DataType *l1Addr,
             uint32_t mTileCeil, uint32_t kPartCeil, uint32_t mSrcStride,
             uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride) {
    for (uint32_t i = 0; i < mTileCeil / BLOCK_SIZE; i++) {
      load_cbuf_to_ca(l0aAddr + i * mDstStride * FRACTAL_SIZE,
                      l1Addr + i * mSrcStride * FRACTAL_SIZE,
                      0,                                 // baseIdx
                      kPartCeil / BLOCK_NUM_PER_FRACTAL, // repeat
                      kSrcStride,                        // srcStride
                      kDstStride - 1,                    // dstStride
                      0,                                 // sid
                      true,                              // transpose
                      inc                                // addr_cal_mode_t
      );
    }
  };
};

// Partial specialization for V220, int8_t, transpose, not vector
template <>
struct l1_to_l0_a<ArchType::ASCEND_V220, int8_t, true, CubeFormat::ZN,
                  CubeFormat::ZZ> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using DataType = int8_t;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);

  __aicore__ __attribute__((always_inline))
  l1_to_l0_a(__ca__ int8_t *l0aAddr, __cbuf__ int8_t *l1Addr,
             uint32_t mTileCeil, uint32_t kPartCeil, uint32_t mSrcStride,
             uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride) {
    for (uint32_t i = 0; i < mTileCeil / BLOCK_SIZE; i++) {
      load_cbuf_to_ca_transpose(l0aAddr + i * mDstStride * FRACTAL_SIZE,
                                l1Addr + i * mSrcStride * FRACTAL_SIZE,
                                0,                         // baseIdx
                                kPartCeil / BLOCK_SIZE,    // repeat
                                kSrcStride,                // srcStride
                                kDstStride - 1,            // dstStride
                                0,                         // addrmode
                                kPartCeil / BLOCK_SIZE - 1 // dstFracStride
      );
    }
  };
};

/////////////////////////////////////////////////////
// l1_to_l0_b
/////////////////////////////////////////////////////

// Partial specialization for V220, no transpose, not vector
template <typename DataType>
struct l1_to_l0_b<ArchType::ASCEND_V220, DataType, false, CubeFormat::ZN,
                  CubeFormat::NZ> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);
  static constexpr uint32_t BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  __aicore__ __attribute__((always_inline))
  l1_to_l0_b(__cb__ DataType *l0bAddr, __cbuf__ DataType *l1Addr,
             uint32_t nTileCeil, uint32_t kPartCeil, uint32_t nSrcStride,
             uint32_t kSrcStride, uint32_t nDstStride, uint32_t kDstStride) {
    for (uint32_t i = 0; i < kPartCeil / BLOCK_NUM_PER_FRACTAL; i++) {
      load_cbuf_to_cb(l0bAddr + i * kDstStride * FRACTAL_SIZE,
                      l1Addr + i * kSrcStride * FRACTAL_SIZE,
                      0,                      // baseIdx
                      nTileCeil / BLOCK_SIZE, // repeat
                      nSrcStride,             // srcStride
                      nDstStride - 1,         // dstStride
                      0,                      // sid
                      true,                   // transpose
                      inc                     // addr_cal_mode_t
      );
    }
  };
};

// Partial specialization for V220, transpose, not vector
template <typename DataType>
struct l1_to_l0_b<ArchType::ASCEND_V220, DataType, true, CubeFormat::ZN,
                  CubeFormat::NZ> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);
  static constexpr uint32_t BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  __aicore__ __attribute__((always_inline))
  l1_to_l0_b(__cb__ DataType *l0bAddr, __cbuf__ DataType *l1Addr,
             uint32_t nTileCeil, uint32_t kPartCeil, uint32_t nSrcStride,
             uint32_t kSrcStride, uint32_t nDstStride, uint32_t kDstStride) {
    for (uint32_t i = 0; i < kPartCeil / BLOCK_SIZE; i++) {
      load_cbuf_to_cb(l0bAddr + i * kDstStride * FRACTAL_SIZE,
                      l1Addr + i * kSrcStride * FRACTAL_SIZE, 0,
                      nTileCeil / BLOCK_NUM_PER_FRACTAL,
                      nSrcStride,     // srcStride
                      nDstStride - 1, // dstStride
                      0,              // sid
                      false,          // transpose
                      inc             // addr_cal_mode_t
      );
    }
  };
};

// Partial specialization for V220, int8_t, no transpose, not vector
template <>
struct l1_to_l0_b<ArchType::ASCEND_V220, int8_t, false, CubeFormat::ZN,
                  CubeFormat::NZ> {
  static constexpr ArchType ArchTag = ArchType::ASCEND_V220;
  using DataType = int8_t;
  using HardwareParams = HardwareInfo<ArchTag>;
  static constexpr uint32_t BLOCK_SIZE =
      HardwareParams::l1l0BlockSize / sizeof(DataType);
  static constexpr uint32_t FRACTAL_SIZE =
      HardwareParams::fractalSize / sizeof(DataType);
  static constexpr uint32_t BLOCK_NUM_PER_FRACTAL =
      HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

  __aicore__ __attribute__((always_inline))
  l1_to_l0_b(__cb__ DataType *l0bAddr, __cbuf__ DataType *l1Addr,
             uint32_t nTileCeil, uint32_t kPartCeil, uint32_t nSrcStride,
             uint32_t kSrcStride, uint32_t nDstStride, uint32_t kDstStride) {
    for (uint32_t i = 0; i < kPartCeil / (BLOCK_NUM_PER_FRACTAL * CONST_2);
         i++) {
      load_cbuf_to_cb_transpose(l0bAddr + i * kDstStride * FRACTAL_SIZE * 2,
                                l1Addr + i * kSrcStride * FRACTAL_SIZE * 2,
                                0,                      // baseIdx
                                nTileCeil / BLOCK_SIZE, // repeat
                                nSrcStride / 2,         // srcStride
                                nDstStride,             // dstStride
                                0,                      // addrmode
                                0                       // dstFracStride
      );
    }
  };
};