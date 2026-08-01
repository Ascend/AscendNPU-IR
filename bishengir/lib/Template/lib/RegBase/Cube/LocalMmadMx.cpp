/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef CATLASS_GEMM_L1MMAD_HPP
#define CATLASS_GEMM_L1MMAD_HPP

#define TILING_KEY_VAR

#include "Cube/LocalMmad/LocalMmadUtils.h"

#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt_a5.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a_a5.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b_a5.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define CATLASS_ARCH_A5_ENABLED
#endif

using ElementMxScaleA = AscendCBisheng::fp8_e8m0_t;
using ElementMxScaleB = AscendCBisheng::fp8_e8m0_t;

namespace Catlass::Gemm {

template <bool Trans> struct TransToTag {};

template <> struct TransToTag<false> {
  using tag = layout::zN;
};

template <> struct TransToTag<true> {
  using tag = layout::nZ;
};

__aicore__ inline uint32_t getL1MmadPingPongId() {
  static uint32_t pingPongId = 1;
  pingPongId = 1 - pingPongId;
  return pingPongId;
}

enum class HIVMMatmulDataformat : uint32_t {
  FP8E5M2_T = 1,
  FP8E4M3_T = 2,
  FP4E2M1_T = 3,
};

CATLASS_DEVICE inline bool isFp4Format(HIVMMatmulDataformat format) {
  return format == HIVMMatmulDataformat::FP4E2M1_T;
}

CATLASS_DEVICE inline uint32_t getMxFormatKFactor(HIVMMatmulDataformat format) {
  return isFp4Format(format) ? 2 : 1;
}

template <class ElementAFp8, class ArchTag, class LayoutTagL1A,
          class LayoutTagL0A, class TensorMxScale>
CATLASS_DEVICE void copyTransposedAInFp8Format(
    __cbuf__ int8_t *l1A, uint32_t l1M, uint32_t l1K, uint32_t actualM,
    uint32_t kL0Actual, uint32_t kL0Idx, uint32_t l0K, uint32_t pingPongId,
    TensorMxScale const &tensorTileL1MxScaleA) {
  using LayoutL1AFp8 = detail::TagToLayout_t<ElementAFp8, LayoutTagL1A>;
  using LayoutL0AFp8 = detail::TagToLayout_t<ElementAFp8, LayoutTagL0A>;
  using TensorL1AFp8 =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementAFp8>, LayoutL1AFp8,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
  using TensorL0AFp8 =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementAFp8>, LayoutL0AFp8,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A2>;
  using CopyL1ToL0AFp8 =
      Gemm::Tile::TileCopyTla<ArchTag, TensorL1AFp8, TensorL0AFp8>;

  AscendCBisheng::LocalTensor<ElementAFp8> l1ATensorFp8{
      AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1A),
      l1M * l1K};
  AscendCBisheng::LocalTensor<ElementAFp8> l0ATensorFp8{
      AscendCBisheng::TPosition::A2, 0, ArchTag::L0A_SIZE};

  auto tensorL1AFp8 = tla::MakeTensor(
      l1ATensorFp8, tla::MakeLayout<ElementAFp8, LayoutTagL1A>(l1M, l1K),
      Arch::PositionL1{});
  auto l0ATileFp8 =
      l0ATensorFp8[pingPongId * (ArchTag::L0A_SIZE / 2) /
                   sizeof(ElementAFp8)];
  auto tensorL0AFp8 = tla::MakeTensor(
      l0ATileFp8,
      tla::MakeLayout<ElementAFp8, LayoutTagL0A>(actualM, kL0Actual),
      Arch::PositionL0A{});
  auto tensorTileL1AFp8 =
      GetTile(tensorL1AFp8, tla::MakeCoord(0, kL0Idx * l0K),
              tla::MakeShape(actualM, kL0Actual));

  CopyL1ToL0AFp8 copyL1ToL0AFp8;
  copyL1ToL0AFp8(tensorL0AFp8, tensorTileL1AFp8, tensorTileL1MxScaleA);
}

template <class ArchTag, class LayoutTagL1A, class LayoutTagL0A,
          class CopyL1ToL0A, class TensorL0A, class TensorL1A,
          class TensorMxScale>
CATLASS_DEVICE void copyTransposedAByFormat(
    CopyL1ToL0A &copyL1ToL0A, TensorL0A const &tensorL0A,
    TensorL1A const &tensorTileL1A, TensorMxScale const &tensorTileL1MxScaleA,
    HIVMMatmulDataformat lhsFormat, __cbuf__ int8_t *l1A, uint32_t l1M,
    uint32_t l1K, uint32_t actualM, uint32_t kL0Actual, uint32_t kL0Idx,
    uint32_t l0K, uint32_t pingPongId) {
  if (lhsFormat == HIVMMatmulDataformat::FP8E5M2_T) {
    copyTransposedAInFp8Format<float8_e5m2_t, ArchTag, LayoutTagL1A,
                               LayoutTagL0A>(l1A, l1M, l1K, actualM,
                                             kL0Actual, kL0Idx, l0K,
                                             pingPongId, tensorTileL1MxScaleA);
  } else if (lhsFormat == HIVMMatmulDataformat::FP8E4M3_T) {
    copyTransposedAInFp8Format<float8_e4m3_t, ArchTag, LayoutTagL1A,
                               LayoutTagL0A>(l1A, l1M, l1K, actualM,
                                             kL0Actual, kL0Idx, l0K,
                                             pingPongId, tensorTileL1MxScaleA);
  } else {
    copyL1ToL0A(tensorL0A, tensorTileL1A, tensorTileL1MxScaleA);
  }
}

// A5 mad_mx has no separate bias operand. Bias table address is packed into
// Xd[63:32], with C address in Xd[31:0] (same as AscendC MmadCal).
template <class ElementACC>
CATLASS_DEVICE __cc__ ElementACC *packCWithBiasAddr(__cc__ ElementACC *l0CPhyAddr,
                                                    uint64_t biasAddr,
                                                    bool hasBias) {
  if (!hasBias)
    return l0CPhyAddr;
  uint64_t xd = (reinterpret_cast<uint64_t>(l0CPhyAddr) & 0xffffffffULL) |
                ((biasAddr & 0xffffffffULL) << 32);
  return reinterpret_cast<__cc__ ElementACC *>(xd);
}

template <class ElementACC, class L0CPhyAddr, class L0APhyAddr,
          class L0BPhyAddr>
CATLASS_DEVICE void madMxByFormat(L0CPhyAddr l0CPhyAddr, L0APhyAddr l0APhyAddr,
                                  L0BPhyAddr l0BPhyAddr, uint32_t actualM,
                                  uint32_t kL0Actual, uint32_t actualN,
                                  uint8_t unitFlag, bool initC,
                                  HIVMMatmulDataformat lhsFormat,
                                  HIVMMatmulDataformat rhsFormat,
                                  uint64_t biasAddr = 0, bool hasBias = false) {
  // With bias: C comes from bias table (cmatrixSource=true), so initC is false.
  const bool cmatrixSource = hasBias;
  const bool cmatrixInitVal = initC && !hasBias;
  auto *cAddr = packCWithBiasAddr<ElementACC>(
      (__cc__ ElementACC *)l0CPhyAddr, biasAddr, hasBias);
  if (lhsFormat == HIVMMatmulDataformat::FP8E5M2_T &&
      rhsFormat == HIVMMatmulDataformat::FP8E5M2_T) {
    INTRINSIC(mad_mx, cAddr, (__ca__ float8_e5m2_t *)l0APhyAddr,
              (__cb__ float8_e5m2_t *)l0BPhyAddr, actualM, kL0Actual, actualN,
              unitFlag, true, cmatrixSource, cmatrixInitVal);
  } else if (lhsFormat == HIVMMatmulDataformat::FP8E5M2_T &&
             rhsFormat == HIVMMatmulDataformat::FP8E4M3_T) {
    INTRINSIC(mad_mx, cAddr, (__ca__ float8_e5m2_t *)l0APhyAddr,
              (__cb__ float8_e4m3_t *)l0BPhyAddr, actualM, kL0Actual, actualN,
              unitFlag, true, cmatrixSource, cmatrixInitVal);
  } else if (lhsFormat == HIVMMatmulDataformat::FP8E4M3_T &&
             rhsFormat == HIVMMatmulDataformat::FP8E5M2_T) {
    INTRINSIC(mad_mx, cAddr, (__ca__ float8_e4m3_t *)l0APhyAddr,
              (__cb__ float8_e5m2_t *)l0BPhyAddr, actualM, kL0Actual, actualN,
              unitFlag, true, cmatrixSource, cmatrixInitVal);
  } else if (lhsFormat == HIVMMatmulDataformat::FP8E4M3_T &&
             rhsFormat == HIVMMatmulDataformat::FP8E4M3_T) {
    INTRINSIC(mad_mx, cAddr, (__ca__ float8_e4m3_t *)l0APhyAddr,
              (__cb__ float8_e4m3_t *)l0BPhyAddr, actualM, kL0Actual, actualN,
              unitFlag, true, cmatrixSource, cmatrixInitVal);
  } else if (lhsFormat == HIVMMatmulDataformat::FP4E2M1_T &&
             rhsFormat == HIVMMatmulDataformat::FP4E2M1_T) {
    INTRINSIC(mad_mx, cAddr, (__ca__ fp4x2_e2m1_t *)l0APhyAddr,
              (__cb__ fp4x2_e2m1_t *)l0BPhyAddr, actualM, kL0Actual, actualN,
              unitFlag, true, cmatrixSource, cmatrixInitVal);
  } else {
    // Mixed FP8/FP4 formats are not registered by the template library.
  }
}

template <class ElementA, class ElementB, class ElementBias, class ElementACC,
          bool TA, bool TB, bool HF32>
CATLASS_DEVICE void
L1MxMmad(__cc__ ElementACC *l0C, __cbuf__ ElementA *l1A, __cbuf__ ElementB *l1B,
         __cbuf__ ElementMxScaleA *l1MxScaleA,
         __cbuf__ ElementMxScaleB *l1MxScaleB, __cbuf__ ElementBias *l1Bias,
         uint32_t l1M, uint32_t l1K, uint32_t l1N, uint32_t actualM,
         uint32_t actualK, uint32_t actualN,
         uint32_t l1AMTE2MTE1EventId, uint32_t l1ScaleAMTE2MTE1EventId,
         uint32_t l1BMTE2MTE1EventId, uint32_t l1ScaleBMTE2MTE1EventId,
         uint32_t l1AMTE1MTE2EventId, uint32_t l1ScaleAMTE1MTE2EventId,
         uint32_t l1BMTE1MTE2EventId, uint32_t l1ScaleBMTE1MTE2EventId,
         bool isL1FirstK, bool isL1LastK, bool enable_unit_flag,
         bool hasBias = false) {
  if constexpr (HF32) {
    AscendCBisheng::SetHF32Mode(true);
  }

  using ArchTag = Arch::AtlasA5;
  using LayoutTagL1A = typename TransToTag<TA>::tag;
  using LayoutTagL1B = typename TransToTag<TB>::tag;
  using LayoutTagL1MxScaleA = layout::zZ;
  using LayoutTagL1MxScaleB = layout::nN;
  using LayoutTagL0A = layout::zN;
  using LayoutTagL0B = layout::nZ;

  using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
  using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
  using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
  using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
  using LayoutL0C = typename detail::LayoutL0C;

  using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagL1A>;
  using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagL1B>;

  using TensorL1A =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementA>, LayoutL1A,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
  using TensorL1B =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementB>, LayoutL1B,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
  using TensorL0A =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementA>, LayoutL0A,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A2>;
  using TensorL0B =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementB>, LayoutL0B,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::B2>;
  using TensorL0C =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementACC>, LayoutL0C,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::CO1>;
  using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
  using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
  CopyL1ToL0A copyL1ToL0A;
  CopyL1ToL0B copyL1ToL0B;

  AscendCBisheng::LocalTensor<ElementA> l1ATensor{
      AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1A),
      l1M * l1K};
  AscendCBisheng::LocalTensor<ElementB> l1BTensor{
      AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1B),
      l1K * l1N};
  AscendCBisheng::LocalTensor<ElementMxScaleA> l1MxScaleATensor{
      AscendCBisheng::TPosition::A1,
      (uint32_t)reinterpret_cast<int64_t>(l1MxScaleA),
      l1M * l1K / MX_SCALE_GROUP_NUM};
  AscendCBisheng::LocalTensor<ElementMxScaleB> l1MxScaleBTensor{
      AscendCBisheng::TPosition::A1,
      (uint32_t)reinterpret_cast<int64_t>(l1MxScaleB),
      l1K * l1N / MX_SCALE_GROUP_NUM};
  AscendCBisheng::LocalTensor<ElementA> l0ATensor{AscendCBisheng::TPosition::A2,
                                                  0, ArchTag::L0A_SIZE};
  AscendCBisheng::LocalTensor<ElementB> l0BTensor{AscendCBisheng::TPosition::B2,
                                                  0, ArchTag::L0B_SIZE};
  AscendCBisheng::LocalTensor<ElementACC> bTTensor{
      AscendCBisheng::TPosition::C2, 0, ArchTag::BIAS_SIZE};
  AscendCBisheng::LocalTensor<ElementACC> l0CTensor{
      AscendCBisheng::TPosition::CO1, (uint32_t)reinterpret_cast<int64_t>(l0C),
      l1M * l1N};

  auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(l1M, l1K);
  auto tensorL1A = tla::MakeTensor(l1ATensor, layoutAInL1, Arch::PositionL1{});
  auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(l1K, l1N);
  auto tensorL1B = tla::MakeTensor(l1BTensor, layoutBInL1, Arch::PositionL1{});
  auto layoutMxScaleAInL1 =
      tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(
          l1M, l1K / MX_SCALE_GROUP_NUM);
  auto tensorL1MxScaleA =
      tla::MakeTensor(l1MxScaleATensor, layoutMxScaleAInL1, Arch::PositionL1{});
  auto layoutMxScaleBInL1 =
      tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
          l1K / MX_SCALE_GROUP_NUM, l1N);
  auto tensorL1MxScaleB =
      tla::MakeTensor(l1MxScaleBTensor, layoutMxScaleBInL1, Arch::PositionL1{});
  auto layoutInL0C = tla::MakeLayoutL0C(actualM, actualN);
  auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
  auto layoutBiasInBT = tla::MakeLayout(actualN);
  auto tensorL0Bias =
      tla::MakeTensor(bTTensor, layoutBiasInBT, Arch::PositionBias{});

  constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / 2;
  constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / 2;
  constexpr uint32_t L0A_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
  constexpr uint32_t L0B_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementB);

  uint32_t l0K =
      RoundDown<64>(min(L0A_PINGPONG_BUF_SIZE / sizeof(ElementA) /
                            RoundUp<L1AAlignHelper::M_ALIGNED>(actualM) /
                            L0A_ELE_NUM_PER_C0 * L0A_ELE_NUM_PER_C0,
                        L0B_PINGPONG_BUF_SIZE / sizeof(ElementB) /
                            RoundUp<L1BAlignHelper::N_ALIGNED>(actualN) /
                            L0B_ELE_NUM_PER_C0 * L0B_ELE_NUM_PER_C0));

  uint32_t kL0Loop = CeilDiv(actualK, l0K);

  for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
    uint32_t kL0Actual =
        (kL0Idx < kL0Loop - 1) ? l0K : (actualK - kL0Idx * l0K);
    // Get ping/pong id (0 or 1) — unified with L1Mmad using ID0/ID1
    uint32_t pingPongId = getL1MmadPingPongId();
    uint32_t l0EventId = !pingPongId ? EVENT_ID0 : EVENT_ID1;

    // Wait for mmad finished
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);

    // Locate the current tile on L0A
    auto l0ATile =
        l0ATensor[pingPongId * L0A_PINGPONG_BUF_SIZE / sizeof(ElementA)];
    auto layoutAInL0 =
        tla::MakeLayout<ElementA, LayoutTagL0A>(actualM, kL0Actual);
    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
    // Locate the current tile of matrix A on L1
    auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * l0K),
                                 tla::MakeShape(actualM, kL0Actual));
    // Locate the current tile of matrix mxScaleA on L1
    auto tensorTileL1MxScaleA = GetTile(
        tensorL1MxScaleA, tla::MakeCoord(0, kL0Idx * l0K / MX_SCALE_GROUP_NUM),
        tla::MakeShape(actualM, CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual)));

    // Load current tile from L1 to L0A.  The transposed A path needs the
    // scale metadata copied with the data copy so the MX scale address follows
    // the same transpose/tail handling as the data tile — Wait for both before
    // the fused copy.  On the non-transposed path each stream has its own
    // independent Wait→Copy→Set chain.
    if constexpr (TA) {
      if (kL0Idx == 0) {
        if (l1AMTE2MTE1EventId      != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AMTE2MTE1EventId);
        if (l1ScaleAMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1ScaleAMTE2MTE1EventId);
      }
      copyL1ToL0A(tensorL0A, tensorTileL1A, tensorTileL1MxScaleA);
      if (kL0Idx == kL0Loop - 1) {
        if (l1AMTE1MTE2EventId      != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AMTE1MTE2EventId);
        if (l1ScaleAMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1ScaleAMTE1MTE2EventId);
      }
    } else {
      // --- A: Wait → Copy → Set ---
      if (kL0Idx == 0) {
        if (l1AMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AMTE2MTE1EventId);
      }
      copyL1ToL0A(tensorL0A, tensorTileL1A);
      if (kL0Idx == kL0Loop - 1) {
        if (l1AMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AMTE1MTE2EventId);
      }
      // --- ScaleA: Wait → Copy → Set ---
      if (kL0Idx == 0) {
        if (l1ScaleAMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1ScaleAMTE2MTE1EventId);
      }
      copyL1ToL0A.copyScaleTensor(tensorL0A, tensorTileL1MxScaleA);
      if (kL0Idx == kL0Loop - 1) {
        if (l1ScaleAMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1ScaleAMTE1MTE2EventId);
      }
    }

    // Locate the current tile on L0B
    auto l0BTile =
        l0BTensor[pingPongId * L0B_PINGPONG_BUF_SIZE / sizeof(ElementB)];
    auto layoutBInL0 =
        tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, actualN);
    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
    auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(kL0Idx * l0K, 0),
                                 tla::MakeShape(kL0Actual, actualN));
    auto tensorTileL1MxScaleB = GetTile(
        tensorL1MxScaleB, tla::MakeCoord(kL0Idx * l0K / MX_SCALE_GROUP_NUM, 0),
        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), actualN));
    // --- B: Wait → Copy → Set ---
    if (kL0Idx == 0) {
      if (l1BMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1BMTE2MTE1EventId);
    }
    copyL1ToL0B(tensorL0B, tensorTileL1B);
    if (kL0Idx == kL0Loop - 1) {
      if (l1BMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BMTE1MTE2EventId);
    }
    // --- ScaleB: Wait → Copy → Set ---
    if (kL0Idx == 0) {
      if (l1ScaleBMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1ScaleBMTE2MTE1EventId);
    }
    copyL1ToL0B.copyScaleTensor(tensorL0B, tensorTileL1MxScaleB);
    if (kL0Idx == kL0Loop - 1) {
      if (l1ScaleBMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1ScaleBMTE1MTE2EventId);
    }

    const bool initC = isL1FirstK && (kL0Idx == 0);
    if constexpr (!std::is_void_v<ElementBias>) {
      if (hasBias && initC) {
        AscendCBisheng::LocalTensor<ElementBias> l1BiasTensor{
            AscendCBisheng::TPosition::A1,
            (uint32_t)reinterpret_cast<int64_t>(l1Bias), actualN};
        auto layoutBiasInL1 = tla::MakeLayout(actualN);
        auto tensorL1Bias =
            tla::MakeTensor(l1BiasTensor, layoutBiasInL1, Arch::PositionL1{});
        using TensorL1Bias = tla::Tensor<
            AscendCBisheng::LocalTensor<ElementBias>,
            detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
            tla::Coord<tla::_0>, AscendCBisheng::TPosition::A1>;
        using TensorL0Bias = tla::Tensor<
            AscendCBisheng::LocalTensor<ElementACC>,
            detail::TagToLayout_t<ElementACC, layout::VectorLayout>,
            tla::Coord<tla::_0>, AscendCBisheng::TPosition::C2>;
        using CopyL1ToBT = Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>;
        CopyL1ToBT copyL1ToBT;
        copyL1ToBT(tensorL0Bias, tensorL1Bias);
      }
    }

    // Notify to do mmad
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);

    const bool applyBias = hasBias && initC;
    auto *cAddr = packCWithBiasAddr<ElementACC>(
        (__cc__ ElementACC *)tensorL0C.data().GetPhyAddr(),
        applyBias ? (uint64_t)tensorL0Bias.data().GetPhyAddr() : 0, applyBias);
    INTRINSIC(mad_mx, cAddr,
              (__ca__ ElementA *)tensorL0A.data().GetPhyAddr(),
              (__cb__ ElementB *)tensorL0B.data().GetPhyAddr(), actualM,
              kL0Actual, actualN,
              /* unitFlag = */ 0b00, true, /* cmatrixSource = */ applyBias,
              /* cmatrixInitVal = */ initC && !applyBias);

    // Notify to move the next L0B tile
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);
  }

  if constexpr (HF32) {
    AscendCBisheng::SetHF32Mode(false);
  }
}

template <class ElementA, class ElementB, class ElementBias, class ElementACC,
          bool TA, bool TB, bool HF32>
CATLASS_DEVICE void
L1MxMmad(__cc__ ElementACC *l0C, __cbuf__ ElementA *l1A, __cbuf__ ElementB *l1B,
         __cbuf__ ElementMxScaleA *l1MxScaleA,
         __cbuf__ ElementMxScaleB *l1MxScaleB, __cbuf__ ElementBias *l1Bias,
         uint32_t l1M, uint32_t l1K, uint32_t l1N, uint32_t actualM,
         uint32_t actualK, uint32_t actualN,
         uint32_t l1AMTE2MTE1EventId, uint32_t l1ScaleAMTE2MTE1EventId,
         uint32_t l1BMTE2MTE1EventId, uint32_t l1ScaleBMTE2MTE1EventId,
         uint32_t l1AMTE1MTE2EventId, uint32_t l1ScaleAMTE1MTE2EventId,
         uint32_t l1BMTE1MTE2EventId, uint32_t l1ScaleBMTE1MTE2EventId,
         bool isL1FirstK, bool isL1LastK, bool enable_unit_flag,
         HIVMMatmulDataformat lhsFormat, HIVMMatmulDataformat rhsFormat,
         bool hasBias = false) {
  if constexpr (HF32) {
    AscendCBisheng::SetHF32Mode(true);
  }

  using ArchTag = Arch::AtlasA5;
  using LayoutTagL1A = typename TransToTag<TA>::tag;
  using LayoutTagL1B = typename TransToTag<TB>::tag;
  using LayoutTagL1MxScaleA = layout::zZ;
  using LayoutTagL1MxScaleB = layout::nN;
  using LayoutTagL0A = layout::zN;
  using LayoutTagL0B = layout::nZ;

  using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
  using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
  using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
  using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
  using LayoutL0C = typename detail::LayoutL0C;

  using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagL1A>;
  using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagL1B>;

  using TensorL1A =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementA>, LayoutL1A,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
  using TensorL1B =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementB>, LayoutL1B,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
  using TensorL0A =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementA>, LayoutL0A,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A2>;
  using TensorL0B =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementB>, LayoutL0B,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::B2>;
  using TensorL0C =
      tla::Tensor<AscendCBisheng::LocalTensor<ElementACC>, LayoutL0C,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::CO1>;
  using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
  using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
  using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1A>;
  TileMmad tileMmad;
  CopyL1ToL0A copyL1ToL0A;
  CopyL1ToL0B copyL1ToL0B;

  AscendCBisheng::LocalTensor<ElementA> l1ATensor{
      AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1A),
      l1M * l1K};
  AscendCBisheng::LocalTensor<ElementB> l1BTensor{
      AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1B),
      l1K * l1N};
  AscendCBisheng::LocalTensor<ElementMxScaleA> l1MxScaleATensor{
      AscendCBisheng::TPosition::A1,
      (uint32_t)reinterpret_cast<int64_t>(l1MxScaleA),
      l1M * l1K * getMxFormatKFactor(lhsFormat) /
          MX_SCALE_GROUP_NUM};
  AscendCBisheng::LocalTensor<ElementMxScaleB> l1MxScaleBTensor{
      AscendCBisheng::TPosition::A1,
      (uint32_t)reinterpret_cast<int64_t>(l1MxScaleB),
      l1K * l1N * getMxFormatKFactor(rhsFormat) /
          MX_SCALE_GROUP_NUM};
  AscendCBisheng::LocalTensor<ElementA> l0ATensor{AscendCBisheng::TPosition::A2,
                                                  0, ArchTag::L0A_SIZE};
  AscendCBisheng::LocalTensor<ElementB> l0BTensor{AscendCBisheng::TPosition::B2,
                                                  0, ArchTag::L0B_SIZE};
  AscendCBisheng::LocalTensor<ElementACC> bTTensor{
      AscendCBisheng::TPosition::C2, 0, ArchTag::BIAS_SIZE};
  AscendCBisheng::LocalTensor<ElementACC> l0CTensor{
      AscendCBisheng::TPosition::CO1, (uint32_t)reinterpret_cast<int64_t>(l0C),
      l1M * l1N};

  auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(l1M, l1K);
  auto tensorL1A = tla::MakeTensor(l1ATensor, layoutAInL1, Arch::PositionL1{});
  auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(l1K, l1N);
  auto tensorL1B = tla::MakeTensor(l1BTensor, layoutBInL1, Arch::PositionL1{});
  auto layoutMxScaleAInL1 =
      tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(
          l1M, l1K * getMxFormatKFactor(lhsFormat) / MX_SCALE_GROUP_NUM);
  auto tensorL1MxScaleA =
      tla::MakeTensor(l1MxScaleATensor, layoutMxScaleAInL1, Arch::PositionL1{});
  auto layoutMxScaleBInL1 =
      tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
          l1K * getMxFormatKFactor(rhsFormat) / MX_SCALE_GROUP_NUM,
          l1N);
  auto tensorL1MxScaleB =
      tla::MakeTensor(l1MxScaleBTensor, layoutMxScaleBInL1, Arch::PositionL1{});
  auto layoutInL0C = tla::MakeLayoutL0C(actualM, actualN);
  auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
  auto layoutBiasInBT = tla::MakeLayout(actualN);
  auto tensorL0Bias =
      tla::MakeTensor(bTTensor, layoutBiasInBT, Arch::PositionBias{});

  constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / 2;
  constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / 2;
  constexpr uint32_t L0A_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
  constexpr uint32_t L0B_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementB);

  uint32_t l0K =
      RoundDown<64>(min(L0A_PINGPONG_BUF_SIZE / sizeof(ElementA) /
                            RoundUp<L1AAlignHelper::M_ALIGNED>(actualM) /
                            L0A_ELE_NUM_PER_C0 * L0A_ELE_NUM_PER_C0,
                        L0B_PINGPONG_BUF_SIZE / sizeof(ElementB) /
                            RoundUp<L1BAlignHelper::N_ALIGNED>(actualN) /
                            L0B_ELE_NUM_PER_C0 * L0B_ELE_NUM_PER_C0));

  actualK *= getMxFormatKFactor(lhsFormat);
  uint32_t kL0Loop = CeilDiv(actualK, l0K);

  for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
    uint32_t kL0Actual =
        (kL0Idx < kL0Loop - 1) ? l0K : (actualK - kL0Idx * l0K);
    // Get ping/pong id (0 or 1) — unified with L1Mmad using ID0/ID1
    uint32_t pingPongId = getL1MmadPingPongId();
    uint32_t l0EventId = !pingPongId ? EVENT_ID0 : EVENT_ID1;

    // Wait for mmad finished
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);

    // Locate the current tile on L0A
    auto l0ATile =
        l0ATensor[pingPongId * L0A_PINGPONG_BUF_SIZE / sizeof(ElementA)];
    auto layoutAInL0 =
        tla::MakeLayout<ElementA, LayoutTagL0A>(actualM, kL0Actual);
    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
    // Locate the current tile of matrix A on L1
    auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * l0K),
                                 tla::MakeShape(actualM, kL0Actual));
    // Locate the current tile of matrix mxScaleA on L1
    auto tensorTileL1MxScaleA = GetTile(
        tensorL1MxScaleA, tla::MakeCoord(0, kL0Idx * l0K / MX_SCALE_GROUP_NUM),
        tla::MakeShape(actualM, CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual)));

    // Load current tile from L1 to L0A — transposed path needs both before
    // the fused copy; non-transposed path has independent Wait→Copy→Set
    // for each stream.
    // FIXME: this need to refactor back into one without if branch.
    if constexpr (TA) {
      if (kL0Idx == 0) {
        if (l1AMTE2MTE1EventId      != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AMTE2MTE1EventId);
        if (l1ScaleAMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1ScaleAMTE2MTE1EventId);
      }
      copyTransposedAByFormat<ArchTag, LayoutTagL1A, LayoutTagL0A>(
          copyL1ToL0A, tensorL0A, tensorTileL1A, tensorTileL1MxScaleA,
          lhsFormat, reinterpret_cast<__cbuf__ int8_t *>(l1A), l1M, l1K,
          actualM, kL0Actual, kL0Idx, l0K, pingPongId);
      if (kL0Idx == kL0Loop - 1) {
        if (l1AMTE1MTE2EventId      != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AMTE1MTE2EventId);
        if (l1ScaleAMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1ScaleAMTE1MTE2EventId);
      }
    } else {
      // --- A: Wait → Copy → Set ---
      if (kL0Idx == 0) {
        if (l1AMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AMTE2MTE1EventId);
      }
      copyL1ToL0A(tensorL0A, tensorTileL1A);
      if (kL0Idx == kL0Loop - 1) {
        if (l1AMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AMTE1MTE2EventId);
      }
      // --- ScaleA: Wait → Copy → Set ---
      if (kL0Idx == 0) {
        if (l1ScaleAMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1ScaleAMTE2MTE1EventId);
      }
      copyL1ToL0A.copyScaleTensor(tensorL0A, tensorTileL1MxScaleA);
      if (kL0Idx == kL0Loop - 1) {
        if (l1ScaleAMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1ScaleAMTE1MTE2EventId);
      }
    }

    // Locate the current tile on L0B
    auto l0BTile =
        l0BTensor[pingPongId * L0B_PINGPONG_BUF_SIZE / sizeof(ElementB)];
    auto layoutBInL0 =
        tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, actualN);
    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
    auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(kL0Idx * l0K, 0),
                                 tla::MakeShape(kL0Actual, actualN));
    auto tensorTileL1MxScaleB = GetTile(
        tensorL1MxScaleB, tla::MakeCoord(kL0Idx * l0K / MX_SCALE_GROUP_NUM, 0),
        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), actualN));
    // --- B: Wait → Copy → Set ---
    if (kL0Idx == 0) {
      if (l1BMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1BMTE2MTE1EventId);
    }
    copyL1ToL0B(tensorL0B, tensorTileL1B);
    if (kL0Idx == kL0Loop - 1) {
      if (l1BMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BMTE1MTE2EventId);
    }
    // --- ScaleB: Wait → Copy → Set ---
    if (kL0Idx == 0) {
      if (l1ScaleBMTE2MTE1EventId != -1) AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1ScaleBMTE2MTE1EventId);
    }
    copyL1ToL0B.copyScaleTensor(tensorL0B, tensorTileL1MxScaleB);
    if (kL0Idx == kL0Loop - 1) {
      if (l1ScaleBMTE1MTE2EventId != -1) AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1ScaleBMTE1MTE2EventId);
    }

    bool initC = isL1FirstK && (kL0Idx == 0);
    if constexpr (!std::is_void_v<ElementBias>) {
      if (hasBias && initC) {
        AscendCBisheng::LocalTensor<ElementBias> l1BiasTensor{
            AscendCBisheng::TPosition::A1,
            (uint32_t)reinterpret_cast<int64_t>(l1Bias), actualN};
        auto layoutBiasInL1 = tla::MakeLayout(actualN);
        auto tensorL1Bias =
            tla::MakeTensor(l1BiasTensor, layoutBiasInL1, Arch::PositionL1{});
        using TensorL1Bias = tla::Tensor<
            AscendCBisheng::LocalTensor<ElementBias>,
            detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
            tla::Coord<tla::_0>, AscendCBisheng::TPosition::A1>;
        using TensorL0Bias = tla::Tensor<
            AscendCBisheng::LocalTensor<ElementACC>,
            detail::TagToLayout_t<ElementACC, layout::VectorLayout>,
            tla::Coord<tla::_0>, AscendCBisheng::TPosition::C2>;
        using CopyL1ToBT = Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>;
        CopyL1ToBT copyL1ToBT;
        copyL1ToBT(tensorL0Bias, tensorL1Bias);
      }
    }

    // Notify to do mmad
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);

    // If the unit flag is enabled, the unit flag is set according to the
    // calculation progress
    uint8_t unitFlag = 0b00;
    // if (enable_unit_flag) {
    //   if (isL1LastK && (kL0Idx == kL0Loop - 1)) {
    //     unitFlag = 0b11;
    //   } else {
    //     unitFlag = 0b10;
    //   }
    // }

    const bool applyBias = hasBias && initC;
    madMxByFormat<ElementACC>(
        tensorL0C.data().GetPhyAddr(), tensorL0A.data().GetPhyAddr(),
        tensorL0B.data().GetPhyAddr(), actualM, kL0Actual, actualN, unitFlag,
        initC, lhsFormat, rhsFormat,
        applyBias ? (uint64_t)tensorL0Bias.data().GetPhyAddr() : 0,
        applyBias);

    // Notify to move the next L0B tile
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);
  }

  if constexpr (HF32) {
    AscendCBisheng::SetHF32Mode(false);
  }
}

} // namespace Catlass::Gemm

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE,
          bool TA = false, bool TB = false>
__aicore__ __attribute__((always_inline)) void mmamx_tile_core(
    memref_t<__cc__ DST_TYPE, 4> *mc, memref_t<__cbuf__ SRC_TYPE, 4> *ma,
    memref_t<__cbuf__ SRC_TYPE, 4> *mb,
    memref_t<__cbuf__ ElementMxScaleA, 1> *l1MxScaleA,
    memref_t<__cbuf__ ElementMxScaleB, 1> *l1MxScaleB, bool init, int64_t m,
    int64_t k, int64_t n,
    int64_t mmad_l1_wait_l1a_event,
    int64_t mmad_l1_wait_l1scalea_event,
    int64_t mmad_l1_wait_l1b_event, int64_t mmad_l1_wait_l1scaleb_event, int64_t l1a_wait_mmad_l1_event,
    int64_t l1scalea_wait_mmad_l1_event,
    int64_t l1b_wait_mmad_l1_event, int64_t l1scaleb_wait_mmad_l1_event) {
  Catlass::Gemm::L1MxMmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, TA, TB,
                          false>(
      mc->aligned + mc->offset, ma->aligned + ma->offset,
      mb->aligned + mb->offset, l1MxScaleA->aligned + l1MxScaleA->offset,
      l1MxScaleB->aligned + l1MxScaleB->offset, nullptr,
      (TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2])),
      (TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3])),
      (TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3])),
      m, k, n,
      mmad_l1_wait_l1a_event, mmad_l1_wait_l1scalea_event,
      mmad_l1_wait_l1b_event, mmad_l1_wait_l1scaleb_event,
      l1a_wait_mmad_l1_event, l1scalea_wait_mmad_l1_event,
      l1b_wait_mmad_l1_event, l1scaleb_wait_mmad_l1_event,
      init, true, false, false);
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE,
          bool TA = false, bool TB = false>
__aicore__ __attribute__((always_inline)) void mmamx_tile_bias(
    memref_t<__cc__ DST_TYPE, 4> *mc, memref_t<__cbuf__ SRC_TYPE, 4> *ma,
    memref_t<__cbuf__ SRC_TYPE, 4> *mb,
    memref_t<__cbuf__ ElementMxScaleA, 1> *l1MxScaleA,
    memref_t<__cbuf__ ElementMxScaleB, 1> *l1MxScaleB, bool init, int64_t m,
    int64_t k, int64_t n, memref_t<__cbuf__ BIAS_TYPE, 4> *bias,
    int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1scalea_event,
    int64_t mmad_l1_wait_l1b_event, int64_t mmad_l1_wait_l1scaleb_event,
    int64_t l1a_wait_mmad_l1_event, int64_t l1scalea_wait_mmad_l1_event,
    int64_t l1b_wait_mmad_l1_event, int64_t l1scaleb_wait_mmad_l1_event) {
  Catlass::Gemm::L1MxMmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, TA, TB,
                          false>(
      mc->aligned + mc->offset, ma->aligned + ma->offset,
      mb->aligned + mb->offset, l1MxScaleA->aligned + l1MxScaleA->offset,
      l1MxScaleB->aligned + l1MxScaleB->offset, bias->aligned + bias->offset,
      (TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2])),
      (TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3])),
      (TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3])),
      m, k, n, mmad_l1_wait_l1a_event, mmad_l1_wait_l1scalea_event,
      mmad_l1_wait_l1b_event, mmad_l1_wait_l1scaleb_event,
      l1a_wait_mmad_l1_event, l1scalea_wait_mmad_l1_event,
      l1b_wait_mmad_l1_event, l1scaleb_wait_mmad_l1_event, init, true, false, true);
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE,
          bool TA = false, bool TB = false>
__aicore__ __attribute__((always_inline)) void
mmamx_tile_core(memref_t<__cc__ DST_TYPE, 4> *mc,
                memref_t<__cbuf__ SRC_TYPE, 4> *ma,
                memref_t<__cbuf__ SRC_TYPE, 4> *mb,
                memref_t<__cbuf__ ElementMxScaleA, 1> *l1MxScaleA,
                memref_t<__cbuf__ ElementMxScaleB, 1> *l1MxScaleB, bool init,
                int64_t m, int64_t k, int64_t n,
                int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1scalea_event,
                int64_t mmad_l1_wait_l1b_event, int64_t mmad_l1_wait_l1scaleb_event,
                int64_t l1a_wait_mmad_l1_event, int64_t l1scalea_wait_mmad_l1_event,
                int64_t l1b_wait_mmad_l1_event, int64_t l1scaleb_wait_mmad_l1_event,
                Catlass::Gemm::HIVMMatmulDataformat lhsFormat,
                Catlass::Gemm::HIVMMatmulDataformat rhsFormat) {
  Catlass::Gemm::L1MxMmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, TA, TB,
                          false>(
      mc->aligned + mc->offset, ma->aligned + ma->offset,
      mb->aligned + mb->offset, l1MxScaleA->aligned + l1MxScaleA->offset,
      l1MxScaleB->aligned + l1MxScaleB->offset, nullptr,
      (TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2])),
      (TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3])),
      (TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3])),
      m, k, n,
      mmad_l1_wait_l1a_event, mmad_l1_wait_l1scalea_event,
      mmad_l1_wait_l1b_event, mmad_l1_wait_l1scaleb_event,
      l1a_wait_mmad_l1_event, l1scalea_wait_mmad_l1_event,
      l1b_wait_mmad_l1_event, l1scaleb_wait_mmad_l1_event,
      init, true, false, lhsFormat, rhsFormat, false);
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE,
          bool TA = false, bool TB = false>
__aicore__ __attribute__((always_inline)) void
mmamx_tile_bias(memref_t<__cc__ DST_TYPE, 4> *mc,
                memref_t<__cbuf__ SRC_TYPE, 4> *ma,
                memref_t<__cbuf__ SRC_TYPE, 4> *mb,
                memref_t<__cbuf__ ElementMxScaleA, 1> *l1MxScaleA,
                memref_t<__cbuf__ ElementMxScaleB, 1> *l1MxScaleB, bool init,
                int64_t m, int64_t k, int64_t n,
                memref_t<__cbuf__ BIAS_TYPE, 4> *bias,
                int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1scalea_event,
                int64_t mmad_l1_wait_l1b_event, int64_t mmad_l1_wait_l1scaleb_event,
                int64_t l1a_wait_mmad_l1_event, int64_t l1scalea_wait_mmad_l1_event,
                int64_t l1b_wait_mmad_l1_event, int64_t l1scaleb_wait_mmad_l1_event,
                Catlass::Gemm::HIVMMatmulDataformat lhsFormat,
                Catlass::Gemm::HIVMMatmulDataformat rhsFormat) {
  Catlass::Gemm::L1MxMmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, TA, TB,
                          false>(
      mc->aligned + mc->offset, ma->aligned + ma->offset,
      mb->aligned + mb->offset, l1MxScaleA->aligned + l1MxScaleA->offset,
      l1MxScaleB->aligned + l1MxScaleB->offset, bias->aligned + bias->offset,
      (TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2])),
      (TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3])),
      (TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3])),
      m, k, n, mmad_l1_wait_l1a_event, mmad_l1_wait_l1scalea_event,
      mmad_l1_wait_l1b_event, mmad_l1_wait_l1scaleb_event,
      l1a_wait_mmad_l1_event, l1scalea_wait_mmad_l1_event,
      l1b_wait_mmad_l1_event, l1scaleb_wait_mmad_l1_event, init, true, false,
      lhsFormat, rhsFormat, true);
}

#endif // CATLASS_GEMM_L1MMAD_HPP

extern "C" {
REGISTER_MMA_MX(float8_e5m2_t, float, float);
REGISTER_MMA_MX(float8_e4m3_t, float, float);
REGISTER_MMA_MX_BIAS(float8_e5m2_t, float, float);
REGISTER_MMA_MX_BIAS(float8_e4m3_t, float, float);
REGISTER_MMA_MX_TRANS(float8_e5m2_t, float, float, _ta, true, false);
REGISTER_MMA_MX_TRANS(float8_e5m2_t, float, float, _tb, false, true);
REGISTER_MMA_MX_TRANS(float8_e5m2_t, float, float, _ta_tb, true, true);
REGISTER_MMA_MX_TRANS(float8_e4m3_t, float, float, _ta, true, false);
REGISTER_MMA_MX_TRANS(float8_e4m3_t, float, float, _tb, false, true);
REGISTER_MMA_MX_TRANS(float8_e4m3_t, float, float, _ta_tb, true, true);
REGISTER_MMA_MX_BIAS_TRANS(float8_e5m2_t, float, float, _ta, true, false);
REGISTER_MMA_MX_BIAS_TRANS(float8_e5m2_t, float, float, _tb, false, true);
REGISTER_MMA_MX_BIAS_TRANS(float8_e5m2_t, float, float, _ta_tb, true, true);
REGISTER_MMA_MX_BIAS_TRANS(float8_e4m3_t, float, float, _ta, true, false);
REGISTER_MMA_MX_BIAS_TRANS(float8_e4m3_t, float, float, _tb, false, true);
REGISTER_MMA_MX_BIAS_TRANS(float8_e4m3_t, float, float, _ta_tb, true, true);
REGISTER_MMA_MX_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t, _ta,
                             true, false,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t, _tb,
                             false, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                             _ta_tb, true, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                                  _ta, true, false,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                                  _tb, false, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e5m2_t,
                                  _ta_tb, true, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t, _ta,
                             true, false,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t, _tb,
                             false, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                             _ta_tb, true, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                                  _ta, true, false,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                                  _tb, false, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e5m2_t, fp8_e4m3_t,
                                  _ta_tb, true, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t, _ta,
                             true, false,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t, _tb,
                             false, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                             _ta_tb, true, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                                  _ta, true, false,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                                  _tb, false, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e5m2_t,
                                  _ta_tb, true, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E5M2_T);
REGISTER_MMA_MX_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                       Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                            Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t, _ta,
                             true, false,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t, _tb,
                             false, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                             _ta_tb, true, true,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                             Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                                  _ta, true, false,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                                  _tb, false, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp8_e4m3_t, fp8_e4m3_t,
                                  _ta_tb, true, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP8E4M3_T);
REGISTER_MMA_MX_FP4(int8_t, float, float, fp4x2_e2m1_t, fp4x2_e2m1_t);
REGISTER_MMA_MX_BIAS_FORMAT(int8_t, float, float, fp4x2_e2m1_t, fp4x2_e2m1_t,
                            Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T,
                            Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T);
REGISTER_MMA_MX_FP4_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                          fp4x2_e2m1_t, _ta, true, false);
REGISTER_MMA_MX_FP4_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                          fp4x2_e2m1_t, _tb, false, true);
REGISTER_MMA_MX_FP4_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                          fp4x2_e2m1_t, _ta_tb, true, true);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                                  fp4x2_e2m1_t, _ta, true, false,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                                  fp4x2_e2m1_t, _tb, false, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T);
REGISTER_MMA_MX_BIAS_FORMAT_TRANS(int8_t, float, float, fp4x2_e2m1_t,
                                  fp4x2_e2m1_t, _ta_tb, true, true,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T,
                                  Catlass::Gemm::HIVMMatmulDataformat::FP4E2M1_T);
}
