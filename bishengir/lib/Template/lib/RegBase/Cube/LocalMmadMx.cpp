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

CATLASS_DEVICE inline uint32_t pingpongIdToEventId(uint32_t pingPongId) {
  return 6 + (!pingPongId ? EVENT_ID0 : EVENT_ID1);
}

template <class ElementA, class ElementB, class ElementBias, class ElementACC,
          bool TA, bool TB, bool HF32>
CATLASS_DEVICE void
L1MxMmad(__cc__ ElementACC *l0C, __cbuf__ ElementA *l1A, __cbuf__ ElementB *l1B,
         __cbuf__ ElementMxScaleA *l1MxScaleA,
         __cbuf__ ElementMxScaleB *l1MxScaleB, __cbuf__ ElementBias *l1Bias,
         uint32_t l1M, uint32_t l1K, uint32_t l1N, uint32_t actualM,
         uint32_t actualK, uint32_t actualN, uint32_t l1AMTE2MTE1EventId,
         uint32_t l1AMTE1MTE2EventId, uint32_t l1BMTE2MTE1EventId,
         uint32_t l1BMTE1MTE2EventId, bool isL1FirstK, bool isL1LastK,
         bool enable_unit_flag) {
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

  uint32_t pingPongId = 1;
  // Wait for mmad finished
  AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(0));
  AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(1));

  for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
    uint32_t kL0Actual =
        (kL0Idx < kL0Loop - 1) ? l0K : (actualK - kL0Idx * l0K);
    // Get ping/pong id (0 or 1)
    uint32_t l0EventId = pingpongIdToEventId(pingPongId);

    // This synchronization can be optimize to use a more fine-grained control
    // of using 4 event-ids. Each for A, B, ScaleA and ScaleB.
    // Now it's a temporary solution to move wait of both A and B earlier.
    if (kL0Idx == 0 && l1AMTE2MTE1EventId != -1) {
      AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(
          l1AMTE2MTE1EventId);
    }
    if (kL0Idx == 0 && l1BMTE2MTE1EventId != -1) {
      AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(
          l1BMTE2MTE1EventId);
    }
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

    // Wait for mmad finished
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);

    // Load current tile from L1 to L0A
    copyL1ToL0A(tensorL0A, tensorTileL1A);
    copyL1ToL0A.copyScaleTensor(tensorL0A, tensorTileL1MxScaleA);
    if (kL0Idx == kL0Loop - 1 && l1AMTE1MTE2EventId != -1) {
      AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(
          l1AMTE1MTE2EventId);
    }

    // Locate the current tile on L0B
    auto l0BTile =
        l0BTensor[pingPongId * L0B_PINGPONG_BUF_SIZE / sizeof(ElementB)];
    auto layoutBInL0 =
        tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, actualN);
    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
    // Locate the current tile of matrix B on L1
    auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(kL0Idx * l0K, 0),
                                 tla::MakeShape(kL0Actual, actualN));
    // Locate the current tile of matrix mxScaleB on L1
    auto tensorTileL1MxScaleB = GetTile(
        tensorL1MxScaleB, tla::MakeCoord(kL0Idx * l0K / MX_SCALE_GROUP_NUM, 0),
        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), actualN));

    // Load current tile from L1 to L0B
    copyL1ToL0B(tensorL0B, tensorTileL1B);
    copyL1ToL0B.copyScaleTensor(tensorL0B, tensorTileL1MxScaleB);
    if (kL0Idx == kL0Loop - 1 && l1BMTE1MTE2EventId != -1) {
      AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(
          l1BMTE1MTE2EventId);
    }

    const bool initC = isL1FirstK && (kL0Idx == 0);

    // Notify to do mmad
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);

    INTRINSIC(mad_mx, (__cc__ ElementACC *)tensorL0C.data().GetPhyAddr(),
              (__ca__ ElementA *)tensorL0A.data().GetPhyAddr(),
              (__cb__ ElementB *)tensorL0B.data().GetPhyAddr(), actualM,
              kL0Actual, actualN,
              /* unitFlag = */ 0b00, true, false, initC);

    // Notify to move the next L0B tile
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);
  }
  // Wait for mmad finished
  AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(0));
  AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(1));

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
         uint32_t actualK, uint32_t actualN, uint32_t l1AMTE2MTE1EventId,
         uint32_t l1AMTE1MTE2EventId, uint32_t l1BMTE2MTE1EventId,
         uint32_t l1BMTE1MTE2EventId, bool isL1FirstK, bool isL1LastK,
         bool enable_unit_flag, bool is_fp4) {
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
      l1M * l1K * 2 / MX_SCALE_GROUP_NUM};
  AscendCBisheng::LocalTensor<ElementMxScaleB> l1MxScaleBTensor{
      AscendCBisheng::TPosition::A1,
      (uint32_t)reinterpret_cast<int64_t>(l1MxScaleB),
      l1K * l1N * 2 / MX_SCALE_GROUP_NUM};
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
          l1M, l1K * 2 / MX_SCALE_GROUP_NUM);
  auto tensorL1MxScaleA =
      tla::MakeTensor(l1MxScaleATensor, layoutMxScaleAInL1, Arch::PositionL1{});
  auto layoutMxScaleBInL1 =
      tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
          l1K * 2 / MX_SCALE_GROUP_NUM, l1N);
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

  actualK *= 2;
  uint32_t kL0Loop = CeilDiv(actualK, l0K);

  uint32_t pingPongId = 1;
  // Wait for mmad finished
  AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(0));
  AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(1));

  for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
    uint32_t kL0Actual =
        (kL0Idx < kL0Loop - 1) ? l0K : (actualK - kL0Idx * l0K);
    // Get ping/pong id (0 or 1)
    uint32_t l0EventId = pingpongIdToEventId(pingPongId);

    // This synchronization can be optimize to use a more fine-grained control
    // of using 4 event-ids. Each for A, B, ScaleA and ScaleB.
    // Now it's a temporary solution to move wait of both A and B earlier.
    if (kL0Idx == 0 && l1AMTE2MTE1EventId != -1) {
      AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(
          l1AMTE2MTE1EventId);
    }
    if (kL0Idx == 0 && l1BMTE2MTE1EventId != -1) {
      AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(
          l1BMTE2MTE1EventId);
    }
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

    // Wait for mmad finished
    AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);

    // Load current tile from L1 to L0A
    copyL1ToL0A(tensorL0A, tensorTileL1A);
    copyL1ToL0A.copyScaleTensor(tensorL0A, tensorTileL1MxScaleA);
    if (kL0Idx == kL0Loop - 1 && l1AMTE1MTE2EventId != -1) {
      AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(
          l1AMTE1MTE2EventId);
    }

    // Locate the current tile on L0B
    auto l0BTile =
        l0BTensor[pingPongId * L0B_PINGPONG_BUF_SIZE / sizeof(ElementB)];
    auto layoutBInL0 =
        tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, actualN);
    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
    // Locate the current tile of matrix B on L1
    auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(kL0Idx * l0K, 0),
                                 tla::MakeShape(kL0Actual, actualN));
    // Locate the current tile of matrix mxScaleB on L1
    auto tensorTileL1MxScaleB = GetTile(
        tensorL1MxScaleB, tla::MakeCoord(kL0Idx * l0K / MX_SCALE_GROUP_NUM, 0),
        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), actualN));

    // Load current tile from L1 to L0B
    copyL1ToL0B(tensorL0B, tensorTileL1B);
    copyL1ToL0B.copyScaleTensor(tensorL0B, tensorTileL1MxScaleB);
    if (kL0Idx == kL0Loop - 1 && l1BMTE1MTE2EventId != -1) {
      AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(
          l1BMTE1MTE2EventId);
    }

    bool initC = isL1FirstK && (kL0Idx == 0);

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

    // if (l1Bias != nullptr) {
    //   if (initC) {
    //     tileMmad(tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, actualM,
    //              actualN, kL0Actual, initC, unitFlag);
    //   } else {
    //     tileMmad(tensorL0C, tensorL0A, tensorL0B, actualM, actualN,
    //     kL0Actual,
    //              initC, unitFlag);
    //   }
    // } else {
    // tileMmad(tensorL0C, tensorL0A, tensorL0B, actualM, actualN, kL0Actual,
    //          initC, unitFlag);
    // }

    if (is_fp4) {
      INTRINSIC(mad_mx, (__cc__ ElementACC *)tensorL0C.data().GetPhyAddr(),
                (__ca__ fp4x2_e2m1_t *)tensorL0A.data().GetPhyAddr(),
                (__cb__ fp4x2_e2m1_t *)tensorL0B.data().GetPhyAddr(), actualM,
                kL0Actual, actualN, unitFlag, true, false, initC);
    } else {
      // TODO need support when b8 as input be it's actually fp8 in format
    }

    // Notify to move the next L0B tile
    AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);
  }
  // Wait for mmad finished
  AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(0));
  AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(
      pingpongIdToEventId(1));

  if constexpr (HF32) {
    AscendCBisheng::SetHF32Mode(false);
  }
}

} // namespace Catlass::Gemm

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE>
__aicore__ __attribute__((always_inline)) void mmamx_tile_core(
    memref_t<__cc__ DST_TYPE, 4> *mc, memref_t<__cbuf__ SRC_TYPE, 4> *ma,
    memref_t<__cbuf__ SRC_TYPE, 4> *mb,
    memref_t<__cbuf__ ElementMxScaleA, 1> *l1MxScaleA,
    memref_t<__cbuf__ ElementMxScaleB, 1> *l1MxScaleB, int64_t m, int64_t k,
    int64_t n, int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,
    int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event) {
  Catlass::Gemm::L1MxMmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, false, false,
                          false>(
      mc->aligned + mc->offset, ma->aligned + ma->offset,
      mb->aligned + mb->offset, l1MxScaleA->aligned + l1MxScaleA->offset,
      l1MxScaleB->aligned + l1MxScaleB->offset, nullptr,
      (ma->sizes[1] * ma->sizes[2]), (ma->sizes[0] * ma->sizes[3]),
      (mb->sizes[0] * mb->sizes[3]), (ma->sizes[1] * ma->sizes[2]), k, n,
      mmad_l1_wait_l1a_event, l1a_wait_mmad_l1_event, mmad_l1_wait_l1b_event,
      l1b_wait_mmad_l1_event, true, true, false);
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE>
__aicore__ __attribute__((always_inline)) void
mmamx_tile_core(memref_t<__cc__ DST_TYPE, 4> *mc,
                memref_t<__cbuf__ SRC_TYPE, 4> *ma,
                memref_t<__cbuf__ SRC_TYPE, 4> *mb,
                memref_t<__cbuf__ ElementMxScaleA, 1> *l1MxScaleA,
                memref_t<__cbuf__ ElementMxScaleB, 1> *l1MxScaleB, int64_t m,
                int64_t k, int64_t n, int64_t mmad_l1_wait_l1a_event,
                int64_t mmad_l1_wait_l1b_event, int64_t l1a_wait_mmad_l1_event,
                int64_t l1b_wait_mmad_l1_event, bool isFp4) {
  Catlass::Gemm::L1MxMmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, false, false,
                          false>(
      mc->aligned + mc->offset, ma->aligned + ma->offset,
      mb->aligned + mb->offset, l1MxScaleA->aligned + l1MxScaleA->offset,
      l1MxScaleB->aligned + l1MxScaleB->offset, nullptr,
      (ma->sizes[1] * ma->sizes[2]), (ma->sizes[0] * ma->sizes[3]),
      (mb->sizes[0] * mb->sizes[3]), (ma->sizes[1] * ma->sizes[2]), k, n,
      mmad_l1_wait_l1a_event, l1a_wait_mmad_l1_event, mmad_l1_wait_l1b_event,
      l1b_wait_mmad_l1_event, true, true, false, isFp4);
}

#endif // CATLASS_GEMM_L1MMAD_HPP

extern "C" {
REGISTER_MMA_MX(float8_e5m2_t, float, float);
REGISTER_MMA_MX(float8_e4m3_t, float, float);
REGISTER_MMA_MX_FP4(int8_t, float, float, fp4x2_e2m1_t, fp4x2_e2m1_t);
}
