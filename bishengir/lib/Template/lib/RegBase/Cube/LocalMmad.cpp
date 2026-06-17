/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
    #define CATLASS_ARCH_A5_ENABLED
#endif

#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt_a5.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a_a5.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b_a5.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"
#include "ascendc_bisheng/ascendc_bisheng.hpp"

template <typename T, size_t Dim> struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

namespace Catlass::Gemm {

template <bool Trans>
struct TransToTag {};

template <>
struct TransToTag<false> {
    using tag = layout::zN;
};

template <>
struct TransToTag<true> {
    using tag = layout::nZ;
};

__aicore__ inline uint32_t getL1MmadPingPongId() {
    static uint32_t pingPongId = 1;
    pingPongId = 1 - pingPongId;
    return pingPongId;
}

template <class ElementA, class ElementB, class ElementBias, class ElementACC, bool TA, bool TB, bool HF32>
CATLASS_DEVICE void L1Mmad(
    __cc__ ElementACC *l0C,
    __cbuf__ ElementA *l1A,
    __cbuf__ ElementB *l1B,
    __cbuf__ ElementBias *l1Bias,
    uint32_t l1M,
    uint32_t l1K,
    uint32_t l1N,
    uint32_t actualM,
    uint32_t actualK,
    uint32_t actualN,
    uint32_t l1AMTE2MTE1EventId,
    uint32_t l1AMTE1MTE2EventId,
    uint32_t l1BMTE2MTE1EventId,
    uint32_t l1BMTE1MTE2EventId,
    bool isL1FirstK,
    bool isL1LastK,
    bool enable_unit_flag,
    bool hasBias
)
{
    if constexpr (HF32) {
        AscendCBisheng::SetHF32Mode(true);
    }

    using ArchTag = Arch::AtlasA5;
    using LayoutTagL1A = typename TransToTag<TA>::tag;
    using LayoutTagL1B = typename TransToTag<TB>::tag;
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
        tla::Tensor<AscendCBisheng::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
    using TensorL1B =
        tla::Tensor<AscendCBisheng::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendCBisheng::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendCBisheng::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::B2>;
    using TensorL0C =
        tla::Tensor<AscendCBisheng::LocalTensor<ElementACC>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::CO1>;
    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1A>;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    AscendCBisheng::LocalTensor<ElementA> l1ATensor{
        AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1A), l1M * l1K};
    AscendCBisheng::LocalTensor<ElementB> l1BTensor{
        AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1B), l1K * l1N};
    AscendCBisheng::LocalTensor<ElementA> l0ATensor{AscendCBisheng::TPosition::A2, 0, ArchTag::L0A_SIZE};
    AscendCBisheng::LocalTensor<ElementB> l0BTensor{AscendCBisheng::TPosition::B2, 0, ArchTag::L0B_SIZE};
    AscendCBisheng::LocalTensor<ElementACC> bTTensor{AscendCBisheng::TPosition::C2, 0, ArchTag::BIAS_SIZE};
    AscendCBisheng::LocalTensor<ElementACC> l0CTensor{
        AscendCBisheng::TPosition::CO1, (uint32_t)reinterpret_cast<int64_t>(l0C), l1M * l1N};

    auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(l1M, l1K);
    auto tensorL1A = tla::MakeTensor(l1ATensor, layoutAInL1, Arch::PositionL1{});
    auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(l1K, l1N);
    auto tensorL1B = tla::MakeTensor(l1BTensor, layoutBInL1, Arch::PositionL1{});
    auto layoutInL0C = tla::MakeLayoutL0C(actualM, actualN);
    auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
    auto layoutBiasInBT = tla::MakeLayout(actualN);
    auto tensorL0Bias = tla::MakeTensor(bTTensor, layoutBiasInBT, Arch::PositionBias{});

    constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / 2;
    constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / 2;
    constexpr uint32_t L0A_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    constexpr uint32_t L0B_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementB);

    uint32_t l0K = RoundDown<C0_NUM_PER_FRACTAL>(
        min(L0A_PINGPONG_BUF_SIZE / sizeof(ElementA) / RoundUp<L1AAlignHelper::M_ALIGNED>(actualM) / L0A_ELE_NUM_PER_C0
                * L0A_ELE_NUM_PER_C0,
            L0B_PINGPONG_BUF_SIZE / sizeof(ElementB) / RoundUp<L1BAlignHelper::N_ALIGNED>(actualN) / L0B_ELE_NUM_PER_C0
                * L0B_ELE_NUM_PER_C0));

    uint32_t kL0Loop = CeilDiv(actualK, l0K);

    for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
        uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? l0K : (actualK - kL0Idx * l0K);

        // Get ping/pong id (0 or 1)
        uint32_t pingPongId = getL1MmadPingPongId();
        uint32_t l0EventId = !pingPongId ? EVENT_ID0 : EVENT_ID1;

        // Locate the current tile on L0A
        auto l0ATile = l0ATensor[pingPongId * L0A_PINGPONG_BUF_SIZE / sizeof(ElementA)];
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(actualM, kL0Actual);
        auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
        // Locate the current tile of matrix A on L1
        auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * l0K), tla::MakeShape(actualM, kL0Actual));

        // Wait for mmad finished
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);

        if (kL0Idx == 0 && l1AMTE2MTE1EventId != -1) {
            AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AMTE2MTE1EventId);
        }
        // Load current tile from L1 to L0A
        copyL1ToL0A(tensorL0A, tensorTileL1A);
        if (kL0Idx == kL0Loop - 1 && l1AMTE1MTE2EventId != -1) {
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AMTE1MTE2EventId);
        }

        // Locate the current tile on L0B
        auto l0BTile = l0BTensor[pingPongId * L0B_PINGPONG_BUF_SIZE / sizeof(ElementB)];
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, actualN);
        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
        // Locate the current tile of matrix B on L1
        auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(kL0Idx * l0K, 0), tla::MakeShape(kL0Actual, actualN));

        if (kL0Idx == 0 && l1BMTE2MTE1EventId != -1) {
            AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1BMTE2MTE1EventId);
        }
        // Load current tile from L1 to L0B
        copyL1ToL0B(tensorL0B, tensorTileL1B);
        if (kL0Idx == kL0Loop - 1 && l1BMTE1MTE2EventId != -1) {
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BMTE1MTE2EventId);
        }

        bool initC = isL1FirstK && (kL0Idx == 0);
        if constexpr (!std::is_void_v<ElementBias>) {
            if (hasBias && initC) {
                AscendCBisheng::LocalTensor<ElementBias> l1BiasTensor{
                    AscendCBisheng::TPosition::A1, (uint32_t)reinterpret_cast<int64_t>(l1Bias), actualN};
                auto layoutBiasInL1 = tla::MakeLayout(actualN);
                auto tensorL1Bias = tla::MakeTensor(l1BiasTensor, layoutBiasInL1, Arch::PositionL1{});
                // Load bias to l0 biastable
                using TensorL1Bias = tla::Tensor<
                    AscendCBisheng::LocalTensor<ElementBias>, detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
                    tla::Coord<tla::_0>, AscendCBisheng::TPosition::A1>;
                using TensorL0Bias = tla::Tensor<
                    AscendCBisheng::LocalTensor<ElementACC>, detail::TagToLayout_t<ElementACC, layout::VectorLayout>,
                    tla::Coord<tla::_0>, AscendCBisheng::TPosition::C2>;
                using CopyL1ToBT = Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>;
                CopyL1ToBT copyL1ToBT;
                copyL1ToBT(tensorL0Bias, tensorL1Bias);
            }
        }

        // Notify to do mmad
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE1_M>(EVENT_ID0);

        // If the unit flag is enabled, the unit flag is set according to the calculation progress
        uint8_t unitFlag = 0b00;
        if (enable_unit_flag) {
            if (isL1LastK && (kL0Idx == kL0Loop - 1)) {
                unitFlag = 0b11;
            } else {
                unitFlag = 0b10;
            }
        }

        if (hasBias) {
            if (initC) {
                tileMmad(tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, actualM, actualN, kL0Actual, initC, unitFlag);
            } else {
                tileMmad(tensorL0C, tensorL0A, tensorL0B, actualM, actualN, kL0Actual, initC, unitFlag);
            }
        } else {
            tileMmad(tensorL0C, tensorL0A, tensorL0B, actualM, actualN, kL0Actual, initC, unitFlag);
        }

        // Notify to move the next L0B tile
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0EventId);
    }

    if constexpr (HF32) {
        AscendCBisheng::SetHF32Mode(false);
    }
}

} // namespace Catlass::Gemm

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE, bool TA,
          bool TB, bool HF32>
__aicore__ __attribute__((always_inline)) void
mma_tile_core(memref_t<__cbuf__ SRC_TYPE, 4> *ma,
              memref_t<__cbuf__ SRC_TYPE, 4> *mb, bool init, int64_t m,
              int64_t k, int64_t n, memref_t<__cc__ DST_TYPE, 4> *mc,
              int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,
              int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,
              int64_t kloop_db_cond,
              int64_t back_pipe_m_pipe_mte1_db_event0,
              int64_t back_pipe_m_pipe_mte1_db_event1,
              uint8_t unit_flag) {
  Catlass::Gemm::L1Mmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, TA, TB, HF32>(
      mc->aligned + mc->offset, // l0C
      ma->aligned + ma->offset, // l1A
      mb->aligned + mb->offset, // l1B
      nullptr,                  // l1Bias
      TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2]), // l1M
      TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3]), // l1K
      TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3]), // l1N
      m,                                  // actualM
      k,                                  // actualK
      n,                                  // actualN
      mmad_l1_wait_l1a_event,             // l1AMTE2MTE1EventId
      l1a_wait_mmad_l1_event,             // l1AMTE1MTE2EventId
      mmad_l1_wait_l1b_event,             // l1BMTE2MTE1EventId
      l1b_wait_mmad_l1_event,             // l1BMTE1MTE2EventId
      init,                               // isL1FirstK
      true,                               // isL1LastK
      unit_flag != 0,                     // enable_unit_flag
      false                               // hasBias
  );
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE, bool TA,
          bool TB, bool HF32>
__aicore__ __attribute__((always_inline)) void
mma_tile_bias(memref_t<__cbuf__ SRC_TYPE, 4> *ma,
              memref_t<__cbuf__ SRC_TYPE, 4> *mb, bool init, int64_t m,
              int64_t k, int64_t n, memref_t<__cbuf__ BIAS_TYPE, 4> *bias,
              memref_t<__cc__ DST_TYPE, 4> *mc, int64_t mmad_l1_wait_l1a_event,
              int64_t mmad_l1_wait_l1b_event, int64_t l1a_wait_mmad_l1_event,
              int64_t l1b_wait_mmad_l1_event, int64_t kloop_db_cond,
              int64_t back_pipe_m_pipe_mte1_db_event0,
              int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag) {
  Catlass::Gemm::L1Mmad<SRC_TYPE, SRC_TYPE, BIAS_TYPE, DST_TYPE, TA, TB, HF32>(
      mc->aligned + mc->offset,     // l0C
      ma->aligned + ma->offset,     // l1A
      mb->aligned + mb->offset,     // l1B
      bias->aligned + bias->offset, // l1Bias
      TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2]), // l1M
      TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3]), // l1K
      TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3]), // l1N
      m,                                  // actualM
      k,                                  // actualK
      n,                                  // actualN
      mmad_l1_wait_l1a_event,             // l1AMTE2MTE1EventId
      l1a_wait_mmad_l1_event,             // l1AMTE1MTE2EventId
      mmad_l1_wait_l1b_event,             // l1BMTE2MTE1EventId
      l1b_wait_mmad_l1_event,             // l1BMTE1MTE2EventId
      init,                               // isL1FirstK
      true,                               // isL1LastK
      unit_flag != 0,                     // enable_unit_flag
      true                                // hasBias
  );
}

#define DECLARE_MMA_TILE(src_scope, dst_scope, dim, src_type, dst_type,        \
                         bias_type)                                            \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type(                        \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE(src_scope, dst_scope, dim, src_type, dst_type,       \
                          bias_type)                                           \
  DECLARE_MMA_TILE(src_scope, dst_scope, dim, src_type, dst_type, bias_type) { \
    mma_tile_core<src_type, dst_type, bias_type, false, false, false>(         \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_BIAS(src_scope, dst_scope, dim, src_type, dst_type,    \
                              bias_type)                                        \
  __aicore__ __attribute__((always_inline)) void                                \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                      \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,           \
          int64_t m, int64_t k, int64_t n,                                      \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                         \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,       \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,       \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,       \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS(src_scope, dst_scope, dim, src_type, dst_type,  \
                               bias_type)                                      \
  DECLARE_MMA_TILE_BIAS(src_scope, dst_scope, dim, src_type, dst_type,         \
                        bias_type) {                                           \
    mma_tile_bias<src_type, dst_type, bias_type, false, false, false>(         \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,          \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_BIAS_TA(src_scope, dst_scope, dim, src_type, dst_type, \
                                 bias_type)                                     \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_ta( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                      \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,           \
          int64_t m, int64_t k, int64_t n,                                      \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                         \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,       \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,       \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,       \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_TA(src_scope, dst_scope, dim, src_type, dst_type, \
                                  bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_TA(src_scope, dst_scope, dim, src_type, dst_type,        \
                           bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, true, false, false>(           \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,           \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                         \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                  \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,       \
        unit_flag);                                                             \
  }

#define DECLARE_MMA_TILE_BIAS_TB(src_scope, dst_scope, dim, src_type, dst_type, \
                                 bias_type)                                     \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_tb( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                      \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,           \
          int64_t m, int64_t k, int64_t n,                                      \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                         \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,       \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,       \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,       \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_TB(src_scope, dst_scope, dim, src_type, dst_type, \
                                  bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_TB(src_scope, dst_scope, dim, src_type, dst_type,        \
                           bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, false, true, false>(           \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,           \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                         \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                  \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,       \
        unit_flag);                                                             \
  }

#define DECLARE_MMA_TILE_BIAS_TA_TB(src_scope, dst_scope, dim, src_type, dst_type, \
                                    bias_type)                                    \
  __aicore__ __attribute__((always_inline)) void                                \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_ta_tb( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                       \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,            \
          int64_t m, int64_t k, int64_t n,                                       \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                      \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                          \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,        \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,        \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,        \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_TA_TB(src_scope, dst_scope, dim, src_type, dst_type, \
                                     bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_TA_TB(src_scope, dst_scope, dim, src_type, dst_type,        \
                              bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, true, true, false>(               \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,              \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                            \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                     \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,          \
        unit_flag);                                                                \
  }

#define DECLARE_MMA_TILE_BIAS_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                      bias_type)                                     \
  __aicore__ __attribute__((always_inline)) void                                    \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_hf32( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                           \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,                \
          int64_t m, int64_t k, int64_t n,                                           \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                          \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                              \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,            \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,            \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,            \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                       bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_HF32(src_scope, dst_scope, dim, src_type, dst_type,        \
                                bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, false, false, true>(                 \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                              \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                       \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,            \
        unit_flag);                                                                  \
  }

#define DECLARE_MMA_TILE_BIAS_TA_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                      bias_type)                                     \
  __aicore__ __attribute__((always_inline)) void                                    \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_ta_hf32( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                           \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,                \
          int64_t m, int64_t k, int64_t n,                                           \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                          \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                              \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,            \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,            \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,            \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_TA_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                       bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_TA_HF32(src_scope, dst_scope, dim, src_type, dst_type,        \
                                bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, true, false, true>(                 \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                              \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                       \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,            \
        unit_flag);                                                                  \
  }

#define DECLARE_MMA_TILE_BIAS_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                      bias_type)                                     \
  __aicore__ __attribute__((always_inline)) void                                    \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_tb_hf32( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                           \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,                \
          int64_t m, int64_t k, int64_t n,                                           \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                          \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                              \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,            \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,            \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,            \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                       bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type,        \
                                bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, false, true, true>(                 \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                              \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                       \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,            \
        unit_flag);                                                                  \
  }

#define DECLARE_MMA_TILE_BIAS_TA_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                         bias_type)                                     \
  __aicore__ __attribute__((always_inline)) void                                       \
      _mlir_ciface_mma_tile_with_##bias_type##_bias_##src_type##_to_##dst_type##_ta_tb_hf32( \
          memref_t<__##src_scope##__ src_type, dim> *src0,                              \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,                   \
          int64_t m, int64_t k, int64_t n,                                              \
          memref_t<__##src_scope##__ bias_type, dim> *bias,                             \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                                 \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,               \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,               \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,               \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_BIAS_TA_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type, \
                                          bias_type)                                     \
  DECLARE_MMA_TILE_BIAS_TA_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type,        \
                                   bias_type) {                                          \
    mma_tile_bias<src_type, dst_type, bias_type, true, true, true>(                     \
        src0, src1, init, m, k, n, bias, dst, mmad_l1_wait_l1a_event,                   \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                                 \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                          \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,               \
        unit_flag);                                                                     \
  }

#define DECLARE_MMA_TILE_TA(src_scope, dst_scope, dim, src_type, dst_type,     \
                            bias_type)                                         \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta(                   \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA(src_scope, dst_scope, dim, src_type, dst_type,    \
                             bias_type)                                        \
  DECLARE_MMA_TILE_TA(src_scope, dst_scope, dim, src_type, dst_type,           \
                      bias_type) {                                             \
    mma_tile_core<src_type, dst_type, bias_type, true, false, false>(          \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TB(src_scope, dst_scope, dim, src_type, dst_type,     \
                            bias_type)                                         \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_tb(                   \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TB(src_scope, dst_scope, dim, src_type, dst_type,    \
                             bias_type)                                        \
  DECLARE_MMA_TILE_TB(src_scope, dst_scope, dim, src_type, dst_type,           \
                      bias_type) {                                             \
    mma_tile_core<src_type, dst_type, bias_type, false, true, false>(          \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA_TB(src_scope, dst_scope, dim, src_type, dst_type,  \
                               bias_type)                                      \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta_tb(                \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA_TB(src_scope, dst_scope, dim, src_type, dst_type, \
                                bias_type)                                     \
  DECLARE_MMA_TILE_TA_TB(src_scope, dst_scope, dim, src_type, dst_type,        \
                         bias_type) {                                          \
    mma_tile_core<src_type, dst_type, bias_type, true, true, false>(           \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_HF32(src_scope, dst_scope, dim, src_type, dst_type,   \
                              bias_type)                                       \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_hf32(                 \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_HF32(src_scope, dst_scope, dim, src_type, dst_type,  \
                               bias_type)                                      \
  DECLARE_MMA_TILE_HF32(src_scope, dst_scope, dim, src_type, dst_type,         \
                        bias_type) {                                           \
    mma_tile_core<src_type, dst_type, bias_type, false, false, true>(          \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA_HF32(src_scope, dst_scope, dim, src_type,          \
                                 dst_type, bias_type)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta_hf32(              \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA_HF32(src_scope, dst_scope, dim, src_type,         \
                                  dst_type, bias_type)                         \
  DECLARE_MMA_TILE_TA_HF32(src_scope, dst_scope, dim, src_type, dst_type,      \
                           bias_type) {                                        \
    mma_tile_core<src_type, dst_type, bias_type, true, false, true>(           \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TB_HF32(src_scope, dst_scope, dim, src_type,          \
                                 dst_type, bias_type)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_tb_hf32(              \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TB_HF32(src_scope, dst_scope, dim, src_type,         \
                                  dst_type, bias_type)                         \
  DECLARE_MMA_TILE_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type,      \
                           bias_type) {                                        \
    mma_tile_core<src_type, dst_type, bias_type, false, true, true>(           \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

#define DECLARE_MMA_TILE_TA_TB_HF32(src_scope, dst_scope, dim, src_type,       \
                                    dst_type, bias_type)                       \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_mma_tile_##src_type##_to_##dst_type##_ta_tb_hf32(           \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          int64_t m, int64_t k, int64_t n,                                     \
          memref_t<__##dst_scope##__ dst_type, 4> *dst,                        \
          int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,      \
          int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,      \
          int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,      \
          int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag)

#define REGISTER_MMA_TILE_TA_TB_HF32(src_scope, dst_scope, dim, src_type,      \
                                     dst_type, bias_type)                      \
  DECLARE_MMA_TILE_TA_TB_HF32(src_scope, dst_scope, dim, src_type, dst_type,   \
                              bias_type) {                                     \
    mma_tile_core<src_type, dst_type, bias_type, true, true, true>(            \
        src0, src1, init, m, k, n, dst, mmad_l1_wait_l1a_event,                \
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event,                        \
        l1b_wait_mmad_l1_event, kloop_db_cond,                                 \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1,      \
        unit_flag);                                                            \
  }

extern "C" {
REGISTER_MMA_TILE(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TB_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_TB_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float8_e4m3_t, float, half);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float8_e5m2_t, float, half);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, half, float, half);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float, float, half);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, bfloat16_t, float, half);

REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, float8_e4m3_t, float, half);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, float8_e5m2_t, float, half);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, half, float, half);
REGISTER_MMA_TILE_BIAS_TA(cbuf, cc, 4, bfloat16_t, float, half);

REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, float8_e4m3_t, float, half);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, float8_e5m2_t, float, half);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, half, float, half);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, bfloat16_t, float, half);
REGISTER_MMA_TILE_BIAS_TB(cbuf, cc, 4, float, float, half);

REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, float8_e4m3_t, float, float);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, float8_e5m2_t, float, float);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, float8_e4m3_t, float, half);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, float8_e5m2_t, float, half);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, half, float, half);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, bfloat16_t, float, half);
REGISTER_MMA_TILE_BIAS_TA_TB(cbuf, cc, 4, float, float, half);

REGISTER_MMA_TILE_BIAS_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS_TA_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS_TB_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS_TA_TB_HF32(cbuf, cc, 4, float, float, float);

}
