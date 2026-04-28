/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_MULTI_BATCH_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_MULTI_BATCH_TLA_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class ArchTag_,
    bool USE_HF32_MODE_,
    uint32_t L0C_STAGES_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadMultiBatch<ArchTag_, USE_HF32_MODE_, L0C_STAGES_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadMultiBatch<ArchTag_, USE_HF32_MODE_, L0C_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy_::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy_::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy_::LayoutC;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy_::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy_::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy_::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy_::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy_::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static_assert(
        tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!"
    );
    static_assert(
        tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!"
    );

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr bool USE_HF32_MODE = DispatchPolicy::USE_HF32_MODE;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // Check HF32_MODE
    static_assert(
        !USE_HF32_MODE || (USE_HF32_MODE && std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>),
        "HF32 MODE only supports in float!"
    );

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value, "LayoutC only support RowMajor yet!");

    // Check L1TileShape
    static_assert((L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N && L1_TILE_K == L0_TILE_K,
        "The situation where the basic blocks of L1 and L0 differ on the m and n and k axes is not supported yet"
    );

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L0A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL0A>(tla::Int<L0_TILE_M>{}, tla::Int<L0_TILE_K>{});
    static constexpr auto L0B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL0B>(tla::Int<L0_TILE_K>{}, tla::Int<L0_TILE_N>{});

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t maxL1Batch = 1, uint32_t l1BufAddrStart = 0)
    {
        // use HF32 when USE_HF32_MODE is true
        if constexpr (USE_HF32_MODE) {
            AscendCBisheng::SetHF32Mode(true);
        }
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + maxL1Batch * L1A_TILE_SIZE * STAGES;
        // Init L1 and L0A/L0B buffers with STAGES
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1AOffset + maxL1Batch * L1A_TILE_SIZE * i
            );
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BOffset + maxL1Batch * L1B_TILE_SIZE * i
            );
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(ArchTag::L0A_SIZE / STAGES * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(ArchTag::L0B_SIZE / STAGES * i);

            // Assign event ID for L1 and L0A/L0B stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0ABEventList[i] = i;

            // The event id that needs to be set before the loop
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0ABEventList[i]);
        }
        
        // init L0C buffers with L0C_STAGES
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(ArchTag::L0C_SIZE / L0C_STAGES * i);

            // Assign event ID for L0C stages
            l0CEventList[i] = i;

            // The event id that needs to be set before the loop for L0C
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        if constexpr (USE_HF32_MODE) {
            AscendCBisheng::SetHF32Mode(false);
        }
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0ABEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            AscendC::WaitFlag<AscendCBisheng::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(
        TensorA &tensorA,
        TensorB &tensorB,
        TensorC &tensorC,
        TensorA &nextTensorA,
        TensorB &nextTensorB,
        GemmCoord const &actualShape,
        uint32_t &l1Batch,
        uint32_t &nextL1Batch,
        uint32_t &l0Batch,
        bool isFirstBlock,
        bool hasNextBlock
    )
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToGm copyL0CToGm;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        auto layoutInL0C = tla::MakeLayoutL0C(mBlockActual, nBlockActual);

        if (isFirstBlock) {
            // load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListId], L1A_LAYOUT, Arch::PositionL1{});
            copyGmToL1A(tensorL1A, tensorA, l1Batch, mBlockActual * kBlockActual, L1_TILE_M * L1_TILE_K);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListId], L1B_LAYOUT, Arch::PositionL1{});
            copyGmToL1B(tensorL1B, tensorB, l1Batch, kBlockActual * nBlockActual, L1_TILE_K * L1_TILE_N);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
        }
        if (hasNextBlock) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            // load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListIdNext], L1A_LAYOUT, Arch::PositionL1{});
            copyGmToL1A(tensorL1A, nextTensorA, nextL1Batch, mBlockActual * kBlockActual, L1_TILE_M * L1_TILE_K);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListIdNext], L1B_LAYOUT, Arch::PositionL1{});
            copyGmToL1B(tensorL1B, nextTensorB, nextL1Batch, kBlockActual * nBlockActual, L1_TILE_K * L1_TILE_N);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
        }

        uint32_t l0BatchLoop = CeilDiv(l1Batch, l0Batch);
        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0BatchLoop; l0BatchIdx++) {
            uint32_t actualL0Batch = (l0BatchIdx == l0BatchLoop - 1) ? (l1Batch - l0BatchIdx * l0Batch) : l0Batch;
            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            auto tensorL1A = tla::MakeTensor(
                l1ATensor[l0BatchIdx * l0Batch * L1_TILE_M * L1_TILE_K], L1A_LAYOUT, Arch::PositionL1{}
            );
            auto tensorL1B = tla::MakeTensor(
                l1BTensor[l0BatchIdx * l0Batch * L1_TILE_K * L1_TILE_N], L1B_LAYOUT, Arch::PositionL1{}
            );

            // Locate the current tile on L0A
            auto tensorL0A = tla::MakeTensor(l0ATensorList[l0ABListId], L0A_LAYOUT, Arch::PositionL0A{});
            AscendC::WaitFlag<AscendCBisheng::HardEvent::M_MTE1>(l0ABEventList[l0ABListId]);
            if (l0BatchIdx == 0) {
                AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            }
            // Load current tile from L1 to L0A
            copyL1ToL0A(tensorL0A, tensorL1A, actualL0Batch);
            if (l0BatchIdx == l0BatchLoop - 1) {
                AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            }

            // Locate the current tile on L0B
            auto tensorL0B = tla::MakeTensor(l0BTensorList[l0ABListId], L0B_LAYOUT, Arch::PositionL0B{});
            if (l0BatchIdx == 0) {
                AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }
            // Load current tile from L1 to L0B
            copyL1ToL0B(tensorL0B, tensorL1B, actualL0Batch);
            if (l0BatchIdx == l0BatchLoop - 1) {
                AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            }

            // Notify to do mmad
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE1_M>(l0ABEventList[l0ABListId]);
            AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE1_M>(l0ABEventList[l0ABListId]);

            // Locate the current tile on L0C
            auto tensorL0C = tla::MakeTensor(l0CTensor[l0CListId], layoutInL0C, Arch::PositionL0C{});

            AscendC::WaitFlag<AscendCBisheng::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            // Perform calculation operations
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mBlockActual, nBlockActual, kBlockActual, actualL0Batch);
            // Notify to move the next L0 tile
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_MTE1>(l0ABEventList[l0ABListId]);

            // copy block out
            auto tensorTileC = tla::MakeTensor(
                tensorC.data()[l0BatchIdx * l0Batch * mBlockActual * nBlockActual], tensorC.layout(),
                Arch::PositionType<TensorC::position>{}
            );
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendCBisheng::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            copyL0CToGm(tensorTileC, tensorL0C, actualL0Batch, mBlockActual * nBlockActual);
            AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::FIX_M>(l0CEventList[l0CListId]);

            l0ABListId = (l0ABListId + 1 < STAGES) ? (l0ABListId + 1) : 0;
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        }
        l1ListId = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0C_STAGES];

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0ABEventList[STAGES];
    int32_t l0CEventList[L0C_STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0ABListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_MULTI_BATCH_TLA_HPP
