/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_ELEMWISE_NO_SOURCE_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_ELEMWISE_NO_SOURCE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Epilogue::Block {
// 部分特化：当DispatchPolicy为EpilogueAtlasA2ElemWiseNoSource时的特化版本
template <
    class CType_, // fp32输入 Gemm::GemmType
    class DType_, // half/bf16输出
    class TileElemWiseEpilogue_,
    // TileCopy的方法
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2ElemWiseNoSource,
    CType_,
    DType_,
    TileElemWiseEpilogue_,
    TileCopy_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2ElemWiseNoSource;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementX = typename CType_::Element; // X是fp32的计算结果，无GM
    using LayoutX = typename CType_::Layout;

    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogue = TileElemWiseEpilogue_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogue::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    using ElementCompute = ElementC;
    using ElementOut = ElementD;

    using LayoutComputeInUb = layout::RowMajor;

    // Check the element type of C and D
    static_assert(std::is_same_v<ElementC, float>,
        "Element type of C must be float");
    // Check the layout type of C and D
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutD, layout::RowMajor>,
        "Layout type of C, D must be RowMajor");

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogue::ArchTag, ArchTag>,
        "Tile epilogue's ArchTag mismatch");
    // Check if compute length is valid
    static_assert(COMPUTE_LENGTH * (OPERANDS_NUM * sizeof(ElementC) + sizeof(ElementD)) <= ArchTag::UB_SIZE,
        "UB out of bounds");

    // Epilogue params definition
    struct Params {
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrD;
        LayoutD layoutD;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrC_, LayoutD const &layoutC_, GM_ADDR ptrD_, LayoutD const &layoutD_)
            : ptrC(ptrC_), layoutC(layoutC_), ptrD(ptrD_), layoutD(layoutD_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, Params const &params) : params(params)
    {
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        ubX = resource.ubBuf.template GetBufferByByte<ElementX>(COMPUTE_LENGTH * sizeof(ElementC));
        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(COMPUTE_LENGTH * sizeof(ElementC) * 2);

        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE2>(EVENT_ID0);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::V_MTE2>(EVENT_ID0);
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator() (
        GemmCoord const &blockShapeMNK,
        GemmCoord const &blockCoordMNK,
        GemmCoord const &actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementCompute> const &gmBlockC,
        LayoutD const &layoutBlockC
    )
    {
        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        // Calculate the offset and the shape of the current subblock
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };

        MatrixCoord subblockCoord{ AscendC::GetSubBlockIdx(), 0 };
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // Get the data and layout of C
        auto gmSubblockC = gmBlockC[layoutBlockC.GetOffset(subblockOffset)];
        auto layoutSubblockC = layoutBlockC.GetTileLayout(actualSubblockShape);

        // Get the data and layout of D
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrD));
        auto gmSubblockD = gmD[params.layoutD.GetOffset(blockOffset + subblockOffset)];
        auto layoutSubblockD = params.layoutD.GetTileLayout(actualSubblockShape);

        // Get the layout on UB
        auto ubTileStride = MakeCoord(static_cast<int64_t>(blockShape.column()), 1L);
        LayoutComputeInUb layoutComputeInUb{actualSubblockShape, ubTileStride};
        LayoutComputeInUb layoutComputeOutUb{actualSubblockShape, ubTileStride};
        // Copy the data of C
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::V_MTE2>(EVENT_ID0);
        copyGmToUbC(ubC, gmSubblockC, layoutComputeInUb, layoutSubblockC);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE2_V>(EVENT_ID0);

        // Perform epilogue calculation
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE2_V>(EVENT_ID0);
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::MTE3_V>(EVENT_ID0);
        tileEpilogue(ubX, ubC);
        AscendC::Cast(ubD, ubX, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE2>(EVENT_ID0);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE3>(EVENT_ID0);

        // Copy the data of D
        AscendCBisheng::WaitFlag<AscendCBisheng::HardEvent::V_MTE3>(EVENT_ID0);
        copyUbToGmD(gmSubblockD, ubD, layoutSubblockD, layoutComputeOutUb);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE3_V>(EVENT_ID0);
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubC;
    AscendC::LocalTensor<ElementC> ubX;
    AscendC::LocalTensor<ElementD> ubD;

    TileElemWiseEpilogue tileEpilogue;
    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_ELEMWISE_NO_SOURCE_HPP
