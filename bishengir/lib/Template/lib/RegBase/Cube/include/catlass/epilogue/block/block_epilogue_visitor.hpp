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
 
#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP
 
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
 
namespace Catlass::Epilogue::Block {
 
template <
    class ArchTag_,
    class ComputeLength_,
    class EVG_,
    class ElementC_
>
class BlockEpilogue<
    EpilogueVisitor,
    ArchTag_,
    ComputeLength_,
    EVG_,
    ElementC_
> {
public:
    static constexpr uint32_t COMPUTE_LENGTH = ComputeLength_::value;
    using EVG = EVG_;
    using ArchTag = ArchTag_;
    using ElementC = ElementC_;
 
    struct Params {
        typename EVG::Params evg_params;
 
        Params() {}
 
        Params(typename EVG::Params const& evg_params_)
            : evg_params(evg_params_) {}
    };
 
    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
        : params(params), evg(params.evg_params), resource_(resource)
    {
        uint32_t ub_offset0 = 0;
        auto callbacks0 = evg.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        callbacks0.begin_epilogue();
        
        int32_t evVMTE2 = 0;   // V_MTE2
        int32_t evMTE2V = 0;   // MTE2_V
        int32_t evMTE3V = 0;   // MTE3_V
        int32_t evVMTE3 = 0;   // V_MTE3
 
        for (int i = 0; i < 2; ++i) {
            eventVMTE2[i] = evVMTE2++;
            eventMTE2V[i] = evMTE2V++;
            eventMTE3V[i] = evMTE3V++;
            eventVMTE3[i] = evVMTE3++;
        }
 
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE2>(eventVMTE2[0]);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE2>(eventVMTE2[1]);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE3_V>(eventMTE3V[0]);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE3_V>(eventMTE3V[1]);
    }
 
    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendCBisheng::HardEvent::V_MTE2>(eventVMTE2[0]);
        AscendC::WaitFlag<AscendCBisheng::HardEvent::V_MTE2>(eventVMTE2[1]);
        AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE3_V>(eventMTE3V[0]);
        AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE3_V>(eventMTE3V[1]);
 
        uint32_t ub_offset0 = 0;
        auto callbacks0 = evg.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        callbacks0.end_epilogue();
    }
 
    template <class TensorC>
    CATLASS_DEVICE
    void operator()(
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        GemmCoord const& actualBlockShapeMNK,
        TensorC const& tensorBlockC
    )
    {
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;
 
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;
 
        auto tensorSubblockC = GetTile(tensorBlockC,
            tla::MakeCoord(subblockOffset.row(), subblockOffset.column()),
            tla::MakeShape(actualSubblockShape.row(), actualSubblockShape.column()));
 
        // 分配 UB 空间并获取两套 callbacks（双缓冲）
        uint32_t ub_offset0 = 0;
        auto callbacks0 = evg.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        uint32_t ub_offset1 = ub_offset0;
        auto callbacks1 = evg.get_callbacks(
            resource_, ub_offset1, COMPUTE_LENGTH
        );
 
        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();
 
        // 遍历所有 tile，实现双缓冲流水
        uint32_t ubListId = 0;  // 0或1，交替使用
        
        for (uint32_t r = 0; r < rows; ) {
            auto& cbs = ((ubListId & 1) ? callbacks1 : callbacks0);
 
            // 检查是否需要列分块
            if (cols <= COMPUTE_LENGTH) {
                // 列宽 <= COMPUTE_LENGTH，可以处理完整行宽，一次做多行
                uint32_t colsAligned = RoundUp<BYTE_PER_C0>(cols);
                uint32_t maxRowsPerTile = COMPUTE_LENGTH / colsAligned;
                if (maxRowsPerTile == 0) maxRowsPerTile = 1;  // 防止除零
                
                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < maxRowsPerTile) ? remainRows : maxRowsPerTile;
                
                MatrixCoord tileShape{tileRows, cols};
                MatrixCoord localTileOffset{r, 0};
                
                // 计算对齐的 tile shape
                MatrixCoord alignedTileShape{
                    tileShape.row(),
                    colsAligned
                };
                
                // 统一流水：执行一次 tile 的 Load-Compute-Store
                // 从tensorSubblockC获取信息，只传递必要参数
                run_tile(cbs, tensorSubblockC, localTileOffset, tileShape, alignedTileShape, ubListId);
                r += tileRows;
            } else { 
                // 列宽 > COMPUTE_LENGTH，需要列分块，每次处理1行
                for (uint32_t c = 0; c < cols; ) {
                    uint32_t remainCols = cols - c;
                    uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                    
                    uint32_t colsAligned = RoundUp<BYTE_PER_C0>(tileCols);
 
                    MatrixCoord tileShape{1, tileCols};
                    MatrixCoord localTileOffset{r, c};
                    
                    // 计算对齐的 tile shape
                    MatrixCoord alignedTileShape{
                        tileShape.row(),
                        colsAligned
                    };
                    
                    // 统一流水：执行一次 tile 的 Load-Compute-Store
                    // 从tensorSubblockC获取信息，只传递必要参数
                    run_tile(cbs, tensorSubblockC, localTileOffset, tileShape, alignedTileShape, ubListId);
                    c += tileCols;
                }
                
                r += 1;  // 处理完一行
            }
 
            ubListId = 1 - ubListId; // Buffer 轮转
        }
    }
 
    private:
    template <class Callbacks, class TensorC>
    CATLASS_DEVICE void run_tile(
        Callbacks& cbs,
        TensorC const& tensorSubblockC,
        MatrixCoord const& localTileOffset,
        MatrixCoord const& actualTileShape,
        MatrixCoord const& alignedTileShape,
        uint32_t ubListId
    ) {
        // 创建acc tile tensor
        auto tensorTile = GetTile(tensorSubblockC,
            tla::MakeCoord(localTileOffset.row(), localTileOffset.column()),
            tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
        
        AscendC::WaitFlag<AscendCBisheng::HardEvent::V_MTE2>(eventVMTE2[ubListId]);
        cbs.template visit<ArchTag>(tensorTile, alignedTileShape, Epilogue::Fusion::VisitStage::LOAD);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE2_V>(eventMTE2V[ubListId]);
 
        AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE2_V>(eventMTE2V[ubListId]);
        AscendC::WaitFlag<AscendCBisheng::HardEvent::MTE3_V>(eventMTE3V[ubListId]);
        cbs.template visit<ArchTag>(tensorTile, alignedTileShape, Epilogue::Fusion::VisitStage::COMPUTE);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE2>(eventVMTE2[ubListId]);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::V_MTE3>(eventVMTE3[ubListId]);
 
        AscendC::WaitFlag<AscendCBisheng::HardEvent::V_MTE3>(eventVMTE3[ubListId]);
        cbs.template visit<ArchTag>(tensorTile, alignedTileShape, Epilogue::Fusion::VisitStage::STORE);
        AscendCBisheng::SetFlag<AscendCBisheng::HardEvent::MTE3_V>(eventMTE3V[ubListId]);
    }
 
    Params params;
    EVG evg;
    Arch::Resource<ArchTag>& resource_;
    int32_t eventVMTE2[2];
    int32_t eventMTE2V[2];
    int32_t eventMTE3V[2];
    int32_t eventVMTE3[2];
};
 
} // namespace Catlass::Epilogue::Block
 
#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP