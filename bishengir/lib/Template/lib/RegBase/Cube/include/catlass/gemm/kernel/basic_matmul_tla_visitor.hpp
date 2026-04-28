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

#ifndef CATLASS_GEMM_KERNEL_BASIC_MATMUL_TLA_VISITOR_HPP
#define CATLASS_GEMM_KERNEL_BASIC_MATMUL_TLA_VISITOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTlaVisitor {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrBias;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        Params() {}

        Params(
            GemmCoord const& problemShape_,
            GM_ADDR ptrA_, LayoutA const& layoutA_,
            GM_ADDR ptrB_, LayoutB const& layoutB_,
            GM_ADDR ptrBias_,
            GM_ADDR ptrWorkspace_, EpilogueParams const& epilogueParams_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
            ptrBias(ptrBias_), ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA; LayoutA layoutA;
        GM_ADDR ptrB; LayoutB layoutB;
        GM_ADDR ptrC; LayoutC layoutC;
        GM_ADDR ptrBias{nullptr};
        typename BlockEpilogue::EVG::Arguments evg_args;
    };

    static bool CanImplement(const Arguments& args)
    {
        return BlockEpilogue::EVG::can_implement(args.problemShape, args.evg_args);
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return sizeof(ElementC) * args.problemShape.m() * args.problemShape.n() +
               BlockEpilogue::EVG::get_workspace_size(args.problemShape, args.evg_args);
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();

        uint8_t* evg_workspace = workspace + sizeof(ElementC) * m * n;
        BlockEpilogue::EVG::initialize_workspace(problemShape, args.evg_args, evg_workspace);

        // 转换 EVG Arguments 到 Params
        typename BlockEpilogue::EVG::Params fusion_params = 
            BlockEpilogue::EVG::to_underlying_arguments(
                problemShape, args.evg_args, 
                evg_workspace  // EVG workspace 在 GEMM workspace 之后
            );
        
        EpilogueParams epilogueParams{fusion_params};
        Params params{problemShape, args.ptrA, args.layoutA, args.ptrB, args.layoutB, args.ptrBias, workspace, epilogueParams};
        return params;
    }

    CATLASS_DEVICE
    BasicMatmulTlaVisitor() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const& params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);
        }

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(params.problemShape.m(), params.problemShape.n());
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});
        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Make tiled views
            auto tensorBlockA = GetTile(tensorA,
                                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(tensorB,
                                        tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(tensorC,
                                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            // Compute block-scoped matrix multiply-add
            if constexpr (std::is_void_v<ElementBias>) {
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            } else {
                auto tensorBlockBias = GetTile(
                    tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n())
                );
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias);
            }

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }
        
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm as TLA tensor
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        
        // Create TLA layout for workspace C
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(params.problemShape.m(), params.problemShape.n());
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        // For TLA tuple, construct GemmCoord manually
        GemmCoord blockShape(L1_TILE_M, L1_TILE_N, L1_TILE_K);
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            
            // 使用 GetTile 创建 tile 视图
            auto tensorBlockC = GetTile(tensorC,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAicFinishStore);
            blockEpilogue(blockShape, blockCoord, actualBlockShape, tensorBlockC);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_MATMUL_TLA_VISITOR_HPP

