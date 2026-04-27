/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B + Bias
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTla {
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrBias_ = nullptr)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrBias(ptrBias_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrC; LayoutC layoutC;
        uint8_t *ptrBias{nullptr};
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.problemShape,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrC, args.layoutC,
            args.ptrBias};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    BasicMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        auto layoutBias = tla::MakeLayout(params.problemShape.n());

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
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
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_TLA_HPP