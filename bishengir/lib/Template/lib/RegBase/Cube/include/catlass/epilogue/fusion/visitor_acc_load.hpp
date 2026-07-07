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
 
#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_ACC_LOAD_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_ACC_LOAD_HPP
 
#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "catlass/layout/layout.hpp"
 
namespace Catlass::Epilogue::Fusion {
 
template<class Element>
struct VisitorAccLoad : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;
 
    // 输出元素类型与输出阶段元信息
    using ElementOutput = Element;
 
    struct Arguments {};
 
    struct Params {};
 
    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const&, void*) {
        return Params();
    }
 
    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }
 
    template <class ProblemShape>
    static bool
    can_implement(ProblemShape const&, Arguments const&) {
        return true;
    }
 
    
    VisitorAccLoad() {}
 
    
    VisitorAccLoad(Params const&) {}
 
    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubAcc;
        Params const* params_ptr;
        uint32_t compute_length;
 
        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<Element> ubAcc_,
                 Params const* params_ptr_,
                 uint32_t compute_length_)
            : ubAcc(ubAcc_), params_ptr(params_ptr_), compute_length(compute_length_) {}
 
        template <class ArchTag, class TensorC, typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            VisitStage stage,
            Args const&... /*unused*/
        ) {
            if (stage == VisitStage::LOAD) {
                // 从tensor获取actualTileShape
                auto actualRows = tla::get<0>(tensorTile.shape());
                auto actualCols = tla::get<1>(tensorTile.shape());
                
                // 创建UB tensor（使用对齐后的形状）
                auto layoutUb = tla::MakeLayout(
                    tla::MakeShape(actualRows, actualCols),
                    tla::MakeStride(alignedTileShape.column(), tla::Int<1>{})
                );
                auto tensorUb = tla::MakeTensor(ubAcc, layoutUb, Arch::PositionUB{});
                
                using CopyGm2UbTlaT = Epilogue::Tile::CopyGm2UbTla<ArchTag, TensorC, decltype(tensorUb)>;
                CopyGm2UbTlaT copyGm2UbTla{};
                copyGm2UbTla(tensorUb, tensorTile);
            }
            return ubAcc;
        }
    };
 
    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        auto ubAcc = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        assert(ub_offset <= ArchTag::UB_SIZE, "ub_offset exceeds ArchTag::UB_SIZE");
        return Callbacks(ubAcc, &params, compute_length);
    }
 
    Params params;
};
 
} // namespace Catlass::Epilogue::Fusion
 
#endif