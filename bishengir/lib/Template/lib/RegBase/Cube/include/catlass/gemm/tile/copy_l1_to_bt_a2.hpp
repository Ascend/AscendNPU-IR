/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_BT_A2_HPP
#define CATLASS_GEMM_TILE_COPY_L1_TO_BT_A2_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

using namespace tla;

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToBT {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to biasTable buffer, can not find the specialization.");
};

template<class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToBT<ArchTag, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendCBisheng::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendCBisheng::TPosition::C2>>{
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C2 =  BYTE_PER_C2 / sizeof(ElementSrc);

    CATLASS_DEVICE
    CopyL1ToBT(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> dstTensor,
        AscendC::LocalTensor<ElementSrc> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendCBisheng::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = (layoutDst.shape(0) + ELE_NUM_PER_C2 - 1) / ELE_NUM_PER_C2;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendCBisheng::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for CopyL1ToBT, AtlasA2, VectorLayout in and VectorLayout out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::C2>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value &&
                     tla::detail::isVector<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C2 = BYTE_PER_C2 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isVector<typename TensorSrc::Layout>::value &&
                      tla::detail::isVector<typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendCBisheng::TPosition::A1 &&
                      TensorDst::position == AscendCBisheng::TPosition::C2,
            "The input parameters do not match. TensorSrc must be L1 and Vector, "
            "while TensorDst must be BT and Vector");

        AscendCBisheng::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.shape()), ELE_NUM_PER_C2);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_BT_A2_HPP
