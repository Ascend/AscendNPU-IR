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

#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_BT_A5_HPP
#define CATLASS_GEMM_TILE_COPY_L1_TO_BT_A5_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyL1ToBT, AtlasA5, VectorLayout in and VectorLayout out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::A1>,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::C2>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isVector<typename TensorSrc::Layout>::value
                && tla::detail::isVector<typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::C2,
            "The input parameters do not match. TensorSrc must be L1 and Vector, "
            "while TensorDst must be BT and Vector"
        );

        AscendCBisheng::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.shape()), ELE_NUM_PER_C0);
        if (sizeof(ElementSrc) == 4) {
            // the burst length should be even when B32
            intriParams.blockLen = RoundUp(intriParams.blockLen, 2);
        }
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_BT_A5_HPP
