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

#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_L1_A5_HPP
#define CATLASS_GEMM_TILE_COPY_GM_TO_L1_A5_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyGmToL1, AtlasA5, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0
    )
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value
                && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::GM && TensorDst::position == AscendCBisheng::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN"
        );

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::A1>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor
    )
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::GM && TensorDst::position == AscendCBisheng::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and zN, while TensorDst must be L1 and zN"
        );

        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());

        AscendCBisheng::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0
    )
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::GM && TensorDst::position == AscendCBisheng::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
            "while TensorDst must be L1 and nZ"
        );

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzC0Stride = dstOuterStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::A1>,
    std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor
    )
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::GM && TensorDst::position == AscendCBisheng::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and nZ, "
            "while TensorDst must be L1 and nZ"
        );

        const uint32_t blockCount = tla::get<0, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        AscendCBisheng::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, VectorLayout in and VectorLayout out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::A1>,
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
                && TensorSrc::position == AscendCBisheng::TPosition::GM && TensorDst::position == AscendCBisheng::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and Vector, "
            "while TensorDst must be L1 and Vector"
        );

        AscendCBisheng::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.shape()), ELE_NUM_PER_C0);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_GM_TO_L1_A5_HPP
