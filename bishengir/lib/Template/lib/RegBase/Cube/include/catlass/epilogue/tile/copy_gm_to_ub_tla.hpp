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

#ifndef CATLASS_EPILOGUE_TILE_COPY_GM_TO_UB_TLA_HPP
#define CATLASS_EPILOGUE_TILE_COPY_GM_TO_UB_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Tile {

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct CopyGm2UbTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyGm2UbTla, can not find the specialization.");
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<Arch::AtlasA2,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    CopyGm2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                      tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendCBisheng::TPosition::GM &&
                      TensorDst::position == AscendCBisheng::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.shape()),
            tla::get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    };
};

/// Partial specialization for AtlasA5, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    CopyGm2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                      tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendCBisheng::TPosition::GM &&
                      TensorDst::position == AscendCBisheng::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.shape()),
            tla::get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    };
};

}  // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_COPY_GM_TO_UB_TLA_HPP
