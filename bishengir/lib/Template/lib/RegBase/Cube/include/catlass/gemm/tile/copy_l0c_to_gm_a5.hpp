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

#ifndef CATLASS_GEMM_TILE_COPY_L0C_TO_GM_A5_HPP
#define CATLASS_GEMM_TILE_COPY_L0C_TO_GM_A5_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendCBisheng::TPosition::GM>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendCBisheng::TPosition::CO1
                && TensorDst::position == AscendCBisheng::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor"
        );

        // AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // // Fixpipe layout information
        // intriParams.nSize = tla::get<1>(dstTensor.shape());
        // intriParams.mSize = tla::get<0>(dstTensor.shape());
        // intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // // Fixpipe auxiliary arguments
        // intriParams.quantPre = quantPre;
        // intriParams.reluEn = reluEn;
        // intriParams.unitFlag = unitFlag;

        // auto dstOffset = dstTensor.layout()(dstTensor.coord());
        // auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // // Call AscendC Fixpipe
        // AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
        //     dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = true;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void
    operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch, uint32_t dstNdStride)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendCBisheng::TPosition::CO1
                && TensorDst::position == AscendCBisheng::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor"
        );

        const uint32_t L0CM = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L0CN = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        // AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // // Fixpipe layout information
        // intriParams.nSize = tla::get<1>(dstTensor.shape());
        // intriParams.mSize = tla::get<0>(dstTensor.shape());
        // intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // // Fixpipe auxiliary arguments
        // intriParams.quantPre = quantPre;
        // intriParams.reluEn = reluEn;
        // intriParams.unitFlag = 0;

        // intriParams.params.ndNum = l0Batch;
        // intriParams.params.srcNdStride = L0CM * L0CN / tla::get<1, 0>(srcTensor.shape());
        // intriParams.params.dstNdStride = dstNdStride;

        // // Call AscendC Fixpipe
        // AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
        //     dstTensor.data(), srcTensor.data(), intriParams);

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = true;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = 0;

        AscendC::SetFixpipeNz2ndFlag(l0Batch, L0CM * L0CN / tla::get<1, 0>(srcTensor.shape()), dstNdStride);
        AscendCBisheng::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendCBisheng::TPosition::GM>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
            && TensorSrc::position == AscendCBisheng::TPosition::CO1 && TensorDst::position == AscendCBisheng::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and zN"
        );

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        intriParams.mSize = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / (BYTE_PER_C0 / sizeof(ElementDst));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = false;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.channelSplit = true;
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendCBisheng::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L0C_TO_GM_A5_HPP
