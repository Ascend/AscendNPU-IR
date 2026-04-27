/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_COPY_GM_TO_L1_HPP
#define CATLASS_CONV_TILE_COPY_GM_TO_L1_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Conv::Tile {

template <
    class ArchTag,
    /// ConvType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

/// Gm to L1A
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::NC1HWC0, AscendCBisheng::TPosition::GM>> {
    using LayoutDst = layout::NC1HWC0; // L1
    using LayoutSrc = layout::NC1HWC0; // GM

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()( // {Cin1, Hi, Wi, C0}
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        uint32_t cin1Actual = layoutSrc.shape(1);
        uint32_t hiActual = layoutSrc.shape(2);
        uint32_t wiActual = layoutSrc.shape(3);
        uint32_t strideCin1 = layoutSrc.stride(1); // Hi * Wi * ELE_NUM_PER_C0
        uint32_t strideHi = layoutSrc.stride(2); // Wi * ELE_NUM_PER_C0
        uint32_t Wi = strideHi / ELE_NUM_PER_C0;

        for (int cin1Idx = 0; cin1Idx < cin1Actual; cin1Idx++) {
            size_t gmOffset = cin1Idx * strideCin1;
            size_t l1Offset = cin1Idx * hiActual * wiActual * ELE_NUM_PER_C0;
            AscendCBisheng::DataCopy(
                dstTensor[l1Offset],
                srcTensor[gmOffset],
                {
                    static_cast<uint16_t>(hiActual), // blockCount 连续传输数据块个数
                    static_cast<uint16_t>(wiActual * ELE_NUM_PER_C0 * sizeof(Element) / 32), // blockLen 每个连续传输数据块长度 单位为datablock(32Bytes)
                    static_cast<uint16_t>((Wi - wiActual) * ELE_NUM_PER_C0 * sizeof(Element) / 32), // srcStride 相邻连续数据块的间隔 单位为datablock(32Bytes)
                    0 // dstStride 相邻连续数据块间的间隔 单位为datablock(32Bytes)
                }
            );
        }
    }
};

/// Gm to L1B
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::CI1KHKWCOCI0, AscendCBisheng::TPosition::GM>> {
    using LayoutDst = layout::CI1KHKWCOCI0; // L1
    using LayoutSrc = layout::CI1KHKWCOCI0; // GM

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()( // {Cin1, Kh, Kw, Cout, C0}
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        uint32_t cin1Actual = layoutSrc.shape(0);
        uint32_t KhKw = layoutSrc.shape(1) * layoutSrc.shape(2);
        uint32_t coutActual = layoutSrc.shape(3);
        uint32_t coutRound = layoutDst.shape(3);
        uint32_t strideKhKw = layoutSrc.stride(2); // Cout * ELE_NUM_PER_C0
        uint32_t Cout = strideKhKw / ELE_NUM_PER_C0;

        AscendCBisheng::DataCopy(
            dstTensor,
            srcTensor,
            AscendCBisheng::DataCopyParams(
                cin1Actual * KhKw, // blockCount 连续传输数据块个数
                coutActual * ELE_NUM_PER_C0 * sizeof(Element) / 32, // blockLen 每个连续传输数据块长度
                (Cout - coutActual) * ELE_NUM_PER_C0 * sizeof(Element) / 32, // 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为datablock(32Bytes)
                (coutRound - coutActual) * ELE_NUM_PER_C0 * sizeof(Element) / 32 // 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为datablock(32Bytes)
            )
        );
    }
};

} // namespace Catlass::Conv::Tile

#endif // CATLASS_CONV_TILE_COPY_GM_TO_L1_HPP
