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

#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_L0B_A5_HPP
#define CATLASS_GEMM_TILE_COPY_L1_TO_L0B_A5_HPP

#include "ascendc_bisheng/utils/std/is_one_of.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"
#include <bits/stdint-uintn.h>

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyL1ToL0B, AtlasA5, not B8, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::A1>,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::B2>,
    std::enable_if_t<
        !AscendCBisheng::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        !AscendCBisheng::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            !AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            !AscendCBisheng::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ"
        );

        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendCBisheng::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = true;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendCBisheng::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch)
    {
        static_assert(
            !AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            !AscendCBisheng::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ"
        );

        const uint32_t L1K = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L1N = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendCBisheng::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = true;

        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendCBisheng::LoadData(
                dstTensor.data()[l0BatchIdx * L0N * L0K], srcTensor.data()[l0BatchIdx * L1N * L1K], loadDataParams
            );
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA5, B8, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::A1>,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::B2>,
    std::enable_if_t<
        AscendCBisheng::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        AscendCBisheng::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t> &&
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            AscendCBisheng::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ"
        );

        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendCBisheng::LoadData2DParamsV2 loadDataParams;
        if (L0N % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
            loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
            loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
            loadDataParams.ifTranspose = true;

            auto dstOffset = dstTensor.layout()(dstTensor.coord());
            AscendCBisheng::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
        } else {
            for (uint32_t kIdx = 0; kIdx < L0K / ELE_NUM_PER_C0; kIdx++) {
                loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord)) + kIdx * 2;
                loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
                loadDataParams.mStep = 2;
                loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
                loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
                loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
                loadDataParams.ifTranspose = true;

                auto dstOffset = dstTensor.layout()(dstTensor.coord());
                AscendCBisheng::LoadData(
                    dstTensor.data()[dstOffset + kIdx * L0N * ELE_NUM_PER_C0], srcTensor.data(), loadDataParams
                );
            }
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch)
    {
        static_assert(
            AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            AscendCBisheng::Std::is_one_of_v<typename TensorDst::Element, int8_t, float8_e4m3_t, float8_e5m2_t> &&
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ"
        );

        const uint32_t L1K = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L1N = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t L0K = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t L0N = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendCBisheng::LoadData2DParamsV2 loadDataParams;
        if (L0N % ELE_NUM_PER_C0 == 0) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
            loadDataParams.ifTranspose = true;

            for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
                AscendCBisheng::LoadData(
                    dstTensor.data()[l0BatchIdx * L0N * L0K], srcTensor.data()[l0BatchIdx * L1N * L1K], loadDataParams
                );
            }
        } else {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = 2;
            loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
            loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
            loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
            loadDataParams.ifTranspose = true;
            for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
                for (uint32_t kIdx = 0; kIdx < L0K / ELE_NUM_PER_C0; kIdx++) {
                    AscendCBisheng::LoadData(
                        dstTensor.data()[l0BatchIdx * L0N * L0K + kIdx * L0N * ELE_NUM_PER_C0],
                        srcTensor.data()[l0BatchIdx * L1N * L1K + kIdx * ELE_NUM_PER_FRACTAL * 2], loadDataParams
                    );
                }
            }
        }
    }

    template <class TensorDst, class TensorSrc, class TensorMxScale>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor,
                                   TensorSrc const &srcTensor,
                                   TensorMxScale const &scaleTensor) {
      static_assert(
          AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element,
                                           float8_e4m3_t, float8_e5m2_t,
                                           float4_e2m1x2_t, float4_e1m2x2_t> &&
              AscendCBisheng::Std::is_one_of_v<
                  typename TensorDst::Element, AscendCBisheng::mx_fp8_e4m3_t,
                  AscendCBisheng::mx_fp8_e5m2_t, float4_e2m1x2_t,
                  float4_e1m2x2_t> &&
              std::is_same_v<typename TensorMxScale::Element,
                             AscendCBisheng::fp8_e8m0_t> &&
              tla::detail::iszN<typename TensorSrc::Element,
                                typename TensorSrc::Layout>::value &&
              tla::detail::isnZ<typename TensorDst::Element,
                                typename TensorDst::Layout>::value &&
              tla::detail::isMxScalenN<typename TensorMxScale::Element,
                                       typename TensorMxScale::Layout>::value &&
              TensorSrc::position == AscendCBisheng::TPosition::A1 &&
              TensorMxScale::position == AscendCBisheng::TPosition::A1 &&
              TensorDst::position == AscendCBisheng::TPosition::B2,
          "The input parameters do not match. TensorSrc must be L1 and zN, "
          "TensorMxScale must be L1 and nN, while "
          "TensorDst must be L0B and nZ");

      const uint32_t L0K =
          tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
      const uint32_t L0N =
          tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
      const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
      const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
      auto srcCoord = srcTensor.coord();
      auto dstOffset = dstTensor.layout()(dstTensor.coord());

      AscendCBisheng::LoadData2DParamsV2 loadDataParams;
      if (L0N % ELE_NUM_PER_C0 == 0) {
        loadDataParams.mStartPosition =
            CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord));
        loadDataParams.kStartPosition =
            CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(L0K);
        loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
        loadDataParams.srcStride =
            CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
        loadDataParams.dstStride =
            CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = true;

        if constexpr (AscendCBisheng::Std::is_one_of_v<
                          typename TensorSrc::Element, float8_e4m3_t,
                          float8_e5m2_t>) {
          load_cbuf_to_cb(
              reinterpret_cast<__cb__ typename TensorSrc::Element *>(
                  dstTensor.data()[dstOffset].GetPhyAddr()),
              (__cbuf__ typename TensorSrc::Element *)srcTensor.data()
                  .GetPhyAddr(),
              loadDataParams.mStartPosition, loadDataParams.kStartPosition,
              loadDataParams.mStep, loadDataParams.kStep,
              loadDataParams.srcStride, loadDataParams.dstStride, 1);
        } else {
          load_cbuf_to_cb_s4(
              (__cb__ typename TensorSrc::Element *)dstTensor.data()[dstOffset]
                  .GetPhyAddr(),
              (__cbuf__ typename TensorSrc::Element *)srcTensor.data()
                  .GetPhyAddr(),
              loadDataParams.mStartPosition, loadDataParams.kStartPosition,
              loadDataParams.mStep, loadDataParams.kStep,
              loadDataParams.srcStride, loadDataParams.dstStride, 1);
        }

        // AscendCBisheng::LoadData(
        //     dstTensor.data()[dstOffset].template ReinterpretCast<typename
        //     TensorSrc::Element>(), srcTensor.data(), loadDataParams
        // );
      } else {
        constexpr uint32_t mStep = ELE_NUM_PER_C0 / C0_NUM_PER_FRACTAL;
        for (uint32_t kIdx = 0; kIdx < L0K / ELE_NUM_PER_C0; kIdx++) {
          loadDataParams.mStartPosition =
              CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<0>(srcCoord)) + kIdx * mStep;
          loadDataParams.kStartPosition =
              CeilDiv<ELE_NUM_PER_C0>(tla::get<1>(srcCoord));
          loadDataParams.mStep = mStep;
          loadDataParams.kStep = CeilDiv<ELE_NUM_PER_C0>(L0N);
          loadDataParams.srcStride =
              CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol);
          loadDataParams.dstStride =
              CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
          loadDataParams.ifTranspose = true;

          if (AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element,
                                               float8_e4m3_t, float8_e5m2_t>) {
            load_cbuf_to_cb(
                reinterpret_cast<__cb__ typename TensorSrc::Element *>(
                    dstTensor.data()[dstOffset + kIdx * L0N * ELE_NUM_PER_C0]
                        .GetPhyAddr()),
                (__cbuf__ typename TensorSrc::Element *)srcTensor.data()
                    .GetPhyAddr(),
                loadDataParams.mStartPosition, loadDataParams.kStartPosition,
                loadDataParams.mStep, loadDataParams.kStep,
                loadDataParams.srcStride, loadDataParams.dstStride, 1);
          } else {
            load_cbuf_to_cb_s4(
                (__cb__ typename TensorSrc::Element *)dstTensor
                    .data()[dstOffset + kIdx * L0N * ELE_NUM_PER_C0]
                    .GetPhyAddr(),
                (__cbuf__ typename TensorSrc::Element *)srcTensor.data()
                    .GetPhyAddr(),
                loadDataParams.mStartPosition, loadDataParams.kStartPosition,
                loadDataParams.mStep, loadDataParams.kStep,
                loadDataParams.srcStride, loadDataParams.dstStride, 1);
          }

          // AscendCBisheng::LoadData(
          //     dstTensor.data()[dstOffset + kIdx * L0N * ELE_NUM_PER_C0]
          //         .template ReinterpretCast<typename TensorSrc::Element>(),
          //     srcTensor.data(), loadDataParams
          // );
        }
      }

      auto scaleCoord = scaleTensor.coord();
      AscendCBisheng::LoadData2DMxParams loadDataMxParams;
      loadDataMxParams.xStartPosition =
          CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(scaleCoord));
      loadDataMxParams.yStartPosition =
          CeilDiv<MX_SCALE_COPY_GROUP_NUM>(tla::get<0>(scaleCoord));
      loadDataMxParams.xStep = tla::get<1, 1>(scaleTensor.shape());
      loadDataMxParams.yStep = tla::get<0, 1>(scaleTensor.shape());
      loadDataMxParams.srcStride =
          CeilDiv<BYTE_PER_C0>(tla::get<1, 1>(scaleTensor.stride()));
      loadDataMxParams.dstStride = tla::get<0, 1>(scaleTensor.shape());

      uint64_t mxDstAddr =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
              (__cb__ typename TensorDst::Element *)dstTensor.data()[dstOffset]
                  .GetPhyAddr())) /
          16;
      load_cbuf_to_cb_mx(
          mxDstAddr,
          static_cast<__cbuf__ void *>(
              (__cbuf__ typename TensorMxScale::Element *)scaleTensor.data()
                  .GetPhyAddr()),
          loadDataMxParams.xStartPosition, loadDataMxParams.yStartPosition,
          loadDataMxParams.xStep, loadDataMxParams.yStep,
          loadDataMxParams.srcStride, loadDataMxParams.dstStride);
    }

    template <class TensorDst, class TensorMxScale>
    CATLASS_DEVICE void copyScaleTensor(TensorDst const &dstTensor,
                                        TensorMxScale const &scaleTensor) {
      auto scaleCoord = scaleTensor.coord();
      AscendCBisheng::LoadData2DMxParams loadDataMxParams;
      loadDataMxParams.xStartPosition =
          CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(scaleCoord));
      loadDataMxParams.yStartPosition =
          CeilDiv<MX_SCALE_COPY_GROUP_NUM>(tla::get<0>(scaleCoord));
      loadDataMxParams.xStep = tla::get<1, 1>(scaleTensor.shape());
      loadDataMxParams.yStep = tla::get<0, 1>(scaleTensor.shape());
      loadDataMxParams.srcStride =
          CeilDiv<BYTE_PER_C0>(tla::get<1, 1>(scaleTensor.stride()));
      loadDataMxParams.dstStride = tla::get<0, 1>(scaleTensor.shape());

      auto dstOffset = dstTensor.layout()(dstTensor.coord());
      uint64_t mxDstAddr =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
              (__cb__ typename TensorDst::Element *)dstTensor.data()[dstOffset]
                  .GetPhyAddr())) /
          16;
      load_cbuf_to_cb_mx(
          mxDstAddr,
          static_cast<__cbuf__ void *>(
              (__cbuf__ typename TensorMxScale::Element *)scaleTensor.data()
                  .GetPhyAddr()),
          loadDataMxParams.xStartPosition, loadDataMxParams.yStartPosition,
          loadDataMxParams.xStep, loadDataMxParams.yStep,
          loadDataMxParams.srcStride, loadDataMxParams.dstStride);
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA5, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendCBisheng::TPosition::A1>,
    tla::Tensor<AscendCBisheng::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendCBisheng::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0B and nZ"
        );

        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
        auto srcCoord = srcTensor.coord();

        AscendCBisheng::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(srcCoord));
        loadDataParams.kStartPosition = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcCoord));
        loadDataParams.mStep = dstOuterShapeCol;
        loadDataParams.kStep = dstOuterShapeRow;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = false;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        AscendCBisheng::LoadData(dstTensor.data()[dstOffset], srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendCBisheng::TPosition::A1 && TensorDst::position == AscendCBisheng::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0B and nZ"
        );

        const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendCBisheng::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = dstOuterShapeCol;
        loadDataParams.kStep = dstOuterShapeRow * l0Batch;
        loadDataParams.srcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
        loadDataParams.dstStride = CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
        loadDataParams.ifTranspose = false;

        AscendCBisheng::LoadData(dstTensor.data(), srcTensor.data(), loadDataParams);
    }

    template <class TensorDst, class TensorSrc, class TensorMxScale>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor,
                                   TensorSrc const &srcTensor,
                                   TensorMxScale const &scaleTensor) {
      static_assert(
          AscendCBisheng::Std::is_one_of_v<typename TensorSrc::Element,
                                           float8_e4m3_t, float8_e5m2_t,
                                           float4_e2m1x2_t, float4_e1m2x2_t> &&
              AscendCBisheng::Std::is_one_of_v<
                  typename TensorDst::Element, AscendCBisheng::mx_fp8_e4m3_t,
                  AscendCBisheng::mx_fp8_e5m2_t, float4_e2m1x2_t,
                  float4_e1m2x2_t> &&
              std::is_same_v<typename TensorMxScale::Element,
                             AscendCBisheng::fp8_e8m0_t> &&
              tla::detail::isnZ<typename TensorSrc::Element,
                                typename TensorSrc::Layout>::value &&
              tla::detail::isnZ<typename TensorDst::Element,
                                typename TensorDst::Layout>::value &&
              tla::detail::isMxScalenN<typename TensorMxScale::Element,
                                       typename TensorMxScale::Layout>::value &&
              TensorSrc::position == AscendCBisheng::TPosition::A1 &&
              TensorMxScale::position == AscendCBisheng::TPosition::A1 &&
              TensorDst::position == AscendCBisheng::TPosition::B2,
          "The input parameters do not match. TensorSrc must be L1 and nZ, "
          "TensorMxScale must be L1 and nN, while "
          "TensorDst must be L0B and nZ");

      const uint32_t dstOuterShapeRow = tla::get<0, 1>(dstTensor.shape());
      const uint32_t dstOuterShapeCol = tla::get<1, 1>(dstTensor.shape());
      const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
      const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());
      auto srcCoord = srcTensor.coord();

      AscendCBisheng::LoadData2DParamsV2 loadDataParams;
      loadDataParams.mStartPosition =
          CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(srcCoord));
      loadDataParams.kStartPosition =
          CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcCoord));
      loadDataParams.mStep = dstOuterShapeCol;
      loadDataParams.kStep = dstOuterShapeRow;
      loadDataParams.srcStride =
          CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideRow);
      loadDataParams.dstStride =
          CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow);
      loadDataParams.ifTranspose = false;

      auto scaleCoord = scaleTensor.coord();
      AscendCBisheng::LoadData2DMxParams loadDataMxParams;
      loadDataMxParams.xStartPosition =
          CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(scaleCoord));
      loadDataMxParams.yStartPosition =
          CeilDiv<MX_SCALE_COPY_GROUP_NUM>(tla::get<0>(scaleCoord));
      loadDataMxParams.xStep = tla::get<1, 1>(scaleTensor.shape());
      loadDataMxParams.yStep = tla::get<0, 1>(scaleTensor.shape());
      loadDataMxParams.srcStride =
          CeilDiv<BYTE_PER_C0>(tla::get<1, 1>(scaleTensor.stride()));
      loadDataMxParams.dstStride = tla::get<0, 1>(scaleTensor.shape());

      auto dstOffset = dstTensor.layout()(dstTensor.coord());
      AscendCBisheng::LoadData(dstTensor.data()[dstOffset], srcTensor.data(),
                               scaleTensor.data(), loadDataParams,
                               loadDataMxParams);
    }

    template <class TensorDst, class TensorMxScale>
    CATLASS_DEVICE void copyScaleTensor(TensorDst const &dstTensor,
                                        TensorMxScale const &scaleTensor) {
      auto scaleCoord = scaleTensor.coord();
      AscendCBisheng::LoadData2DMxParams loadDataMxParams;
      loadDataMxParams.xStartPosition =
          CeilDiv<C0_NUM_PER_FRACTAL>(tla::get<1>(scaleCoord));
      loadDataMxParams.yStartPosition =
          CeilDiv<MX_SCALE_COPY_GROUP_NUM>(tla::get<0>(scaleCoord));
      loadDataMxParams.xStep = tla::get<1, 1>(scaleTensor.shape());
      loadDataMxParams.yStep = tla::get<0, 1>(scaleTensor.shape());
      loadDataMxParams.srcStride =
          CeilDiv<BYTE_PER_C0>(tla::get<1, 1>(scaleTensor.stride()));
      loadDataMxParams.dstStride = tla::get<0, 1>(scaleTensor.shape());

      auto dstOffset = dstTensor.layout()(dstTensor.coord());
      uint64_t mxDstAddr =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
              (__cb__ typename TensorDst::Element *)dstTensor.data()[dstOffset]
                  .GetPhyAddr())) /
          16;
      load_cbuf_to_cb_mx(
          mxDstAddr,
          static_cast<__cbuf__ void *>(
              (__cbuf__ typename TensorMxScale::Element *)scaleTensor.data()
                  .GetPhyAddr()),
          loadDataMxParams.xStartPosition, loadDataMxParams.yStartPosition,
          loadDataMxParams.xStep, loadDataMxParams.yStep,
          loadDataMxParams.srcStride, loadDataMxParams.dstStride);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_L0B_A5_HPP
