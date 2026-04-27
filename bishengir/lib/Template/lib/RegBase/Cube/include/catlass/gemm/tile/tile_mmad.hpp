/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_MMAD_HPP
#define CATLASS_GEMM_TILE_TILE_MMAD_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm/helper.hpp"
#include "tla/tensor.hpp"
#include "ascendc_bisheng/utils/kernel_struct_mm.h"
#include "ascendc_bisheng/mmad_pipe_barrier.hpp"
namespace Catlass::Gemm::Tile {

///////////////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_
>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // Methods

    CATLASS_DEVICE
    TileMmad() {}

    CATLASS_DEVICE
    void operator()(AscendCBisheng::LocalTensor<ElementAccumulator> const &l0CTensor,
         AscendCBisheng::LocalTensor<ElementA> const &l0ATensor,
         AscendCBisheng::LocalTensor<ElementB> const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
#if defined(CATLASS_ARCH_A2_ENABLED)
        AscendCBisheng::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
        if constexpr (std::is_same_v<ElementA, float> &&
                      (std::is_same_v<typename AType_::Layout, layout::ColumnMajor> ||
                          std::is_same_v<typename AType_::Layout, layout::nZ>)) {
            mmadParams.kDirectionAlign = true;
        }

        AscendCBisheng::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendCBisheng::PipeBarrier<PIPE_M>();
        }
#endif
    }

    CATLASS_DEVICE
    void operator()(AscendCBisheng::LocalTensor<ElementAccumulator> const &l0CTensor,
         AscendCBisheng::LocalTensor<ElementA> const &l0ATensor,
         AscendCBisheng::LocalTensor<ElementB> const &l0BTensor,
         AscendCBisheng::LocalTensor<ElementAccumulator> const &l0BiasTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
#if defined(CATLASS_ARCH_A2_ENABLED)
        AscendCBisheng::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
        if constexpr (std::is_same_v<ElementA, float> &&
                      (std::is_same_v<typename AType_::Layout, layout::ColumnMajor> ||
                          std::is_same_v<typename AType_::Layout, layout::nZ>)) {
            mmadParams.kDirectionAlign = true;
        }

        AscendCBisheng::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      l0BiasTensor,
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendCBisheng::PipeBarrier<PIPE_M>();
        }
#endif
    }
};

///////////////////////////////////////////TileMmadTla/////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag,
    /// Element for A matrix operand
    class ElementA,
    /// LayoutTag for A matrix operand in L1
    class LayoutTagL1A
>
struct TileMmadTla {
    // Methods

    CATLASS_DEVICE
    TileMmadTla() {}

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE
    void operator()(TensorC const &l0CTensor,
         TensorA const &l0ATensor,
         TensorB const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendCBisheng::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
#if defined(CATLASS_ARCH_A2_ENABLED)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if defined(CATLASS_ARCH_A5_ENABLED)
        mmadParams.disableGemv = true;
#endif

        AscendCBisheng::Mmad(l0CTensor.data(),
                      l0ATensor.data(),
                      l0BTensor.data(),
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendCBisheng::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB, class TensorBias>
    CATLASS_DEVICE
    void operator()(TensorC const &l0CTensor,
         TensorA const &l0ATensor,
         TensorB const &l0BTensor,
         TensorBias const &l0BiasTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendCBisheng::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
#if defined(CATLASS_ARCH_A2_ENABLED)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if defined(CATLASS_ARCH_A5_ENABLED)
        mmadParams.disableGemv = true;
#endif

        AscendCBisheng::Mmad(l0CTensor.data(),
                      l0ATensor.data(),
                      l0BTensor.data(),
                      l0BiasTensor.data(),
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendCBisheng::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE
    void operator()(TensorC const &l0CTensor,
         TensorA const &l0ATensor,
         TensorB const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         uint32_t l0Batch)
    {
        const uint32_t L0AM = tla::get<0, 0>(l0ATensor.shape()) * tla::get<0, 1>(l0ATensor.shape());
        const uint32_t L0AK = tla::get<1, 0>(l0ATensor.shape()) * tla::get<1, 1>(l0ATensor.shape());
        const uint32_t L0BK = tla::get<0, 0>(l0BTensor.shape()) * tla::get<0, 1>(l0BTensor.shape());
        const uint32_t L0BN = tla::get<1, 0>(l0BTensor.shape()) * tla::get<1, 1>(l0BTensor.shape());
        const uint32_t L0CM = tla::get<0, 0>(l0CTensor.shape()) * tla::get<0, 1>(l0CTensor.shape());
        const uint32_t L0CN = tla::get<1, 0>(l0CTensor.shape()) * tla::get<1, 1>(l0CTensor.shape());

        AscendCBisheng::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
#if defined(CATLASS_ARCH_A5_ENABLED)
        mmadParams.disableGemv = true;
#endif

        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendCBisheng::Mmad(l0CTensor.data()[l0BatchIdx * L0CM * L0CN],
                l0ATensor.data()[l0BatchIdx * L0AM * L0AK],
                l0BTensor.data()[l0BatchIdx * L0BK * L0BN],
                mmadParams);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_MMAD_HPP