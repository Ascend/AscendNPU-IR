/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_mm_base_impl.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BASE_IMPL_H
#define ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BASE_IMPL_H
// #include "kernel_tensor.h"
#include "ascendc_bisheng/local_tensor.hpp"

#include "./kernel_operator_mm_impl.h"
#include "./kernel_struct_mm.h"
namespace AscendCBisheng {

template <typename T, typename U, typename S>
__aicore__ inline void MmadImpl(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const MmadParams& mmadParams)
{
    MmadCal((__cc__ PrimT<T>*)dst.GetPhyAddr(), (__ca__ PrimT<U>*)fm.GetPhyAddr(),
        (__cb__ PrimT<S>*)filter.GetPhyAddr(), mmadParams);
}

template <typename T, typename U, typename S, typename V>
__aicore__ inline void MmadImpl(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const LocalTensor<V>& bias, const MmadParams& mmadParams)
{
    const Hardware biasScope = GetPhyType((TPosition)bias.GetPosition());
    bool cmatrixSource = false;
    if (biasScope == Hardware::BIAS) {
        cmatrixSource = true;
    } else if (biasScope == Hardware::L0C) {
        cmatrixSource = false;
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
            "Failed to check bias tensor position in Mmad, supported positions are CO1 or C2"); });
    }
    MmadCal((__cc__ PrimT<T>*)dst.GetPhyAddr(), (__ca__ PrimT<U>*)fm.GetPhyAddr(),
        (__cb__ PrimT<S>*)filter.GetPhyAddr(), (uint64_t)bias.GetPhyAddr(), mmadParams, cmatrixSource);
}

} // namespace AscendCBisheng
#endif // ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BASE_IMPL_H