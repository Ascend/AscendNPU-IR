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
 * \file kernel_operator_mm_intf.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_MMAD_INTF_HPP
#define ASCENDC_BISHENG_MMAD_INTF_HPP
#include "local_tensor.hpp"

#include "utils/kernel_operator_mm_bitmode_intf.h"

#include "utils/kernel_operator_mm_base_impl.h"
#include "utils/kernel_struct_mm.h"


namespace AscendCBisheng  {
template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const AscendCBisheng::MmadParams& mmadParams);

template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const LocalTensor<V>& bias, const AscendCBisheng::MmadParams& mmadParams);

// #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const AscendCBisheng::MmadBitModeParams& mmadParams);

template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const LocalTensor<V>& bias, AscendCBisheng::MmadBitModeParams& mmadParams);
// #endif

template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const AscendCBisheng::MmadParams& mmadParams)
{
    MmadImpl(dst, fm, filter, mmadParams);
}

template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const LocalTensor<V>& bias, const AscendCBisheng::MmadParams& mmadParams)
{
    MmadImpl(dst, fm, filter, bias, mmadParams);
}

template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const AscendCBisheng::MmadBitModeParams& mmadParams)
{
    MmadImpl(dst, fm, filter, mmadParams);
}

template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm,
    const LocalTensor<S>& filter, const LocalTensor<V>& bias, AscendCBisheng::MmadBitModeParams& mmadParams)
{
    MmadImpl(dst, fm, filter, bias, mmadParams);
}

}

#endif
