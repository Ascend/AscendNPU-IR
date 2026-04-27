/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file integral_constant.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_STD_INTEGRAL_CONSTANT_H
#define ASCENDC_BISHENG_STD_INTEGRAL_CONSTANT_H

namespace AscendCBisheng {
namespace Std {

template <typename Tp, Tp v>
struct integral_constant
{
    static constexpr const Tp value = v;

    using value_type = Tp;
    using type = integral_constant;

    ASCENDC_HOST_AICORE inline constexpr operator value_type() const noexcept {
        return value;
    }

    ASCENDC_HOST_AICORE inline constexpr value_type operator()() const noexcept {
        return value;
    }
};

template <typename Tp, Tp v>
constexpr const Tp integral_constant<Tp, v>::value;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool b>
using bool_constant = integral_constant<bool, b>;

template <size_t v>
using Int = integral_constant<size_t, v>;

#define STD_INT_BINARY_OP(OP) \
template <auto t, auto u> \
__aicore__ inline constexpr Int<(t OP u)> operator OP (Int<t>, Int<u>) { \
    return {}; \
}

STD_INT_BINARY_OP(+);
STD_INT_BINARY_OP(-);
STD_INT_BINARY_OP(*);
STD_INT_BINARY_OP(/);
STD_INT_BINARY_OP(%);
#undef STD_INT_BINARY_OP

}
}

#endif // ASCENDC_BISHENG_STD_INTEGRAL_CONSTANT_H
