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
 * \file conditional.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_STD_CONDITIONAL_H
#define ASCENDC_BISHENG_STD_CONDITIONAL_H

namespace AscendCBisheng {
namespace Std {

namespace conditional_impl {

template <bool>
struct IfImpl;

template <>
struct IfImpl<true>
{
    template <typename IfRes, typename ElseRes>
    using Select = IfRes;
};

template <>
struct IfImpl<false>
{
    template <typename IfRes, typename ElseRes>
    using Select = ElseRes;
};

template <bool Cond, typename IfRes, typename ElseRes>
using If = typename IfImpl<Cond>::template Select<IfRes, ElseRes>;

} // namespace conditional_impl

template <bool Bp, typename If, typename Then>
struct conditional {
    using type = If;
};

template <typename If, typename Then>
struct conditional<false, If, Then> {
    using type = Then;
};

template <bool Bp, typename If, typename Then>
using conditional_t = typename conditional<Bp, If, Then>::type;

}
}

#endif // ASCENDC_BISHENG_STD_CONDITIONAL_H
