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
 * \file is_one_of.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_STD_IS_ONE_OF_H
#define ASCENDC_BISHENG_STD_IS_ONE_OF_H

#include "integral_constant.h"
#include "conditional.h"
#include "is_same.h"

namespace AscendCBisheng {
namespace Std {
template <typename T, typename... Args> struct is_one_of : public false_type {};

template <typename T, typename Head, typename... Tail>
struct is_one_of<T, Head, Tail...>
    : Std::conditional_t<Std::is_same_v<T, Head>, Std::true_type, is_one_of<T, Tail...>> {};

template <typename T, typename... Args> inline constexpr bool is_one_of_v = is_one_of<T, Args...>::value;
}
}

#endif // ASCENDC_BISHENG_STD_IS_ONE_OF_H
