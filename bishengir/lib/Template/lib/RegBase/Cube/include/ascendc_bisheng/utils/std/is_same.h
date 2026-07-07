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
 * \file is_same.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_STD_IS_SAME_H
#define ASCENDC_BISHENG_STD_IS_SAME_H

#include "integral_constant.h"

namespace AscendCBisheng {
namespace Std {

template <typename TP, typename Up>
struct is_same : public false_type {};

template <typename TP>
struct is_same<TP, TP> : public true_type {};

template <typename TP, typename Up>
constexpr bool is_same_v = false;

template <typename TP>
constexpr bool is_same_v<TP, TP> = true;

template <typename Tp, typename Up>
using IsSame = bool_constant<is_same_v<Tp, Up>>;

template <typename Tp, typename Up>
using IsNotSame = bool_constant<!is_same_v<Tp, Up>>;

}
}

#endif // ASCENDC_BISHENG_STD_IS_SAME_H
