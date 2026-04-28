/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_utils.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_UTILS_H_
#define ASCENDC_BISHENG_KERNEL_UTILS_H_

namespace ops {

template<typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template<typename, typename>
struct IsSame
    : public falseType {
};
template<typename Tp>
struct IsSame<Tp, Tp>
    : public trueType {
};

template <typename T>
__aicore__ inline T Ceil(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b;
}

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

}
#endif  // ASCENDC_BISHENG_KERNEL_UTILS_H_