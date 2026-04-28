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
 * \file kernel_reg.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_REG_H
#define ASCENDC_BISHENG_KERNEL_REG_H

#include "./kernel_utils.h"
#include "./kernel_struct_aipp.h"

namespace AscendCBisheng {
constexpr uint64_t MASK_PLACEHOLDER = 0;
constexpr uint64_t MASK_PLACEHOLDER_LIST[2] = {0, 0};

enum class MaskMode : uint8_t {
    NORMAL = 0,
    COUNTER
};


template <pipe_t pipe> __aicore__ inline void PipeBarrierImpl()
{

    if ASCEND_IS_AIC {
        if constexpr (pipe == PIPE_MTE3) {
            return;
        }
    }
    if constexpr (pipe != PIPE_V) {
        pipe_barrier(pipe);
    }
    return;
}

} // namespace AscendCBisheng
#endif // ASCENDC_BISHENG_KERNEL_REG_H