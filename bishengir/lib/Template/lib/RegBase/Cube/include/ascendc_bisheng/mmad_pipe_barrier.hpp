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
 * \file kernel_operator_block_sync_intf.h
 * \brief
 */

#ifndef ASCENDC_BISHENG_MMAD_PIPE_BARRIER_HPP
#define ASCENDC_BISHENG_MMAD_PIPE_BARRIER_HPP

#include "utils/kernel_reg.h"
// #include "utils/kernel_tensor_base.h"

namespace AscendCBisheng {
template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
{
    PipeBarrierImpl<pipe>();
}

} 
#endif // ASCENDC_BISHENG_PIPEBARRIER_HPP