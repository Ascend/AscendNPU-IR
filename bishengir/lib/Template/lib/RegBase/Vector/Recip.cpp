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

#include "RegBase/SimtUtils.h"
#include "Utils.h"

#if defined(__DAV_C310__)
// reciprocal : 1 / x
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_recip(TYPE srcX) {
    return (TYPE)(1.0f / (float)srcX);
}

extern "C" {
REGISTER_SIMT_OP(recip, half, half);
REGISTER_SIMT_OP(recip, float, float);
}

#endif
