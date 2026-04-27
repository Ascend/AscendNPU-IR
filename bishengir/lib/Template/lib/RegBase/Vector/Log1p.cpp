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
/// normalize log1p(x) to ln(x + 1)
/// eg.
/// y = hfusion elemwise unary {log1p} (x)
///  is normalized to
///  y = linalg.elemwise_unary {log}(x + 1)
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_log1p(TYPE src) {
  return (TYPE)(__logf(1 + src));
}

extern "C" {
REGISTER_SIMT_OP(log1p, half, half);
REGISTER_SIMT_OP(log1p, float, float);
}
#endif