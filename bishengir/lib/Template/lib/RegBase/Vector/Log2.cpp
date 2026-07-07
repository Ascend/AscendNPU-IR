/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#if defined(__DAV_C310__)
/// normalize logb(x) to ln(x) * (1 / ln(b)) when log base b is not e
/// eg.
/// y = log2(x)  is normalized to
/// y = ln(x) * (1 / ln(b))
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_log2(TYPE src) {
  float x = (float) src;
  return (TYPE)(__logf(x) * REC_LN2);
}

extern "C" {
REGISTER_SIMT_OP(log2, float, float);
}
#endif