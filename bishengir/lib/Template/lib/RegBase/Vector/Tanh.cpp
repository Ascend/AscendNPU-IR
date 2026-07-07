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
/// Normalize tanh(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))
///                  =(exp(2x)-1)/(exp(2x)+1)
///                  =(exp(2x')-1)/(exp(2x')+1)
/// where x' = clip(x, [-8.8, 8.8])
/// Reference: HFusion/Transforms/Normalize.cpp NormalizeTanhOp
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_tanh(TYPE src) {
  float x = src;
  // When x's value is too large, exp(2x) will be overflow.
  // So clip it to [-8.8, 8.8], the epison is ie-8.
  float clip_x = __fmaxf(__fminf(x, 8.8), -8.8);
  // expf with float for high precision.
  float e = __expf(2 * clip_x);
  float y = (e - 1) / (e + 1);
  return (TYPE)(y);
}

extern "C" {
REGISTER_SIMT_OP(tanh, bfloat16_t, bfloat16_t);
REGISTER_SIMT_OP(tanh, half, half);
REGISTER_SIMT_OP(tanh, float, float);
}
#endif
