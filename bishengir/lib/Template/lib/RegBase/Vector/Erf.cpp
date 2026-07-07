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
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_erf(TYPE src) {
  float a1 = 0.254829592f;
  float a2 = -0.284496736f;
  float a3 = 1.421413741f;
  float a4 = -1.453152027f;
  float a5 = 1.061405429f;
  float p = 0.3275911f;

  float x = src;
  int sign = 1;
  if (x < 0) {
    sign = -1;
    x = -x;
  }

  float t = 1 / (1 + p * x);
  float y =
      1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * __expf(-x * x);

  return (TYPE)(sign * y);
}

extern "C" {
REGISTER_SIMT_OP(erf, bfloat16_t, bfloat16_t);
REGISTER_SIMT_OP(erf, half, half);
REGISTER_SIMT_OP(erf, float, float);
}
#endif
