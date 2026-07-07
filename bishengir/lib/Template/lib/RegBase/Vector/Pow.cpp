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
/// normalize powf(baseNum, exponent) as below
/// powf(x, y) = 1, when abs(x) = 1 and abs(y) = inf
///            = nan, when x = -inf and y is not integer value or y is finite
///            = nan, when x < 0 and x is finite. and y is finite and y is not
///            integer
///            = x ^ y = exp(y * ln(|x|)), when x > 0
///            = x ^ y = ((-1) ^ y) * exp(y * ln|x|), when x <  0
///            = 1, when y == 0
/// so
/// partialRes0 = x ^ y = exp(y * ln(|x|)), when x > 0
///             = x ^ y = ((-1) ^ y) * exp(y * ln|x|), when x <  0
/// partialRes1 = select(abs(x)==1 && abs(y)==inf, 1, partialRes0)
/// partialRes2 = select((abs(x) != inf and x < 0 and abs(y) != inf
///               and floor(y) != y), nan, partialRes1), namely when x is
///               negative finite and y is finite and not integer, result is nan
/// pow(x, y) = select(y == 0, 1, partialRes2)
/// note: hardware vln will output -inf when x == 0
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_pow(TYPE srcX, TYPE srcY) {
  float x = srcX;
  float y = srcY;

  float fabsx = __fabsf(x);
  float fabsy = __fabsf(y);
  int inty = (int32_t)fabsy;

  float partialRes0 = x >= 0.0f
                          ? __expf(y * __logf(fabsx))
                          : (1 - (inty & 1) * 2) * __expf(y * __logf(fabsx));
  float partialRes1 =
      (__fabsf(fabsx - 1.0f) < 1e-6f && fabsy == INFINITY) ? 1.0f : partialRes0;
  float partialRes2 =
      (fabsx != INFINITY && x < 0.0f && fabsy != INFINITY && __floorf(y) != y)
          ? NAN
          : partialRes1;
  return (TYPE)((y == 0.0f) ? 1 : partialRes2);
}

extern "C" {
REGISTER_SIMT_OP2(pow, half, half);
REGISTER_SIMT_OP2(pow, float, float);
REGISTER_SIMT_OP2(pow, bfloat16_t, bfloat16_t);
REGISTER_SIMT_OP2(pow, int32_t, int32_t);
}
#endif
