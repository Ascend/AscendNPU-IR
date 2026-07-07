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

// High-precision Taylor series for atan, up to 13th order
__aicore__ __attribute__((always_inline)) static float
do_taylor_series(float val) {
  float val_sq = val * val;
  float result = val;
  float term = val;

  for (int i = 3; i <= 13; i += 2) {
    term = term * (-val_sq);
    float next_term = term / static_cast<float>(i);
    result += next_term;
  }

  return result;
}

template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_atan(TYPE src) {
  float x = static_cast<float>(src);

  // Step 1: Clamp input to avoid overflow
  if (x > 10000.0f)
    x = 10000.0f;
  if (x < -10000.0f)
    x = -10000.0f;

  // Step 2: Extract sign and use absolute value
  float sign = (x >= 0.0f) ? 1.0f : -1.0f;
  float abs_x = __fabsf(x);

  // Step 3: Multi-level range reduction for accuracy
  float result;

  if (abs_x <= tan_pi_12) {
    // Very small: direct Taylor
    result = do_taylor_series(x);
  } else if (abs_x <= tan_pi_8) {
    // π/12 reduction: atan(x) = π/12 + atan((x - tan(π/12))/(1 + x*tan(π/12)))
    const float pi_12 = PI / 12.0f;
    float reduced = (abs_x - tan_pi_12) / (1.0f + abs_x * tan_pi_12);
    result = sign * (pi_12 + do_taylor_series(reduced));
  } else if (abs_x <= 1.0f) {
    // π/8 reduction: atan(x) = π/8 + atan((x - tan(π/8))/(1 + x*tan(π/8)))
    float reduced = (abs_x - tan_pi_8) / (1.0f + abs_x * tan_pi_8);
    result = sign * (PI_OVER_8 + do_taylor_series(reduced));
  } else if (abs_x <= 2.0f) {
    // π/4 reduction: atan(x) = π/4 + atan((x - 1)/(x + 1))
    float reduced = (abs_x - 1.0f) / (abs_x + 1.0f);
    result = sign * (PI_OVER_4 + do_taylor_series(reduced));
  } else {
    // Large values: atan(x) = π/2 − atan(1/x)
    float inv_x = 1.0f / abs_x;
    const float pi_12 = PI / 12.0f;

    if (inv_x > tan_pi_12) {
      // Reciprocal π/12 reduction: atan(x) = π/2 - π/12 - atan((1/x -
      // tan(π/12))/(1 + (1/x)*tan(π/12)))
      float reduced = (inv_x - tan_pi_12) / (1.0f + inv_x * tan_pi_12);
      result = sign * (PI_OVER_2 - pi_12 - do_taylor_series(reduced));
    } else {
      // Reciprocal direct Taylor: atan(x) = π/2 - atan(1/x) with small 1/x
      result = sign * (PI_OVER_2 - do_taylor_series(inv_x));
    }
  }
  return static_cast<TYPE>(result);
}

extern "C" {
REGISTER_SIMT_OP(atan, half, half);
REGISTER_SIMT_OP(atan, float, float);
}
#endif