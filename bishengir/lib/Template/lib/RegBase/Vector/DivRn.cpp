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
#include "Utils.h"
 
#if defined(__DAV_C310__)

__aicore__ __attribute__((always_inline)) float div_rn_fp32(float a, float b) {
    // Standard hardware division result used for fallback in special cases (b=0, inf, etc.)
    float base_res = a / b;

    // Threshold and Scale Factor: 
    // If |b| > 1.0e37, 1/b becomes a subnormal number (~2.9e-39).
    // Apply scaling: if b is too large, scale down both a and b by 0.125.
    // This preserves the ratio (a/b = (a*S)/(b*S)) while keeping 1/(b*S) normal.
    static constexpr float EPSILON = 1e-6f;
    static constexpr float LARGE_THRESHOLD = 1.0e37f; 
    static constexpr float SCALE_FACTOR = 0.125f; 
    static constexpr float INF_THRESHOLD = 3.40282e38f;

    float abs_b = (b < 0.0f) ? -b : b; 

    // Use selection instead of if-branch to maintain SIMT synchronization
    bool need_scale = (abs_b > LARGE_THRESHOLD);
    float s = need_scale ? SCALE_FACTOR : 1.0f;
    float a_scaled = a * s;
    float b_scaled = b * s;

    float neg_b_scaled = -b_scaled;
    // Initial reciprocal estimate for scaled divisor
    float y = 1.0f / b_scaled;

    // Newton-Raphson iteration to refine the reciprocal
    // err = 1.0 - b_scaled * y
    float err = __fma(neg_b_scaled, y, 1.0f);
    // y = y + y * err
    y = __fma(y, err, y);

    // Compute preliminary quotient
    float q = a_scaled * y;
    
    // Check for overflow or special values in quotient
    float abs_q = (q < 0.0f) ? -q : q; 

    // Final residual correction (Markstein step) for 0-ULP accuracy
    // rem = a_scaled - b_scaled * q
    float rem = __fma(neg_b_scaled, q, a_scaled);
    // result = q + rem * y
    float result = __fma(rem, y, q);

    // Selection Logic: Fallback to hardware result if:
    // 1. Division by zero (b == 0.0f)
    // 2. Infinite divisor (abs_b > INF_THRESHOLD)
    // 3. Infinite or overflowed quotient (abs_q > INF_THRESHOLD)
    bool use_base = (abs_b < EPSILON) || (abs_b > INF_THRESHOLD) || (abs_q > INF_THRESHOLD);

    return use_base ? base_res : result;
}

template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE div_rn(TYPE a, TYPE b) {
    float res_f = div_rn_fp32(static_cast<float>(a), static_cast<float>(b));
    return static_cast<TYPE>(res_f);
}

template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_divrn(const TYPE src1, const TYPE src2) {
    return div_rn<TYPE>(src1, src2); 
}

extern "C" {
REGISTER_SIMT_OP2(divrn, float, float);
}
#endif