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
__aicore__ __attribute__((always_inline)) TYPE simt_sin(TYPE src) {
  // Step 1: Reduce the angle to the range [0, pi/2) and determine the quadrant
  float x = src;
  x = __fma(x, 0.0f, x);
  int quadrant = 0;
  float reducedAngle = 0.0f;
  // Uses Payne-Hanek for large angles
  if (__fabsf(x) > SIN_COS_THRESHOLD) {
    // Step 1: Extract raw bits of the input angle
    uint32_t inputBits = reinterpret_cast<uint32_t &>(x);

    // Step 2: Extract exponent and compute index into 2/pi table
    int32_t exponent = ((inputBits & 0x7F800000) >> 23) - 127;
    uint32_t exponentIndex = static_cast<uint32_t>(exponent) >> 5;

    // Step 3: Get the 2/pi table entries for this exponent index
    constexpr uint32_t twoOverPiTable[] = {0x517cc1b7, 0x27220a94, 0xfe13abe8,
                                           0xfa9a6ee0, 0x6db14acc, 0x9e21c820};
    uint32_t highTerm = exponentIndex ? twoOverPiTable[exponentIndex - 1] : 0;
    uint32_t midTerm = twoOverPiTable[exponentIndex];
    uint32_t lowTerm = twoOverPiTable[exponentIndex + 1];
    uint32_t lastTerm = twoOverPiTable[exponentIndex + 2];

    // Step 4: Compute exponent remainder and shift table entries accordingly
    int32_t exponentRemainder = static_cast<uint32_t>(exponent) & 0x1F;
    if (exponentRemainder != 0) {
      highTerm = (highTerm << exponentRemainder) |
                 (midTerm >> (32 - exponentRemainder));
      midTerm = (midTerm << exponentRemainder) |
                (lowTerm >> (32 - exponentRemainder));
      lowTerm = (lowTerm << exponentRemainder) |
                (lastTerm >> (32 - exponentRemainder));
    }

    // Step 5: Extract and normalize the mantissa
    uint32_t mantissa = (inputBits & 0x007FFFFF) | 0x4F000000;
    uint32_t normalizedMantissa =
        static_cast<uint32_t>(reinterpret_cast<float &>(mantissa));

    // Step 6: Compute product = (mantissa * highTerm) << 32 + mantissa *
    // midTerm + mantissa * lowTerm
    uint64_t product = static_cast<uint64_t>(normalizedMantissa) * lowTerm;
    product =
        static_cast<uint64_t>(normalizedMantissa) * midTerm + (product >> 32);
    product =
        (static_cast<uint64_t>(normalizedMantissa * highTerm) << 32) + product;

    // Step 7: Extract quotient and remainder
    int32_t quotient = static_cast<int32_t>(product >> 62);
    product = product & 0x3FFFFFFFFFFFFFFFULL;

    // Step 8: Handle carry
    if (product & 0x2000000000000000ULL) {
      product -= 0x4000000000000000ULL;
      quotient += 1;
    }

    // Step 9: Split product into high and low
    int64_t productInt64 = static_cast<int64_t>(product);
    int64_t highFloat = static_cast<float>(productInt64);
    productInt64 = productInt64 - static_cast<int64_t>(highFloat);
    int64_t lowFloat = static_cast<float>(productInt64);

    // Step 10: Compute final result = (high + low) * pi/2 * 2^-62
    reducedAngle = (highFloat + lowFloat) * PI_OVER_TWO_LOW;

    // Step 11: Handle negative input
    if (x < 0.0f) {
      reducedAngle = -reducedAngle;
      quotient = -quotient;
    }

    quadrant = quotient;
  } else {
    // Uses Cody-Waite for small angles
    float y = __fma(x, TWO_OVER_PI, TRUNCATE_VALUE);
    quadrant = reinterpret_cast<int &>(y);
    y = y - TRUNCATE_VALUE;
    x = __fma(y, NEG_HIGN_OF_PI_OVER_2, x);
    x = __fma(y, NEG_MIDDLE_OF_PI_OVER_2, x);
    reducedAngle = __fma(y, NEG_LOW_OF_PI_OVER_2, x);
  }

  // Step 2: Compute cosine and sine of the reduced angle using polynomial
  // approximations
  // cosPoly
  float c = reducedAngle;
  c = c * c;
  float t = __fma(c, ONE_OVER_8_FACT, NEG_ONE_OVER_6_FACT);
  t = __fma(c, t, ONE_OVER_4_FACT);
  t = __fma(c, t, NEG_ONE_OVER_2_FACT);
  c = __fma(c, t, ONE);

  // sinpoly
  float s = reducedAngle;
  float p = s * s;
  float m = __fma(s, p, 0.0f);

  float z = __fma(p, ONE_OVER_9_FACT, NEG_ONE_OVER_7_FACT);
  z = __fma(p, z, ONE_OVER_5_FACT);
  z = __fma(p, z, NEG_ONE_OVER_3_FACT);

  s = __fma(z, m, s); // * x^3 + x

  // Step 3: Adjust the sine value based on the quadrant
  if (quadrant & 2) { // Quadrants 2 and 3: sin(pi + x) = -sin(x)
    s = -s;
    c = -c;
  }
  if (quadrant & 1) { // Quadrants 1 and 3: sin(pi/2 + x) = cos(x)
    s = c;
  }

  return (TYPE)s;
}

extern "C" {
REGISTER_SIMT_OP(sin, bfloat16_t, bfloat16_t);
REGISTER_SIMT_OP(sin, half, half);
REGISTER_SIMT_OP(sin, float, float);
}
#endif
