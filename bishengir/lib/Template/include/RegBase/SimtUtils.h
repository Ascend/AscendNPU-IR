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

#ifndef HIVM_MLIR_TEMPLATE_SIMT_UTILS_H
#define HIVM_MLIR_TEMPLATE_SIMT_UTILS_H

#include "Utils.h"
#if defined(__DAV_C310__)

static constexpr float PI_0 = 3.140625;
static constexpr float PI_1 = 0.0009670257568359375;
static constexpr float PI_2 = 6.2771141529083251953125e-7;
static constexpr float PI_3 = 1.21644916362129151821136474609375e-10;
static constexpr float PI_4 = -1.0290623200529979163359041220560e-13;

static constexpr float PI = PI_0 + PI_1 + PI_2 + PI_3 + PI_4;
static constexpr float PI_OVER_2 = PI * 0.5f;
static constexpr float PI_OVER_4 = PI * 0.25f;
static constexpr float PI_OVER_8 = PI * 0.125f;

static constexpr float tan_pi_12 = 0.26794919243112270f; // tan(π/12)
static constexpr float tan_pi_8 = 0.41421356237309503f;  // tan(π/8)

// The threshold to decide between Payne-Hanek and Cody-Waite.
static constexpr float SIN_COS_THRESHOLD = 71476.0625f;

// pi/2 * 2^-62
static constexpr float PI_OVER_TWO_LOW = 3.4061215800865545e-19f;

static constexpr float TWO_OVER_PI = 0.636619747f; // 2/Pi

// used to truncate mantissa of x*(2/pi)
static constexpr float TRUNCATE_VALUE = 12582912.0f;

// -high of pi/2
static constexpr float NEG_HIGN_OF_PI_OVER_2 = -1.57079601e+00f;
// -middle of pi/2
static constexpr float NEG_MIDDLE_OF_PI_OVER_2 = -3.13916473e-07f;
// -low of pi/2
static constexpr float NEG_LOW_OF_PI_OVER_2 = -5.39030253e-15f;

static constexpr float ONE_OVER_8_FACT = 2.44677067e-5f;      // 1/8!
static constexpr float NEG_ONE_OVER_6_FACT = -1.38877297e-3f; // -1/6!
static constexpr float ONE_OVER_4_FACT = 4.16666567e-2f;      // 1/4!
static constexpr float NEG_ONE_OVER_2_FACT = -5.00000000e-1f; // -1/2!
static constexpr float ONE = 1.00000000e+0f;                  // 1

static constexpr float ONE_OVER_9_FACT = 2.86567956e-6f;      // 1/9!
static constexpr float NEG_ONE_OVER_7_FACT = -1.98559923e-4f; // -1/7!
static constexpr float ONE_OVER_5_FACT = 8.33338592e-3f;      // 1/5!
static constexpr float NEG_ONE_OVER_3_FACT = -1.66666672e-1f; // -1/3!

static constexpr float LN2 = 0.693147182;
static constexpr float REC_LN2 = 1. / LN2;

#define DECLARE_SIMT_OP(name, opType, returnType)                              \
  __aicore__ __attribute__((always_inline)) returnType                         \
  _mlir_ciface_simt_##name##_##opType(opType src)

#define DECLARE_SIMT_OP2(name, opType, returnType)                             \
  __aicore__ __attribute__((always_inline)) returnType                         \
  _mlir_ciface_simt_##name##_##opType(opType src1, opType src2)

#define DECLARE_SIMT_OPWithInt32(name, opType, returnType)                     \
  __aicore__ __attribute__((always_inline)) returnType                         \
  _mlir_ciface_simt_##name##_##opType(opType src1, int32_t src2)

#define REGISTER_SIMT_OP(name, opType, returnType)                             \
  DECLARE_SIMT_OP(name, opType, returnType) { return simt_##name<opType>(src); }

#define REGISTER_SIMT_OP2(name, opType, returnType)                            \
  DECLARE_SIMT_OP2(name, opType, returnType) {                                 \
    return simt_##name<opType>(src1, src2);                                    \
  }

#define REGISTER_SIMT_OPWithInt32(name, opType, returnType)                    \
  DECLARE_SIMT_OPWithInt32(name, opType, returnType) {                         \
    return simt_##name<opType>(src1, src2);                                    \
  }

extern "C" {
// ERF declarations
DECLARE_SIMT_OP(erf, bfloat16_t, bfloat16_t);
DECLARE_SIMT_OP(erf, half, half);
DECLARE_SIMT_OP(erf, float, float);

// RSQRT declarations
DECLARE_SIMT_OP(rsqrt, bfloat16_t, bfloat16_t);
DECLARE_SIMT_OP(rsqrt, half, half);
DECLARE_SIMT_OP(rsqrt, float, float);

// TANH declarations
DECLARE_SIMT_OP(tanh, bfloat16_t, bfloat16_t);
DECLARE_SIMT_OP(tanh, half, half);
DECLARE_SIMT_OP(tanh, float, float);

// SIN declarations
DECLARE_SIMT_OP(sin, bfloat16_t, bfloat16_t);
DECLARE_SIMT_OP(sin, half, half);
DECLARE_SIMT_OP(sin, float, float);

// COS declarations
DECLARE_SIMT_OP(cos, bfloat16_t, bfloat16_t);
DECLARE_SIMT_OP(cos, half, half);
DECLARE_SIMT_OP(cos, float, float);

DECLARE_SIMT_OP(isnan, float, bool);
DECLARE_SIMT_OP(isinf, float, bool);
DECLARE_SIMT_OP(isfinite, float, bool);

// TAN declarations
DECLARE_SIMT_OP(tan, half, half);
DECLARE_SIMT_OP(tan, float, float);

// POWF declarations
DECLARE_SIMT_OP2(pow, half, half);
DECLARE_SIMT_OP2(pow, float, float);
DECLARE_SIMT_OP2(pow, bfloat16_t, bfloat16_t);

// POWI declarations
DECLARE_SIMT_OP2(pow, int32_t, int32_t);

// UMULHI declarations
DECLARE_SIMT_OP2(umulhi, uint32_t, uint32_t);

// LDEXP declarations
DECLARE_SIMT_OPWithInt32(ldexp, float, float);
DECLARE_SIMT_OPWithInt32(ldexp, half, half);

// LOG1P declarations
DECLARE_SIMT_OP(log1p, half, half);
DECLARE_SIMT_OP(log1p, float, float);

// LOG2 declarations
DECLARE_SIMT_OP(log2, float, float);

// RECIP declarations
DECLARE_SIMT_OP(recip, half, half);
DECLARE_SIMT_OP(recip, float, float);

// ATAN declarations
DECLARE_SIMT_OP(atan, half, half);
DECLARE_SIMT_OP(atan, float, float);

// ILOGB declarations
DECLARE_SIMT_OP(ilogb, half, half);
DECLARE_SIMT_OP(ilogb, float, float);

// RELU declarations
DECLARE_SIMT_OP(relu, half, half);
DECLARE_SIMT_OP(relu, float, float);

// ROUND declarations
DECLARE_SIMT_OP(round, float, float);

// DIVRN declarations
DECLARE_SIMT_OP2(divrn, float, float);

// DIV MAGIC SHIFT declaration
DECLARE_SIMT_OP(div_magic_shift, uint32_t, uint32_t);
DECLARE_SIMT_OP2(div_magic_mul, uint32_t, uint32_t);

// FLOAT_AS_INT declaration
DECLARE_SIMT_OP(float_as_int, float, int32_t);
}
#endif
#endif
