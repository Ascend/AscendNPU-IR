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
/// y = tan(x)
/// step1: xround = round(x / pi)
/// step2: Calculate res_down1 res_down2
///     p0=3.140625 p1=0.0009670257568359375 p2=6.2771141529083251953125e-7
///     p3=1.21644916362129151821136474609375e-10
///     p4=-1.0290623200529979163359041220560e-13
///     kpi0 = xround * p0; kpi1 = xround * p1...
///     res_down1=x-kpi0-kpi1+1.57079637050628662109375-kpi2+(-0.00000004371139000189375)-kpi_3-kpi_4
///     res_down2=x-kpi0-kpi1+(-1.57079637050628662109375)-kpi2+0.00000004371139000189375-kpi_3-kpi_4
/// step3: z = x - kpi0 - kpi1 - kpi2 - kpi3 - kpi4 z2 = z * z
/// step4: Calculate res_up res_down
///     CST0 = 0.0698520831551998762793
///     T1 = -6.8711573651634203789 T2 = 61.20362572811089435388
///     res_up = ((((z2*CST0)+T1)*z2)+T2)*z
///     res_down = (z2 - 24.8048928861126769186219) * res_down1 * res_down2
/// step5: y = res_up / res_down
/// Reference: HFusion/Transforms/Normalize.cpp NormalizeTanOp
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_tan(TYPE src) {
  float x = src;
  float round_x = __roundf(x / (float)M_PI);
  float kpi[] = {
    round_x * PI_0,
    round_x * PI_1,
    round_x * PI_2,
    round_x * PI_3,
    round_x * PI_4
  };
  float pio2_hi = 1.57079637050628662109375f;
  float pio2_lo = -0.00000004371139000189375f;
  float res_down1 = x - kpi[0] - kpi[1] + pio2_hi - kpi[2] + pio2_lo - kpi[3] - kpi[4];
  float res_down2 = x - kpi[0] - kpi[1] - pio2_hi - kpi[2] - pio2_lo - kpi[3] - kpi[4];
  float z = x - kpi[0] - kpi[1] - kpi[2] - kpi[3] - kpi[4];
  float z2 = z * z;
  float CST0 = 0.0698520831551998762793f;
  float T1 = -6.8711573651634203789f;
  float T2 = 61.20362572811089435388f;
  float res_up = ((((z2 * CST0) + T1) * z2) + T2) * z;
  float res_down = (z2 - 24.8048928861126769186219f) * res_down1 * res_down2;
  float y = res_up / res_down;
  return (TYPE)(y);
}

extern "C" {
REGISTER_SIMT_OP(tan, half, half);
REGISTER_SIMT_OP(tan, float, float);
}
#endif
