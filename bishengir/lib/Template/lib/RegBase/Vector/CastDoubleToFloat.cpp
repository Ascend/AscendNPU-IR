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

#include "ArchDefs.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/Cast/DoubleToFloatRNE.h"

#if BISHENG_TPL_REGBASE

// f64 -> f32 vector cast, registered as
// `_mlir_ciface_vcast_double_to_float_<rank>d_with_mode`.
//
// Same software implementation as the Ascend910B template (see
// DoubleToFloatRNE.h): no fp64 vector lanes are involved, the conversion is
// done on the raw integer bits so that no __truncdfsf2 reference is emitted.
extern "C" {
DECLARE_CAST(double, float, 1) {
  (void)cast_mode; // always rint for this pair, see the comment above.
  vcast_double_to_float_rn_nd<1>(src, dst);
}
DECLARE_CAST(double, float, 2) {
  (void)cast_mode; // always rint for this pair, see the comment above.
  vcast_double_to_float_rn_nd<2>(src, dst);
}
}

#endif
