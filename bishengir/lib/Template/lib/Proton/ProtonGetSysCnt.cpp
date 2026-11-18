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

#include "Utils.h"

#if defined(PROTON_AIC)
#define PROTON_CORE_ATTR __aicore__
#define PROTON_FN_NAME(base) base##_cube
#else
#define PROTON_CORE_ATTR __aiv__
#define PROTON_FN_NAME(base) base
#endif

extern "C" {
PROTON_CORE_ATTR __attribute__((always_inline)) int64_t
PROTON_FN_NAME(proton_get_sys_cnt)() {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  return 0;
#else
  return INTRINSIC_NO_ARGS(get_sys_cnt);
#endif
}
}
