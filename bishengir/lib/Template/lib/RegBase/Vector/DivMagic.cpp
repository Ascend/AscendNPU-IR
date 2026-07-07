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
__aicore__ __attribute__((always_inline)) TYPE
simt_div_magic_shift(TYPE divisor) {
  int64_t log_2 = 64 - clz((uint64_t)divisor);
  int64_t cnt1 = bcnt1((uint64_t)divisor);
  return cnt1 == 1 ? log_2 - 1 : log_2;
}

template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_div_magic_mul(TYPE divisor,
                                                                  TYPE shift) {
  return (1l << 32) * ((1l << shift) - divisor) / divisor + 1;
}

extern "C" {
REGISTER_SIMT_OP(div_magic_shift, uint32_t, uint32_t);
REGISTER_SIMT_OP2(div_magic_mul, uint32_t, uint32_t);
}
#endif
