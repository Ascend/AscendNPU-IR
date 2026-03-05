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

#include "Utils.h"
#include "Vector/VecUtils.h"

/// mask of num = (1 << num) - 1.
///
/// constraints:
/// 1. num <= 64
__aiv__ __attribute__((always_inline)) uint64_t getMaskOfNumber(int64_t num) {
  // To avoid overflow when num is 64, change the formula as bellow
  uint64_t value = (uint64_t)1 << (num - 1);
  return value - 1 + value;
}

/// Take the r0 numbers sequentially starting from the low bit.
/// This is a general way to set the mask by count.
///
/// constraints:
/// 1. num <= 128
__aiv__ __attribute__((always_inline)) void SetMaskValueByCount(int64_t num) {
  if (num > 64) {
    INTRINSIC(set_vector_mask, getMaskOfNumber(num - 64), 0xffffffffffffffff);
  } else {
    INTRINSIC(set_vector_mask, 0x0, getMaskOfNumber(num));
  }
}
