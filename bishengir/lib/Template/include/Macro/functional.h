/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#pragma once

#include "limits"
#include "type_traits"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <uint32_t ALIGN, typename T>
__aicore__ __attribute__((always_inline)) T RoundUp(const T val) {
  static_assert(ALIGN != 0, "align must not be zero");
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  T align = ALIGN;
  return (val + align - 1) / align * align;
}

template <typename T>
__aicore__ __attribute__((always_inline)) T RoundUp(const T val,
                                                    const T align) {
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  if (align == 0) {
    return 0;
  } else {
    return (val + align - 1) / align * align;
  }
}

template <uint32_t DIVISOR, typename T>
__aicore__ __attribute__((always_inline)) T CeilDiv(const T dividend) {
  static_assert(DIVISOR != 0, "align must not be zero");
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  T divisor = DIVISOR;
  return (dividend + divisor - 1) / divisor;
}

template <typename T> constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
__aicore__ __attribute__((always_inline)) T CeilDiv(const T dividend,
                                                    const T divisor) {
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  if (divisor == 0) {
    return T_MAX<T>;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}
