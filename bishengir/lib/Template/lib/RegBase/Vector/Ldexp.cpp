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
#include <cstdint>
#include <type_traits>

// --- Constant Definitions ---

template <> struct FpTraits<uint32_t> {
  static constexpr uint32_t E_MASK = 0x7F800000UL;
  static constexpr uint32_t S_MASK = 0x80000000UL;
  static constexpr uint32_t M_MASK = 0x007FFFFFUL;
  static constexpr int SHIFT = 23;
  static constexpr int MAX_E = 255;
};
template <> struct FpTraits<uint16_t> {
  static constexpr uint16_t E_MASK = 0x7C00U;
  static constexpr uint16_t S_MASK = 0x8000U;
  static constexpr uint16_t M_MASK = 0x03FFU;
  static constexpr int SHIFT = 10;
  static constexpr int MAX_E = 31;
};


// ----------------------------------------------------------------------
// Helper Function: Generic Bitwise ldexp Core Logic
// ----------------------------------------------------------------------
/**
 * @brief Performs the core bitwise ldexp logic (exponent adjustment, overflow, underflow).
 * @tparam IntType The integer width type representing the float (uint32_t) or half (uint16_t).
 * @param bits The current floating-point bit pattern.
 * @param exp The integer exponent 'n' (shift amount).
 * @param E_MASK, S_MASK, M_MASK Type-specific bitmasks.
 * @param SHIFT The type-specific mantissa bit width.
 * @param MAX_E The type-specific maximum exponent encoding.
 * @return IntType The resulting bit pattern.
 */
template <typename IntType>
__aicore__ __attribute__((always_inline)) IntType ldexp_process_bits(IntType bits, int32_t exp) {
  using Traits = FpTraits<IntType>;
  if ((bits & ~Traits::S_MASK) == 0x0000 || (bits & Traits::E_MASK) == Traits::E_MASK) {
    return bits;
  }
  IntType current_exp_bits = (bits & Traits::E_MASK) >> Traits::SHIFT;
  uint32_t mantissa = bits & Traits::M_MASK;
  int32_t effective_exp = (int32_t)current_exp_bits;

  if (current_exp_bits == 0) {
    int shift_cnt = 0;
    while (!(mantissa & (1U << Traits::SHIFT))) {
      mantissa <<= 1;
      shift_cnt++;
    }
    mantissa &= Traits::M_MASK;
    effective_exp = 1 - shift_cnt;
  }

  int32_t new_exp_bits = effective_exp + exp;

  if (new_exp_bits >= Traits::MAX_E) {
    bits = (bits & Traits::S_MASK) | Traits::E_MASK;
  } else if (new_exp_bits <= 0) {
    if (new_exp_bits < -(int32_t)Traits::SHIFT) {
      bits &= Traits::S_MASK;
    } else {
      uint32_t full_mantissa = mantissa | (1U << Traits::SHIFT);
      int shift_amount = 1 - new_exp_bits;
      bits = (bits & Traits::S_MASK) | static_cast<IntType>(full_mantissa >> shift_amount);
    }
  } else {
    bits = (bits & Traits::S_MASK) |
       (static_cast<IntType>(new_exp_bits) << Traits::SHIFT) |
       static_cast<IntType>(mantissa);
  }
  return bits;
}


/**
 * @brief simt_ldexp core function master template declaration.
 * * Implements the IEEE 754 standard ldexp operation: srcX * 2^(int)srcY.
 * @tparam TYPE Input and output data type (float or half).
 */
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_ldexp(TYPE srcX, int32_t srcY) {
  using IntType = std::conditional_t<sizeof(TYPE) == 4, uint32_t, uint16_t>;
  union { TYPE f; IntType i; } u;
  u.f = srcX;
  u.i = ldexp_process_bits<IntType>(u.i, static_cast<int>(srcY));
  return u.f;
}

extern "C" {
REGISTER_SIMT_OPWithInt32(ldexp, half, half);
REGISTER_SIMT_OPWithInt32(ldexp, float, float);
}
#endif