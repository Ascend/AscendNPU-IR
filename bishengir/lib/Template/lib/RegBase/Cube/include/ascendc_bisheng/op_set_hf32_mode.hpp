#ifndef ASCENDC_BISHENG_OP_SET_HF32_MODE_HPP
#define ASCENDC_BISHENG_OP_SET_HF32_MODE_HPP

#include "__clang_cce_types.h"
#include "catlass/catlass.hpp"
#include "utils/kernel_macros.h"

namespace AscendCBisheng {

__aicore__ inline void SetHF32Mode(bool hf32Mode) {
  if (hf32Mode) {
    set_ctrl(sbitset1(get_ctrl(), HF32_MODE_BIT));
  } else {
    set_ctrl(sbitset0(get_ctrl(), HF32_MODE_BIT));
  }
}

} // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_OP_SET_HF32_MODE_HPP
