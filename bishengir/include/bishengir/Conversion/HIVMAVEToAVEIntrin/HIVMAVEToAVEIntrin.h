//===- HIVMAVEToAVEIntrin.h - Convert HIVMAVE dialect to AVEIntrin dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMAVETOAVEINTRIN_HIVMAVETOAVEINTRIN_H_
#define BISHENGIR_CONVERSION_HIVMAVETOAVEINTRIN_HIVMAVETOAVEINTRIN_H_

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTHIVMAVETOAVEINTRIN
#include "bishengir/Conversion/Passes.h.inc"

/// Create a pass to convert HIVMAVE ops to AVEIntrin ops.
std::unique_ptr<Pass> createConvertHIVMAVEToAVEIntrinPass();

// Helper struct to pass Op Type as a value
template <typename T>
struct OpTag {
  using type = T;
};

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMAVETOAVEINTRIN_HIVMAVETOAVEINTRIN_H_
