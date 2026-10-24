//===- NPUTargetSpec.cpp - Target spec table -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Owns the generated target spec table. This translation unit is the single
// place that includes NPUTargetSpec.cpp.inc, so the non-inline symbols it
// defines are emitted exactly once.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Targets/NPUTargetSpecUtils.h"

#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.cpp.inc"

namespace mlir {
namespace hacc {

std::optional<const TargetSpec *> lookupTargetSpec(TargetDevice target) {
  return getTargetSpec(target);
}

} // namespace hacc
} // namespace mlir
