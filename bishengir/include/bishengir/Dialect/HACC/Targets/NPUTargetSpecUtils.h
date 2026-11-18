//===- NPUTargetSpecUtils.h - Target spec lookup ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPECUTILS_H
#define BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPECUTILS_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.h.inc"

#include <optional>

namespace mlir {
namespace hacc {

/// Look up the static target spec for `target`.
std::optional<const TargetSpec *> lookupTargetSpec(TargetDevice target);

} // namespace hacc
} // namespace mlir

#endif // BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPECUTILS_H
