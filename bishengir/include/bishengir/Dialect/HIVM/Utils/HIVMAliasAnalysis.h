//===- HIVMAliasAnalysis.h - Memory region alias queries -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_UTILS_HIVMALIASANALYSIS_H
#define BISHENGIR_DIALECT_HIVM_UTILS_HIVMALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"

namespace mlir::utils {

/// Compare memref ranges using their MLIR address and view operations.
/// Returns NoAlias when proven disjoint, otherwise MayAlias. No IR is changed.
/// Assumes signed address arithmetic, including intermediates, never overflows.
class HIVMAliasAnalysis {
public:
  explicit HIVMAliasAnalysis(Operation *op) : aliasAnalysis(op), scope(op) {}

  /// Use only assumptions dominating context; omit it to ignore assumptions.
  AliasResult alias(Value lhs, Value rhs, Operation *context = nullptr);

private:
  mlir::AliasAnalysis aliasAnalysis;
  Operation *scope;
};

} // namespace mlir::utils

#endif // BISHENGIR_DIALECT_HIVM_UTILS_HIVMALIASANALYSIS_H
