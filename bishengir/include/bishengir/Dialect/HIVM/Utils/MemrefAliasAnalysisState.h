//===- MemrefAliasAnalysisState.h - Utilities to support the HIVM dialect
//-----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_MEMREFALIASANALYSISSTATE_H
#define MLIR_DIALECT_HIVM_UTILS_MEMREFALIASANALYSISSTATE_H

#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir {
namespace utils {
// Alias analysis of memref, greatly borrowed from OneShotAnalysisState.
class MemrefAliasAnalysisState {
public:
  explicit MemrefAliasAnalysisState(Operation *op);

  // Union the alias sets of `v1` and `v2`.
  void unionAliasSets(Value v1, Value v2);

  // Add a new entry for `v` in the `aliasInfo`
  void createAliasInfoEntry(Value v);

  void printAlias();

  bool isAlias(Value v1, Value v2);

private:
  using EquivalenceClassRangeType = llvm::iterator_range<
      llvm::EquivalenceClasses<Value, ValueComparator>::member_iterator>;
  /// Check that aliasInfo for `v` exists and return a reference to it.
  EquivalenceClassRangeType getAliases(Value v) const;

  llvm::EquivalenceClasses<Value, ValueComparator> aliasInfo;
};
} // namespace utils
} // namespace mlir
#endif
