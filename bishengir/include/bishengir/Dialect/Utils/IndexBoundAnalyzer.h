//===- IndexBoundAnalyzer.h - Small index value bound analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_UTILS_INDEXBOUNDANALYZER_H
#define BISHENGIR_DIALECT_UTILS_INDEXBOUNDANALYZER_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace utils {

struct IndexBounds {
  std::optional<int64_t> lower;
  std::optional<int64_t> upper;

  void print(llvm::raw_ostream &os) const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IndexBounds &bounds);

/// Computes conservative bounds for simple index expressions.
///
/// The analyzer intentionally stays local and cheap: it looks through index
/// constants and arith.maxsi/arith.minsi chains, and treats other dynamic
/// values as unknown.
class IndexBoundAnalyzer {
public:
  static constexpr unsigned kDefaultMaxAnalysisDepth = 8;

  explicit IndexBoundAnalyzer(
      unsigned maxAnalysisDepth = kDefaultMaxAnalysisDepth)
      : maxAnalysisDepth(maxAnalysisDepth) {}

  IndexBounds get(OpFoldResult value) const;
  IndexBounds get(Value value) const;

  bool hasUpperBoundAtMost(OpFoldResult value, int64_t bound) const;

private:
  IndexBounds get(Value value, unsigned depth) const;

  unsigned maxAnalysisDepth;
};

} // namespace utils
} // namespace mlir

#endif // BISHENGIR_DIALECT_UTILS_INDEXBOUNDANALYZER_H
