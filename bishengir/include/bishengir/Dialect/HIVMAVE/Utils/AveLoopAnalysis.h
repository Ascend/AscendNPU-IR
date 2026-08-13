//===- AveLoopAnalysis.h - AVE loop analysis utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVMAVE_UTILS_AVELOOPANALYSIS_H
#define BISHENGIR_DIALECT_HIVMAVE_UTILS_AVELOOPANALYSIS_H

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace hivmave {

enum class LoopAccessContinuity {
  Contiguous,
  NonContiguous,
  Unknown,
};

/// Reusable analysis for memory accesses derived from one scf.for IV.
class AveLoopAnalysis {
public:
  explicit AveLoopAnalysis(scf::ForOp loop);

  /// Return the coefficient of the loop IV in value, or std::nullopt when the
  /// expression cannot be represented as a linear function of the IV.
  std::optional<int64_t> getInductionVarCoefficient(Value value) const;

  /// Return the linearized memref address stride per unit IV increment. Memref
  /// strides and the result are measured in elements.
  std::optional<int64_t> getLinearizedAccessStride(Value base,
                                                   ValueRange indices,
                                                   MemRefType memrefType) const;

  /// Prove whether adjacent loop iterations access adjacent vectors.
  LoopAccessContinuity analyzeAccessContinuity(Value base, ValueRange indices,
                                               MemRefType memrefType,
                                               VectorType vectorType) const;

private:
  std::optional<int64_t> getAffineExprCoefficient(
      AffineExpr expr, ArrayRef<std::optional<int64_t>> dimCoefficients,
      ArrayRef<std::optional<int64_t>> symbolCoefficients) const;

  Operation *loopOp;
  Block *loopBody;
  Value inductionVar;
  Value step;
};

} // namespace hivmave
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVMAVE_UTILS_AVELOOPANALYSIS_H
