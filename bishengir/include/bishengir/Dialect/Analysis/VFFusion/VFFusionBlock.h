//===- VFFusionBlock.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_BLOCK_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_BLOCK_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace analysis {

/// Represents a block of operations that can be fused together in vector
/// flow analysis.
///
/// This class manages a collection of operations along with their input and
/// output values, supporting fusion of operations in topological order. It only
/// stores first-layer operations (operations with regions are not recursively
/// stored), but tracks all values including those from nested regions.
///
/// @note For performance optimization, getInputs/Outputs can be improved by
///       considering the specific case of outlining two consecutive blocks.
/// @note In order to avoid dependency issue, Operations should always be fused
///       in topological order.
class VFFusionBlock {
public:
  VFFusionBlock() = default;
  SmallVector<Operation *> getOps() const;

  SetVector<Value> recomputeInputs();
  SetVector<Value> recomputeOutputs();

  SetVector<Value> getInputs() const;
  SetVector<Value> getOutputs() const;

  void fuseOp(Operation *op);

private:
  // Only store operation directly after a function
  SetVector<Operation *> ops;

  // These consider all operations including the operations inside regions.
  SetVector<Value> inputs;
  SetVector<Value> outputs;

  bool hasFusedUser(Operation *op) const;
};

} // namespace analysis
} // namespace mlir
#endif