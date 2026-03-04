//===- ReshapeAnalyzer.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_RESHAPE_ANALYZER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_RESHAPE_ANALYZER_H

namespace mlir {
namespace hfusion {
namespace detail {

struct ReshapeValue {
  explicit ReshapeValue(Value sourceParent, OpOperand &endTarget, int d)
      : source(sourceParent), endTarget(&endTarget), depth(d){};
  Value source;
  OpOperand *endTarget;
  int depth;
};

/// Will run an analysis on the func graph
class ReshapeAnalyzer {
public:
  ReshapeAnalyzer(func::FuncOp funcOp);

  /// Builds a chain of reshape operations starting from a given value.
  /// Traces upward through expand/collapse operations to find the original
  /// value.
  ///
  /// @param val The value to start tracing from
  /// @return A vector containing the reshape chain, with the unreshaped value
  /// at the back, and reshaped at the front
  SmallVector<Value> getReshapeChain(Value val);

  /// @param chain A vector of Value objects to analyze
  /// @return A vector of Operation pointers corresponding to reshape operations
  /// in the chain
  ///
  /// Iterates through the provided vector of values and collects the operations
  /// that define these values, but only if they are reshape operations.
  /// Non-reshape operations are skipped.
  static SmallVector<Operation *>
  getOpsFromReshapeValue(SmallVector<Value> chain);

  /// Gets the original value (reshape.end()) from the beginning of a reshape
  /// chain.
  ///
  /// @param val The value to find the reshape head for
  /// @return The original value before any reshape operations
  Value getReshapeHead(Value val);

  /// Gets the first reshape appearing from the original value
  ///
  /// @param val The value to find the first reshape from
  /// @return The first reshape
  Value getFirstReshape(Value val);

  /// Gets the first reshape appearing from the chain
  ///
  /// @param reshapeChain The chain to find the first reshape from
  /// @return The first reshape
  Value getFirstReshape(SmallVector<Value> &reshapeChain);

  /// Collects all operations that use the results of reshape operations
  /// starting from a value.
  /// Uses breadth-first search to find all descendants of reshape operations.
  ///
  /// @param val The starting value
  /// @param descendants Vector to store the results in
  void getReshapeDescendants(Value val, SmallVector<ReshapeValue> &descendants);

  /// Overloaded version of getReshapeDescendants that stores results in a set.
  ///
  /// @param val The starting value
  /// @param descendants SetVector to store the results in
  void getReshapeDescendants(Value val, SetVector<Value> &descendants);

  /// Traces through a chain of reshape operations and creates inverse
  /// operations for them. This allows transforming the IR by inserting inverse
  /// operations where needed.
  ///
  /// @param rewriter The pattern rewriter to use for creating operations
  /// @param outValue The operand to start tracing from
  /// @return The updated operand after creating inverse operations
  static OpOperand *traceReshapeAndRewriteInverse(PatternRewriter &rewriter,
                                                  OpOperand *outValue);

protected:
  /// Computes and tracks reshape operations that originate from function
  /// arguments. Builds a mapping between function arguments and their reshaped
  /// descendants.
  void computeReshapeInputs();

  /// Computes and tracks the original values (unreshaped values) that are
  /// returned from the function.
  /// Maps return values to their reshape head values.
  void computeUnreshapedOutputs();

  DenseMap<Value, int64_t> valueDependency;

  SmallVector<DenseSet<Value>> argIdxToReshapedInput;
  DenseMap<Value, int64_t> reshapedInputToArgIdx;
  DenseSet<Value> reshapedInputs;

  DenseSet<Value> unreshapedOutputs;
  DenseMap<Value, int64_t> unreshapedOutputToRetIdx;
  SmallVector<Value> retIdxToUnreshapedOutputs;
  func::FuncOp func;
};

} // namespace detail
} // namespace hfusion
} // namespace mlir
#endif