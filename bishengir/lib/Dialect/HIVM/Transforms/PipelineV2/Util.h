//===- Util.h - Common utilities for CV Pipelining V2 -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PIPELINEV2_UTIL_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PIPELINEV2_UTIL_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace hivm {

using scope::ScopeOp;

constexpr llvm::StringLiteral kPipelineIfCondAttrName = "pipeline.if.cond";

/// Return true if `op` is a core HIVM DPS operation.
bool isCoreOp(Operation *op);

/// Return `TCoreType::CUBE` or `TCoreType::VECTOR` if `op` has a concrete core
/// type, or `std::nullopt` otherwise.
std::optional<TCoreType> queryConcreteCoreType(Operation *op);

/// Return the `{thenVal, elseVal}` yielded for `resIdx` of `ifOp`.
std::pair<Value, Value> getIfYieldedValues(scf::IfOp ifOp, unsigned resIdx);

/// Return `elseVal` if non-null, or materialize a default zero/empty value
/// matching `thenVal.getType()`.
Value getOrCreateElseYieldValue(OpBuilder &builder, Location loc, Value thenVal,
                                Value elseVal);

/// Recreate `srcScope` with `returnOperands` as its `scope.return` operands and
/// result types, moving `srcScope`'s body into the new `ScopeOp`.
ScopeOp recreateScope(ScopeOp srcScope, ValueRange returnOperands,
                      Operation *insertBefore = nullptr);

/// Move `op` (and its non-CoreOp operand dependencies within the same scope if
/// `op` is a CoreOp) into `target`, updating the original scope's return values
/// and results accordingly.
///
/// NOTE: This operation is expensive as it traces dependencies and recreates
/// the source `scope.scope` whenever its return signature changes. Prefer
/// merging scopes whenever possible.
void moveOpToScope(Operation *op, ScopeOp target, bool insertAtEnd = false);

/// Merge `scope1` and `scope2` into a new `ScopeOp` inserted before `at`,
/// ordering operations by dominance, merging `scf.if` operations that share the
/// same `pipeline.if.cond` ID, and pruning scope results that become unused.
ScopeOp mergeScopes(ScopeOp scope1, ScopeOp scope2, Operation *at);

/// Write a Graphviz DOT file depicting the dependence graph between scopes in
/// `forOp` (including loop-carried dependencies), modeled after LLVM's
/// `writeCFGToDotFile` (`-dot-cfg`). If `filename` is empty, defaults to
/// `scope-dep.<func>.dot`.
void writeScopeDependenceGraphToDotFile(scf::ForOp forOp,
                                        StringRef filename = "",
                                        bool shortNames = false);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PIPELINEV2_UTIL_H
