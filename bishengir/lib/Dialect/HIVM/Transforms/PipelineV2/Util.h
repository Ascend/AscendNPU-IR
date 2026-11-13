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

/// Private clone of a function taken before OutlineCVScopes. The value is a
/// symbol reference to the function this snapshot restores.
constexpr llvm::StringLiteral kCVPipelineSnapshotAttrName =
    "hivm.cv_pipeline_snapshot";

inline bool isCVPipelineSnapshot(Operation *op) {
  return op && op->hasAttr(kCVPipelineSnapshotAttrName);
}

/// Return true if some op on its ancestor chain carries LegacyCVPipelineAttr.
bool isPipelineReject(Operation *op);

/// Return true if `op` is a core HIVM DPS operation.
bool isCoreOp(Operation *op);

/// Return `TCoreType::CUBE` or `TCoreType::VECTOR` if `op` has a concrete core
/// type, or `std::nullopt` otherwise.
std::optional<TCoreType> queryConcreteCoreType(Operation *op);

/// Return the `hivm.loop_core_type` attribute value of `scope` if present.
std::optional<TCoreType> getScopeCoreType(ScopeOp scope);

/// Trace `v` backward through view-like, reshape, cast, bufferization, and loop
/// iter_arg operations to its underlying defining value.
Value traceValueDef(Value v);

/// Trace `v` backward to its underlying `memref.alloc` operation if present.
memref::AllocOp traceAlloc(Value v);

/// Return true if `copy` is an `hivm.hir.copy` writing to an L1 (`cbuf`) alloc.
bool isCrossCoreCopy(Operation *copy);

/// Return true if `ct1` and `ct2` can be merged into a single core scope
/// (`CUBE` and `VECTOR` cannot be merged together; `CUBE_AND_VECTOR` and
/// `CUBE_OR_VECTOR` are compatible with any core type).
bool areCompatibleCoreTypes(TCoreType ct1, TCoreType ct2);

/// Resolve the resulting core type when merging compatible `ct1` and `ct2`,
/// or return `failure()` if they are incompatible.
FailureOr<TCoreType> resolveMergedCoreType(TCoreType ct1, TCoreType ct2);

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

/// Merge `scope1` and `scope2` (which must reside in the same parent block)
/// into a new `ScopeOp` inserted before `at`, ordering operations by
/// dominance, merging `scf.if` operations that share the same
/// `pipeline.if.cond` ID, and pruning scope results that become unused.
ScopeOp mergeScopes(ScopeOp scope1, ScopeOp scope2, Operation *at);

/// Write a Graphviz DOT file depicting the dependence graph between scopes
/// (including nested scopes and loop-carried dependencies) in `forOp`, modeled
/// after LLVM's `writeCFGToDotFile` (`-dot-cfg`). If `filename` is empty,
/// defaults to `[<prefix>.]<func>[.<idx>].dot`.
void writeScopeDependenceGraphToDotFile(scf::ForOp forOp, StringRef prefix = "",
                                        StringRef filename = "",
                                        bool shortNames = false);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PIPELINEV2_UTIL_H
