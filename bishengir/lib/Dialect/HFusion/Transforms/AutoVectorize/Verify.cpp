//===- Verify.cpp - Auto vectorization verification helpers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Verify.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

bool hasVectorOperandOrResult(Operation *op) {
  auto isVectorType = [](Type type) { return isa<VectorType>(type); };
  return llvm::any_of(op->getOperandTypes(), isVectorType) ||
         llvm::any_of(op->getResultTypes(), isVectorType);
}

bool hasAutoVectorizeTargetAttr(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().starts_with("hfusion-auto-vectorize-target-"))
      return true;
  }
  return false;
}

bool isAutoVectorizeTargetOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  return hasAutoVectorizeTargetAttr(op) &&
         (isa<linalg::LinalgOp>(op) ||
          (dialect && isa<hfusion::HFusionDialect>(dialect)));
}

bool hasOutlinedLoopTargetAttr(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().starts_with("outlined-loop-target-"))
      return true;
  }
  return false;
}

bool isNestedInOutlinedLoopTarget(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (hasOutlinedLoopTargetAttr(current))
      return true;
  }
  return false;
}

WalkResult emitVerifierError(Operation *op, StringRef message,
                             bool emitDiagnostics) {
  if (emitDiagnostics)
    op->emitError(message);
  return WalkResult::interrupt();
}

LogicalResult runFreeVectorRegionCheck(Operation *root, bool emitDiagnostics) {
  WalkResult result = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool isVectorOp = hasVectorOperandOrResult(op);
    bool isTargetOp = isAutoVectorizeTargetOp(op);
    if (!isVectorOp && !isTargetOp)
      return WalkResult::advance();

    if (isNestedInOutlinedLoopTarget(op))
      return WalkResult::advance();

    if (isVectorOp) {
      return emitVerifierError(
          op, "unexpected vector operation outside outlined vector region",
          emitDiagnostics);
    }
    return emitVerifierError(
        op, "unexpected hfusion/linalg operation outside outlined vector region",
        emitDiagnostics);
  });

  return failure(result.wasInterrupted());
}

LogicalResult runFreeVectorFuncCheck(Operation *root, bool emitDiagnostics) {
  WalkResult result = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      if (hivm::isVF(func))
        return WalkResult::skip();
      return WalkResult::advance();
    }

    if (!hasVectorOperandOrResult(op))
      return WalkResult::advance();

    return emitVerifierError(
        op, "unexpected vector operation outside vector function",
        emitDiagnostics);
  });

  return failure(result.wasInterrupted());
}

} // namespace

LogicalResult hfusion::AutoVectorizeVerifier::check(Operation *root) const {
  if (enabledStrategies.verifyFreeVectorRegion &&
      failed(runFreeVectorRegionCheck(root, emitDiagnosticsFlag)))
    return failure();

  if (enabledStrategies.verifyFreeVectorFunc &&
      failed(runFreeVectorFuncCheck(root, emitDiagnosticsFlag)))
    return failure();

  return success();
}
