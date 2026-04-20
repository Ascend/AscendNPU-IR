//===- MaterializeSimtVFMemScope.cpp - Materialize SIMT VF Mem Scopes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InferHIVMMemScope.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
#define GEN_PASS_DEF_MATERIALIZESIMTVFMEMSCOPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

AddressSpaceAttr getAddressSpaceAttr(MLIRContext *ctx, hivm::AddressSpace space) {
  return AddressSpaceAttr::get(ctx, space);
}

std::optional<AddressSpaceAttr> getSimtMemScopeHint(func::FuncOp func,
                                                    unsigned argIdx) {
  auto hint = func.getArgAttrOfType<SimtMemScopeHintAttr>(
      argIdx, SimtMemScopeHintAttr::name);
  if (!hint)
    return std::nullopt;
  return getAddressSpaceAttr(func.getContext(), hint.getAddressSpace());
}

LogicalResult updateFuncTypeFromBody(func::FuncOp func) {
  if (func.isExternal())
    return success();
  auto newFuncType = func.getFunctionType().clone(
      func.front().getArgumentTypes(), func.getResultTypes());
  func.setFunctionType(newFuncType);
  return success();
}

struct MaterializeSimtVFMemScopePass
    : public impl::MaterializeSimtVFMemScopeBase<
          MaterializeSimtVFMemScopePass> {
  void runOnOperation() override;

private:
  LogicalResult materializeFunc(func::FuncOp func);
};

} // namespace

LogicalResult MaterializeSimtVFMemScopePass::materializeFunc(func::FuncOp func) {
  MemScopeInferAndPropagateHelper helper;
  // Split SIMT modules materialize the previously inferred hints into explicit
  // address spaces right before TTIR lowering.
  for (BlockArgument arg : func.getArguments()) {
    if (!isa<BaseMemRefType>(arg.getType()))
      continue;

    auto hint = getSimtMemScopeHint(func, arg.getArgNumber());
    if (!hint.has_value()) {
      return func.emitOpError()
             << "missing simt memory scope hint for argument #"
             << arg.getArgNumber();
    }
    if (failed(helper.Run(arg, hint.value()))) {
      return func.emitOpError()
             << "failed to materialize memory scope for simt argument #"
             << arg.getArgNumber();
    }
  }

  if (failed(updateFuncTypeFromBody(func)))
    return failure();

  // After argument types are fixed, propagate any explicit GM semantics carried
  // by pointer casts inside the outlined SIMT function.
  bool hasFailure = false;
  func.walk([&](hivm::PointerCastOp op) {
    if (failed(hivm::inferAndPropagateMemScopeForPointerCast(op)))
      hasFailure = true;
  });
  if (hasFailure)
    return failure();

  auto funcCoreType = queryFuncCoreType(func);
  hivm::AddressSpace defaultAllocSpace = hivm::AddressSpace::UB;
  if (funcCoreType.has_value() &&
      funcCoreType.value() == TFuncCoreType::AIC) {
    defaultAllocSpace = hivm::AddressSpace::L1;
  }

  // Remaining local allocs inherit the default memory space of the SIMT core.
  hasFailure = false;
  func.walk([&](memref::AllocOp op) {
    if (failed(hivm::inferAndPropagateMemScopeForAlloc(op, defaultAllocSpace)))
      hasFailure = true;
  });
  if (hasFailure)
    return failure();
  return success();
}

void MaterializeSimtVFMemScopePass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<func::FuncOp> simtFuncs;
  module.walk([&](func::FuncOp func) {
    if (util::isSIMTVF(func))
      simtFuncs.push_back(func);
  });

  for (func::FuncOp func : simtFuncs) {
    if (failed(materializeFunc(func))) {
      signalPassFailure();
      return;
    }
  }
}

std::unique_ptr<Pass> mlir::hivm::createMaterializeSimtVFMemScopePass() {
  return std::make_unique<MaterializeSimtVFMemScopePass>();
}
