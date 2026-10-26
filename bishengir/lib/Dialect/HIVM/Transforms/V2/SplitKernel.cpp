//===- V2SplitKernel.cpp - MIX split -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memref-IR split: clone the MIX into AIC/AIV and drop the other core's ops.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/SplitKernelUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <optional>

namespace mlir {
#define GEN_PASS_DEF_V2SPLITKERNEL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-v2-split-kernel"

using namespace mlir;

namespace {

/// Erase ops owned by `filteredCore`; fail on filtered ops with live results.
static LogicalResult filterCoreOps(func::FuncOp func,
                                   hivm::TCoreType filteredCore) {
  WalkResult result = func.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op == func.getOperation())
      return WalkResult::advance();
    if (op->hasTrait<OpTrait::IsTerminator>())
      return WalkResult::advance();

    auto coreType = hivm::getCoreType(op);
    if (failed(coreType) || *coreType != filteredCore)
      return WalkResult::advance();

    if (op->getNumResults() != 0) {
      op->emitError()
          << "hivm-v2-split-kernel cannot erase a filtered op with results "
             "on memref IR";
      return WalkResult::interrupt();
    }

    op->erase();
    return WalkResult::skip();
  });
  return success(!result.wasInterrupted());
}

static std::optional<hivm::TFuncCoreType> getFuncCoreType(func::FuncOp func) {
  auto attr = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  if (!attr)
    return std::nullopt;
  return attr.getFuncCoreType();
}

static bool isMixFunction(func::FuncOp func) {
  if (hacc::utils::isHost(func) || func.isExternal())
    return false;
  auto coreType = getFuncCoreType(func);
  return coreType && *coreType == hivm::TFuncCoreType::MIX &&
         !func->hasAttr(hivm::TPartOfMixAttr::name);
}

static LogicalResult splitMixKernel(func::FuncOp func) {
  if (!isMixFunction(func))
    return success();

  StringRef funcName = func.getSymName();
  if (failed(hivm::generateMixKernelDecl(func)))
    return failure();

  OpBuilder builder(func);
  builder.setInsertionPointAfter(func.getOperation());
  auto vecFunc = cast<func::FuncOp>(builder.clone(*func.getOperation()));

  func.setSymNameAttr(
      builder.getStringAttr((Twine(funcName) + hivm::kMixFuncAicSuffix).str()));
  vecFunc.setSymNameAttr(
      builder.getStringAttr((Twine(funcName) + hivm::kMixFuncAivSuffix).str()));

  func->setAttr(hivm::TPartOfMixAttr::name, builder.getUnitAttr());
  vecFunc->setAttr(hivm::TPartOfMixAttr::name, builder.getUnitAttr());

  func->setAttr(hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(func->getContext(),
                                             hivm::TFuncCoreType::AIC));
  vecFunc->setAttr(hivm::TFuncCoreTypeAttr::name,
                   hivm::TFuncCoreTypeAttr::get(vecFunc->getContext(),
                                                hivm::TFuncCoreType::AIV));

  if (failed(filterCoreOps(func, hivm::TCoreType::VECTOR)))
    return failure();
  if (failed(filterCoreOps(vecFunc, hivm::TCoreType::CUBE)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[hivm-v2-split-kernel] AIC:\n"
                          << func << "\nAIV:\n"
                          << vecFunc << "\n");
  return success();
}

struct V2SplitKernelPass : public impl::V2SplitKernelBase<V2SplitKernelPass> {
  using Base::Base;
  void runOnOperation() override {
    if (ghost)
      return;

    SmallVector<func::FuncOp> mixFunctions;
    getOperation().walk([&](func::FuncOp func) {
      if (isMixFunction(func))
        mixFunctions.push_back(func);
    });

    for (func::FuncOp func : mixFunctions)
      if (failed(splitMixKernel(func)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::hivm::createV2SplitKernelPass(const V2SplitKernelOptions &options) {
  return std::make_unique<V2SplitKernelPass>(options);
}
