//===- SplitKernelUtils.cpp - MIX split helpers ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/SplitKernelUtils.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::hivm;

namespace {

FailureOr<bool> hasRequiredCoreWork(func::FuncOp func,
                                    hivm::TCoreType requiredCore) {
  bool foundWork = false;

  WalkResult result = func.walk([&](Operation *op) {
    if (op == func.getOperation())
      return WalkResult::advance();

    FailureOr<hivm::TCoreType> coreType = hivm::getCoreType(op);
    if (failed(coreType))
      return WalkResult::interrupt();

    if (*coreType == requiredCore ||
        *coreType == hivm::TCoreType::CUBE_AND_VECTOR) {
      foundWork = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted() && !foundWork)
    return failure();

  return foundWork;
}

LogicalResult generateMixKernelDecl(func::FuncOp funcOp) {
  auto module = funcOp->getParentOfType<ModuleOp>();
  std::optional<SymbolTable::UseRange> maybeUses = funcOp.getSymbolUses(module);
  if (!maybeUses)
    return funcOp.emitError("cannot determine MIX kernel symbol uses");
  if (maybeUses->empty())
    return success();

  if (!llvm::all_of(*maybeUses, [&](const SymbolTable::SymbolUse &use) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          return false;
        auto caller = call->getParentOfType<hacc::HACCFunction>();
        return caller && caller.isHost();
      }))
    return funcOp.emitError(
        "Currently, MIX kernels can only be called by host functions!");

  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  auto funcDeclOp = builder.create<func::FuncOp>(
      funcOp->getLoc(), funcOp.getSymName(), funcOp.getFunctionType());
  funcDeclOp.setPrivate();
  funcDeclOp->setAttr(hacc::HACCFuncTypeAttr::name,
                      hacc::HACCFuncTypeAttr::get(funcOp.getContext(),
                                                  hacc::HACCFuncType::DEVICE));
  funcDeclOp->setAttr(hivm::TFuncCoreTypeAttr::name,
                      hivm::TFuncCoreTypeAttr::get(funcOp.getContext(),
                                                   hivm::TFuncCoreType::MIX));
  if (auto coreRatio = hivm::getCoreRatioAttr(funcOp))
    funcDeclOp->setAttr(hivm::TCoreRatioAttr::name, coreRatio);
  if (hacc::utils::isDeviceEntry(funcOp))
    funcDeclOp->setAttr(
        hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::MIX_ENTRY),
        builder.getUnitAttr());
  return success();
}

} // namespace

LogicalResult mlir::hivm::verifyCoreRatio(func::FuncOp func) {
  auto ratio = hivm::getCoreRatioAttr(func);
  auto coreType = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  if (!ratio || !coreType)
    return success();

  TFuncCoreType funcCoreType = coreType.getFuncCoreType();
  if ((funcCoreType == TFuncCoreType::MIX ||
       funcCoreType == TFuncCoreType::AIC) &&
      ratio.getCube() == 0) {
    auto hasCube = hasRequiredCoreWork(func, hivm::TCoreType::CUBE);
    if (failed(hasCube)) {
      return failure();
    }
    if (*hasCube) {
      return func->emitError()
             << "hivm.core_ratio<0, " << ratio.getVector()
             << "> reserves no cube core, but the kernel has cube work";
    }
  }

  if ((funcCoreType == TFuncCoreType::MIX ||
       funcCoreType == TFuncCoreType::AIV) &&
      ratio.getVector() == 0) {
    auto hasVec = hasRequiredCoreWork(func, hivm::TCoreType::VECTOR);
    if (failed(hasVec)) {
      return failure();
    }
    if (*hasVec) {
      return func->emitError()
             << "hivm.core_ratio<" << ratio.getCube()
             << ", 0> reserves no vector core, but the kernel has vector work";
    }
  }

  return success();
}

bool mlir::hivm::isSingleCoreRatio(hivm::TCoreRatioAttr ratio) {
  return ratio && ((ratio.getCube() == 0 && ratio.getVector() == 1) ||
                   (ratio.getCube() == 1 && ratio.getVector() == 0));
}

FailureOr<std::pair<func::FuncOp, func::FuncOp>>
mlir::hivm::createSplitMixFunctions(func::FuncOp func) {
  StringRef funcName = func.getSymName();
  auto module = func->getParentOfType<ModuleOp>();
  for (StringRef suffix : {kMixFuncAicSuffix, kMixFuncAivSuffix})
    if (module.lookupSymbol((funcName + suffix).str())) {
      func.emitError("split kernel symbol already exists: ")
          << funcName << suffix;
      return failure();
    }
  if (failed(generateMixKernelDecl(func)))
    return failure();

  OpBuilder builder(func);
  builder.setInsertionPointAfter(func);
  auto vecFunc = cast<func::FuncOp>(builder.clone(*func));

  func.setSymName((funcName + kMixFuncAicSuffix).str());
  vecFunc.setSymName((funcName + kMixFuncAivSuffix).str());
  for (auto [half, coreType] : {std::pair{func, TFuncCoreType::AIC},
                                std::pair{vecFunc, TFuncCoreType::AIV}}) {
    half->setAttr(TPartOfMixAttr::name, builder.getUnitAttr());
    half->setAttr(TFuncCoreTypeAttr::name,
                  TFuncCoreTypeAttr::get(func.getContext(), coreType));
  }
  return std::make_pair(func, vecFunc);
}
