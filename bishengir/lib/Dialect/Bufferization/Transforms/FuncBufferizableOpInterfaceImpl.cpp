//===- FuncBufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::bufferization::func_ext;
using namespace mlir::func;

static FuncOp getCalledFunction(func::CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  if (!isa<OneShotAnalysisState>(state))
    return FuncOpAnalysisState::NotAnalyzed;
  auto *funcState = static_cast<const OneShotAnalysisState &>(state)
                        .getExtension<FuncAnalysisState>();
  if (!funcState)
    return FuncOpAnalysisState::NotAnalyzed;
  const auto &analyzedFuncOps = funcState->analyzedFuncOps;
  auto it = analyzedFuncOps.find(funcOp);
  if (it == analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

namespace CallOpInterfaceForOpReuseInPlanMemory {
static bool isNotConflicting(Operation *op, OpOperand *uRead, OpOperand *uWrite,
                             const AnalysisState &state) {
  if (uRead->getOwner() != uWrite->getOwner())
    return false;

  func::CallOp callOp = cast<func::CallOp>(op);
  FuncOp funcOp = getCalledFunction(callOp);
  if (!funcOp)
    return false;

  if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
    return false;

  auto readArg = funcOp.getArgument(uRead->getOperandNumber());
  auto writeArg = funcOp.getArgument(uWrite->getOperandNumber());
  DominanceInfo domInfo(funcOp);
  return !wouldCreateReadAfterWriteInterference(
      readArg, writeArg, domInfo,
      const_cast<OneShotAnalysisState &>(
          static_cast<const OneShotAnalysisState &>(state)));
}
} // namespace CallOpInterfaceForOpReuseInPlanMemory

RegisterOpInterfaceOverride(
    /*Op=*/func::CallOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/isNotConflicting,
    /*Impl=*/
    &CallOpInterfaceForOpReuseInPlanMemory::isNotConflicting);

void mlir::bufferization_ext::registerFuncBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {}