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
  /// Same callOp and different value are the preconditions to have no
  /// conflict. Because OpOperand can be read and written both.
  ///
  /// case:
  ///
  /// @test_vf(%arg: tensor<16xf32>) -> tensor<16xf32> {
  ///   %0 = read %arg : tensor<16xf32>
  ///   %1 = vadd %0, %cst
  ///   %2 = write %1 to %arg : tensor<16xf32>
  ///   return %2 : tensor<16xf32>
  ///  }

  /// @AIV (%arg0 = memref<16xf32>) {
  ///   %1 = vbrc %0 : tensor<16xf32>
  ///   for
  ///     %2 = call @test_vf(%1) {hivm.vector_function, no_inline} :
  ///     store %2 to %arg0 : memref<16xf32>
  ///   endfor
  ///  }

  /// def:vbrcOp, uRead:callOp OpOperand #0(%1), uWrite:callOp OpOperand #0(%1).
  /// In this case, we expect a RaW conflict because def is out of loop.
  /// But wouldCreateReadAfterWriteInterference will return noConflict. So we
  /// add extra check to make sure the uRead and uWrite are not the same value.

  if (uRead->getOwner() != uWrite->getOwner() || uRead->get() == uWrite->get())
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