//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::scf;

namespace ConditionOpInterfaceForOpReuseInPlanMemory {
AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                    const AnalysisState &state) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op->getParentOp())) {
    auto args = cast<scf::ConditionOp>(op).getArgs();
    auto argIndex = llvm::find(args, opOperand.get()) - args.begin();
    AliasingValueList aliases;
    aliases.addAlias(AliasingValue(whileOp.getAfterArguments()[argIndex],
                                   BufferRelation::Equivalent,
                                   /*isDefinite=*/false));
    return aliases;
  }
  return {};
}
} // namespace ConditionOpInterfaceForOpReuseInPlanMemory

namespace WhileOpInterfaceForOpReuseInPlanMemory {
static BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                     const AnalysisState &state) {
  unsigned int resultNumber = opResult.getResultNumber();
  auto whileOp = cast<scf::WhileOp>(op);

  if (resultNumber >= whileOp.getBeforeArguments().size())
    return BufferRelation::Unknown;
  if (opResult.getType() !=
      whileOp.getBeforeArguments()[resultNumber].getType())
    return BufferRelation::Unknown;

  auto conditionOp = whileOp.getConditionOp();
  BlockArgument beforeBbArg = whileOp.getBeforeArguments()[resultNumber];
  Value conditionOperand = conditionOp.getArgs()[resultNumber];
  auto yieldOp = whileOp.getYieldOp();
  BlockArgument afterBbArg = whileOp.getAfterArguments()[resultNumber];
  Value yieldOperand = yieldOp.getOperand(resultNumber);

  bool equivCondition = false;
  auto &oneShotState = static_cast<const OneShotAnalysisState &>(state);
  oneShotState.applyOnAliases(conditionOperand,
                              [&equivCondition, beforeBbArg](Value alias) {
                                if (alias == beforeBbArg)
                                  equivCondition = true;
                              });
  bool equivYield = false;
  oneShotState.applyOnAliases(yieldOperand,
                              [&equivYield, afterBbArg](Value alias) {
                                if (alias == afterBbArg)
                                  equivYield = true;
                              });
  return equivCondition || equivYield ? BufferRelation::Equivalent
                                      : BufferRelation::Unknown;
}

LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                               const AnalysisState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(op);
  return bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state);
}

AliasingOpOperandList getAliasingOpOperands(Operation *op, Value value,
                                            const AnalysisState &state) {

  if (auto opResult = llvm::dyn_cast<OpResult>(value))
    return bufferization::detail::defaultGetAliasingOpOperands(value, state);
  // value need to be blockArgument.
  auto bbArg = cast<BlockArgument>(value);
  auto whileOp = dyn_cast<scf::WhileOp>(op);
  auto opResult = cast<OpResult>(whileOp.getResult(bbArg.getArgNumber()));
  BufferRelation relation = bufferRelation(op, opResult, state);
  if (bbArg.getOwner() == whileOp.getBeforeBody()) {
    auto conditionOp = whileOp.getConditionOp();
    auto &conditionOperand = conditionOp.getArgsMutable()[bbArg.getArgNumber()];
    return {{&conditionOperand, relation,
             /*isDefinite=*/relation == BufferRelation::Equivalent}};
  } else {
    auto yieldOp = whileOp.getYieldOp();
    auto &yieldOperand = yieldOp->getOpOperand(bbArg.getArgNumber());
    return {{&yieldOperand, relation,
             /*isDefinite=*/relation == BufferRelation::Equivalent}};
  }
}
} // namespace WhileOpInterfaceForOpReuseInPlanMemory

namespace ForOpInterfaceForOpReuseInPlanMemory {
// getOwnerOfValue ensure that value should be opResult or bbArg
AliasingOpOperandList getAliasingOpOperands(Operation *op, Value value,
                                            const AnalysisState &state) {

  if (auto opResult = llvm::dyn_cast<OpResult>(value))
    return bufferization::detail::defaultGetAliasingOpOperands(value, state);
  // value need to be blockArgument.
  auto bbArg = cast<BlockArgument>(value);
  auto forOp = cast<scf::ForOp>(op);
  auto *yieldOpOperand = forOp.getTiedLoopYieldedValue(bbArg);
  return {{yieldOpOperand, BufferRelation::Equivalent}};
}

// We believe there should not be conflict casued by forOp. Conlict in VF should
// be caused by read or write ops.
bool isNotConflicting(Operation *op, OpOperand *uRead, OpOperand *uWrite,
                      const AnalysisState &state) {
  return hivm::isVF(op->getParentOfType<func::FuncOp>());
}
} // namespace ForOpInterfaceForOpReuseInPlanMemory

namespace YieldOpInterfaceForOpReuseInPlanMemory {
bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                             const AnalysisState &state) {
  return isa<scf::IfOp>(op->getParentOp());
}

AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                    const AnalysisState &state) {
  AliasingValueList aliases;
  auto opResult = op->getParentOp()->getResult(opOperand.getOperandNumber());
  if (isa<scf::IfOp>(op->getParentOp())) {
    aliases.addAlias(AliasingValue(opResult, BufferRelation::Equivalent,
                                   /*isDefinite=*/false));
  } else if (isa<scf::ForOp>(op->getParentOp())) {
    auto iterArg = dyn_cast<scf::ForOp>(op->getParentOp())
                       .getTiedLoopRegionIterArg(opResult);
    aliases.addAlias(AliasingValue(iterArg, BufferRelation::Equivalent));
  } else if (isa<scf::WhileOp>(op->getParentOp())) {
    auto whileOp = dyn_cast<scf::WhileOp>(op->getParentOp());
    auto beforeArg = whileOp.getBeforeArguments()[opResult.getResultNumber()];
    aliases.addAlias(AliasingValue(beforeArg, BufferRelation::Equivalent,
                                   /*isDefinite=*/false));
  } else if (isa<scf::ExecuteRegionOp>(op->getParentOp())) {
    aliases.addAlias(AliasingValue(opResult, BufferRelation::Equivalent));
  }
  return aliases;
}

bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                          const AnalysisState &state) {
  return false;
}

/// This is designed to handle cases with sameYieldOperands, like the following:
///
/// scf.if ... {
///     ...
///     scf.yield %1, %2, %3
/// } else {
///     ...
///     scf.yield %4, %4, %4
/// }

/// When analysing the second opOperand of "scf.yield %4, %4, %4", please note
/// that in `wouldCreateReadAfterWriteInterference`, there is no opOperand added
/// to usesWrite through `getAliasingInplaceWrites` becasue yieldOp is not must
/// bufferized in-place. So usesWrite only has one opOperand, which must be the
/// opOperand analysising currently.
/// If uRead is the first opOperand, it must have been analysised before and if
/// it doesn't bufferize in-place, it must bufferize out-of-place. We consider
/// that uRead bufferized out-of-place should not be conflicting with uWrite.

/// Without this interface, all opOperands of "scf.yield %4, %4, %4" will
/// bufferize out-of-place. And now, the last opOperand will bufferize in-place.
bool isNotConflicting(Operation *op, OpOperand *uRead, OpOperand *uWrite,
                      const AnalysisState &state) {
  return uRead->get() == uWrite->get() &&
         uRead->getOwner() == uWrite->getOwner() &&
         uRead->getOperandNumber() < uWrite->getOperandNumber() &&
         !state.isInPlace(*uRead);
}

/// change copy insertion point for preload, like the following:
/// scf.for ... {
///     %0 = scope.scope ... {
///         ...
///         scope.return %2
///     }
///     ...
///     memref.copy %0, %1
///     scf.yield %1
/// }

/// After change insertion point, the IR will be like the following:

/// scf.for ... {
///     %0 = scope.scope ... {
///         ...
///         memref.copy %2, %3
///         scope.return %3
///     }
///     ...
///     scf.yield %0
/// }
LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                               const AnalysisState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(op);
  if (failed(bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state)))
    return failure();
  if (!isa<scf::ForOp>(op->getParentOp()))
    return success();

  OpBuilder::InsertionGuard g(rewriter);
  auto yieldOp = cast<scf::YieldOp>(op);
  for (const auto &pair : llvm::enumerate(yieldOp->getOperands())) {
    auto defOp = pair.value().getDefiningOp();
    if (!isa_and_nonnull<bufferization::AllocTensorOp>(defOp)) {
      continue;
    }
    auto scopeResult = defOp->getOperand(0);
    if (!isa_and_nonnull<scope::ScopeOp>(scopeResult.getDefiningOp()) ||
        !scopeResult.getDefiningOp()->hasAttr(hivm::PreloadNumAttr::name)) {
      continue;
    }
    // Remove old copy out of scopeOp
    rewriter.setInsertionPoint(yieldOp);
    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldOp.setOperand(pair.index(), scopeResult); });
    rewriter.eraseOp(defOp);
    // Add new copy in scopeOp
    auto scopeOp = cast<scope::ScopeOp>(scopeResult.getDefiningOp());
    auto *returnOp = &scopeOp.getBody()->back();
    auto returnOpIdx = cast<OpResult>(scopeResult).getResultNumber();
    auto returnValue = returnOp->getOperand(returnOpIdx);
    rewriter.setInsertionPoint(returnOp);
    FailureOr<Value> alloc = allocateTensorForShapedValue(
        rewriter, returnOp->getLoc(), returnValue, state.getOptions());
    if (failed(alloc))
      return failure();
    rewriter.modifyOpInPlace(
        returnOp, [&]() { returnOp->setOperand(returnOpIdx, *alloc); });
  }
  return success();
}
} // namespace YieldOpInterfaceForOpReuseInPlanMemory

RegisterOpInterfaceOverride(
    /*Op=*/scf::YieldOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/bufferizesToMemoryWrite,
    /*Impl=*/
    &YieldOpInterfaceForOpReuseInPlanMemory::bufferizesToMemoryWrite);

RegisterOpInterfaceOverride(
    /*Op=*/scf::YieldOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/getAliasingValues,
    /*Impl=*/
    &YieldOpInterfaceForOpReuseInPlanMemory::getAliasingValues);

RegisterOpInterfaceOverride(
    /*Op=*/scf::YieldOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/mustBufferizeInPlace,
    /*Impl=*/
    &YieldOpInterfaceForOpReuseInPlanMemory::mustBufferizeInPlace);

RegisterOpInterfaceOverride(
    /*Op=*/scf::YieldOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/isNotConflicting,
    /*Impl=*/
    &YieldOpInterfaceForOpReuseInPlanMemory::isNotConflicting);

RegisterOpInterfaceOverride(
    /*Op=*/scf::YieldOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/resolveConflicts,
    /*Impl=*/
    &YieldOpInterfaceForOpReuseInPlanMemory::resolveConflicts);

RegisterOpInterfaceOverride(
    /*Op=*/scf::ForOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/getAliasingOpOperands,
    /*Impl=*/
    &ForOpInterfaceForOpReuseInPlanMemory::getAliasingOpOperands);

RegisterOpInterfaceOverride(
    /*Op=*/scf::ForOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/isNotConflicting,
    /*Impl=*/
    &ForOpInterfaceForOpReuseInPlanMemory::isNotConflicting);

RegisterOpInterfaceOverride(
    /*Op=*/scf::ConditionOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/getAliasingValues,
    /*Impl=*/
    &ConditionOpInterfaceForOpReuseInPlanMemory::getAliasingValues);

RegisterOpInterfaceOverride(
    /*Op=*/scf::WhileOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/getAliasingOpOperands,
    /*Impl=*/
    &WhileOpInterfaceForOpReuseInPlanMemory::getAliasingOpOperands);

RegisterOpInterfaceOverride(
    /*Op=*/scf::WhileOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/resolveConflicts,
    /*Impl=*/
    &WhileOpInterfaceForOpReuseInPlanMemory::resolveConflicts);

void mlir::scf_ext::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {}
