//===----------------------- SimplifyVFArgs.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_SIMPLIFYVFARGS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static FailureOr<SmallVector<func::CallOp>>
collectDirectCallSites(func::FuncOp funcOp, ModuleOp module) {
  auto symbolUses = funcOp.getSymbolUses(module);
  if (!symbolUses)
    return failure();

  SmallVector<func::CallOp> callSites;
  for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
    auto callOp = dyn_cast<func::CallOp>(symbolUse.getUser());
    if (!callOp)
      return failure();
    callSites.push_back(callOp);
  }
  return callSites;
}

static void simplifyUnusedArguments(func::FuncOp funcOp,
                                    ArrayRef<func::CallOp> callSites) {
  Block &entryBlock = funcOp.getBody().front();
  auto funcType = funcOp.getFunctionType();
  auto funcInputTypes = funcType.getInputs();
  SmallVector<unsigned> unusedArgumentIndices;
  SmallVector<Type> usedFunctionTypeInputs;
  for (BlockArgument blockArg : entryBlock.getArguments()) {
    unsigned argIdx = blockArg.getArgNumber();
    if (!blockArg.use_empty()) {
      usedFunctionTypeInputs.push_back(funcInputTypes[argIdx]);
      continue;
    }
    unusedArgumentIndices.push_back(argIdx);
  }

  if (unusedArgumentIndices.empty())
    return;

  entryBlock.eraseArguments(
      [&](BlockArgument blockArg) { return blockArg.use_empty(); });
  funcOp.setFunctionType(FunctionType::get(
      funcOp.getContext(), usedFunctionTypeInputs, funcType.getResults()));

  for (func::CallOp callOp : callSites) {
    SmallVector<Value> newOperands;
    for (auto [index, operand] : llvm::enumerate(callOp.getOperands())) {
      if (!llvm::is_contained(unusedArgumentIndices, index))
        newOperands.push_back(operand);
    }
    callOp->setOperands(newOperands);
  }
}

struct SimplifyVFArgsPass
    : public impl::SimplifyVFArgsBase<SimplifyVFArgsPass> {
  using SimplifyVFArgsBase<SimplifyVFArgsPass>::SimplifyVFArgsBase;

public:
  void runOnOperation() override;
};
} // namespace

void SimplifyVFArgsPass::runOnOperation() {
  auto mod = getOperation();
  mod->walk([&](func::FuncOp funcOp) {
    if (!hivm::isVF(funcOp) || funcOp.getBody().empty())
      return;

    FailureOr<SmallVector<func::CallOp>> callSites =
        collectDirectCallSites(funcOp, mod);
    if (failed(callSites))
      return;

    simplifyUnusedArguments(funcOp, *callSites);
  });
}

std::unique_ptr<Pass> hfusion::createSimplifyVFArgsPass() {
  return std::make_unique<SimplifyVFArgsPass>();
}
