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
#include <cassert>

namespace mlir {
#define GEN_PASS_DEF_SIMPLIFYVFARGS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

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
    if (hivm::isVF(funcOp)) {
      SmallVector<int> unusedArgumentInd;
      auto funcType = funcOp.getFunctionType();
      Block &entryBlock = funcOp.getBody().front();
      auto funcInputTypes = funcType.getInputs();
      SmallVector<Type> usedFunctionTypeInputs;
      for (BlockArgument blockArg : entryBlock.getArguments()) {
        int argIdx = blockArg.getArgNumber();
        if (!blockArg.use_empty()) {
          usedFunctionTypeInputs.push_back(funcInputTypes[argIdx]);
          continue;
        }
        unusedArgumentInd.push_back(argIdx);
      }

      if (unusedArgumentInd.empty()) {
        return;
      }

      entryBlock.eraseArguments(
          [&](BlockArgument bArg) { return bArg.use_empty(); });

      auto callSites = funcOp.getSymbolUses(mod);

      assert(callSites.has_value() && "A VF must at least be invoked once");
      FunctionType allUsedFunctionType = FunctionType::get(
          funcOp.getContext(), usedFunctionTypeInputs, funcType.getResults());
      funcOp.setFunctionType(allUsedFunctionType);

      for (SymbolTable::SymbolUse callSite : callSites.value()) {
        func::CallOp call = cast<func::CallOp>(callSite.getUser());
        SmallVector<Value> operands = call->getOperands();
        SmallVector<Value> newOperands;
        for (size_t i = 0; i < operands.size(); ++i) {
          if (!llvm::is_contained(unusedArgumentInd, i)) {
            newOperands.push_back(operands[i]);
          }
        }
        call->setOperands(newOperands);
      }
    }
  });
}

std::unique_ptr<Pass> hfusion::createSimplifyVFArgsPass() {
  return std::make_unique<SimplifyVFArgsPass>();
}
