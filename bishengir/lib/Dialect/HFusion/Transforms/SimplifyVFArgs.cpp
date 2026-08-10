//===----------------------- SimplifyVFArgs.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

static bool isSupportedFP8Type(Type type) {
  return type.isFloat8E4M3FN() || type.isFloat8E5M2();
}

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

// CCEC accepts FP8 SSA values, but it cannot parse a typed FP8 literal at a
// call boundary. LLVM constant folding can turn a scalar FP8 argument into
// exactly such a literal. Keep the numerical conversion at the caller, carry
// its raw eight-bit payload through the internal VF ABI, and restore the FP8
// type inside the callee.
static void legalizeFP8ScalarABI(func::FuncOp funcOp,
                                 ArrayRef<func::CallOp> callSites) {
  if (callSites.empty())
    return;

  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> inputTypes(funcType.getInputs());
  SmallVector<std::pair<unsigned, Type>> fp8Arguments;
  for (auto [index, type] : llvm::enumerate(inputTypes)) {
    if (isSupportedFP8Type(type))
      fp8Arguments.emplace_back(index, type);
  }
  if (fp8Arguments.empty())
    return;

  for (func::CallOp callOp : callSites) {
    if (callOp.getNumOperands() != inputTypes.size())
      return;
    for (auto [index, fp8Type] : fp8Arguments) {
      if (callOp.getOperand(index).getType() != fp8Type)
        return;
    }
  }

  Type i8Type = IntegerType::get(funcOp.getContext(), 8);
  for (func::CallOp callOp : callSites) {
    OpBuilder builder(callOp);
    for (auto [index, fp8Type] : fp8Arguments) {
      Value payload = builder.create<arith::BitcastOp>(
          callOp.getLoc(), i8Type, callOp.getOperand(index));
      callOp->setOperand(index, payload);
    }
  }

  for (auto [index, fp8Type] : fp8Arguments)
    inputTypes[index] = i8Type;
  funcOp.setFunctionType(FunctionType::get(funcOp.getContext(), inputTypes,
                                           funcType.getResults()));

  Block &entryBlock = funcOp.getBody().front();
  OpBuilder builder = OpBuilder::atBlockBegin(&entryBlock);
  for (auto [index, fp8Type] : fp8Arguments) {
    BlockArgument payload = entryBlock.getArgument(index);
    payload.setType(i8Type);
    Value fp8Value =
        builder.create<arith::BitcastOp>(payload.getLoc(), fp8Type, payload);
    payload.replaceAllUsesExcept(fp8Value, fp8Value.getDefiningOp());
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
    legalizeFP8ScalarABI(funcOp, *callSites);
  });
}

std::unique_ptr<Pass> hfusion::createSimplifyVFArgsPass() {
  return std::make_unique<SimplifyVFArgsPass>();
}
