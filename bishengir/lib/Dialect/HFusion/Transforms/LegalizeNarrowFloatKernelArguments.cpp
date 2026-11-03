//===- LegalizeNarrowFloatKernelArguments.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace {

bool isLauncherEntryCall(func::CallOp call) {
  auto module = call->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
  return callee && (hacc::isEntry(callee) || hacc::isMixEntry(callee));
}

void extendNarrowFloatCallArguments(func::FuncOp func,
                                    ArrayRef<unsigned> narrowFloatArgs) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (auto symbolUses = func.getSymbolUses(module)) {
    for (SymbolTable::SymbolUse use : *symbolUses) {
      auto call = dyn_cast<func::CallOp>(use.getUser());
      if (!call)
        continue;
      for (unsigned index : narrowFloatArgs) {
        Value arg = call.getOperand(index);
        if (!isa<Float16Type, BFloat16Type>(arg.getType()) ||
            isLauncherEntryCall(call))
          continue;
        OpBuilder callBuilder(call);
        call.setOperand(index,
                        callBuilder.create<arith::ExtFOp>(
                            call.getLoc(), callBuilder.getF32Type(), arg));
      }
    }
  }
}

void narrowFunctionArguments(func::FuncOp func, FunctionType oldType,
                             ArrayRef<unsigned> narrowFloatArgs) {
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(&func.getBody().front());
  for (unsigned index : narrowFloatArgs) {
    BlockArgument arg = func.getArgument(index);
    arg.setType(builder.getF32Type());
    auto narrow = builder.create<arith::TruncFOp>(func.getLoc(),
                                                  oldType.getInput(index), arg);

    SmallVector<OpOperand *> uses;
    for (OpOperand &use : arg.getUses())
      uses.push_back(&use);
    for (OpOperand *use : uses) {
      if (use->getOwner() == narrow.getOperation())
        continue;
      if (auto call = dyn_cast<func::CallOp>(use->getOwner())) {
        if (isLauncherEntryCall(call))
          continue;
      }
      use->set(narrow.getResult());
    }
  }
}

void legalizeNarrowFloatKernelArgumentsImpl(func::FuncOp func) {
  if (!hacc::isEntry(func) && !hacc::isMixEntry(func))
    return;

  SmallVector<unsigned> narrowFloatArgs;
  FunctionType oldType = func.getFunctionType();
  for (auto [index, type] : llvm::enumerate(oldType.getInputs()))
    if (isa<Float16Type, BFloat16Type>(type))
      narrowFloatArgs.push_back(index);
  if (narrowFloatArgs.empty())
    return;

  extendNarrowFloatCallArguments(func, narrowFloatArgs);

  SmallVector<Type> inputTypes(oldType.getInputs());
  for (unsigned index : narrowFloatArgs)
    inputTypes[index] = Float32Type::get(func.getContext());
  func.setFunctionType(
      FunctionType::get(func.getContext(), inputTypes, oldType.getResults()));

  if (func.isDeclaration())
    return;

  narrowFunctionArguments(func, oldType, narrowFloatArgs);
}

class LegalizeNarrowFloatKernelArguments
    : public PassWrapper<LegalizeNarrowFloatKernelArguments,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LegalizeNarrowFloatKernelArguments)

  StringRef getArgument() const final {
    return "hfusion-legalize-narrow-float-kernel-arguments";
  }
  StringRef getDescription() const final {
    return "Legalize narrow scalar launcher arguments to f32";
  }

  void runOnOperation() override {
    legalizeNarrowFloatKernelArgumentsImpl(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::hfusion::createLegalizeNarrowFloatKernelArgumentsPass() {
  return std::make_unique<LegalizeNarrowFloatKernelArguments>();
}
