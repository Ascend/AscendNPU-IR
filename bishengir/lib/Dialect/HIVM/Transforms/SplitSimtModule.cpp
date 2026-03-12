//===------------- SplitSimtModule.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement create module for every simt vf.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_SPLITSIMTMODULE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct SplitSimtModulePass
    : public impl::SplitSimtModuleBase<SplitSimtModulePass> {
  void runOnOperation() override;
};

} // namespace

void SplitSimtModulePass::runOnOperation() {
  auto mod = llvm::cast<ModuleOp>(getOperation());
  auto ctx = &getContext();
  OpBuilder builder(ctx);
  auto modBlock = mod.getBody();
  builder.setInsertionPointToStart(modBlock);
  Type gridSizeType;
  mod->walk([&mod, &gridSizeType](func::CallOp callOp) {
    auto parentFuncOp = callOp->getParentOfType<func::FuncOp>();
    assert(parentFuncOp && "call op to simt vf should in func body");
    auto funcOp = llvm::cast<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(mod, callOp.getCalleeAttr()));
    if (util::isSIMTVF(funcOp)) {
      auto gridSizeArgs = parentFuncOp.getArguments().take_back(3);
      for (auto arg : gridSizeArgs) {
        auto argType = llvm::dyn_cast<IntegerType>(arg.getType());
        assert(argType && argType.getWidth() == 32 &&
               "grid size arg type should be i32");
      }
      gridSizeType = gridSizeArgs[0].getType();
      callOp.getOperandsMutable().append(gridSizeArgs);
    }
  });
  mod->walk([&builder, modBlock, &mod, &gridSizeType](func::FuncOp funcOp) {
    if (util::isSIMTVF(funcOp)) {
      auto newMod = builder.create<ModuleOp>(funcOp->getLoc());
      builder.setInsertionPointToStart(newMod.getBody());
      builder.clone(*funcOp);
      funcOp.eraseBody();
      funcOp.setSymVisibility("private");
      builder.setInsertionPointToStart(modBlock);
      newMod->setAttrs(mod->getAttrs());
      newMod->setAttr(hacc::SIMTModuleAttr::name, builder.getUnitAttr());
      auto funcType = funcOp.getFunctionType();
      auto newInputs = funcType.getInputs().vec();
      // only add grid size args for simt vf declare in simd module
      // because AdaptTritonIRKernel pass will add args for simt vf
      newInputs.insert(newInputs.end(),
                       {gridSizeType, gridSizeType, gridSizeType});
      auto newFuncType =
          builder.getFunctionType(newInputs, funcType.getResults());
      funcOp.setFunctionType(newFuncType);
    }
  });
  llvm::SmallVector<Operation *> simdFuncs;
  for (auto &op : mod.getBody()->getOperations()) {
    if (!llvm::isa<ModuleOp>(&op)) {
      simdFuncs.emplace_back(&op);
    }
  }
  auto newMod = builder.create<ModuleOp>(mod->getLoc());
  newMod->setAttrs(mod->getAttrs());
  mod->setAttrs(llvm::ArrayRef<NamedAttribute>{});
  builder.setInsertionPointToStart(newMod.getBody());
  for (auto op : simdFuncs) {
    builder.clone(*op);
    op->erase();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSplitSimtModulePass() {
  return std::make_unique<SplitSimtModulePass>();
}
