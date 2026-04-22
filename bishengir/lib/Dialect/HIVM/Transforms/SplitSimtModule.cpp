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

static void updateFuncSignatrueTo1D(func::FuncOp func) {
  auto* ctx = func.getContext();
  auto oldType = func.getFunctionType();
  SmallVector<Type> newArgTypes;
  for (auto argType : oldType.getInputs()) {
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      if (memrefType.getRank() > 1) {
        int64_t totalSize = 1;
        bool hasDynamicDim = false;
        for (auto dim : memrefType.getShape()) {
          if (ShapedType::isDynamic(dim)) {
            hasDynamicDim = true;
            break;
          }
          totalSize *= dim;
      }
      int64_t finalSize = hasDynamicDim ? ShapedType::kDynamic : totalSize;
      auto elemType = memrefType.getElementType();
      auto collapsedMemrefType = MemRefType::get({finalSize}, elemType, MemRefLayoutAttrInterface(),
                                         memrefType.getMemorySpace());
      newArgTypes.push_back(collapsedMemrefType);
      } else {
        newArgTypes.push_back(argType);
      }
    } else {
      newArgTypes.push_back(argType);
    }
  }
  auto newFuncType = FunctionType::get(ctx, newArgTypes, oldType.getResults());
  func.setType(newFuncType);
}

static void fixMultiDimMemrefToSIMTFunc(ModuleOp module, func::FuncOp func) {
  updateFuncSignatrueTo1D(func);
  StringRef funcName = func.getName();
  module.walk([&funcName](func::CallOp callOp) {
    if (callOp.getCallee() != funcName) {
      WalkResult::advance();
    }
    for (auto& operand : callOp->getOpOperands()) {
      Value val = operand.get();
      if (auto memrefType = dyn_cast<MemRefType>(val.getType())) {
        auto rank = memrefType.getRank();
        if (rank > 1) {
          OpBuilder b(callOp);
          int64_t totalSize = 1;
          SmallVector<ReassociationIndices> reassociation(1);
          bool hasDynamicDim = false;
          for(int64_t i = 0; i < rank; i++) {
            reassociation[0].push_back(i);
            int64_t dimSize = memrefType.getDimSize(i);
            if (ShapedType::isDynamic(dimSize)) {
              hasDynamicDim = true;
            }else{
              totalSize *= dimSize;
            }
          }
          int64_t finalSize = hasDynamicDim ? ShapedType::kDynamic : totalSize;
          auto elemType = memrefType.getElementType();
          auto collapsedMemrefType = MemRefType::get({finalSize}, elemType, MemRefLayoutAttrInterface(),
                                              memrefType.getMemorySpace());
          Value collapsed = b.create<memref::CollapseShapeOp>(callOp.getLoc(), collapsedMemrefType, val, reassociation);
          operand.set(collapsed);
          }
        }
      }
      return WalkResult::advance();
    });
}

void SplitSimtModulePass::runOnOperation() {
  static constexpr int GridSizeCount = 3;
  static constexpr int GridSizeWidth = 32;
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
      auto gridSizeArgs = parentFuncOp.getArguments().take_back(GridSizeCount);
      for (auto arg : gridSizeArgs) {
        auto argType = llvm::dyn_cast<IntegerType>(arg.getType());
        assert(argType && argType.getWidth() == GridSizeWidth &&
               "grid size arg type should be i32");
      }
      gridSizeType = gridSizeArgs[0].getType();
      callOp.getOperandsMutable().append(gridSizeArgs);
    }
  });
  if (!gridSizeType) {
    llvm::report_fatal_error("gridSizeType is not initialized for SIMT vf");
  }

  // Collect simt vf
  SmallVector<func::FuncOp> simtVFs;
  mod->walk([&simtVFs](func::FuncOp funcOp) {
    if (util::isSIMTVF(funcOp)) {
      simtVFs.push_back(funcOp);
    }
  });
  // Create simt modules and then put simt vfs into their respective simt
  // modules.
  for (auto funcOp : simtVFs) {
    auto newMod = builder.create<ModuleOp>(funcOp->getLoc());
    builder.setInsertionPointToStart(newMod.getBody());
    builder.clone(*funcOp);

    // only add grid size args for simt vf declare in simd module
    // because AdaptTritonIRKernel pass will add args for simt vf
    auto emptyAttr = DictionaryAttr::get(funcOp.getContext());
    auto insertPos = funcOp.getNumArguments();
    auto loc = funcOp.getLoc();
    for (int i = 0; i < GridSizeCount; i++) {
      funcOp.insertArgument(insertPos, gridSizeType, emptyAttr, loc);
    }

    funcOp.eraseBody();
    funcOp.setSymVisibility("private");
    builder.setInsertionPointToStart(modBlock);
    newMod->setAttrs(mod->getAttrs());
    newMod->setAttr(hacc::SIMTModuleAttr::name, builder.getUnitAttr());

    // Fix multi-dim memref to 1-d memref for simt vf and its call sites.
    // This is a workaround for the fact that current SIMT VF lowering assumes
    // all memref arguments are 1-d.
    fixMultiDimMemrefToSIMTFunc(mod,funcOp);
  }

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
