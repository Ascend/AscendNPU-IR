//===- OutlineAllocInVF.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "hivm-outline-alloc-in-VF"

namespace mlir {
#define GEN_PASS_DEF_OUTLINEALLOCINVF
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

struct OutlineAllocInVFPass
    : public mlir::impl::OutlineAllocInVFBase<OutlineAllocInVFPass> {
public:
  void runOnOperation() override;
};
} // namespace

/// If there are multiple writes to the same tensor in a VF, one-shot-bufferize
/// may introduce memref.alloc which VF cannot handle. Therefore, it needs to be
/// moved outside of VF and passed as an argument into VF, see issue:
/// https://codehub-y.huawei.com/CompilerKernel/BiShengKernel/BiSheng/issues/3395
void OutlineAllocInVFPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (!hivm::isVF(funcOp))
      continue;
    funcOp.walk([&](memref::AllocOp allocOp) {
      if (auto memRefType =
              dyn_cast<MemRefType>(allocOp->getResultTypes()[0])) {
        // FIXME: here only handle allocOp with static shape.
        if (!memRefType.hasStaticShape()) {
          // Dynamic memory allocation is not allowed within VF
          return;
        }
        auto oldFuncType = funcOp.getFunctionType();
        SmallVector<Type, 4> inputTypes(oldFuncType.getInputs().begin(),
                                        oldFuncType.getInputs().end());
        inputTypes.push_back(memRefType);
        auto newFuncType = FunctionType::get(funcOp.getContext(), inputTypes,
                                             oldFuncType.getResults());
        funcOp.setType(newFuncType);
        funcOp.getFunctionBody().addArgument(memRefType, funcOp->getLoc());
        IRRewriter rewriter(funcOp.getContext());
        rewriter.replaceAllUsesWith(
            allocOp->getResults()[0],
            funcOp.getFunctionBody().getArguments().back());
        auto callSites =
            funcOp.getSymbolUses(funcOp->getParentOfType<ModuleOp>());
        assert(callSites.has_value() && "A VF must at least be invoked once");
        for (SymbolTable::SymbolUse callSite : callSites.value()) {
          func::CallOp callOp = cast<func::CallOp>(callSite.getUser());
          SmallVector<Value> operands(callOp->getOperands().begin(),
                                      callOp->getOperands().end());
          rewriter.setInsertionPoint(callOp);
          Value outlinedAllocOp =
              rewriter.create<memref::AllocOp>(allocOp->getLoc(), memRefType);
          operands.push_back(outlinedAllocOp);
          callOp->setOperands(operands);
        }
        rewriter.eraseOp(allocOp);
      }
    });
  }
}

std::unique_ptr<Pass> mlir::hivm::createOutlineAllocInVFPass() {
  return std::make_unique<OutlineAllocInVFPass>();
}
