//===- WriteBackShared.cpp - ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_WRITEBACKSHARED
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
class WriteBackSharedPass
    : public impl::WriteBackSharedBase<WriteBackSharedPass> {
public:
  void runOnOperation() override;
};
} // namespace

void WriteBackSharedPass::runOnOperation() {
  auto topM = getOperation();
  ModuleOp mainMod = nullptr;
  llvm::StringMap<int64_t> funcToShared;
  for (auto nested : topM.getOps<ModuleOp>()) {
    // Create the map from SIMT kernel to shared value
    if (nested->hasAttr(hacc::SIMTModuleAttr::name)) {
      auto sharedAttr = nested->getAttrOfType<IntegerAttr>("ttg.shared");
      if (sharedAttr) {
        nested->walk([&](LLVM::LLVMFuncOp func) {
          if (func->hasAttr("nvvm.kernel")) {
            auto name = func.getName();
            if (funcToShared.contains(name)) {
              llvm::report_fatal_error("SIMT kernel " + name +
                                       " exists in more than one SIMT module");
            }
            funcToShared[name] = sharedAttr.getInt();
          }
        });
      } else {
        llvm::report_fatal_error("Fail to find ttg.shared in simt vf");
      }
    } else {
      assert(!mainMod && "only one main module shall exist");
      mainMod = nested;
    }
  }

  // Traverse each alloc op in main module to complete shared value writeback.
  mainMod->walk([&](memref::AllocOp op) {
    if (!op->hasAttr(SharedMemoryAttr::name))
      return;

    // Find the call op which uses the alloc op as a operand.
    func::CallOp targetCall = nullptr;
    for (auto user : op->getUsers()) {
      if (auto callOp = dyn_cast<func::CallOp>(user)) {
        if (!funcToShared.contains(callOp.getCallee())) {
          llvm::report_fatal_error(
              "A callee which is not a SIMT kernel is unexpected");
        }

        if (targetCall) {
          llvm::report_fatal_error("Fond more than one call op");
        }
        targetCall = callOp;
      }
    }
    if (!targetCall) {
      llvm::report_fatal_error("Can't find corresponding call op");
    }

    auto shared = funcToShared[targetCall.getCallee()];
    OpBuilder builder(op->getContext());

    // Find the position of the alloc op in the call op's operands.
    int opIdx = -1;
    for (auto [idx, operand] : llvm::enumerate(targetCall.getOperands())) {
      if (operand == op) {
        assert(opIdx == -1 && "Expect as only one operand");
        opIdx = idx;
      }
    }
    assert(opIdx != -1 && "Expect one operand exists");

    // Get the declaration of the callee
    auto calleeDecl = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        targetCall, targetCall.getCalleeAttr());
    if (!calleeDecl || !calleeDecl.isDeclaration()) {
      llvm::report_fatal_error("Callee declaration not found");
    }

    // Update the signature of the SIMT Kernel declaration.
    auto origFuncTy = calleeDecl.getFunctionType();
    auto newInputTys = origFuncTy.getInputs().vec();
    auto origTy = op.getType();
    assert(origTy.getRank() == 1 && "Expect 1D shape memref");
    assert(newInputTys[opIdx] == origTy && "Expect the same memerf type");
    auto newTy = MemRefType::get({shared}, origTy.getElementType(),
                                 origTy.getLayout(), origTy.getMemorySpace());

    newInputTys[opIdx] = newTy;
    auto newFuncTy = FunctionType::get(calleeDecl->getContext(), newInputTys,
                                       origFuncTy.getResults());
    calleeDecl.setFunctionType(newFuncTy);

    // Write back the actually used shared memory value
    builder.setInsertionPoint(op);
    auto newOp = builder.create<memref::AllocOp>(op.getLoc(), newTy);
    op.replaceAllUsesWith(newOp.getOperation());
    op.erase();
  });
}

std::unique_ptr<Pass> mlir::hivm::createWriteBackSharedPass() {
  return std::make_unique<WriteBackSharedPass>();
}
