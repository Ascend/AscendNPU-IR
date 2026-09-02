//===- ValidPropagatedReshape.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "valid-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace bishengir_test {
using namespace mlir;
using namespace mlir::utils;
using namespace mlir::hfusion::reshape_utils;
struct ValidPropagatedReshapePass
    : public PassWrapper<ValidPropagatedReshapePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidPropagatedReshapePass)

  StringRef getArgument() const final { return "valid-propagate"; }
  StringRef getDescription() const final { return "valid-propagate"; }
  FailureOr<Value> getSrc(Operation *op) {
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      return collapseOp.getSrc();
    }
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      return expandOp.getSrc();
    }
    return failure();
  }
  bool isIllegal(Operation *op) {
    if (isSkippableOp(op))
      return false;
    return isa<DestinationStyleOpInterface, tensor::ExtractOp, tensor::ConcatOp,
               tensor::BitcastOp, tensor::PadOp, tensor::ExtractSliceOp,
               tensor::InsertSliceOp>(op);
  }
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool isValid = true;
    moduleOp.walk([&](Operation *op) {
      if (isReshapingOp(op)) {
        bool isValidUser = true;
        bool isValidDef = true;
        LLVM_DEBUG(llvm::dbgs() << "Checking " << *op << "\n";);
        // Check the usage of this op
        for (auto *user : op->getUsers()) {
          if (isIllegal(user)) {
            LLVM_DEBUG(llvm::dbgs() << "Failing user " << *user << "\n";);
            isValidUser = false;
          }
        }
        FailureOr<Value> src = getSrc(op);
        if (succeeded(src)) {
          if (Operation *srcOp = src.value().getDefiningOp()) {
            if (isIllegal(srcOp)) {
              LLVM_DEBUG(llvm::dbgs()
                             << "Failing def " << src.value() << "\n";);
              isValidDef = false;
            }
          }
        }
        LDBG(isValidUser << " " << isValidDef);
        isValid &= (isValidUser || isValidDef);
      }
    });
    if (isValid)
      llvm::outs() << "Valid propagation\n";
    else
      llvm::outs() << "Failed propagation\n";
  }
};
void registerValidPropagatedReshapePass() {
  PassRegistration<ValidPropagatedReshapePass>();
}
} // namespace bishengir_test
