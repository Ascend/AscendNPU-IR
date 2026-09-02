//===- TestFlattenInterface.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "test-flatten-interface"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")
using namespace mlir::utils::debugger;

namespace bishengir_test {
using namespace mlir;
using namespace mlir::hivm;
struct TestFlattenInterface
    : public PassWrapper<TestFlattenInterface, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFlattenInterface)

  StringRef getArgument() const final { return DEBUG_TYPE; }
  StringRef getDescription() const final { return "Flatten Interface"; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](FlattenInterface hivmFlattenInterface) {
      Operation *currentOperation = hivmFlattenInterface.getOperation();
      auto res = hivmFlattenInterface.getFlattened(FlattenOptions());
      LDBG("Current operation: " << *currentOperation);
      if (failed(res)) {
        LDBG("Failed to flatten");
        return;
      }
      LDBG(to_string(res->reassociation));
      for (auto ty : res->operandTypes)
        LDBG((ty.first ? "DpsInput" : "DpsInit") << " " << ty.second);
    });
  }
};
void registerTestFlattenInterface() {
  PassRegistration<TestFlattenInterface>();
}
} // namespace bishengir_test
