//===- TestFlattenInterface.cpp -------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

    [[maybe_unused]] unsigned instructionCounter = 0;

    funcOp.walk([&](FlattenInterface hivmFlattenInterface) {
      [[maybe_unused]] Operation *currentOperation =
          hivmFlattenInterface.getOperation();
      auto res = hivmFlattenInterface.getFlattened(FlattenOptions());
      LDBG("Current operation: " << *currentOperation);
      if (failed(res)) {
        LDBG("Failed to flatten");
        return;
      }
      LDBG(to_string(res->reassociation));
      for ([[maybe_unused]] auto ty : res->operandTypes)
        LDBG((ty.first ? "DpsInput" : "DpsInit") << " " << ty.second);
    });
  }
};
void registerTestFlattenInterface() {
  PassRegistration<TestFlattenInterface>();
}
} // namespace bishengir_test
