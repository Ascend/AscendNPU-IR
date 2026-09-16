//===- InstructionMarker.cpp - Pass to number instrutions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Test/TestPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include <set>

namespace bishengir_test {
using namespace mlir;
using ArgsSet = std::set<std::string>;
struct InstructionMarkerPass
    : public PassWrapper<InstructionMarkerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstructionMarkerPass)

  StringRef getArgument() const final { return "instruction-marker"; }
  StringRef getDescription() const final { return "Instruction marker"; }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    unsigned instructionCounter = 0;

    moduleOp.walk([&](Operation *op) {
      if (isa<ModuleOp>(op))
        return;
      op->setAttr("debug_instruction_number",
                  IntegerAttr::get(IntegerType::get(op->getContext(),
                                                    32), /* 32-bit integer */
                                   instructionCounter++));
    });
  }
};
void registerInstructionMarkerPass() {
  PassRegistration<InstructionMarkerPass>();
}
} // namespace bishengir_test
