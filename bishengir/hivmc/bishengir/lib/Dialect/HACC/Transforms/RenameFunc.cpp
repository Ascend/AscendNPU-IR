//===- RenameFunc.cpp ---- Rename Function Pass ------------------*- C++-*-===//
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

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#define GEN_PASS_DEF_HACCRENAMEFUNCTION
#include "bishengir/Dialect/HACC/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hacc;

struct RenameFuncPass : public impl::HACCRenameFunctionBase<RenameFuncPass> {
public:
  void runOnOperation() override;
};

void RenameFuncPass::runOnOperation() {
  auto module = getOperation();
  module->walk([&](func::FuncOp funcOp) {
    auto renameFuncAttr =
        funcOp->getAttrOfType<hacc::RenameFuncAttr>(hacc::RenameFuncAttr::name);
    if (!renameFuncAttr)
      return;

    auto newName = renameFuncAttr.getTargetName();
    if (newName.name.empty()) {
      funcOp.emitOpError() << "invalid target function name: " << newName;
      signalPassFailure();
      return;
    }
    // There must not be an existing function with the target name.
    auto maybeExistingFuncOp =
        SymbolTable::lookupNearestSymbolFrom(funcOp, newName);
    if (maybeExistingFuncOp) {
      funcOp.emitOpError()
          << "failed to rename function to @" << newName.getValue()
          << " because there is already a function with the same name!";
      signalPassFailure();
      return;
    }

    // Update all the symbol reference.
    if (failed(SymbolTable::replaceAllSymbolUses(funcOp.getSymNameAttr(),
                                                 newName.getAttr(), module))) {
      funcOp.emitOpError() << "failed to rename function!";
      signalPassFailure();
      return;
    }

    // Update the function name and remove the attribute.
    funcOp->removeAttr(hacc::RenameFuncAttr::name);
    funcOp.setSymName(newName.getAttr());
  });
}

std::unique_ptr<Pass> mlir::hacc::createRenameFuncPass() {
  return std::make_unique<RenameFuncPass>();
}
