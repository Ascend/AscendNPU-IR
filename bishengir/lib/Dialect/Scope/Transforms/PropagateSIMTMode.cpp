//===- PropagateSIMTMode.cpp - Propagate SIMT VF mode to callers ----------===//
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
//
// This file implements a pass to propagate the SIMT VF mode of outlined
// functions to their callers: a caller that invokes a SIMT VF function is
// marked MIX.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DEF_PROPAGATESIMTMODE
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"

using namespace impl;
namespace mlir {
namespace scope {

namespace {

class PropagateSIMTModePass
    : public PropagateSIMTModeBase<PropagateSIMTModePass> {
public:
  explicit PropagateSIMTModePass() : PropagateSIMTModeBase() {}
  void runOnOperation() final {
    ModuleOp module = getOperation();
    module->walk([&](func::FuncOp funcOp) {
      // A caller already marked SIMT executes only SIMT paths; calling
      // another SIMT VF must not downgrade it to MIX.
      if (hivm::util::isSIMTVF(funcOp))
        return;
      bool callsSIMTVF = false;
      funcOp.walk([&](func::CallOp callOp) -> WalkResult {
        auto *calleeOp = SymbolTable::lookupNearestSymbolFrom(
            callOp, callOp.getCalleeAttr());
        if (auto callee = llvm::dyn_cast_if_present<func::FuncOp>(calleeOp)) {
          if (hivm::util::isSIMTVF(callee)) {
            callsSIMTVF = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      // Current propagation checks direct callees only
      if (callsSIMTVF)
        funcOp->setAttr(hivm::VFModeAttr::name,
                        hivm::VFModeAttr::get(funcOp->getContext(),
                                              hivm::VFMode::MIX));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createPropagateSIMTModePass() {
  return std::make_unique<PropagateSIMTModePass>();
}

} // namespace scope
} // namespace mlir
