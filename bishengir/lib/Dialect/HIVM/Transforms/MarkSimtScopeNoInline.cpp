//===- MarkSimtScopeNoInline.cpp ------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// Mark SIMT scopes as `no_inline` so the generic inline-scope pass keeps them
// outlined.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
#define GEN_PASS_DEF_MARKSIMTSCOPENOINLINE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct MarkSimtScopeNoInlinePass
    : public impl::MarkSimtScopeNoInlineBase<MarkSimtScopeNoInlinePass> {
  void runOnOperation() override;
};

void MarkSimtScopeNoInlinePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  moduleOp.walk([](scope::ScopeOp scopeOp) {
    if (auto vectorMode =
            scopeOp->getAttrOfType<StringAttr>("vector_mode");
        vectorMode && vectorMode.getValue() == "simt") {
      scopeOp.setNoInline(true);
    }
  });
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkSimtScopeNoInlinePass() {
  return std::make_unique<MarkSimtScopeNoInlinePass>();
}
