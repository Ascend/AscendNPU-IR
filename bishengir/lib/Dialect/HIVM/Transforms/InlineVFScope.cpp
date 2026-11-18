//===- InlineVFScope.cpp - Inline VF Scope Pass ----------------*- C++ -*-===//
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
// NOTE: This is a temporary solution.

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
#define GEN_PASS_DEF_INLINEVFSCOPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static bool isPreloadScopeWithCoreType(scope::ScopeOp scopeOp) {
  if (!scopeOp || !scopeOp->hasAttr(hivm::PreloadNumAttr::name))
    return false;
  return scopeOp->hasAttr(hivm::kPipelinedLoopCoreTypeAttrName) ||
         scopeOp->hasAttr(hivm::TCoreTypeAttr::name);
}

struct InlineVFScopePass : public impl::InlineVFScopeBase<InlineVFScopePass> {
  void runOnOperation() override;
};

void InlineVFScopePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SmallVector<scope::ScopeOp> vfScopes;
  moduleOp.walk([&](scope::ScopeOp scopeOp) {
    if (!scopeOp->hasAttr("vf-scope"))
      return;
    auto parentScope =
        dyn_cast_if_present<scope::ScopeOp>(scopeOp->getParentOp());
    if (!isPreloadScopeWithCoreType(parentScope))
      return;
    if (!scopeOp.getRegion().hasOneBlock())
      return;
    vfScopes.push_back(scopeOp);
  });

  for (scope::ScopeOp scopeOp : vfScopes) {
    Block &block = scopeOp.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(block.getTerminator());
    OpBuilder builder(scopeOp);
    builder.setInsertionPointAfter(scopeOp);
    IRMapping mapping;
    for (Operation &op : block.without_terminator())
      builder.clone(op, mapping);
    for (auto [res, retVal] :
         llvm::zip_equal(scopeOp.getResults(), returnOp.getResults()))
      res.replaceAllUsesWith(mapping.lookupOrDefault(retVal));
    scopeOp.erase();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInlineVFScopePass() {
  return std::make_unique<InlineVFScopePass>();
}
