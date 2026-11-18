//===- WrapCVPreloadScopes.cpp - Wrap CV preload scopes ----------------===//
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

#include "Util.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {
#define GEN_PASS_DEF_WRAPCVPRELOADSCOPES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace hivm
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct StageGroup {
  IntegerAttr preload;
  IntegerAttr maxPreload;
  SmallVector<scope::ScopeOp> scopes;
};

static LogicalResult wrapStages(Block &block) {
  SmallVector<StageGroup> groups;
  for (auto scopeOp : block.getOps<scope::ScopeOp>()) {
    auto preload = scopeOp->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto maxPreload =
        scopeOp->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name);
    if (!preload || !maxPreload || preload.getInt() < 0 ||
        preload.getInt() >= maxPreload.getInt())
      return scopeOp.emitOpError(
          "expected valid preload_num and max_preload_num");
    if (!groups.empty() && maxPreload != groups.back().maxPreload)
      return scopeOp.emitOpError("expected consistent max_preload_num");
    if (groups.empty() || preload != groups.back().preload) {
      if (!groups.empty() && preload.getInt() >= groups.back().preload.getInt())
        return scopeOp.emitOpError("expected scopes ordered by descending "
                                   "preload_num with contiguous stages");
      groups.push_back({preload, maxPreload, {}});
    }
    groups.back().scopes.push_back(scopeOp);
  }

  for (StageGroup &group : groups) {
    if (group.scopes.size() < 2)
      continue;

    // Include the complete span: staging may have hoisted buffer operations
    // between the core scopes. Their order and escaping results matter too.
    SmallVector<Operation *> operations;
    for (Operation *op = group.scopes.front();; op = op->getNextNode()) {
      operations.push_back(op);
      if (op == group.scopes.back().getOperation())
        break;
    }
    // Include nested users once, rather than walking their ancestors for each
    // result use. Group discovery and use classification remain linear.
    DenseSet<Operation *> members;
    for (Operation *op : operations)
      op->walk([&](Operation *nested) { members.insert(nested); });
    SmallVector<Value> returns;
    SmallVector<SmallVector<OpOperand *>> externalUses;
    for (Operation *op : operations) {
      for (Value result : op->getResults()) {
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : result.getUses()) {
          if (!members.contains(use.getOwner()))
            uses.push_back(&use);
        }
        if (!uses.empty()) {
          returns.push_back(result);
          externalUses.push_back(std::move(uses));
        }
      }
    }

    OpBuilder builder(group.scopes.front());
    auto wrapper = builder.create<scope::ScopeOp>(
        group.scopes.front().getLoc(), TypeRange(ValueRange(returns)));
    wrapper->setAttr(PreloadNumAttr::name, group.preload);
    wrapper->setAttr(MaxPreloadNumAttr::name, group.maxPreload);
    wrapper->setAttr(
        kPipelinedLoopCoreTypeAttrName,
        TCoreTypeAttr::get(builder.getContext(), TCoreType::CUBE_AND_VECTOR));
    wrapper.setNoInline(true);
    Block *body = builder.createBlock(&wrapper.getRegion());
    for (Operation *op : operations)
      op->moveBefore(body, body->end());
    for (scope::ScopeOp inner : group.scopes) {
      inner->removeAttr(PreloadNumAttr::name);
      inner->removeAttr(MaxPreloadNumAttr::name);
    }
    builder.setInsertionPointToEnd(body);
    builder.create<scope::ReturnOp>(wrapper.getLoc(), returns);
    for (auto [result, uses] : llvm::zip(wrapper.getResults(), externalUses))
      for (OpOperand *use : uses)
        use->set(result);
  }
  return success();
}

struct WrapCVPreloadScopesPass
    : mlir::hivm::impl::WrapCVPreloadScopesBase<WrapCVPreloadScopesPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (isCVPipelineSnapshot(funcOp) || hasLegacyCVPipelinedLoop(funcOp))
      return;
    SmallVector<LoopLikeOpInterface> loops;
    funcOp.walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
      if (!loop->hasAttr(CVPipelinedLoopAttr::name))
        return WalkResult::advance();
      loops.push_back(loop);
      return WalkResult::skip();
    });
    for (LoopLikeOpInterface loop : loops)
      for (Region &region : loop->getRegions())
        for (Block &block : region)
          if (failed(wrapStages(block))) {
            signalPassFailure();
            return;
          }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createWrapCVPreloadScopesPass() {
  return std::make_unique<WrapCVPreloadScopesPass>();
}
