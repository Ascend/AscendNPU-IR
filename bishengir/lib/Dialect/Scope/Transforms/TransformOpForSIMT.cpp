//===- TransformOpForSIMT.cpp - Transform Op For SIMT Pass ----------------===//
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
// This file implements a pass to transform operations for SIMT execution.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "transform-op-for-simt"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_TRANSFORMOPFORSIMT
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"

using namespace impl;

namespace mlir {
namespace scope {

class TransformOpForSIMTPass
    : public TransformOpForSIMTBase<TransformOpForSIMTPass> {
public:
  explicit TransformOpForSIMTPass() : TransformOpForSIMTBase() {}
  void runOnOperation() final;
};

// Collect everything an op depends on: its direct operands plus the values its
// nested regions capture from an enclosing block.
static void appendOpDependencies(Operation *op,
                                 SmallVectorImpl<Value> &worklist) {
  llvm::append_range(worklist, op->getOperands());
  if (op->getNumRegions() == 0)
    return;
  SetVector<Value> captured;
  getUsedValuesDefinedAbove(op->getRegions(), captured);
  llvm::append_range(worklist, captured);
}

// Ops nested in a hoisted op travel with it, so they count as hoisted too.
static bool isHoistedWithAncestors(Operation *op,
                                   const SetVector<Operation *> &toHoist) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (toHoist.contains(cur))
      return true;
  return false;
}

// A local_load materializes a SIMD-to-SIMT transfer at a scope boundary. If
// its users are hoisted back to SIMD, restore their use of the original SIMD
// tensor instead of leaking the SIMT-only boundary op into the SIMD module.
static void rewriteHoistedLocalLoads(scope::ScopeOp scopeOp,
                                     SetVector<Operation *> &toHoist) {
  SmallVector<hivm::LocalLoadOp> localLoads;
  for (Operation *op : toHoist)
    if (auto localLoad = dyn_cast<hivm::LocalLoadOp>(op))
      localLoads.push_back(localLoad);

  SmallVector<std::pair<hivm::LocalLoadOp, Value>> recoverableLoads;
  SetVector<Operation *> blocked;
  auto isDefinedInsideScope = [&](Value value) {
    if (Operation *defOp = value.getDefiningOp())
      return scopeOp.getRegion().isAncestor(defOp->getParentRegion());
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return scopeOp.getRegion().isAncestor(blockArg.getOwner()->getParent());
    return false;
  };

  auto isDirectlyInScope = [&](Operation *op) {
    return op->getBlock() == &scopeOp.getRegion().front();
  };

  for (hivm::LocalLoadOp localLoad : localLoads) {
    for (Operation *user : localLoad->getUsers()) {
      if (!toHoist.contains(user) || isDirectlyInScope(user))
        continue;
      blocked.insert(localLoad.getOperation());
      break;
    }

    Value addr = localLoad.getAddr();
    while (auto castOp = addr.getDefiningOp<memref::CastOp>())
      addr = castOp.getSource();

    Value tensorSource;
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    if (auto toMemref = addr.getDefiningOp<bufferization::ToMemrefOp>())
      tensorSource = toMemref.getTensor();
#else
    if (auto toBuffer = addr.getDefiningOp<bufferization::ToBufferOp>())
      tensorSource = toBuffer.getTensor();
#endif

    if (blocked.contains(localLoad.getOperation()) || !tensorSource ||
        isDefinedInsideScope(tensorSource) ||
        tensorSource.getType() != localLoad.getResult().getType()) {
      blocked.insert(localLoad.getOperation());
      continue;
    }

    recoverableLoads.emplace_back(localLoad, tensorSource);
  }

  // Classify every boundary load before mutating any uses. Otherwise an op
  // that consumes both a recoverable and an unrecoverable load can be rewired
  // to the raw SIMD tensor before it is later blocked from hoisting.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : toHoist) {
      if (blocked.contains(op))
        continue;
      SmallVector<Value> deps;
      appendOpDependencies(op, deps);
      for (Value dep : deps) {
        if (blocked.contains(dep.getDefiningOp())) {
          changed |= blocked.insert(op);
          break;
        }
      }
    }
  }
  for (Operation *op : blocked)
    toHoist.remove(op);

  for (auto [localLoad, tensorSource] : recoverableLoads) {
    if (!toHoist.contains(localLoad.getOperation()))
      continue;
    localLoad.getResult().replaceUsesWithIf(tensorSource, [&](OpOperand &use) {
      return isHoistedWithAncestors(use.getOwner(), toHoist);
    });
    toHoist.remove(localLoad.getOperation());
    if (localLoad->use_empty())
      localLoad.erase();
  }
}

// 1. Move tensor.from_elements and related ops outside simt_scope:
//    Before:
//        scope {vf_mode="simt"} {
//          %97 = memref.load %reinterpret_cast_2[%c0]
//          %98 = arith.cmpi slt, %97, %c0_i32
//          %from_elem = tensor.from_elements %98, %98 : tensor<2xi1>
//          %empty = tensor.empty() : tensor<1xf16>
//          %vcast = hivm.hir.vcast ins(%from_elem) outs(%empty) ->
//          tensor<1xf16> hivm.hir.local_store ins(%buf, %vcast)
//          ...
//        }
//    After:
//        %97 = memref.load %reinterpret_cast_2[%c0]
//        %98 = arith.cmpi slt, %97, %c0_i32
//        %from_elem = tensor.from_elements %98, %98 : tensor<2xi1>
//        %empty = tensor.empty() : tensor<1xf16>
//        %vcast = hivm.hir.vcast ins(%from_elem) outs(%empty) -> tensor<1xf16>
//        scope {vf_mode="simt"} {
//          hivm.hir.local_store ins(%buf, %vcast)
//          ...
//        }

static void moveFromElementsOutsideScope(scope::ScopeOp scopeOp) {
  // Find all tensor.from_elements with more than 1 operand inside the scope.
  SmallVector<tensor::FromElementsOp> fromElementsOps;
  scopeOp.walk([&](tensor::FromElementsOp op) {
    if (op->getNumOperands() != 1) {
      fromElementsOps.push_back(op);
    }
  });

  if (fromElementsOps.empty())
    return;

  SetVector<Operation *> toHoist;

  auto isInsideScope = [&](Operation *op) -> bool {
    return scopeOp->isAncestor(op);
  };

  for (auto fromElem : fromElementsOps) {
    toHoist.insert(fromElem);

    // Backward slice: collect all defining ops of operands inside the scope.
    SmallVector<Value> worklist;
    appendOpDependencies(fromElem, worklist);
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      auto *defOp = v.getDefiningOp();
      if (!defOp || !isInsideScope(defOp))
        continue;
      if (toHoist.insert(defOp))
        appendOpDependencies(defOp, worklist);
    }
  }

  rewriteHoistedLocalLoads(scopeOp, toHoist);

  // Move ops before the scope, maintaining original block order.
  for (Operation &op :
       llvm::make_early_inc_range(scopeOp.getRegion().front())) {
    if (toHoist.count(&op))
      op.moveBefore(scopeOp);
  }
}

// 2. Hoist hivm.hir.get_sub_block_idx outside simt_scope:
//    TileAndBindSubBlock pins an untiled write to a destination shared by both
//    AIV sub-blocks with scf.if(get_sub_block_idx() == 0), and builds that
//    guard next to the write. For an explicit simt scope that puts the guard
//    inside inside the scope. To avoid different get_sub_block_idx() behavior
//    between simt vf and main function, hoist the get_sub_block_idx() outside
//    the simt scope. The guard remains inside the scope, but uses the hoisted
//    value.
//      Before:
//          scope {vf_mode="simt"} {
//            %idx = hivm.hir.get_sub_block_idx -> i64
//            %i = arith.index_cast %idx : i64 to index
//            %c = arith.cmpi eq, %i, %c0 : index
//            scf.if %c { hivm.hir.scatter_store ... } {limit_sub_block_id0}
//          }
//      After:
//          %idx = hivm.hir.get_sub_block_idx -> i64
//          scope {vf_mode="simt"} {
//            %i = arith.index_cast %idx : i64 to index
//            %c = arith.cmpi eq, %i, %c0 : index
//            scf.if %c { hivm.hir.scatter_store ... } {limit_sub_block_id0}
//          }

static void hoistSubBlockIdxOutsideScope(scope::ScopeOp scopeOp) {
  SmallVector<hivm::GetSubBlockIdxOp> idxOps;

  scopeOp.walk([&](hivm::GetSubBlockIdxOp idxOp) { idxOps.push_back(idxOp); });
  if (idxOps.empty())
    return;

  // no CSE runs before OutlineScope. Fold them here so the outlined function
  // gains one argument instead of one per guard.
  hivm::GetSubBlockIdxOp hoisted = idxOps.front();
  hoisted->moveBefore(scopeOp);
  for (hivm::GetSubBlockIdxOp duplicate : llvm::drop_begin(idxOps)) {
    duplicate.getResult().replaceAllUsesWith(hoisted.getResult());
    duplicate.erase();
  }
}

void TransformOpForSIMTPass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([&](scope::ScopeOp scopeOp) {
    if (!hivm::util::isSIMTVF(scopeOp))
      return;

    // --- Transformation 1: Move tensor.from_elements outside scope ---
    moveFromElementsOutsideScope(scopeOp);
  });

  // --- Transformation 2: Hoist get_sub_block_idx outside simt scope ---
  SmallVector<scope::ScopeOp> simtScopes;
  module.walk([&](scope::ScopeOp scopeOp) {
    if (hivm::util::isSIMTVF(scopeOp))
      simtScopes.push_back(scopeOp);
  });
  for (scope::ScopeOp scopeOp : simtScopes)
    hoistSubBlockIdxOutsideScope(scopeOp);
}

std::unique_ptr<Pass> createTransformOpForSIMTPass() {
  return std::make_unique<TransformOpForSIMTPass>();
}

} // namespace scope
} // namespace mlir
