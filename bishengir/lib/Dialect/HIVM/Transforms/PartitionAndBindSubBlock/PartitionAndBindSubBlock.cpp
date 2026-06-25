//------------------------PartitionAndBindSubBlock.cpp------------------------//
//
// Pass driver: runs core analysis, sub-block lowering, then guard cleanup.
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

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/CoreDependencyAnalysis.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/CoreLegality.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/SubBlockLowering.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-partition-and-bind-sub-block"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_PARTITIONANDBINDSUBBLOCK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::partition_and_bind;

namespace {

//===----------------------------------------------------------------------===//
// PartitionAndBindSubBlockPass
//===----------------------------------------------------------------------===//

struct PartitionAndBindSubBlockPass
    : public mlir::impl::PartitionAndBindSubBlockBase<
          PartitionAndBindSubBlockPass> {
  using Base::Base;
  void runOnOperation() override;

private:
  bool moduleHasSubBlockScopes(ModuleOp moduleOp) const;
  LogicalResult runOnFunc(func::FuncOp func);
};

} // namespace

bool PartitionAndBindSubBlockPass::moduleHasSubBlockScopes(
    ModuleOp moduleOp) const {
  bool found = false;
  moduleOp.walk([&](scope::ScopeOp scopeOp) {
    if (isSingleCore(getSubBlockCoreOf(scopeOp.getOperation()))) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Per-func Steps
//===----------------------------------------------------------------------===//

LogicalResult PartitionAndBindSubBlockPass::runOnFunc(func::FuncOp func) {
  // Skip host-side and declaration-only functions
  if (func.isExternal() || hacc::utils::isHost(func))
    return success();

  // Skip funcs with no `{sub_block}` scope
  bool hasScope = false;
  func.walk([&](scope::ScopeOp scopeOp) {
    if (isSingleCore(getSubBlockCoreOf(scopeOp.getOperation()))) {
      hasScope = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasScope)
    return success();

  // Cube gate: sub-core (VEC0/VEC1) parallelism is a MIX-CV feature
  bool hasCube = false;
  func.walk([&](Operation *op) {
    if (isCubeOrSharedOp(op)) {
      hasCube = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasCube) {
    func.emitWarning(
        "[hivm-partition-and-bind-sub-block]: @" + func.getName().str() +
        " has no cube op; sub-core binding needs a cube to drive the two "
        "AIV sub-blocks -- dropping the {sub_block} partition hint");
    SubBlockLowering::inlineSubBlockScopesAsFallback(func);
    LDBG("no cube op in @" << func.getName()
                           << "; inlined {sub_block} scopes and continued");
    return success();
  }

  // (1) Verify legality
  MergeOnConflictHook conflictHook; // default policy: bail on conflict.
  CoreLegalityChecker checker(func, conflictHook);
  LegalityResult legality = checker.check();
  if (!legality) {
    Operation *blame =
        legality.offendingOp ? legality.offendingOp : func.getOperation();
    blame->emitWarning(
        "[hivm-partition-and-bind-sub-block]: " + legality.message +
        "; dropping the {sub_block} partition hint for @" +
        func.getName().str());
    SubBlockLowering::inlineSubBlockScopesAsFallback(func);
    LDBG("legality failed for @"
         << func.getName()
         << "; inlined {sub_block} scopes and continued: " << legality.message);
    return success();
  }

  // (2) Resolve the core assignment.
  DefaultFreeNodePlacementPolicy defaultPolicy;
  LoadBalancedFreeNodePlacementPolicy balancedPolicy;
  FreeNodePlacementPolicy &placementPolicy =
      enableLoadBalanced
          ? static_cast<FreeNodePlacementPolicy &>(balancedPolicy)
          : static_cast<FreeNodePlacementPolicy &>(defaultPolicy);
  CoreDependencyAnalysis analysis(func, placementPolicy);
  CoreAssignment assignment = analysis.run();

  // (3) Lower scopes to scf.if guards. Lowering also stamps the fixpipe
  // sub-block destinations recorded in the assignment.
  SubBlockLowering lowering(func, assignment);
  if (failed(lowering.run())) {
    func.emitError("[hivm-partition-and-bind-sub-block]: lowering failed");
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// runOnOperation
//===----------------------------------------------------------------------===//

void PartitionAndBindSubBlockPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // (0) Self-gate: no-op unless the module has a `{sub_block}` scope.
  if (!moduleHasSubBlockScopes(moduleOp)) {
    LDBG("no {sub_block} scopes in module; skipping");
    return;
  }

  // Per func. Legality failure is handled in runOnFunc (warn + inline) and
  // returns success, so a failure here is a genuine structural/lowering error.
  for (auto func : moduleOp.getOps<func::FuncOp>()) {
    if (failed(runOnFunc(func))) {
      signalPassFailure();
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::hivm::createPartitionAndBindSubBlockPass(
    const PartitionAndBindSubBlockOptions &options) {
  return std::make_unique<PartitionAndBindSubBlockPass>(options);
}
