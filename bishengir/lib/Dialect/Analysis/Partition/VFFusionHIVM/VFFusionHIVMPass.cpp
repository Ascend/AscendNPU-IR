//===- VFFusionHIVMPass.cpp - VF fusion on HIVM ops ----------------------===//
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

#include "bishengir/Dialect/Analysis/Partition/HIVM/VFFusionHIVM.h"
#include "bishengir/Dialect/Analysis/Partition/Partition.h"
#include "bishengir/Dialect/Analysis/VFFusion/Passes.h"
#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionOutliner.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <memory>

#define DEBUG_TYPE "vf-fusion-hivm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {
#define GEN_PASS_DEF_VFFUSIONHIVM
#include "bishengir/Dialect/Analysis/VFFusion/Passes.h.inc"

namespace partition::hivm {

thread_local Groups toOutline{};

void handoverGroupForOutlining(Group &&g) {
  assert(!g.empty());
  toOutline.emplace_back(g);
}

namespace {

bool isFusionCandidate(func::FuncOp funcOp) {
  if (::mlir::hivm::isVF(funcOp))
    return false;
  if (!hacc::utils::isDevice(funcOp))
    return false;
  if (isCubeFunc(funcOp))
    return false;
  auto parallelMode = funcOp->getAttrOfType<StringAttr>("parallel_mode");
  return parallelMode && parallelMode.getValue() == "simd";
}

void stampVectorFunction(Operation *op, OpBuilder &builder) {
  op->setAttr(::mlir::hivm::VectorFunctionAttr::name, builder.getUnitAttr());
  op->setAttr("no_inline", builder.getUnitAttr());
}

LogicalResult outlineBlock(func::FuncOp funcOp, OpBuilder &builder,
                           VFFusionOutliner &outliner, VFFusionBlock &&block) {
  assert(!block.getOps().empty());

  auto maybeFusedFunc = outliner.outline(funcOp, block, builder);
  if (failed(maybeFusedFunc)) {
    return failure();
  }
  auto fusedFunc = *maybeFusedFunc;
  stampVectorFunction(fusedFunc, builder);

  builder.setInsertionPointAfter(block.getOps().back());
  auto callOp =
      builder.create<func::CallOp>(block.getOps().back()->getLoc(), fusedFunc,
                                   block.recomputeInputs().takeVector());

  for (auto [oldOut, newOut] :
       llvm::zip_equal(block.recomputeOutputs(), callOp.getResults())) {
    Value out = oldOut;
    out.replaceAllUsesWith(newOut);
  }
  stampVectorFunction(callOp, builder);

  for (auto *op : llvm::reverse(block.getOps()))
    op->erase();
  return success();
}

LogicalResult tryRewrite(func::FuncOp funcOp, hivm::VFFusionMode fusionMode,
                         OpBuilder &builder) {
  LDBG("rewriting " << funcOp.getName().str());

  auto result = failure();
  switch (fusionMode) {
  case hivm::VFFusionMode::LinearScan:
    result =
        LinearScanFusion::run(funcOp, {
                                          sameBlockRelation,
                                          filterFuncOpRelation,
                                          linearScanVectorizableOpsRelation,
                                      });
    break;
  default:
    llvm_unreachable("unsupported fusion mode");
  }
  if (failed(result))
    return result;

  VFFusionOutliner outliner{};
  for (auto &group : toOutline) {
    VFFusionBlock block{};
    for (auto *op : group)
      block.fuseOp(op);
    if (failed(outlineBlock(funcOp, builder, outliner, std::move(block)))) {
      assert(false);
      toOutline.clear();
      return failure();
    }
  }
  toOutline.clear();
  return success();
}

class VFFusionHIVMPass : public impl::VFFusionHIVMBase<VFFusionHIVMPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(&getContext());

    auto result = module->walk([this, &builder](func::FuncOp funcOp) {
      if (!isFusionCandidate(funcOp))
        return WalkResult::skip();
      if (failed(tryRewrite(funcOp, fusionMode.getValue(), builder)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
    assert(!result.wasInterrupted());
  }
};

} // namespace
} // namespace partition::hivm

std::unique_ptr<Pass> createVFFusionHIVMPass() {
  return std::make_unique<partition::hivm::VFFusionHIVMPass>();
}

} // namespace mlir::analysis
