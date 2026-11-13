//===- SaveCVPipelineSnapshot.cpp - Snapshot IR before OutlineCVScopes ---===//
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
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {
#define GEN_PASS_DEF_SAVECVPIPELINESNAPSHOT
#define GEN_PASS_DEF_ERASECVPIPELINESNAPSHOTS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace hivm
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

constexpr llvm::StringLiteral kHaccEntryAttrName = "hacc.entry";

bool containsLoop(func::FuncOp func) {
  bool found = false;
  func.walk([&](scf::ForOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

struct SaveCVPipelineSnapshotPass
    : public mlir::hivm::impl::SaveCVPipelineSnapshotBase<
          SaveCVPipelineSnapshotPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal() || isCVPipelineSnapshot(func) || !containsLoop(func))
      return;
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;
    for (func::FuncOp candidate : module.getOps<func::FuncOp>()) {
      auto ref = candidate->getAttrOfType<FlatSymbolRefAttr>(
          kCVPipelineSnapshotAttrName);
      if (ref && ref.getValue() == func.getSymName())
        return;
    }

    func::FuncOp snapshot = func.clone();
    snapshot.setSymName(func.getSymName().str() + ".cv_pipeline_snapshot");
    snapshot.setPrivate();
    snapshot->setAttr(kCVPipelineSnapshotAttrName,
                      FlatSymbolRefAttr::get(func.getSymNameAttr()));
    snapshot->removeAttr(kHaccEntryAttrName);
    module.push_back(snapshot);
  }
};

struct EraseCVPipelineSnapshotsPass
    : public mlir::hivm::impl::EraseCVPipelineSnapshotsBase<
          EraseCVPipelineSnapshotsPass> {
  void runOnOperation() override {
    SmallVector<func::FuncOp> snapshots;
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>())
      if (isCVPipelineSnapshot(func))
        snapshots.push_back(func);
    for (func::FuncOp snapshot : snapshots)
      snapshot.erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createSaveCVPipelineSnapshotPass() {
  return std::make_unique<SaveCVPipelineSnapshotPass>();
}

std::unique_ptr<Pass> mlir::hivm::createEraseCVPipelineSnapshotsPass() {
  return std::make_unique<EraseCVPipelineSnapshotsPass>();
}
