//-------------------------SubBlockGuardResultFree.cpp------------------------//
//
// Rewrites value-returning sub-block guards into result-free guards, keeping an
// anchor-only else when the else branch carries cross-core-GSS anchors.
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

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/SubBlockGuardCleanup.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

/// A then-yielded value is hoistable iff produced by a memref.alloc/alloca
/// directly in the then block with all operands defined outside the guard
bool isHoistableThenOutput(Value yielded, scf::IfOp ifOp) {
  Operation *def = yielded.getDefiningOp();
  if (!def || def->getBlock() != ifOp.thenBlock())
    return false;
  if (!isa<memref::AllocOp, memref::AllocaOp>(def))
    return false;
  for (Value operand : def->getOperands())
    if (Operation *operandDef = operand.getDefiningOp())
      if (ifOp->isAncestor(operandDef))
        return false;
  return true;
}

} // namespace

void mlir::hivm::partition_and_bind::makeSubBlockGuardResultFree(
    scf::IfOp ifOp) {
  if (ifOp.getNumResults() == 0 || !ifOp.elseBlock())
    return;
  auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());

  for (Value result : ifOp.getResults()) {
    auto memrefTy = dyn_cast<MemRefType>(result.getType());
    if (!memrefTy || !memrefTy.hasStaticShape())
      return; // bail: leave value-returning
  }

  OpBuilder builder(ifOp);

  llvm::SmallVector<Value> storage;
  storage.reserve(ifOp.getNumResults());
  for (auto [result, yielded] :
       llvm::zip_equal(ifOp.getResults(), thenYield.getOperands())) {
    if (isHoistableThenOutput(yielded, ifOp)) {
      yielded.getDefiningOp()->moveBefore(ifOp);
      storage.push_back(yielded);
      continue;
    }
    builder.setInsertionPoint(ifOp);
    Value fresh = builder.create<memref::AllocOp>(
        ifOp.getLoc(), cast<MemRefType>(result.getType()), ValueRange{},
        /*alignment=*/builder.getI64IntegerAttr(64));
    // Reconcile with an hivm.hir.copy
    builder.setInsertionPoint(thenYield);
    builder.create<hivm::CopyOp>(ifOp.getLoc(), TypeRange{},
                                               /*src=*/yielded, /*dst=*/fresh);
    storage.push_back(fresh);
  }

  for (auto [result, s] : llvm::zip_equal(ifOp.getResults(), storage))
    result.replaceAllUsesWith(s);

  // InsertAnchorsAndBackup may have placed cross-core-GSS anchors in the else
  // branch.
  auto elseAnchors =
      llvm::to_vector(ifOp.elseBlock()->getOps<hivm::AnchorOp>());
  bool keepElse = !elseAnchors.empty();

  // Rebuild as a result-less guard; move the then body (minus its scf.yield).
  builder.setInsertionPoint(ifOp);
  auto newIf = builder.create<scf::IfOp>(ifOp.getLoc(), ifOp.getCondition(),
                                         /*withElseRegion=*/keepElse);
  Block *newThen = newIf.thenBlock();
  newThen->getOperations().splice(newThen->getTerminator()->getIterator(),
                                  ifOp.thenBlock()->getOperations(),
                                  ifOp.thenBlock()->begin(),
                                  thenYield->getIterator());
  if (keepElse) {
    Operation *elseYield = newIf.elseBlock()->getTerminator();
    for (hivm::AnchorOp anchorOp : elseAnchors)
      anchorOp->moveBefore(elseYield);
  }
  newIf->setAttrs(ifOp->getAttrs());
  ifOp.erase();
}
