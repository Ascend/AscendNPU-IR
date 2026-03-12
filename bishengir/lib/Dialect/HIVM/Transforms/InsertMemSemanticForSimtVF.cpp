//===------------- InsertMemSemanticForSimtVF.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement insert memory semantic for simt vf.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>

namespace mlir {
#define GEN_PASS_DEF_INSERTMEMSEMANTICFORSIMTVF
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct InsertMemSemanticForSimtVFPass
    : public impl::InsertMemSemanticForSimtVFBase<
          InsertMemSemanticForSimtVFPass> {
  void runOnOperation() override;

private:
  void dealWithReferenceOutOfScope(scope::ScopeOp scopeOp, OpBuilder &builder);
  void dealWithReturnValue(scope::ScopeOp &scopeOp, OpBuilder &builder);
};

void InsertMemSemanticForSimtVFPass::dealWithReferenceOutOfScope(
    scope::ScopeOp scopeOp, OpBuilder &builder) {
  llvm::DenseMap<Value, llvm::SmallVector<Operation *>> valsNeedLoad;
  for (auto &op : scopeOp.getBody()->getOperations()) {
    for (auto operand : op.getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == scopeOp.getBody()) {
        continue;
      }
      if (llvm::isa<RankedTensorType>(operand.getType())) {
        if (valsNeedLoad.count(operand)) {
          valsNeedLoad[operand].emplace_back(&op);
        } else {
          valsNeedLoad[operand] = {&op};
        }
      }
    }
  }
  auto insertLoadOp = [&scopeOp, &builder](Value val, Operation *op) {
    auto tensorType = llvm::cast<RankedTensorType>(val.getType());
    builder.setInsertionPoint(scopeOp);
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto memrefVal = builder.create<bufferization::ToMemrefOp>(
        scopeOp->getLoc(), memrefType, val);

    // Transfer data from simd(UB) to simt(Register)
    builder.setInsertionPoint(op);
    auto loadOp =
        builder.create<hivm::LocalLoadOp>(op->getLoc(), tensorType, memrefVal);

    val.replaceUsesWithIf(loadOp->getResult(0), [&op](OpOperand &operand) {
      return operand.getOwner() == op;
    });
  };
  for (const auto &p : valsNeedLoad) {
    for (auto op : p.second) {
      insertLoadOp(p.first, op);
    }
  }
}

void InsertMemSemanticForSimtVFPass::dealWithReturnValue(
    scope::ScopeOp &scopeOp, OpBuilder &builder) {
  auto returnOp =
      llvm::cast<scope::ReturnOp>(scopeOp.getBody()->getTerminator());
  if (returnOp->getNumOperands() == 0) {
    return;
  }
  for (size_t i = 0; i < returnOp->getNumOperands(); i++) {
    auto val = returnOp->getOperand(i);
    auto tensorType = llvm::dyn_cast<RankedTensorType>(val.getType());
    assert(tensorType && "simt vf return value should be tensor");

    // Alloc unified buffer for caching data produced from simt
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    builder.setInsertionPoint(scopeOp);
    auto memrefOfUB =
        builder.create<memref::AllocOp>(scopeOp->getLoc(), memrefType);

    // Transfer data from simt(Register) to simd(UB)
    builder.setInsertionPoint(returnOp);
    builder.create<hivm::LocalStoreOp>(returnOp->getLoc(), memrefOfUB, val);

    // Get latest data snapshot after simt vf scope
    builder.setInsertionPointAfter(scopeOp);
    auto tensorOfUB = builder.create<bufferization::ToTensorOp>(
        scopeOp->getLoc(), tensorType, memrefOfUB, true);
    scopeOp->getResult(i).replaceAllUsesWith(tensorOfUB);

    returnOp->eraseOperand(i);
  }
  builder.setInsertionPoint(scopeOp);
  auto newScopeOp =
      builder.create<scope::ScopeOp>(scopeOp->getLoc(), TypeRange{});
  newScopeOp->setAttrs(scopeOp->getAttrs());
  builder.createBlock(&newScopeOp->getRegion(0));
  newScopeOp.getBody()->getOperations().splice(
      newScopeOp.getBody()->end(), scopeOp.getBody()->getOperations());
  scopeOp->erase();
  scopeOp = newScopeOp;
}

} // namespace

void InsertMemSemanticForSimtVFPass::runOnOperation() {
  auto mod = getOperation();
  auto ctx = &getContext();
  OpBuilder builder(ctx);
  mod->walk([this, &builder](scope::ScopeOp scopeOp) {
    if (!util::isSIMTVF(scopeOp)) {
      return;
    }
    dealWithReferenceOutOfScope(scopeOp, builder);
    dealWithReturnValue(scopeOp, builder);
  });
}

std::unique_ptr<Pass> mlir::hivm::createInsertMemSemanticForSimtVFPass() {
  return std::make_unique<InsertMemSemanticForSimtVFPass>();
}
