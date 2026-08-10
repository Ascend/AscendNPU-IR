//===------------- LegalizeBoolForSimtVF.cpp   ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement Legalize Bool for simt vf.
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
#define GEN_PASS_DEF_LEGALIZEBOOLFORSIMTVF
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct LegalizeBoolForSimtVFPass
    : public impl::LegalizeBoolForSimtVFBase<LegalizeBoolForSimtVFPass> {
  void runOnOperation() override;

 private:
  void dealWithReferenceOutOfScope(scope::ScopeOp scopeOp, OpBuilder &builder);
  void dealWithReturnValue(scope::ScopeOp &scopeOp, OpBuilder &builder);
};

// Get VCastOp with empty tensor.
Value getVCastOpWithEmpty(
    Type typeToCast, Value valToCast, OpBuilder &builder) {
  auto defOp = valToCast.getDefiningOp();
  Value dstEmpty = utils::createEmptyOpWithTargetElemType(
      builder, defOp->getLoc(), valToCast, typeToCast);
  auto roundModeAttr =
      builder.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);
  Value castedRet =
      builder
          .create<hivm::VCastOp>(defOp->getLoc(),
              TypeRange(dstEmpty.getType()),
              valToCast,
              dstEmpty,
              roundModeAttr,
              builder.getAttr<hivm::TypeFnAttr>(hivm::TypeFn::cast_signed))
          .getResults()[0];
  return castedRet;
}

void LegalizeBoolForSimtVFPass::dealWithReferenceOutOfScope(
    scope::ScopeOp scopeOp, OpBuilder &builder) {
  scopeOp.getBody()->walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto operand = op->getOperand(i);
      auto defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      // If the operand is a ranked tensor and the defOp is not in the scope,
      // we need to extend the tensor to i8 when the element type is i1.
      if (llvm::isa<RankedTensorType>(operand.getType())
          && !scopeOp->isAncestor(defOp)) {
        auto defOpElemTy =
            llvm::dyn_cast<RankedTensorType>(defOp->getResult(0).getType())
                .getElementType();
        if (!defOpElemTy.isInteger(1))
          continue;
        builder.setInsertionPointAfter(defOp);
        // Cast the i1 tensor to i8 tensor followed its defining op.
        Value extVal = getVCastOpWithEmpty(
            IntegerType::get(&getContext(), 8), defOp->getResult(0), builder);
        builder.setInsertionPoint(op);
        // Inside the scope, cast back to i1 tensor to keep consistent with
        // the original type.
        Value truncVal = getVCastOpWithEmpty(
            IntegerType::get(&getContext(), 1), extVal, builder);
        op->setOperand(i, truncVal);
      }
    }
  });
}

void LegalizeBoolForSimtVFPass::dealWithReturnValue(
    scope::ScopeOp &scopeOp, OpBuilder &builder) {
  auto returnOp =
      llvm::cast<scope::ReturnOp>(scopeOp.getBody()->getTerminator());
  if (returnOp->getNumOperands() == 0) {
    return;
  }
  SmallVector<Value> scopeRetVals;
  SmallVector<int> legalizedRetIds;
  for (unsigned i = 0; i < returnOp->getNumOperands(); i++) {
    auto val = returnOp->getOperand(i);
    auto tensorType = llvm::dyn_cast<RankedTensorType>(val.getType());
    assert(tensorType && "simt vf return value should be tensor");

    Type elementType = tensorType.getElementType();
    Value storeVal = val;
    if (elementType.isInteger(1)) {
      // If the element type of tensor to return is i1, we need to legalize it
      // to i8.
      builder.setInsertionPoint(returnOp);
      auto i8Type = IntegerType::get(&getContext(), 8);
      Type newTensorType = RankedTensorType::get(tensorType.getShape(), i8Type);
      storeVal = builder.create<arith::ExtUIOp>(
          returnOp->getLoc(), newTensorType, val);
      returnOp->setOperand(i, storeVal);
      legalizedRetIds.push_back(i);
    }
    scopeRetVals.push_back(returnOp->getOperand(i));
  }
  builder.setInsertionPointAfter(scopeOp);

  // If any of the return value of old scope is legalized,we need to
  // create a new scope with legalized return values.
  if (!legalizedRetIds.empty()) {
    // Create a new scope with legalized return values.
    auto newScopeOp = builder.create<scope::ScopeOp>(
        scopeOp->getLoc(), TypeRange(scopeRetVals));
    newScopeOp->setAttrs(scopeOp->getAttrs());

    builder.cloneRegionBefore(scopeOp.getRegion(),
        newScopeOp.getRegion(),
        newScopeOp.getRegion().begin());

    builder.setInsertionPointAfter(newScopeOp);
    // Update the use of the old scope with legalized new scope return values.
    for (size_t i = 0; i < scopeRetVals.size(); i++) {
      Value newScopeRetVal = newScopeOp->getResult(i);
      if (llvm::is_contained(legalizedRetIds, i)) {
        // If the return value is legalized, we need to cast it back to i1 to keep
        // consistent with the old scope's use.
        newScopeRetVal = getVCastOpWithEmpty(IntegerType::get(&getContext(), 1),
            newScopeOp->getResult(i),
            builder);
      }
      scopeOp->getResult(i).replaceAllUsesWith(newScopeRetVal);
    }
    scopeOp->erase();
    scopeOp = newScopeOp;
  }
}
}  // namespace

void LegalizeBoolForSimtVFPass::runOnOperation() {
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

std::unique_ptr<Pass> mlir::hivm::createLegalizeBoolForSimtVFPass() {
  return std::make_unique<LegalizeBoolForSimtVFPass>();
}
