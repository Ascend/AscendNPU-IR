//===- ConvertToHIVMOp.cpp - Convert ops to HIVM Ops ----------------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOHIVMOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
//===---------------------------------------------------------------------===//
// Patterns that convert ops from other dialects to HIVM ops.
//===---------------------------------------------------------------------===//

std::optional<Value> getPadValue(std::optional<memref::AllocOp> maybeAlloc) {
  if (!maybeAlloc.has_value())
    return std::nullopt;

  for (auto *user : maybeAlloc.value()->getUsers()) {
    if (llvm::isa_and_nonnull<hivm::VBrcOp>(user) &&
        user->getOperand(0).getType().isIntOrFloat()) {
      return user->getOperand(0);
    }
  }
  return std::nullopt;
}

std::optional<Value> getLeftPadNum(PatternRewriter &rewriter,
                                   std::optional<memref::AllocOp> maybeAlloc) {
  if (!maybeAlloc.has_value())
    return std::nullopt;

  for (auto *user : maybeAlloc.value()->getUsers()) {
    if (auto subviewOp = llvm::dyn_cast<memref::SubViewOp>(user)) {
      auto offsets = subviewOp.getMixedOffsets();
      return mlir::getValueOrCreateConstantIndexOp(
          rewriter, subviewOp->getLoc(), offsets.back());
    }
  }
  return std::nullopt;
}

std::pair<std::optional<Operation *>, std::optional<Value>>
getInitInfo(Operation *op, hivm::LoadOp loadOp) {
  if (!llvm::isa<hivm::VBrcOp>(op))
    return {std::nullopt, std::nullopt};
  if (!op->getOperand(0).getType().isIntOrFloat())
    return {std::nullopt, std::nullopt};

  if (op->getBlock() == loadOp->getBlock())
    return {op, std::nullopt};
  auto *opParentOp = op->getParentOp();
  if (opParentOp == nullptr)
    llvm::report_fatal_error("unhandled case for null opParentOp");
  if (opParentOp->getBlock() == loadOp->getBlock() &&
      isa<scf::IfOp>(opParentOp)) {
    auto ifOp = cast<scf::IfOp>(opParentOp);
    return {op, ifOp.getCondition()};
  }

  return {std::nullopt, std::nullopt};
}

std::pair<std::optional<Operation *>, std::optional<Value>>
getUniqueInitInfo(std::optional<memref::AllocOp> maybeAlloc,
                  hivm::LoadOp loadOp) {
  if (!maybeAlloc.has_value())
    return {std::nullopt, std::nullopt};

  std::optional<Operation *> initOp = std::nullopt;
  std::optional<Value> initCondition = std::nullopt;
  for (auto *user : (*maybeAlloc)->getUsers()) {
    if (llvm::isa<hivm::LoadOp>(user))
      continue;
    auto maybeInitOp = getInitInfo(user, loadOp).first;
    if (maybeInitOp.has_value() && !initOp.has_value()) {
      std::tie(initOp, initCondition) = getInitInfo(user, loadOp);
    } else if (user->getDialect()->getNamespace() ==
               HIVMDialect::getDialectNamespace()) {
      // there are other write access op among alloc and load op, cannot
      // inline load with init
      return {std::nullopt, std::nullopt};
    }
  }

  return {initOp, initCondition};
}

LogicalResult replaceMemCopyByHIVMLoadOp(memref::CopyOp copyOp,
                                         PatternRewriter &rewriter) {
  Value dst = copyOp.getTarget();
  auto maybeAlloc = utils::tracebackMemRefToAlloc(dst);
  auto maybePadValue = getPadValue(maybeAlloc);
  auto maybeLeftPadNum = getLeftPadNum(rewriter, maybeAlloc);

  auto loadOp = rewriter.create<hivm::LoadOp>(copyOp->getLoc(), TypeRange(),
                                              copyOp.getSource(), dst);
  if (maybeLeftPadNum.has_value()) {
    loadOp.getLeftPaddingNumMutable().assign(maybeLeftPadNum.value());
  }
  if (maybePadValue.has_value()) {
    auto padModeAttr =
        rewriter.getAttr<hivm::PadModeAttr>(hivm::PadMode::PadValue);
    loadOp.setPadModeAttr(padModeAttr);
    loadOp.getPadValueMutable().assign(maybePadValue.value());
    auto [inlineInitOp, inlineInitCond] = getUniqueInitInfo(maybeAlloc, loadOp);
    if (inlineInitOp.has_value()) {
      loadOp.setInitOutBuffer(true);
      rewriter.eraseOp(inlineInitOp.value());
    }
    if (inlineInitCond.has_value()) {
      loadOp.getInitConditionMutable().assign(inlineInitCond.value());
    }
  }

  // TODO: change TA to create hivm.load/store op directly
  Value allocRef = utils::tracebackMemRef(copyOp.getTarget());
  auto implicitTransposeAttrForAlloc = utils::getAnnotateOpWithAttr(
    allocRef, "MayImplicitTransposeWithLastAxis");
  // We require the attribute to be marked on AllocOp
  if (implicitTransposeAttrForAlloc.has_value()) {
    loadOp.setMayImplicitTransposeWithLastAxis(true);
  }
 
  rewriter.replaceOp(copyOp, loadOp);
  return success();
}

bool isAllocLikeOrGMPointerCastOp(Value v) {
  return utils::isAllocLikeOp(v) || util::isGMPointerCastOp(v.getDefiningOp());
}

bool isFromGMSpace(Value v) {
  SmallVector<Value> targetOPVec =
      utils::tracebackMemRefVecByTargetFn(v, isAllocLikeOrGMPointerCastOp);
  for (auto targetOP : targetOPVec) {
    auto defOp = targetOP.getDefiningOp();
    if (defOp != nullptr && !isa<hivm::PointerCastOp>(defOp)) {
      return false;
    }
  }
  return true;
}

struct MemrefCopyOpLowering : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSource();
    bool convertToLoad = isFromGMSpace(src);
    if (convertToLoad) {
      return replaceMemCopyByHIVMLoadOp(copyOp, rewriter);
    }

    Value dst = copyOp.getTarget();
    bool convertToStore = isFromGMSpace(dst);
    if (convertToStore) {
      auto storeOp = rewriter.replaceOpWithNewOp<hivm::StoreOp>(
          copyOp, TypeRange(), src, dst);
      // TODO: change TA to create hivm.load/store op directly
      auto implicitTransposeAttr = utils::getAnnotateOpWithAttr(
          copyOp.getTarget(), "MayImplicitTransposeWithLastAxis");
      if (implicitTransposeAttr.has_value()) {
        storeOp.setMayImplicitTransposeWithLastAxis(true);
      }
      return success();
    }

    rewriter.replaceOpWithNewOp<hivm::CopyOp>(copyOp, TypeRange(), src, dst);
    return success();
  }
};

struct BufferizeMaterializeOpLowering
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern<
      bufferization::MaterializeInDestinationOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp bufMIDOp,
                  PatternRewriter &rewriter) const override {
    Value dst = bufMIDOp.getDest();
    bool convertToStore = isFromGMSpace(dst);
    if (convertToStore) {
      rewriter.replaceOpWithNewOp<hivm::StoreOp>(bufMIDOp, TypeRange(),
                                                 bufMIDOp.getSource(), dst);
      return success();
    }
    return failure();
  }
};

void populateHIVMOpRewritingRule(RewritePatternSet &patterns) {
  patterns.add<MemrefCopyOpLowering, BufferizeMaterializeOpLowering>(
      patterns.getContext());
}

struct ConvertToHIVMOpPass
    : public impl::ConvertToHIVMOpBase<ConvertToHIVMOpPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertToHIVMOpPass::runOnOperation() {
  auto *ctx = &getContext();
  Operation *moduleOp = getOperation();
  moduleOp->walk([&](func::FuncOp funcOp) {
    if (hacc::utils::isHost(funcOp))
      // avoid convert host op to hivm op
      return;

    // rewrite op within cur funcOp
    RewritePatternSet patterns(ctx);
    populateHIVMOpRewritingRule(patterns);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  });
}

std::unique_ptr<Pass> mlir::hivm::createConvertToHIVMOpPass() {
  return std::make_unique<ConvertToHIVMOpPass>();
}
