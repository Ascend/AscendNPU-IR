//===- ConvertToHIVMOp.cpp - Convert ops to HIVM Ops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-to-hivm-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
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

std::optional<Value> getPadValue(PatternRewriter &rewriter,
                                 std::optional<memref::AllocOp> maybeAlloc) {
  if (!maybeAlloc.has_value())
    return std::nullopt;
  // Compatible with older versions
  for (auto *user : maybeAlloc.value()->getUsers()) {
    if (llvm::isa_and_nonnull<hivm::VBrcOp>(user) &&
        user->getOperand(0).getType().isIntOrFloat()) {
      return user->getOperand(0);
    }
  }

  auto allocValue = maybeAlloc.value();
  auto padMarkOp = utils::getAnnotateOpWithAttr(allocValue, "pad_const");
  if (!padMarkOp.has_value())
    return std::nullopt;
  auto markOp = dyn_cast<annotation::MarkOp>(padMarkOp.value());
  auto padValue = markOp.getDynamicAttrValue("pad_const");
  removeMarkOpDynamicAttr(markOp, "pad_const", rewriter);

  return std::optional<Value>(padValue);
}

Value buildLeftPadNumForSubview(PatternRewriter &rewriter,
                                memref::SubViewOp subviewOp) {
  auto normalizeOffset = [](OpBuilder &builder, Location loc,
                            int64_t offsetRawInt,
                            int64_t numElemPerBlock) -> Value {
    LDBG("offsetRawInt = " << offsetRawInt << " from arith.constant\n");
    if (offsetRawInt < 0) {
      llvm::report_fatal_error("negative offset is invalid");
    }
    auto offsetInt = offsetRawInt % numElemPerBlock;
    return builder.create<arith::ConstantIndexOp>(loc, offsetInt);
  };

  auto loc = subviewOp.getLoc();
  auto numElemPerBlock = mlir::utils::getNumPerBlock(subviewOp.getType());
  auto offsets = subviewOp.getMixedOffsets();
  auto offset = offsets.back();
  LDBG("offset = " << offset << "\n");
  auto offsetValue = dyn_cast<Value>(offset);
  if (offsetValue) {
    // The maximum padding size is 32B. So we need to process the raw
    // left_pad.
    APSInt intVal;
    if (matchPattern(offsetValue, m_ConstantInt(&intVal))) {
      offsetValue = normalizeOffset(rewriter, loc, intVal.getSExtValue(),
                                    numElemPerBlock);
      return offsetValue;
    }
    // offset is Value but not arith.constant
    auto offsetTy = offsetValue.getType();
    if (!offsetTy.isIndex()) {
      llvm::report_fatal_error("offset must be index type");
    }
    auto numElemPerBlockVal =
        rewriter.create<arith::ConstantIndexOp>(loc, numElemPerBlock);
    auto leftPadValue =
        rewriter.create<arith::RemUIOp>(loc, offsetValue, numElemPerBlockVal);
    LDBG("leftPadValue = " << leftPadValue << "\n");
    return leftPadValue;
  }
  // handle the case where offset is IntegerAttr
  auto maybeConstantInt = getConstantIntValue(offset);
  if (!maybeConstantInt.has_value())
    llvm::report_fatal_error("offset as integer attr is not obtained");
  offsetValue = normalizeOffset(rewriter, loc, maybeConstantInt.value(),
                                numElemPerBlock);
  return offsetValue;
}

std::optional<Value> getLeftPadNum(PatternRewriter &rewriter,
                                   std::optional<memref::AllocOp> maybeAlloc,
                                   Value dst, Operation *anchorOp) {
  if (auto subviewOp = dst.getDefiningOp<memref::SubViewOp>()) {
    return buildLeftPadNumForSubview(rewriter, subviewOp);
  }
  if (!maybeAlloc.has_value())
    return std::nullopt;

  DominanceInfo dom(anchorOp->getParentOfType<func::FuncOp>());
  for (auto *user : maybeAlloc.value()->getUsers()) {
    auto subviewOp = dyn_cast<memref::SubViewOp>(user);
    if (!subviewOp)
      continue;
    if (subviewOp.getSource() != dst)
      continue;
    if (!dom.properlyDominates(subviewOp.getOperation(), anchorOp))
      continue;
    return buildLeftPadNumForSubview(rewriter, subviewOp);
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

std::optional<StringRef>
getEvictionPolicy(PatternRewriter &rewriter,
                  std::optional<memref::AllocOp> maybeAlloc) {
  if (!maybeAlloc.has_value())
    return std::nullopt;

  auto allocValue = maybeAlloc.value();
  std::optional<Operation *> evictionPolicy =
      utils::getAnnotateOpWithAttr(allocValue, "eviction_policy");
  if (!evictionPolicy.has_value()) {
    return std::nullopt;
  }
  StringAttr evictionAttrVal =
      evictionPolicy.value()->getAttrOfType<StringAttr>("eviction_policy");
  auto markOp = dyn_cast<annotation::MarkOp>(evictionPolicy.value());
  rewriter.eraseOp(markOp);
  return evictionAttrVal.getValue();
}

LogicalResult replaceMemCopyByHIVMLoadOp(memref::CopyOp copyOp,
                                         PatternRewriter &rewriter) {
  Value dst = copyOp.getTarget();
  auto maybeAlloc = utils::tracebackMemRefToAlloc(dst);
  auto maybePadValue = getPadValue(rewriter, maybeAlloc);
  auto maybeLeftPadNum = getLeftPadNum(rewriter, maybeAlloc, dst, copyOp);
  auto maybeEvictionPolicy = getEvictionPolicy(rewriter, maybeAlloc);
  LDBG(*copyOp << "\n");
  rewriter.setInsertionPoint(copyOp);
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
  // Default set to evict_first
  auto evictionPolicyAttr = rewriter.getAttr<hivm::EvictionPolicyAttr>(
      hivm::EvictionPolicy::EvictFirst);
  if (maybeEvictionPolicy.has_value()) {
    if (maybeEvictionPolicy->ends_with("evict_last")) {
      evictionPolicyAttr = rewriter.getAttr<hivm::EvictionPolicyAttr>(
          hivm::EvictionPolicy::EvictLast);
    }
  }
  loadOp.setEvictionPolicyAttr(evictionPolicyAttr);
  rewriter.replaceOp(copyOp, loadOp);

  // ensure pad value and left pad num are defined before load op
  // to avoid the violation of SSA dominance
  Operation *latestDef = nullptr;
  auto updateLatestDef = [&loadOp, &latestDef](Value v) {
    if (v) {
      if (Operation *defOp = v.getDefiningOp()) {
        if (defOp->getBlock() == loadOp->getBlock()) {
          if (!latestDef || latestDef->isBeforeInBlock(defOp)) {
            latestDef = defOp;
          }
        }
      }
    }
  };

  if (maybePadValue.has_value())
    updateLatestDef(maybePadValue.value());
  if (maybeLeftPadNum.has_value())
    updateLatestDef(maybeLeftPadNum.value());

  if (latestDef && loadOp->isBeforeInBlock(latestDef)) {
    rewriter.moveOpAfter(loadOp, latestDef);
  }
  return success();
}

bool isGMSafeSource(Value v) {
  Operation *defOp = v.getDefiningOp();
  return utils::isAllocLikeOp(v) || util::isGMPointerCastOp(defOp) ||
         (defOp && isa<memref::GetGlobalOp>(defOp));
}

bool isFromGMSpace(Value v) {
  SmallVector<Value> targetOPVec =
      utils::tracebackMemRefVecByTargetFn(v, isGMSafeSource);
  for (auto targetOP : targetOPVec) {
    auto defOp = targetOP.getDefiningOp();
    if (defOp != nullptr && !isa<hivm::PointerCastOp>(defOp) &&
        !isa<memref::GetGlobalOp>(defOp)) {
      return false;
    }
  }
  return true;
}

struct MemrefCopyOpLowering : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // do not convert mem copy inside simt_vf
    if (copyOp->getParentOfType<scf::ForallOp>() != nullptr) {
      return failure();
    }
    bool isInVectorFunction = false;
    if (auto funcOp = copyOp->getParentOfType<func::FuncOp>()) {
      isInVectorFunction = funcOp->hasAttr("hivm.vector_function");
    }

    Value src = copyOp.getSource();
    bool convertToLoad = !isInVectorFunction && isFromGMSpace(src);
    if (convertToLoad) {
      return replaceMemCopyByHIVMLoadOp(copyOp, rewriter);
    }

    Value dst = copyOp.getTarget();
    bool convertToStore = !isInVectorFunction && isFromGMSpace(dst);
    if (convertToStore) {
      rewriter.replaceOpWithNewOp<hivm::StoreOp>(copyOp, TypeRange(), src, dst);
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

struct RemovePadConstAnnotation : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp op,
                                PatternRewriter &rewriter) const override {
    if (op.isAnnotatedByDynamicAttr("pad_const")) {
      rewriter.eraseOp(op);
    }
    return success();
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

  // Avoid missing cleanup of annotations("pad_const") on memrefs.
  // Avoid the conflict with MemrefCopyOpLowering and
  // BufferizeMaterializeOpLowering.
  moduleOp->walk([&](func::FuncOp funcOp) {
    if (hacc::utils::isHost(funcOp))
      // avoid convert host op to hivm op
      return;

    // rewrite op within cur funcOp
    RewritePatternSet patterns(ctx);
    patterns.add<RemovePadConstAnnotation>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  });
}

std::unique_ptr<Pass> mlir::hivm::createConvertToHIVMOpPass() {
  return std::make_unique<ConvertToHIVMOpPass>();
}
