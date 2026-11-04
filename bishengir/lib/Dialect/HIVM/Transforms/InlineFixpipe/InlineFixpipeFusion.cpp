//===- InlineFixpipeFusion.cpp - Inline fixpipe fusions -------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeFusion.h"

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "hivm-inline-fixpipe"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::hivm {

static bool traceDownStoreOpWithSingleChain(Value v);

static bool hasCompatibleShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  if (!lhsType || !rhsType || !lhsType.hasRank() || !rhsType.hasRank())
    return false;
  if (lhsType.getRank() != rhsType.getRank())
    return false;
  return succeeded(
      verifyCompatibleShape(lhsType.getShape(), rhsType.getShape()));
}

static bool isInsertedToStatic(tensor::ExtractSliceOp extractSlice) {
  auto sliceType =
      dyn_cast<RankedTensorType>(extractSlice.getResult().getType());
  if (!sliceType || sliceType.hasStaticShape() ||
      !extractSlice.getResult().hasOneUse())
    return false;

  auto insertSlice =
      dyn_cast<tensor::InsertSliceOp>(*extractSlice->user_begin());
  if (!insertSlice)
    return false;

  auto destType = dyn_cast<RankedTensorType>(insertSlice.getDest().getType());
  return destType && destType.hasStaticShape();
}

bool isUserSliceSwappable(hivm::FixpipeOp fixpipe,
                          tensor::ExtractSliceOp extractSlice) {
  if (!extractSlice)
    return false;
  if (!isOnRegBasedArch(fixpipe))
    return true;
  return hasCompatibleShape(fixpipe.getSource(), extractSlice.getSource()) &&
         !isInsertedToStatic(extractSlice);
}

bool isUserTransposeInlinable(hivm::FixpipeOp op, Operation *user) {
  if (!isOnRegBasedArch(op))
    return false;
  auto transpose = dyn_cast<hivm::VTransposeOp>(user);
  if (!transpose || op.getDmaMode() != FixpipeDMAMode::NZ2ND)
    return false;
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  return permutation.size() == 2 && permutation[0] == 1 && permutation[1] == 0;
}

void inlineFixpipeWithQuant(PatternRewriter &rewriter, Location loc,
                            hivm::FixpipeOp op,
                            ArrayRef<hivm::VCastOp> castChain,
                            Value newFixpipeSrcTensor) {
  std::optional<FixpipePreQuantMode> quantMode =
      getQuantModeForCastChain(castChain, op.getPreQuant());
  if (!quantMode) {
    LDBG("cast op quant mode is null");
    return;
  }
  auto quantModeAttr =
      FixpipePreQuantModeAttr::get(op.getContext(), quantMode.value());

  hivm::VCastOp lastCast = castChain.back();
  rewriter.setInsertionPointAfter(lastCast);
  Value fixpipeInit =
      utils::createEmptyOp(rewriter, loc, lastCast.getResult()[0]);
  MLIRContext *ctx = rewriter.getContext();
  bool regBased = isOnRegBasedArch(op);
  FixpipeDMAModeAttr dmaModeAttr =
      regBased ? op.getDmaModeAttr()
               : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
  auto newFixpipeOp = FixpipeOp::clone(
      rewriter, op, newFixpipeSrcTensor, fixpipeInit,
      {rewriter.getNamedAttr(op.getDmaModeAttrName(), dmaModeAttr),
       rewriter.getNamedAttr(op.getPreQuantAttrName(), quantModeAttr)});
  rewriter.replaceAllUsesWith(lastCast.getResult()[0],
                              newFixpipeOp.getResultTensor());
  for (hivm::VCastOp castOp : llvm::reverse(castChain))
    rewriter.eraseOp(castOp);
  rewriter.eraseOp(op);
  LDBG("InlineFixpipeWithPreQuant");
}

void inlineFixpipeWithRelu(PatternRewriter &rewriter, Location loc,
                           hivm::FixpipeOp op, hivm::VReluOp reluOp) {
  std::optional<FixpipePreReluMode> reluMode = getReluMode(reluOp);
  rewriter.modifyOpInPlace(op, [&]() { op.setPreRelu(reluMode); });
  rewriter.replaceAllUsesWith(reluOp.getResult()[0], op.getResult(0));
  rewriter.eraseOp(reluOp);
  LDBG("InlineFixpipeWithPreRelu");
}

void inlineFixpipeWithStore(PatternRewriter &rewriter, Location loc,
                            hivm::FixpipeOp op, hivm::StoreOp storeOp,
                            Value fixpipeSrcTensor) {
  assert(storeOp->getNumResults() == 0 && "StoreOp must have 0 results");
  rewriter.setInsertionPointAfter(storeOp);
  auto dst = storeOp.getDst();
  auto storeAttr = storeOp.getAtomicKindAttr();
  auto noneAtomicAttr =
      AtomicKindAttr::get(op->getContext(), ::mlir::hivm::AtomicKind::NONE);
  // The fused fixpipe writes the store destination, so it carries no result.
  auto newFixpipeOp = FixpipeOp::clone(rewriter, op, fixpipeSrcTensor, dst);
  if (storeAttr) {
    auto typeAttr =
        TypeAttr::get(mlir::cast<ShapedType>(dst.getType()).getElementType());
    rewriter.setInsertionPoint(newFixpipeOp);
    rewriter.create<SetAtomicOp>(loc, storeAttr, typeAttr);
    rewriter.setInsertionPointAfter(newFixpipeOp);
    rewriter.create<SetAtomicOp>(loc, noneAtomicAttr, typeAttr);
  }
  rewriter.eraseOp(storeOp);
  rewriter.eraseOp(op);
  LDBG("InlineFixpipeEnd");
}

bool hasQuantScaleCompileHint(hivm::VMulOp op) {
  return llvm::any_of(op->getUsers(), [](Operation *userOp) {
    auto markOp = dyn_cast<annotation::MarkOp>(userOp);
    return markOp && markOp->hasAttr(utils::kInlinableQuantScaleAttr);
  });
}

bool isUserQuantScaleInlinable(hivm::FixpipeOp op, hivm::VMulOp vMulOp) {
  auto dualDstMode = op.getDualDstModeAttr();
  if ((dualDstMode &&
       dualDstMode.getDualDstMode() != FixpipeDualDstMode::NO_DUAL) ||
      op.getQuantScale())
    return false;
  if (llvm::count_if(vMulOp->getUsers(), [](Operation *userOp) {
        return !isa<annotation::MarkOp>(userOp);
      }) != 1)
    return false;
  Value result = vMulOp.getResult()[0];
  if (auto cast =
          dyn_cast_if_present<hivm::VCastOp>(getSingleSiftedUser(result))) {
    auto casts = collectVCastChain(cast);
    if (getQuantModeForCastChain(casts, FixpipePreQuantMode::QF322F32_PRE) !=
        FixpipePreQuantMode::QF322F16_PRE)
      return false;
  } else if (!traceDownStoreOpWithSingleChain(result)) {
    return false;
  }

  unsigned fixpipeOperand =
      vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op ? 0 : 1;
  Value quantScale = vMulOp.getDpsInputOperand(1 - fixpipeOperand)->get();
  return utils::isScalarLike(quantScale);
}

void inlineFixpipeWithQuantScale(PatternRewriter &rewriter, hivm::FixpipeOp op,
                                 hivm::VMulOp vMulOp) {
  unsigned fixpipeOperand =
      vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op ? 0 : 1;
  Value quantScale = vMulOp.getDpsInputOperand(1 - fixpipeOperand)->get();
  Value dst = vMulOp.getDpsInitOperand(0)->get();

  auto preQuant = op.getPreQuant();
  if (preQuant == FixpipePreQuantMode::NO_QUANT &&
      getElementTypeOrSelf(op.getSrcOperandType()).isF32() &&
      getElementTypeOrSelf(dst.getType()).isF32()) {
    preQuant = FixpipePreQuantMode::QF322F32_PRE;
  }

  rewriter.setInsertionPointAfter(vMulOp);
  quantScale =
      utils::extractScalarValue(rewriter, op.getLoc(), quantScale).value();
  auto newFixpipe = FixpipeOp::clone(
      rewriter, op, op.getSource(), dst,
      {rewriter.getNamedAttr(
          op.getPreQuantAttrName(),
          FixpipePreQuantModeAttr::get(rewriter.getContext(), preQuant))});
  newFixpipe.getQuantScaleMutable().assign(quantScale);
  for (Operation *user : llvm::make_early_inc_range(vMulOp->getUsers())) {
    if (isa<annotation::MarkOp>(user)) {
      newFixpipe->setAttr(utils::kInlinedQuantScaleAttr,
                          rewriter.getUnitAttr());
      rewriter.eraseOp(user);
    }
  }
  rewriter.replaceOp(vMulOp, newFixpipe.getResultTensor());
  rewriter.eraseOp(op);
  LDBG("InlineFixpipeWithQuantScale");
}

bool inlineFixpipeWithTranspose(PatternRewriter &rewriter, hivm::FixpipeOp op,
                                hivm::VTransposeOp transpose) {
  rewriter.setInsertionPointAfter(transpose);
  auto dmaMode =
      FixpipeDMAModeAttr::get(rewriter.getContext(), FixpipeDMAMode::NZ2DN);
  auto newFixpipe = FixpipeOp::clone(
      rewriter, op, op.getSource(), transpose.getDst(),
      {rewriter.getNamedAttr(op.getDmaModeAttrName(), dmaMode)});
  Value newResult = newFixpipe.getResultTensor();
  if (newResult) {
    // The mark usually sits on the old fixpipe, which is before this new
    // result. Clone it after the result so the use dominates.
    for (Operation *user :
         llvm::make_early_inc_range(op.getResultTensor().getUsers())) {
      if (!isa<annotation::MarkOp>(user))
        continue;
      rewriter.setInsertionPointAfterValue(newResult);
      Operation *cloned = rewriter.clone(*user);
      cloned->replaceUsesOfWith(op.getResultTensor(), newResult);
      rewriter.eraseOp(user);
    }
    rewriter.replaceOp(transpose, newResult);
  } else {
    // A memref destination yields the buffer form of the fixpipe, which has no
    // result. The buffer-form transpose has no result either, so there is
    // nothing to replace: dropping it lets the fused fixpipe write the
    // destination directly.
    rewriter.eraseOp(transpose);
  }
  rewriter.eraseOp(op);
  LDBG("InlineFixpipeWithTranspose");
  return true;
}

void swapFixpipeWithExtractSlice(PatternRewriter &rewriter, Location loc,
                                 hivm::FixpipeOp op,
                                 tensor::ExtractSliceOp extractSliceOp) {
  rewriter.setInsertionPointAfter(extractSliceOp);
  auto fixpipeSrc = op.getDpsInputOperand(0)->get();

  auto newExtractSliceResType =
      extractSliceOp.getResultType().clone(getElementTypeOrSelf(fixpipeSrc));
  auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      extractSliceOp.getLoc(), newExtractSliceResType, fixpipeSrc,
      extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
      extractSliceOp.getMixedStrides());

  auto newExtractSliceResult = newExtractSliceOp->getResult(0);
  Value fixpipeInit = utils::createEmptyOpWithTargetElemType(
      rewriter, extractSliceOp.getLoc(), newExtractSliceResult,
      getElementTypeOrSelf(extractSliceOp.getResult()));

  MLIRContext *ctx = rewriter.getContext();
  bool regBased = isOnRegBasedArch(op);
  FixpipeDMAModeAttr dmaModeAttr =
      regBased ? op.getDmaModeAttr()
               : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
  auto newFixpipeOp = FixpipeOp::clone(
      rewriter, op, newExtractSliceResult, fixpipeInit,
      {rewriter.getNamedAttr(op.getDmaModeAttrName(), dmaModeAttr)});
  rewriter.replaceOp(extractSliceOp, newFixpipeOp.getResultTensor());
  rewriter.eraseOp(op);
  LDBG("InlineFixpipeWithExtractSliceReshape");
}

bool traceDownStoreOpWithSingleChain(Value v) {
  auto isMachedOp = [](Operation *op, Value v) {
    return isa<hivm::StoreOp>(op);
  };
  return traceSingleChainUser(v, isMachedOp);
}

bool moveFixpipeOutOfScfFor(PatternRewriter &rewriter, Location loc,
                            hivm::FixpipeOp fixPipeOp, scf::ForOp scfForOp,
                            Value fixpipeResTensor) {
  SmallVector<Value> yieldValues = llvm::to_vector(scfForOp.getYieldedValues());
  auto idx = findIdx(yieldValues, fixpipeResTensor);
  if (!idx.has_value())
    return false;

  // A value defined inside the loop body is not available after it.
  auto isLoopLocal = [scfForOp](Value operand) {
    if (Operation *defOp = operand.getDefiningOp())
      return scfForOp->isProperAncestor(defOp);
    auto blockArg = dyn_cast<BlockArgument>(operand);
    return blockArg && scfForOp->isAncestor(blockArg.getOwner()->getParentOp());
  };
  for (Value operand : fixPipeOp->getOperands()) {
    if (operand == fixPipeOp.getSrc() || operand == fixPipeOp.getDst())
      continue;
    if (isLoopLocal(operand)) {
      LDBG("Cannot hoist fixpipe with loop-local operand " << operand);
      return false;
    }
  }

  LDBG("InlineFixpipeWithYield");
  rewriter.replaceAllUsesWith(fixpipeResTensor,
                              fixPipeOp.getDpsInputOperand(0)->get());

  rewriter.setInsertionPointAfter(scfForOp);
  auto fixpipeInit =
      utils::createEmptyOp(rewriter, scfForOp->getLoc(), fixpipeResTensor);
  auto newFixpipeOp = FixpipeOp::clone(
      rewriter, fixPipeOp, scfForOp->getResult(idx.value()), fixpipeInit);
  rewriter.replaceAllUsesExcept(scfForOp->getResult(idx.value()),
                                newFixpipeOp.getResultTensor(), newFixpipeOp);
  LDBG("moveFixpipeOutOfScfFor");
  return true;
}

} // namespace mlir::hivm
