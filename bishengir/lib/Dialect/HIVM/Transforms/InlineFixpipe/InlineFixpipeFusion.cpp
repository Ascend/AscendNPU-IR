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

Type getInitType(Value v, hivm::FixpipePreQuantMode quant,
                 PatternRewriter &rewriter) {
  if (quant == FixpipePreQuantMode ::NO_QUANT)
    return getElementTypeOrSelf(v);
  if (quant == FixpipePreQuantMode ::F322F16)
    return rewriter.getF16Type();
  if (quant == FixpipePreQuantMode ::F322BF16)
    return rewriter.getBF16Type();
  if (quant == FixpipePreQuantMode::S322I8)
    return rewriter.getI8Type();
  if (quant == FixpipePreQuantMode::QF322F32_PRE)
    return rewriter.getF32Type();
  llvm::report_fatal_error("unsupported QuantMode");
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
      getQuantModeForCastChain(castChain);
  if (!quantMode) {
    LDBG("cast op quant mode is null");
    return;
  }
  auto quantModeAttr =
      FixpipePreQuantModeAttr::get(op.getContext(), quantMode.value());
  auto reluModeAttr = op.getPreReluAttr();

  hivm::VCastOp lastCast = castChain.back();
  rewriter.setInsertionPointAfter(lastCast);
  Value fixpipeInit =
      utils::createEmptyOp(rewriter, loc, lastCast.getResult()[0]);
  MLIRContext *ctx = rewriter.getContext();
  bool regBased = isOnRegBasedArch(op);
  FixpipeDMAModeAttr dmaModeAttr =
      regBased ? op.getDmaModeAttr()
               : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
  auto newFixpipeOp = rewriter.create<FixpipeOp>(
      loc, fixpipeInit.getType(), /*src=*/newFixpipeSrcTensor,
      /*dst=*/fixpipeInit, dmaModeAttr, op.getDualDstModeAttr(),
      op.getSubBlockIdxAttr(), quantModeAttr, reluModeAttr,
      op.getChannelSplitAttr(), op.getC0PadEnAttr(), op.getQuantScale());
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
  OperationState state(loc, op->getName());
  state.addAttributes(op->getAttrs());
  SmallVector<Value> operands(op->getOperands());
  operands[0] = fixpipeSrcTensor;
  operands[1] = dst;
  state.addOperands(operands);
  auto newFixpipeOp = cast<hivm::FixpipeOp>(rewriter.create(state));
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
  if (!traceDownStoreOpWithSingleChain(vMulOp.getResult()[0]))
    return false;

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
  auto newFixpipe = rewriter.create<FixpipeOp>(
      op.getLoc(), dst.getType(), op.getSource(), dst, op.getUnitFlagCond(),
      op.getDmaModeAttr(), op.getDualDstModeAttr(), op.getSubBlockIdxAttr(),
      FixpipePreQuantModeAttr::get(rewriter.getContext(), preQuant),
      op.getPreReluAttr(), op.getChannelSplitAttr(), op.getC0PadEnAttr(),
      quantScale, op.getUnitFlagModeAttr(), op.getUnitFlagGroupIdAttr());
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

void inlineFixpipeWithTranspose(PatternRewriter &rewriter, hivm::FixpipeOp op,
                                hivm::VTransposeOp transpose) {
  rewriter.setInsertionPointAfter(transpose);
  auto dmaMode =
      FixpipeDMAModeAttr::get(rewriter.getContext(), FixpipeDMAMode::NZ2DN);
  auto newFixpipe = rewriter.create<FixpipeOp>(
      op.getLoc(), transpose.getResult()[0].getType(), op.getSource(),
      transpose.getDst(), dmaMode, op.getDualDstModeAttr(),
      op.getSubBlockIdxAttr(), op.getPreQuantAttr(), op.getPreReluAttr(),
      op.getChannelSplitAttr(), op.getC0PadEnAttr(), op.getQuantScale());
  Value newResult = newFixpipe.getResultTensor();
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
  rewriter.eraseOp(op);
  LDBG("InlineFixpipeWithTranspose");
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
  auto quantModeAttr = op.getPreQuantAttr();
  auto reluModeAttr = op.getPreReluAttr();
  Value fixpipeInit = nullptr;
  fixpipeInit = utils::createEmptyOpWithTargetElemType(
      rewriter, extractSliceOp.getLoc(), newExtractSliceResult,
      getInitType(newExtractSliceResult, op.getPreQuant(), rewriter));

  MLIRContext *ctx = rewriter.getContext();
  bool regBased = isOnRegBasedArch(op);
  FixpipeDMAModeAttr dmaModeAttr =
      regBased ? op.getDmaModeAttr()
               : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
  auto newFixpipeOp = rewriter.create<FixpipeOp>(
      extractSliceOp.getLoc(), fixpipeInit.getType(),
      /*src=*/newExtractSliceResult, /*dst=*/fixpipeInit, dmaModeAttr,
      op.getDualDstModeAttr(), op.getSubBlockIdxAttr(), quantModeAttr,
      reluModeAttr, op.getChannelSplitAttr(), op.getC0PadEnAttr(),
      op.getQuantScale());
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

void moveFixpipeOutOfScfFor(PatternRewriter &rewriter, Location loc,
                            hivm::FixpipeOp fixPipeOp, scf::ForOp scfForOp,
                            Value fixpipeResTensor) {
  SmallVector<Value> yieldValues = llvm::to_vector(scfForOp.getYieldedValues());
  auto idx = findIdx(yieldValues, fixpipeResTensor);
  if (idx.has_value()) {
    LDBG("InlineFixpipeWithYield");
    rewriter.replaceAllUsesWith(fixpipeResTensor,
                                fixPipeOp.getDpsInputOperand(0)->get());

    rewriter.setInsertionPointAfter(scfForOp);
    auto fixpipeInit =
        utils::createEmptyOp(rewriter, scfForOp->getLoc(), fixpipeResTensor);
    auto quantModeAttr = fixPipeOp.getPreQuantAttr();
    auto reluModeAttr = fixPipeOp.getPreReluAttr();
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        fixPipeOp.getLoc(), TypeRange{fixpipeInit},
        scfForOp->getResult(idx.value()), fixpipeInit, dmaModeAttr,
        FixpipeDualDstModeAttr{}, fixPipeOp.getSubBlockIdxAttr(), quantModeAttr,
        reluModeAttr);
    rewriter.replaceAllUsesExcept(scfForOp->getResult(idx.value()),
                                  newFixpipeOp.getResultTensor(), newFixpipeOp);
  }
  LDBG("moveFixpipeOutOfScfFor");
}

} // namespace mlir::hivm
