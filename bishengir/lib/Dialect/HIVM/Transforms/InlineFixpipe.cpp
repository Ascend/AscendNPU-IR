//===---------------------- InlineFixpipe.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts ops to hivm.fixpipe .
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INLINEFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-inline-fixpipe"

namespace {
static constexpr llvm::StringLiteral fixpipeAlreadyInserted =
    "fixpipe_already_inserted";
} // namespace

namespace {
struct InlineFixpipe : public impl::InlineFixpipeBase<InlineFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

std::optional<bool> isStoreOp(Operation *dstOp) {
  if (isa<hivm::StoreOp>(dstOp)) {
    return true;
  }
  bool isPreOp = isa<hivm::VCastOp>(dstOp) || isa<hivm::VReluOp>(dstOp);

  if (dstOp->getDialect()->getNamespace() ==
          HIVMDialect::getDialectNamespace() &&
      !isPreOp) {
    return false;
  }
  return std::nullopt;
}

Operation *getInsertPoint(Operation *op, int &resultIndx) {
  auto users = op->getResult(resultIndx).getUsers();
  std::set<scf::YieldOp> yieldOperands;
  for (auto *user : users) {
    // TODO: add auto tracedDownUser = traceDown(user) and use tracedDownUser to
    // judge
    if (!isa<scf::YieldOp>(user)) {
      continue;
    } else {
      yieldOperands.emplace(user);
    }
  }

  if (yieldOperands.empty()) {
    return op;
  }

  if (yieldOperands.size() > 1) {
    op->emitError("unsupport cases");
    return op;
  }
  auto yieldOperand = *yieldOperands.begin();
  auto yieldParentOp = yieldOperand->getParentOp();
  auto yieldValueIndx = findIdx(llvm::to_vector(yieldOperand->getOperands()),
                                op->getResult(resultIndx));
  if (!yieldValueIndx.has_value())
    llvm_unreachable("yield value must have user");
  resultIndx = yieldValueIndx.value();
  return getInsertPoint(yieldParentOp, resultIndx);
}

/// Insert fixpipe when there is hivm::MmadL1Op or hivm::BatchMmadL1Op.
template <typename OpType>
struct InsertFixpipeOpPattern : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op.getResultTensors()[0];

    if (op.shouldDecomposeBiasByElementAdd()) {
      // the op will decompose to mmadL1 + vadd, so fixpipe cannot be inserted
      // now, and fixpipe should be inserted after the decomposition
      return failure();
    }

    // the following check is needed because now there could be two fixpipe ops
    // from the same mmad; you cannot rely on traceSingleChainUser to avoid
    // duplicating fixpipes anymore
    if (op->getAttr(fixpipeAlreadyInserted))
      return failure();

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isa<hivm::FixpipeOp>(op)) {
        // already insert fixpipe, no need to insert fixpipe again
        return true;
      }
      if (isLocalMatmulInit(op, v)) {
        // no need to insert fixpipe because the single user can directly use
        // result stay in local buffer.
        return true;
      }
      return false;
    };
    if (traceSingleChainUser(mmadLikeOpRes, isMatchedOp))
      return failure();

    int resultIndx = 0;
    auto insertAfterOp = getInsertPoint(op, resultIndx);
    rewriter.setInsertionPointAfter(insertAfterOp);

    Value fixpipeInit =
        utils::createEmptyOp(rewriter, insertAfterOp->getLoc(), mmadLikeOpRes);
    LDBG("Replacing fix pipe for " << op);
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto res = rewriter.create<FixpipeOp>(
        op.getLoc(), /*result_tensor=*/fixpipeInit.getType(),
        /*src=*/insertAfterOp->getResult(resultIndx),
        /*dst=*/fixpipeInit, dmaModeAttr, /*dual_dst_mode=*/nullptr,
        /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);
    op->setAttr(fixpipeAlreadyInserted, rewriter.getBoolAttr(true));

    SmallPtrSet<Operation *, 4> exceptedOps;
    exceptedOps.insert(res);
    for (Operation *userOp : insertAfterOp->getResult(resultIndx).getUsers()) {
      if (isa<DebugOp>(userOp) || isa<FixpipeOp>(userOp) ||
          isa<annotation::MarkOp>(userOp)) {
        exceptedOps.insert(userOp);
      }
    }
    rewriter.replaceAllUsesExcept(insertAfterOp->getResult(resultIndx),
                                  res.getResultTensor(), exceptedOps);
    return success();
  }
};

std::optional<FixpipePreQuantMode> getQuantMode(hivm::VCastOp castOp) {
  auto intputType = getElementTypeOrSelf(castOp.getSrc()[0].getType());
  auto outputType = getElementTypeOrSelf(castOp.getDst()[0].getType());
  if (intputType.isF32() && outputType.isF16()) {
    return symbolizeFixpipePreQuantMode("F322F16");
  }
  if (intputType.isF32() && outputType.isBF16()) {
    return symbolizeFixpipePreQuantMode("F322BF16");
  } else if (intputType.isInteger(32) && outputType.isInteger(8)) {
    return symbolizeFixpipePreQuantMode("S322I8");
  }
  llvm_unreachable("unsupported QuantMode");
}

/// when all the activationOps are ready, there should be relu, leaky-relu and
/// p-relu
bool isActivationOp(Operation *op) { return isa<hivm::VReluOp>(op); }

template <typename OpType>
std::optional<FixpipePreReluMode> getReluMode(OpType op) {
  if constexpr (std::is_same_v<OpType, hivm::VReluOp>) {
    return hivm::symbolizeFixpipePreReluMode("NORMAL_RELU");
  }
  llvm_unreachable("unsupported ReluValue");
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
  llvm_unreachable("unsupported QuantMode");
}

//===----------------------------------------------------------------------===//
// InlineFixpipeOpPattern
//===----------------------------------------------------------------------===//
// Fixpipe can complete 3 inner action with origin matrixC operand following
// conditions
//   1. cast or quantization
//   2. relu and other activation function
//   3. store or layout
// Potential optimization is to fuse condition 1&2&3 into fixpipe.
struct InlineFixpipeOpPattern : public OpRewritePattern<FixpipeOp> {
public:
  using OpRewritePattern<FixpipeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResultTensor())
      return failure();

    auto fixpipeResTensor = op.getResultTensor();
    if (fixpipeResTensor.getUsers().empty())
      return failure();

    if (getUsersNum(fixpipeResTensor) != 1)
      return failure();

    return inlineFixpipeOp(rewriter, op);
  }

private:
  LogicalResult inlineFixpipeOp(PatternRewriter &rewriter, FixpipeOp op) const {
    bool matched = false;
    auto loc = op.getLoc();
    auto *curOp = *(op.getResultTensor().getUsers().begin());

    // 1. cast or quantization
    auto castOp = dyn_cast_if_present<hivm::VCastOp>(curOp);
    if (op.getFixpipeState() <= op.needFixpipePreFuse() && castOp) {
      matched = true;
      inlineFixPipeWithRreQuant(rewriter, loc, op, castOp,
                                op.getDpsInputOperand(0)->get());
    } else if (op.getFixpipeState() <= op.needFixpipePreFuse() &&
               isActivationOp(curOp)) {
      // 2. relu and other activation function
      matched = true;
      auto reluOp = llvm::dyn_cast_if_present<hivm::VReluOp>(curOp);
      inlineFixPipeWithRreRelu(rewriter, loc, op, reluOp);
    } else if (auto storeOp = llvm::dyn_cast_if_present<hivm::StoreOp>(curOp)) {
      //   3. store or layout
      auto storeAttr = storeOp.getAtomicKindAttr();
      hivm::AtomicKind atomicKind = hivm::AtomicKind::NONE;
      if (storeAttr)
        atomicKind = storeAttr.getValue();
      if (atomicKind == AtomicKind::NONE || atomicKind == AtomicKind::ADD ||
          atomicKind == AtomicKind::MAX || atomicKind == AtomicKind::MIN) {
        matched = true;
        inlineFixPipeWithStoreOp(rewriter, loc, op, storeOp,
                                 op.getDpsInputOperand(0)->get());
      }
    } else if (auto extractSliceOp =
                   dyn_cast_if_present<tensor::ExtractSliceOp>(curOp)) {
      // change to fixpipe op + extract_slice to extract_slice + fixpipe op
      if (op->getBlock() == extractSliceOp->getBlock()) {
        // only swap when fixpipe op and extract slice op are in same block,
        // otherwise, extract slice op may be in sub block loop and fixpipe
        // cannot be fused into.
        matched = true;
        swapFixpipeAndExtractSliceOp(rewriter, loc, op, extractSliceOp);
      }
    } else if (auto insertSliceOp =
                   dyn_cast_if_present<tensor::InsertSliceOp>(curOp)) {
      // change to fixpipe op + insert_slice + store op to insert_slice +
      // fixpipe op + store op, and besides store op, there is no anther user
      // for insert_slice
      if (traceDownStoreOpWithSingleChain(insertSliceOp.getResult())) {
        matched = true;
        swapFixpipeAndInsertSliceOp(rewriter, loc, op, insertSliceOp);
      }
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::ForOp>(curOp->getParentOp())) {
      // move fixpipe out of scf.for
      matched = true;
      auto scfForOp = dyn_cast_if_present<scf::ForOp>(curOp->getParentOp());
      moveFixpipeOutOfScfFor(rewriter, loc, op, scfForOp, op.getResultTensor());
    }
    return matched ? success() : failure();
  }

  void inlineFixPipeWithRreQuant(PatternRewriter &rewriter, Location loc,
                                 hivm::FixpipeOp op, hivm::VCastOp castOp,
                                 Value newFixpipeSrcTensor) const {
    std::optional<FixpipePreQuantMode> quantMode = getQuantMode(castOp);
    auto quantModeAttr =
        FixpipePreQuantModeAttr::get(op.getContext(), quantMode.value());
    auto reluModeAttr = op.getPreReluAttr();

    rewriter.setInsertionPointAfter(castOp);
    Value fixpipeInit =
        utils::createEmptyOp(rewriter, loc, castOp.getResult()[0]);
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        loc, fixpipeInit.getType(), /*src=*/newFixpipeSrcTensor,
        /*dst=*/fixpipeInit, dmaModeAttr, FixpipeDualDstModeAttr{},
        quantModeAttr, reluModeAttr);
    rewriter.replaceAllUsesWith(castOp.getResult()[0],
                                newFixpipeOp.getResultTensor());
    rewriter.eraseOp(castOp);
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithPreQuant");
  }

  void inlineFixPipeWithRreRelu(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op,
                                hivm::VReluOp reluOp) const {
    std::optional<FixpipePreReluMode> reluMode = getReluMode(reluOp);
    rewriter.modifyOpInPlace(op, [&]() { op.setPreRelu(reluMode); });
    rewriter.replaceAllUsesWith(reluOp.getResult()[0], op.getResult(0));
    rewriter.eraseOp(reluOp);
    LDBG("InlineFixpipeWithPreRelu");
  }

  void inlineFixPipeWithStoreOp(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op, hivm::StoreOp storeOp,
                                Value fixpipeSrcTensor) const {
    assert(storeOp->getNumResults() == 0 && "StoreOp must have 0 results");
    rewriter.setInsertionPointAfter(storeOp);
    auto dst = storeOp.getDst();
    auto storeAttr = storeOp.getAtomicKindAttr();
    auto noneAtomicAttr =
        AtomicKindAttr::get(op->getContext(), ::mlir::hivm::AtomicKind::NONE);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        loc, TypeRange{}, ValueRange{fixpipeSrcTensor, dst}, op->getAttrs());
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

  void
  swapFixpipeAndExtractSliceOp(PatternRewriter &rewriter, Location loc,
                               hivm::FixpipeOp op,
                               tensor::ExtractSliceOp extractSliceOp) const {
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
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(),
        /*src=*/newExtractSliceResult, /*dst=*/fixpipeInit, dmaModeAttr,
        FixpipeDualDstModeAttr{}, quantModeAttr, reluModeAttr);
    rewriter.replaceOp(extractSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithExtractSliceReshape");
  }

  void swapFixpipeAndInsertSliceOp(PatternRewriter &rewriter, Location loc,
                                   hivm::FixpipeOp op,
                                   tensor::InsertSliceOp insertSliceOp) const {
    rewriter.setInsertionPointAfter(insertSliceOp);
    auto fixpipeSrc = op.getDpsInputOperand(0)->get();

    auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        insertSliceOp.getLoc(), fixpipeSrc, insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());

    auto newInsertSliceResult = newInsertSliceOp->getResult(0);
    auto quantModeAttr = op.getPreQuantAttr();
    auto reluModeAttr = op.getPreReluAttr();
    Value fixpipeInit = utils::createEmptyOpWithTargetElemType(
        rewriter, insertSliceOp.getLoc(), newInsertSliceResult,
        getInitType(newInsertSliceResult, op.getPreQuant(), rewriter));
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        insertSliceOp.getLoc(), TypeRange{fixpipeInit}, newInsertSliceResult,
        fixpipeInit, dmaModeAttr, FixpipeDualDstModeAttr{}, quantModeAttr,
        reluModeAttr);
    rewriter.replaceOp(insertSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithInsertSliceOpReshape");
  }

  bool traceDownStoreOpWithSingleChain(Value v) const {
    auto isMachedOp = [](Operation *op, Value v) {
      return isa<hivm::StoreOp>(op);
    };
    return traceSingleChainUser(v, isMachedOp);
  }

  void moveFixpipeOutOfScfFor(PatternRewriter &rewriter, Location loc,
                              hivm::FixpipeOp fixPipeOp, scf::ForOp scfForOp,
                              Value fixpipeResTensor) const {
    SmallVector<Value> yieldValues =
        llvm::to_vector(scfForOp.getYieldedValues());
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
          FixpipeDualDstModeAttr{}, quantModeAttr, reluModeAttr);
      rewriter.replaceAllUsesExcept(scfForOp->getResult(idx.value()),
                                    newFixpipeOp.getResultTensor(),
                                    newFixpipeOp);
    }
    LDBG("moveFixpipeOutOfScfFor");
  }
};

void populateInlineFixpipePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeOpPattern<hivm::MmadL1Op>>(ctx);
  patterns.add<InsertFixpipeOpPattern<hivm::BatchMmadL1Op>>(ctx);
  patterns.add<InlineFixpipeOpPattern>(ctx);
}

struct InsertFixpipeForDevicePrint : public OpRewritePattern<DebugOp> {
public:
  using OpRewritePattern<DebugOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DebugOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeMmadRes = op.getArg();
    if (!traceDefOp<MmadL1Op>(maybeMmadRes).has_value() &&
        !traceDefOp<BatchMmadL1Op>(maybeMmadRes).has_value())
      return failure();

    Operation *definingOp = maybeMmadRes.getDefiningOp();
    rewriter.setInsertionPointAfter(definingOp);

    TensorType tensorType = cast<TensorType>(maybeMmadRes.getType());
    Value workSpaceTensor = getLocalWorkSpaceTensor(
        rewriter, definingOp->getLoc(), tensorType.getShape(),
        getElementTypeOrSelf(tensorType));
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto fixpipeOp = rewriter.create<FixpipeOp>(
        op.getLoc(), /*result_tensor=*/workSpaceTensor.getType(),
        /*src=*/maybeMmadRes,
        /*dst=*/workSpaceTensor, dmaModeAttr, /*dual_dst_mode=*/nullptr,
        /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);
    rewriter.modifyOpInPlace(op, [&]() {
      OpOperand &arg = op.getArgMutable();
      arg.assign(fixpipeOp.getResultTensor());
    });

    return success();
  }
};

void populateDevicePrintPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeForDevicePrint>(ctx);
}

void InlineFixpipe::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateInlineFixpipePatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }

  // special treatment for device print fixpipe
  RewritePatternSet devicePrintPatterns(&getContext());
  populateDevicePrintPatterns(devicePrintPatterns);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(devicePrintPatterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInlineFixpipePass() {
  return std::make_unique<InlineFixpipe>();
}
