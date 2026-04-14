//===-------------------- InsertLoadStoreForMixCV.cpp----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts load/store op for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTLOADSTOREFORMIXCV
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "insert-load-store"

namespace {
struct InsertLoadStoreForMixCVPass
    : public impl::InsertLoadStoreForMixCVBase<InsertLoadStoreForMixCVPass> {
  using Base::Base;
  void runOnOperation() override;
};

enum class InsertMode { LoadOnly = 0, StoreOnly, LoadAndStore };

LogicalResult
InsertOpHelper(PatternRewriter &rewriter, Location loc,
               const llvm::SmallVector<OpOperand *> &consumerOperands,
               InsertMode insertMode,
               std::optional<Value> insertInit = std::nullopt) {
  if (consumerOperands.empty()) {
    return failure();
  }

  for (OpOperand *consumerOperand : consumerOperands) {
    Operation *lastInsertOp = nullptr;
    rewriter.setInsertionPointAfterValue(consumerOperand->get());
    Type type = consumerOperand->get().getType();
    Type elemType = getElementTypeOrSelf(type);
    if (insertMode == InsertMode::LoadOnly) {
      Value loadInit =
          insertInit.has_value()
              ? insertInit.value()
              : mlir::utils::createEmptyOpWithTargetElemType(
                    rewriter, loc, consumerOperand->get(), elemType);
      lastInsertOp = rewriter.create<hivm::LoadOp>(
          loc, TypeRange(type), consumerOperand->get(), loadInit);
    } else if (insertMode == InsertMode::StoreOnly) {
      Value storeInit =
          insertInit.has_value()
              ? insertInit.value()
              : utils::createEmptyOp(rewriter, loc, consumerOperand->get());
      lastInsertOp = rewriter.create<hivm::StoreOp>(
          loc, TypeRange(type), consumerOperand->get(), storeInit);
    } else if (insertMode == InsertMode::LoadAndStore) {
      Value storeInit =
          utils::createEmptyOp(rewriter, loc, consumerOperand->get());
      auto storeOp = rewriter.create<hivm::StoreOp>(
          loc, TypeRange(type), consumerOperand->get(), storeInit);
      Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
          rewriter, loc, consumerOperand->get(), elemType);
      lastInsertOp = rewriter.create<hivm::LoadOp>(
          loc, TypeRange(type), storeOp->getResults()[0], loadInit);
    }
    if (!lastInsertOp) {
      llvm_unreachable("lastInsertOp not defined");
      return failure();
    }
    rewriter.modifyOpInPlace(consumerOperand->getOwner(), [&]() {
      consumerOperand->set(lastInsertOp->getResult(0));
    });
  }

  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// InsertLoadStoreOp
//===----------------------------------------------------------------------===//
/// pattern1 : fixpipe op / store op + vector/mmadl1
/// convert into fixpipe op / store op  + hivm.load + vector/mmadl1
template <typename OpType>
struct InsertLoadOpBetweenStoreLikeAndVectorOrCube
    : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!isa<hivm::HIVMStructuredOp>(op.getOperation())) {
      return failure();
    }

    auto hivmOp = cast<HIVMStructuredOp>(op.getOperation());
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : hivmOp->getOpOperands()) {
      if (traceDefOp<hivm::FixpipeOp>(operand.get()).has_value() ||
          traceDefOp<hivm::StoreOp>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, hivmOp.getLoc(), consumerOperands,
                          InsertMode::LoadOnly);
  }
};

/// pattern2 : vector(src, dst) + hivm.load(dst)
/// convert into  vector(src, dst) + hivm.store(dst) + hivm.load
template <typename OpType>
struct InsertStoreOpBetweenVectorAndLoad
    : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;
  ~InsertStoreOpBetweenVectorAndLoad() override = default;
  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::StoreOnly);
  }
};

/// pattern3 : vector(dst) + cube(dst)
/// convert into vector + hivm.store + hivm.load  +cube
template <typename OpType>
struct InsertLoadStoreOpBetweenVectorAndCube
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  ~InsertLoadStoreOpBetweenVectorAndCube() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern3 : scf::for{discrete load src using scalar} with attr {ExtractedLoadOrStore} + cube(src)
/// ExtractedLoadOrStore describes the process of discretely loading scalars on ub.
/// convert into scf::for{discrete load src using scalar} + hivm.store + hivm.load  + cube
/// for example
/// dst = tensor.empty : 16
/// for i in 16
///   dst[i]=src[offset[i]]
/// mmadl1(dst)
/// convert into
/// dst = tensor.empty : 16
/// for i in 16
///    dst[i]=src[offset[i]]
///  gm_dst = store dst to gm
///  l1_dst = load dst from gm
///  mmadl1(l1_dst)
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto scfForDef = traceDefOp<scf::ForOp>(operand.get());
      if (scfForDef.has_value()) {
        auto forOp = llvm::cast<scf::ForOp>(scfForDef.value());
        if (forOp->getAttr("ExtractedLoadOrStore") != nullptr) {
          consumerOperands.push_back(&operand);
        }
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern3 : bufferization::ToTensorOp with ToBeTransposed mark + cube
/// describe there is an implicit transpose for the input of cube. store & load
/// op will be added here in order to make transpose operation happen in vector
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto toTensorOpDef = traceDefOp<bufferization::ToTensorOp>(operand.get());
      if (!toTensorOpDef.has_value())
        continue;
      auto toTensorOp =
          llvm::cast<bufferization::ToTensorOp>(toTensorOpDef.value());
      auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
          toTensorOp->getResult(0), "ToBeTransposed");
      if (maybeAnnotateOp.has_value()) {
        consumerOperands.push_back(&operand);
        rewriter.eraseOp(maybeAnnotateOp.value());
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern4
/// %1 = fixpipe
/// %4 = scf.for iter_args(%arg0 = %1) {
///    %2 = load(%arg0)
///    %3 = vadd(%2, ...)
///    scf.yield %3
/// }
///
/// is converted into
///
/// %1 = fixpipe
/// %5 = scf.for iter_args(%arg0 = %1) {
///    %2 = load(%arg0)
///    %3 = vadd(%2, ...)
///    %4 = %store (%3)
///    scf.yield %4
/// }

struct InsertStoreForSCFYield : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.hasPureTensorSemantics()) {
      return failure();
    }
    auto blockArg = dyn_cast_if_present<BlockArgument>(loadOp.getSrc());
    if (!blockArg) {
      return failure();
    }
    auto scfForOp =
        dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!scfForOp) {
      return failure();
    }
    OpOperand *yieldOperand = scfForOp.getTiedLoopYieldedValue(blockArg);
    if (!yieldOperand) {
      return failure();
    }
    auto correspondYieldVal = yieldOperand->get();
    if (traceDefOp<hivm::FixpipeOp>(correspondYieldVal).has_value() ||
        traceDefOp<hivm::StoreOp>(correspondYieldVal).has_value()) {
      return failure();
    }
    auto yieldOp = cast<scf::YieldOp>(scfForOp.getBody()->getTerminator());
    return InsertOpHelper(rewriter, yieldOp.getLoc(),
                          llvm::SmallVector<OpOperand *>{yieldOperand},
                          InsertMode::StoreOnly, blockArg);
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<OpType>,
               InsertStoreOpBetweenVectorAndLoad<OpType>,
               InsertLoadOpBetweenStoreLikeAndVectorOrCube<OpType>>(
      patterns.getContext());
}

template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateInsertLoadStorePattern(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
  registerOne<func::CallOp>(patterns);
  registerOne<hivm::IndirectLoadOp>(patterns);
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::IndirectStoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>>(
      patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>>(
          patterns.getContext());
}

void InsertLoadStoreForMixCVPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  populateInsertLoadStorePattern(patterns);
  patterns.insert<InsertStoreForSCFYield>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertLoadStoreForMixCVPass() {
  return std::make_unique<InsertLoadStoreForMixCVPass>();
}
