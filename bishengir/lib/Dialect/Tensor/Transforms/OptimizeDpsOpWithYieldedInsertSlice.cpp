//===- OptimizeDpsOpWithYieldedInsertSlice.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZEDPSOPWITHYIELDEDINSERTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tensor;

namespace {
struct OptimizeDpsOpWithYieldedInsertSlicePass
    : public impl::OptimizeDpsOpWithYieldedInsertSliceBase<
          OptimizeDpsOpWithYieldedInsertSlicePass> {
public:
  void runOnOperation() override;
};

/// Pattern to modify dps op's inits to extract slice if the result
/// is being inserted and yielded.
struct ModifyDpsInitToSlicedIterArg : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern<InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto insertSrc = insertSliceOp.getSource();
    auto *srcDefiningOp = insertSrc.getDefiningOp();
    if (!isa_and_nonnull<DestinationStyleOpInterface>(srcDefiningOp))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "source is not destination style");

    auto dpsSrc = cast<DestinationStyleOpInterface>(srcDefiningOp);
    auto resultNumber = cast<OpResult>(insertSrc).getResultNumber();
    auto tyingInit = dpsSrc.getDpsInitOperand(resultNumber);
    if (!isa_and_nonnull<EmptyOp>(tyingInit->get().getDefiningOp()))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "init is not empty tensor");

    auto enclosingFunc = insertSliceOp->getParentOfType<func::FuncOp>();
    DominanceInfo domInfo(enclosingFunc);
    for (auto operand : insertSliceOp->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;
      auto *operandDef = operand.getDefiningOp();
      if (!domInfo.dominates(operandDef, srcDefiningOp))
        return rewriter.notifyMatchFailure(
            insertSliceOp, "insert slice operand doesn't dominate dps, cannot "
                           "create extract slice");
    }

    rewriter.setInsertionPoint(srcDefiningOp);
    auto extractSlice = rewriter.create<ExtractSliceOp>(
        insertSliceOp.getLoc(), insertSrc.getType(), insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());

    rewriter.modifyOpInPlace(
        srcDefiningOp, [&dpsSrc, &resultNumber, &extractSlice]() {
          dpsSrc.getDpsInitsMutable()[resultNumber].set(extractSlice);
        });
    return success();
  }
};

/// Pattern to hoist allocs out of scf.for loops when the pattern is:
///   alloc -> load -> to_tensor -> insert_slice -> yield
///
/// Uses a two-pass approach on the original forOp body:
/// 1. First pass: insert big-alloc subviews and redirect load destinations.
/// 2. Then remove dead ops and adjust the forOp type.
struct HoistAllocForInsertSliceLoad : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto resultTypes = forOp.getResultTypes();
    if (llvm::none_of(resultTypes, [](Type t) {
          return isa<RankedTensorType>(t);
        }))
      return failure();

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto iterArgs = forOp.getRegionIterArgs();

    struct InsertSliceInfo {
      tensor::InsertSliceOp insertOp;
      bufferization::ToTensorOp toTensorOp;
      memref::AllocOp allocOp;
      Value memcast;
      int resultIdx;
      Value vbrcScalar = nullptr; // scalar src of vbrc if iter_arg init is vbrc
    };
    SmallVector<InsertSliceInfo> infos;

    for (auto [idx, iterArg] : llvm::enumerate(iterArgs)) {
      auto tensorType = dyn_cast<RankedTensorType>(iterArg.getType());
      if (!tensorType)
        continue;

      Value yieldVal = yieldOp.getOperand(idx);
      auto insertOp = yieldVal.getDefiningOp<tensor::InsertSliceOp>();
      if (!insertOp || insertOp.getDest() != iterArg)
        return failure();

      auto toTensorOp =
          insertOp.getSource().getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensorOp)
        return failure();

      Value memref = toTensorOp.getMemref();
      while (auto castOp =
                 memref.getDefiningOp<memref::MemorySpaceCastOp>())
        memref = castOp.getSource();

      auto allocOp = memref.getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        return failure();

      // Verify the alloc/memcast is used as a load destination.
      SmallVector<Value> workList = {toTensorOp.getMemref()};
      bool hasLoadUser = false;
      while (!workList.empty()) {
        Value val = workList.pop_back_val();
        for (auto *user : val.getUsers()) {
          if (user->getParentOp() != forOp)
            continue;
          if (isa<DestinationStyleOpInterface>(user)) {
            hasLoadUser = true;
            break;
          }
          if (isa<memref::MemorySpaceCastOp, memref::SubViewOp>(user))
            workList.push_back(user->getResults().front());
        }
        if (hasLoadUser)
          break;
      }
      if (!hasLoadUser)
        return failure();

      // Check if the iter_arg init is a vbrc (scalar broadcast fill).
      // If so, record the fill scalar to replicate the init on the big alloc.
      Value vbrcScalar = nullptr;
      if (auto vbrcOp =
              forOp.getInitArgs()[idx].getDefiningOp<hivm::VBrcOp>()) {
        auto src = vbrcOp.getSrc();
        // Only track scalar vbrc (fill a scalar value into the entire tensor).
        if (isa<FloatType, IntegerType>(src.getType()))
          vbrcScalar = src;
      }

      infos.push_back(
          {insertOp, toTensorOp, allocOp, toTensorOp.getMemref(), (int)idx,
           vbrcScalar});
    }

    if (infos.empty())
      return failure();

    // ---- Perform the transformation ----

    // 1. Create big allocs + to_tensors before the forOp.
    rewriter.setInsertionPoint(forOp);
    SmallVector<Value> bigMemcasts;
    SmallVector<Value> toTensorResults;

    for (auto &info : infos) {
      auto tensorType =
          cast<RankedTensorType>(iterArgs[info.resultIdx].getType());
      auto allocMemRefType = cast<MemRefType>(info.allocOp.getType());
      Type elementType = allocMemRefType.getElementType();
      Attribute memorySpace = allocMemRefType.getMemorySpace();

      auto bigAllocType = MemRefType::get(
          tensorType.getShape(), elementType, MemRefLayoutAttrInterface{},
          memorySpace);
      auto bigAlloc = rewriter.create<memref::AllocOp>(
          info.allocOp.getLoc(), bigAllocType);
      if (auto align = info.allocOp.getAlignment())
        bigAlloc.setAlignment(align.value());

      // If the iter_arg was initialized with a scalar vbrc (fill), replicate
      // that initialization on the big alloc to avoid losing the fill semantic.
      if (info.vbrcScalar) {
        rewriter.create<hivm::VBrcOp>(info.allocOp.getLoc(),
                                      /*resultTypes=*/TypeRange{},
                                      info.vbrcScalar,
                                      bigAlloc.getResult());
      }

      Value bigMemref = bigAlloc.getResult();
      if (info.memcast.getDefiningOp<memref::MemorySpaceCastOp>()) {
        auto origCastResultType = cast<MemRefType>(info.memcast.getType());
        auto memCastType = MemRefType::get(
            tensorType.getShape(), elementType, origCastResultType.getLayout(),
            origCastResultType.getMemorySpace());
        bigMemref = rewriter.create<memref::MemorySpaceCastOp>(
            info.memcast.getLoc(), memCastType, bigMemref);
      }
      bigMemcasts.push_back(bigMemref);

      auto bigTensor = rewriter.create<bufferization::ToTensorOp>(
          info.toTensorOp.getLoc(), bigMemref, /*restrict=*/true,
          /*writable=*/true);
      toTensorResults.push_back(bigTensor.getResult());
    }

    // 2. Inside the forOp body: insert subview of big buffer before each
    //    alloc, redirect the load's destination, then clean up.
    rewriter.setInsertionPointToStart(forOp.getBody());

    // IRMapping for offset/size/stride SSA values. We'll clone them early.
    IRMapping mapping;
    DenseSet<Operation *> alreadyCloned;

    std::function<Value(Value)> cloneValueChain =
        [&](Value val) -> Value {
      if (mapping.contains(val))
        return mapping.lookup(val);
      if (isa<BlockArgument>(val))
        return val;
      auto *defOp = val.getDefiningOp();
      if (!defOp)
        return val;
      for (auto operand : defOp->getOperands())
        cloneValueChain(operand);
      if (alreadyCloned.contains(defOp))
        return mapping.lookup(val);
      auto *cloned = rewriter.clone(*defOp, mapping);
      alreadyCloned.insert(defOp);
      for (auto [orig, clonedRes] :
           llvm::zip(defOp->getResults(), cloned->getResults()))
        mapping.map(orig, clonedRes);
      return mapping.lookup(val);
    };

    for (auto [i, info] : llvm::enumerate(infos)) {
      // Pre-clone offset computation chain at the start of the loop body.
      for (auto off : info.insertOp.getMixedOffsets())
        if (off.is<Value>())
          cloneValueChain(off.get<Value>());
      for (auto sz : info.insertOp.getMixedSizes())
        if (sz.is<Value>())
          cloneValueChain(sz.get<Value>());
      for (auto st : info.insertOp.getMixedStrides())
        if (st.is<Value>())
          cloneValueChain(st.get<Value>());

      // Create subview right before the original alloc.
      rewriter.setInsertionPoint(info.allocOp);

      auto mapMix = [&](ArrayRef<OpFoldResult> mix) -> SmallVector<OpFoldResult> {
        SmallVector<OpFoldResult> result;
        for (auto v : mix)
          result.push_back(
              v.is<Value>() && mapping.contains(v.get<Value>())
                  ? OpFoldResult(mapping.lookup(v.get<Value>()))
                  : v);
        return result;
      };

      auto newOffsets = mapMix(info.insertOp.getMixedOffsets());
      auto newSizes = mapMix(info.insertOp.getMixedSizes());
      auto newStrides = mapMix(info.insertOp.getMixedStrides());

      auto allocType = cast<MemRefType>(info.allocOp.getType());
      auto bigMemRefType = cast<MemRefType>(bigMemcasts[i].getType());
      auto subviewType = memref::SubViewOp::inferRankReducedResultType(
          allocType.getShape(), bigMemRefType, newOffsets, newSizes,
          newStrides);
      auto bigSubview = rewriter.create<memref::SubViewOp>(
          info.insertOp.getLoc(), cast<MemRefType>(subviewType),
          bigMemcasts[i], newOffsets, newSizes, newStrides);

      // Redirect all users of the old memcast to the big subview.
      rewriter.replaceAllUsesWith(info.memcast, bigSubview.getResult());

      // Replace insert_slice result with the iter_arg (identity passthrough).
      // This keeps the forOp's SSA valid while making its tensor results
      // dead code — the actual tensor data comes from the big buffer's
      // to_tensor result.
      rewriter.replaceAllUsesWith(info.insertOp.getResult(),
                                  iterArgs[info.resultIdx]);
    }

    // 3. Replace uses of old forOp tensor results with to_tensor results.
    //    The forOp itself stays in place (with dead tensor results).
    unsigned toTensorResIdx = 0;
    for (auto [idx, iterArg] : llvm::enumerate(iterArgs)) {
      if (isa<RankedTensorType>(iterArg.getType())) {
        rewriter.replaceAllUsesWith(forOp.getResult(idx),
                                    toTensorResults[toTensorResIdx++]);
      }
    }

    // 4. If we created a vbrc on the big alloc, replace the corresponding
    //    iter_arg init with tensor.empty(). The vbrc init value is no longer
    //    needed (the big alloc already has it), and this lets one-shot-bufferize
    //    avoid inserting memref.copy to materialize the vbrc result.
    for (auto &info : infos) {
      if (info.vbrcScalar) {
        auto tensorType =
            cast<RankedTensorType>(iterArgs[info.resultIdx].getType());
        rewriter.setInsertionPoint(forOp);
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            forOp.getLoc(), tensorType.getShape(),
            tensorType.getElementType());
        forOp.getInitsMutable()[info.resultIdx].set(emptyOp.getResult());
      }
    }

    return success();
  }
};

} // namespace

void OptimizeDpsOpWithYieldedInsertSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  bishengir::tensor::populateOptimizeDpsOpWithYieldedInsertSlicePattern(
      patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass>
mlir::tensor::createOptimizeDpsOpWithYieldedInsertSlicePass() {
  return std::make_unique<OptimizeDpsOpWithYieldedInsertSlicePass>();
}

void bishengir::tensor::populateOptimizeDpsOpWithYieldedInsertSlicePattern(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<ModifyDpsInitToSlicedIterArg,
                  HoistAllocForInsertSliceLoad>(patterns.getContext());
}
