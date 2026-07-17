//===- TileUtils.cpp - Tile/bind pass run pipeline helpers -----===//
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

#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/TileUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BufferizationBubbleUp.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hivm-bind-sub-block-tile-utils"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << X << "\n")

namespace mlir {
namespace hivm {

using namespace mlir;
using mlir::hivm::detail::BufferizationPropagateDownPattern;
using mlir::hivm::detail::BufferizationPropagateUpPattern;


LogicalResult computeFixpipeSplitInfo(FixpipeOp op, int64_t tilingDim,
                                      Value allocVal,
                                      FixpipeDualDstMode &splitMode,
                                      SmallVectorImpl<int64_t> &splitShape,
                                      bool &invalidTilingDim) {
  invalidTilingDim = false;
  auto allocTy = cast<MemRefType>(allocVal.getType());
  int64_t rank = allocTy.getRank();
  if (tilingDim != rank - 2 && tilingDim != rank - 1) {
    op.emitWarning(
        "The tilingDim in AIC does not match row_split or column split!");
    invalidTilingDim = true;
    return failure();
  }

  splitShape = llvm::to_vector(allocTy.getShape());

  int64_t constraints = 0;

  if (op.getDmaMode() == FixpipeDMAMode::NZ2DN) {
    /// FIXME: please double checkout the constraint of nz2dn.
    constexpr int64_t nz2dnRowSplitConstraint = 2;
    constexpr int64_t nz2dnColSplitConstraint = 32;
    if (tilingDim == rank - 2) {
      splitMode = FixpipeDualDstMode::COLUMN_SPLIT;
      constraints = nz2dnColSplitConstraint;
    } else if (tilingDim == rank - 1) {
      splitMode = FixpipeDualDstMode::ROW_SPLIT;
      constraints = nz2dnRowSplitConstraint;
    } else {
      op.emitWarning("The tilingDim in AIC does not match row_split or "
                     "column split for NZ2DN fixpipe!");
      invalidTilingDim = true;
      return failure();
    }
  } else {
    /// FIXME: please double checkout the constraint of nz2nd.
    constexpr int64_t nz2ndRowSplitConstraint = 2;
    constexpr int64_t nz2ndColSplitConstraint = 32;
    if (tilingDim == rank - 2) {
      splitMode = FixpipeDualDstMode::ROW_SPLIT;
      constraints = nz2ndRowSplitConstraint;
    } else if (tilingDim == rank - 1) {
      splitMode = FixpipeDualDstMode::COLUMN_SPLIT;
      constraints = nz2ndColSplitConstraint;
    } else {
      op.emitWarning("The tilingDim in AIC does not match row_split or "
                     "column split for NZ2ND fixpipe!");
      invalidTilingDim = true;
      return failure();
    }
  }

  int64_t size = splitShape[tilingDim];
  if (ShapedType::isDynamicShape(size) || (size % constraints) != 0)
    return failure();
  splitShape[tilingDim] = size / 2;
  return success();
}

namespace {

struct CanonicalizeAllocToTensor : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return failure();
    auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(*op->user_begin());
    if (!toTensorOp)
      return failure();
    auto tensorType = toTensorOp.getType();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        toTensorOp, tensorType.getShape(), tensorType.getElementType());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DetachFixpipeDstReadView : public OpRewritePattern<FixpipeOp> {
public:
  explicit DetachFixpipeDstReadView(MLIRContext *context)
      : OpRewritePattern<FixpipeOp>(context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    auto dstAlloc = traceDefOp<memref::AllocOp>(op.getDst());
    if (!dstAlloc)
      return failure();

    annotation::MarkOp finalMarkOp;
    auto func = op->getParentOfType<func::FuncOp>();
    func.walk([&](annotation::MarkOp markOp) {
      auto tensorType = dyn_cast<RankedTensorType>(markOp.getSrc().getType());
      if (!tensorType || !tensorType.hasStaticShape())
        return WalkResult::advance();

      auto memorySpaceCasts =
          traceDefOps<memref::MemorySpaceCastOp>(markOp.getSrc());
      if (memorySpaceCasts.empty())
        return WalkResult::advance();

      bool allCastsFromDst = llvm::all_of(
          memorySpaceCasts, [&](Operation *memorySpaceCast) {
            auto castOp = cast<memref::MemorySpaceCastOp>(memorySpaceCast);
            auto castAllocs =
                traceDefOps<memref::AllocOp>(castOp.getSource());
            return !castAllocs.empty() &&
                   llvm::all_of(castAllocs, [&](Operation *castAlloc) {
                     return castAlloc == *dstAlloc;
                   });
          });
      if (!allCastsFromDst)
        return WalkResult::advance();

      finalMarkOp = markOp;
      return WalkResult::interrupt();
    });
    if (!finalMarkOp)
      return failure();

    auto tensorType = cast<RankedTensorType>(finalMarkOp.getSrc().getType());
    rewriter.setInsertionPoint(finalMarkOp);
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        finalMarkOp.getLoc(), tensorType.getShape(), tensorType.getElementType());
    rewriter.modifyOpInPlace(finalMarkOp, [&]() {
      finalMarkOp.getSrcMutable().set(emptyOp.getResult());
    });
    return success();
  }
};

static FailureOr<int64_t> getHalvedStaticDim(MemRefType type,
                                             int64_t tilingDim) {
  if (tilingDim < 0 || tilingDim >= type.getRank())
    return failure();
  int64_t size = type.getDimSize(tilingDim);
  if (ShapedType::isDynamic(size))
    return failure();
  return size / 2;
}

static FailureOr<int64_t> getFixpipeDstTilingDim(Value dst,
                                                 memref::AllocOp allocOp,
                                                 int64_t allocTilingDim) {
  auto dstType = dyn_cast<MemRefType>(dst.getType());
  if (!dstType)
    return failure();
  int64_t rankReducedCount = allocOp.getType().getRank() - dstType.getRank();
  if (rankReducedCount < 0)
    return failure();

  int64_t dstTilingDim = allocTilingDim - rankReducedCount;
  if (dstTilingDim < 0 || dstTilingDim >= dstType.getRank())
    return failure();
  return dstTilingDim;
}

} // namespace

InsertFixpipeDstPropagateUp::InsertFixpipeDstPropagateUp(
    MLIRContext *context,
    const DenseMap<int32_t, int64_t> &tightlyCoupledMapIn)
    : OpRewritePattern<FixpipeOp>(context),
      tightlyCoupledBufferToTilingDim(tightlyCoupledMapIn) {}

LogicalResult InsertFixpipeDstPropagateUp::matchAndRewrite(
    FixpipeOp op, PatternRewriter &rewriter) const {
    if (op->hasAttr(tileAndSliceFailure)) {
      return failure();
    }
    Value dst = op.getDst();
    if (auto dualDstModeAttr = op.getDualDstModeAttr();
        dualDstModeAttr &&
        dualDstModeAttr.getDualDstMode() != FixpipeDualDstMode::NO_DUAL) {
      return failure();
    }
    auto dstMemrefType = dyn_cast<MemRefType>(dst.getType());
    if (!dstMemrefType)
      return failure();

    auto dstMemorySpace = dstMemrefType.getMemorySpace();
    if (!dstMemorySpace) 
      return failure();
  
    auto toAddrSpace = cast<AddressSpaceAttr>(dstMemorySpace).getAddressSpace();
    if ((!dstMemorySpace) || (toAddrSpace != AddressSpace::UB)) {
      return success();
    }

    auto maybeAllocOp = traceDefOp<memref::AllocOp>(dst);
    if (!maybeAllocOp)
      return failure();
    
    memref::AllocOp allocOp = cast<memref::AllocOp>(*maybeAllocOp);
    mlir::Value allocVal = allocOp.getResult();
    auto maybeMarkOpRaw =
        utils::getAnnotateOpWithAttr(allocVal, tilghlyCoupledBufferAttr);
    if (!maybeMarkOpRaw)
      return failure();

    auto markOp = dyn_cast<annotation::MarkOp>(*maybeMarkOpRaw);
    if (!markOp) 
      return failure();
    
    auto attr = markOp->getAttrOfType<HIVMTightlyCoupledBufferAttr>(
        tilghlyCoupledBufferAttr);
    if (!attr || !attr.getId().has_value())
      return failure();

    /// FIXME: If the fixpipe dual dst mode is not specified, will defautly
    /// fixpipe the whole data into two aiv cores. So if ub is not tiled, just
    /// keep fixpipe as default.
    if (!tightlyCoupledBufferToTilingDim.contains(attr.getId().value())) {
      LDBG("AIC fixpipe skip propagate-up insertion: buffer id "
           << attr.getId().value() << " is not in tiling map");
      return failure();
    }

    auto tilingDimAttr = markOp->getAttrOfType<IntegerAttr>(AICAttrTilingDim);
    if (!tilingDimAttr)
      return failure();

    int64_t tilingDim = tilingDimAttr.getValue().getSExtValue();
    if (tilingDim == -1) {
      op->emitWarning("The tilingDim in AIC does not exist! Maybe because AIV "
                      "tightly coupled alloc is not tiled!");
      return failure();
    }

    // compute the fixpipe split info
    FixpipeDualDstMode splitMode;
    SmallVector<int64_t> splitShape;
    bool invalidTilingDim = false;
    if (failed(computeFixpipeSplitInfo(op, tilingDim, allocVal, splitMode,
                                       splitShape, invalidTilingDim))) {
      if (invalidTilingDim)
        return failure();
      op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }

    auto maybeDstTilingDim =
        getFixpipeDstTilingDim(dst, allocOp, tilingDim);
    if (failed(maybeDstTilingDim)) {
      op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }
    // `tilingDim` indexes the alloc; `dstTilingDim` indexes fixpipe outs.
    int64_t dstTilingDim = maybeDstTilingDim.value();
    LDBG("The dst tiling dim is: " << dstTilingDim);

    auto dstType = cast<MemRefType>(dst.getType());
    auto slicedDstShape = llvm::to_vector(dstType.getShape());
    auto maybeHalvedDim = getHalvedStaticDim(dstType, dstTilingDim);
    if (failed(maybeHalvedDim)) {
      op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }
    slicedDstShape[dstTilingDim] = maybeHalvedDim.value();
    // After tiling, the dst is a subview of a newly allocated compact UB
    // buffer. Rebuild contiguous strides for the halved shape (and keep a
    // dynamic offset) so the UCC type matches the subview that bubble-up
    // will create. Keeping the pre-tile strides (e.g. [64,1] after 64→32)
    // causes memref.subview layout verification failures.
    int64_t dstOffset = 0;
    SmallVector<int64_t> unusedStrides;
    if (failed(getStridesAndOffset(dstType, unusedStrides, dstOffset)))
      dstOffset = ShapedType::kDynamic;
    SmallVector<int64_t> compactStrides(slicedDstShape.size());
    int64_t running = 1;
    for (int64_t i = static_cast<int64_t>(slicedDstShape.size()) - 1; i >= 0;
         --i) {
      compactStrides[i] = running;
      if (ShapedType::isDynamic(slicedDstShape[i]))
        running = ShapedType::kDynamic;
      else if (!ShapedType::isDynamic(running))
        running *= slicedDstShape[i];
    }
    auto slicedDstType = MemRefType::get(
        slicedDstShape, dstType.getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), dstOffset, compactStrides),
        dstType.getMemorySpace());

    rewriter.setInsertionPoint(op);
    auto up = mlir::hivm::detail::createBubblePropagatorUpLink(
        dst, slicedDstType, rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(slicedDstShape[dstTilingDim]), dstTilingDim,
        rewriter);
    rewriter.modifyOpInPlace(op, [&]() {
      op.getDstMutable().assign(up.getResult(0)); // change the newFixpipe to the ucc result
      op.setDualDstModeAttr(
          FixpipeDualDstModeAttr::get(rewriter.getContext(), splitMode));
    });
    LDBG("AIC fixpipe inserted dst propagate-up for:\n " << op);
    return success();
}

namespace {

static bool hasUnpropagatedBubblePropagator(func::FuncOp func) {
  return func
      .walk([](UnrealizedConversionCastOp propagateOp) {
        if (propagateOp->hasAttr(mlir::hivm::detail::kBubbleUpPropagateUp) ||
            propagateOp->hasAttr(mlir::hivm::detail::kBubbleUpPropagateDown)) {
          LDBG("AIC fixpipe propagation did not converge, remaining UCC is:\n "
               << propagateOp);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static LogicalResult tileAndSliceOpAIC(
    func::FuncOp func,
    const DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim) {
  RewritePatternSet patterns(func.getContext());
  // DetachFixpipeDstReadView is to handle the Op pattern by Preload
  // Please read UT case: trace_def_ops_fixpipe_readview_mix_aic 
  patterns.add<DetachFixpipeDstReadView>(func.getContext());
  patterns.add<InsertFixpipeDstPropagateUp>(
      func.getContext(), tightlyCoupledBufferToTilingDim);
  patterns.add<BufferizationPropagateUpPattern,
               BufferizationPropagateDownPattern>(func.getContext());
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    return failure();
  }
  if (hasUnpropagatedBubblePropagator(func)) {
    return failure();
  }
  if (func
          .walk([](FixpipeOp fixpipeOp) {
            return fixpipeOp->hasAttrOfType<UnitAttr>(tileAndSliceFailure)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }
  return success();
}

} // namespace

void runTileAndBindSubBlockEarlyPatterns(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<CanonicalizeAllocToTensor>(moduleOp.getContext());
  (void)applyPatternsGreedily(moduleOp, std::move(patterns));
}

void createFuncBackups(ArrayRef<func::FuncOp> funcs,
                       SmallVectorImpl<FuncRollbackBackup> &backups) {
  backups.reserve(backups.size() + funcs.size());
  for (func::FuncOp func : funcs) {
    backups.push_back({func.getSymNameAttr().str(), func->clone()});
  }
}

void destroyFuncBackups(SmallVectorImpl<FuncRollbackBackup> &backups) {
  for (auto &entry : backups) {
    if (entry.backupOp) {
      entry.backupOp->destroy();
      entry.backupOp = nullptr;
    }
  }
  backups.clear();
}

LogicalResult
restoreFunctionsFromBackups(ModuleOp moduleOp,
                            SmallVectorImpl<FuncRollbackBackup> &backups,
                            bool limitSubBlockToStore) {
  for (auto &entry : backups) {
    if (!entry.backupOp) {
      continue;
    }
    if (auto currentFunc =
            moduleOp.lookupSymbol<func::FuncOp>(entry.originalName)) {
      currentFunc.erase();
    }
    moduleOp.push_back(entry.backupOp);
    auto restoredFunc = cast<func::FuncOp>(entry.backupOp);
    restoredFunc.setName(entry.originalName);
    entry.backupOp = nullptr;

    if (limitSubBlockToStore &&
        failed(limitUniqueSubBlockToStore(restoredFunc)))
      return failure();
  }
  backups.clear();
  return success();
}

void removeTilingDimMappingMarksFromModule(ModuleOp moduleOp) {
  moduleOp->walk([](annotation::MarkOp markOp) {
    if (markOp.isAnnotatedBy(kTilingDimMappingAttrName))
      removeMarkOpAttr(markOp, kTilingDimMappingAttrName);
  });
}

void collectMixAicAndAivFuncs(ModuleOp moduleOp,
                              SmallVectorImpl<func::FuncOp> &aicFunctions,
                              SmallVectorImpl<func::FuncOp> &aivFunctions) {
  moduleOp.walk([&aicFunctions, &aivFunctions](func::FuncOp func) {
    auto funcCoreType = queryFuncCoreType(func);
    if (!funcCoreType.has_value() ||
        !func->hasAttrOfType<UnitAttr>(TPartOfMixAttr::name)) {
      return;
    }
    if (funcCoreType.value() == TFuncCoreType::AIC) {
      aicFunctions.push_back(func);
    } else if (funcCoreType.value() == TFuncCoreType::AIV) {
      aivFunctions.push_back(func);
    }
  });
}

bool hasBatchMatmulLoopInAicFuncs(ArrayRef<func::FuncOp> aicFunctions) {
  return llvm::any_of(aicFunctions, [](func::FuncOp aicFunc) {
    return aicFunc
        .walk([](MmadL1Op mmad) {
          return mmad->hasAttrOfType<UnitAttr>(batchMatmulAttr)
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        })
        .wasInterrupted();
  });
}

bool hasImplicitTransposeWithLastAxisInAiv(
    ArrayRef<func::FuncOp> aivFunctions) {
  return llvm::any_of(aivFunctions, [](func::FuncOp aivFunc) {
    return aivFunc
        .walk([](annotation::MarkOp markOp) {
          return markOp.isAnnotatedBy(kMayImplicitTransposeWithLastAxis)
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        })
        .wasInterrupted();
  });
}

// Scalar or single-element UB tightly-coupled buffers may stay untiled.
static bool canSkipTilingForTrivialUbAlloc(annotation::MarkOp markOp) {
  auto memrefType = dyn_cast<MemRefType>(markOp.getSrc().getType());
  if (!memrefType)
    return false;

  auto maybeSpace = getOptionalHIVMAddressSpace(memrefType);
  if (!maybeSpace || *maybeSpace != AddressSpace::UB)
    return false;

  return memrefType.getRank() < 1 ||
         (memrefType.hasStaticShape() && memrefType.getNumElements() == 1);
}

LogicalResult pruneTightlyCoupledBufferToTilingDimAfterAivBubbleUp(
    func::FuncOp newFunc,
    llvm::DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim) {
  bool erasedAny = false;
  newFunc.walk([&](annotation::MarkOp markOp) {
    auto attr = markOp->getAttrOfType<HIVMTightlyCoupledBufferAttr>(
        HIVMTightlyCoupledBufferAttr::name);
    if (!attr || !attr.getId().has_value())
      return;
    // Cbuf tightly-coupled buffers are not tiled; their IDs should not be in
    // the map and must not affect the tiling decision.
    {
      auto maybeSpace =
          getOptionalHIVMAddressSpace(markOp.getSrc().getType());
      if (maybeSpace && *maybeSpace == AddressSpace::L1)
        return;
    }
    int32_t id = attr.getId().value();
    if (!markOp->hasAttrOfType<UnitAttr>(kTiledTightlyCoupledAlloc) &&
        !canSkipTilingForTrivialUbAlloc(markOp) &&
        tightlyCoupledBufferToTilingDim.erase(id))
      erasedAny = true;
    auto tilingDimAttr = markOp->getAttrOfType<IntegerAttr>(AICAttrTilingDim);
    if (tilingDimAttr) {
      int64_t tilingDim = tilingDimAttr.getValue().getSExtValue();
      tightlyCoupledBufferToTilingDim[id] = tilingDim;
    }
  });
  return erasedAny ? failure() : success();
}

LogicalResult tileAicFixpipeFuncsIfNeeded(
    ArrayRef<func::FuncOp> aicFunctions,
    const llvm::DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim) {

  for (func::FuncOp originalFunc : aicFunctions) {
    originalFunc->walk([&](annotation::MarkOp markOp) {
      if (auto attr = markOp->getAttrOfType<HIVMTightlyCoupledBufferAttr>(
              HIVMTightlyCoupledBufferAttr::name)) {
        auto maybeId = attr.getId();
        if (!maybeId) {
          markOp.emitError() << "Missing id in HIVMTightlyCoupledBufferAttr";
          return;
        }
        auto id = maybeId.value();
        int64_t tilingDim = -1;
        if (tightlyCoupledBufferToTilingDim.contains(id)) {
          tilingDim = tightlyCoupledBufferToTilingDim.at(id);
        }
        markOp->setAttr(
            AICAttrTilingDim,
            IntegerAttr::get(IndexType::get(markOp.getContext()), tilingDim));
      }
    });
    if (failed(
            tileAndSliceOpAIC(originalFunc, tightlyCoupledBufferToTilingDim))) {
      return failure();
    }
  }
  return success();
}

} // namespace hivm
} // namespace mlir
