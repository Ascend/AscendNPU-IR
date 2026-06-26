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

namespace mlir {
namespace hivm {

using namespace mlir;

 bool isTensorSingleElement(Value tensorValue) {
  auto shapedType = dyn_cast<ShapedType>(tensorValue.getType());
  if (!shapedType)
    return false;
  
  int64_t totalSize = 1;
  for (int64_t dim : shapedType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      return false;
    }
    totalSize *= dim;
  }
  return totalSize == 1;
}

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

struct splitFixpipe : public OpRewritePattern<FixpipeOp> {
  const DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim;

public:
  splitFixpipe(MLIRContext *context,
               const DenseMap<int32_t, int64_t> &tightlyCoupledMapIn)
      : OpRewritePattern<FixpipeOp>(context),
        tightlyCoupledBufferToTilingDim(tightlyCoupledMapIn) {}

  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
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

    bool cvpipeFlag = true;
    auto subviewOp = dst.getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      cvpipeFlag = false;
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
    if (!tightlyCoupledBufferToTilingDim.contains(attr.getId().value()))
      return failure();

    auto tilingDimAttr = markOp->getAttrOfType<IntegerAttr>(AICAttrTilingDim);
    if (!tilingDimAttr)
      return failure();
    int64_t tilingDim = tilingDimAttr.getValue().getSExtValue();
    auto rank = allocOp.getType().getRank();
    if (tilingDim == -1) {
      op->emitWarning("The tilingDim in AIC does not exist! Maybe because AIV "
                      "tightly coupled alloc is not tiled!");
      return failure();
    }
    if (tilingDim != rank - 2 && tilingDim != rank - 1) {
      op->emitWarning(
          "The tilingDim in AIC does not match row_split or column split!");
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

    auto oldTy = cast<MemRefType>(allocVal.getType());
    if (!cvpipeFlag) {
      auto newTy = MemRefType::get(splitShape, oldTy.getElementType(),
                                   oldTy.getLayout(), oldTy.getMemorySpace());
      rewriter.setInsertionPoint(allocOp);
      auto newAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), newTy);

      rewriter.setInsertionPoint(markOp);
      auto newMark =
          rewriter.create<annotation::MarkOp>(markOp->getLoc(), newAlloc);
      rewriter.modifyOpInPlace(newMark,
                               [&] { newMark->setAttrs(markOp->getAttrs()); });
      auto dualAttr =
          FixpipeDualDstModeAttr::get(rewriter.getContext(), splitMode);
      rewriter.setInsertionPoint(op);
      SmallVector<Value> oprs({op.getSrc(), newAlloc});
      if (auto quantScale = op.getQuantScale())
        oprs.push_back(quantScale);
      auto newFixpipeOp = rewriter.create<FixpipeOp>(op.getLoc(), TypeRange{},
                                                     oprs, op->getAttrs());
      newFixpipeOp.setDualDstModeAttr(dualAttr);

      rewriter.replaceAllUsesWith(allocVal, newAlloc.getResult());
      rewriter.replaceOp(op, newFixpipeOp->getResults());
      rewriter.eraseOp(markOp);
      rewriter.eraseOp(allocOp);
      return success();
    }

    auto newTy = MemRefType::get(splitShape, oldTy.getElementType(),
                                 oldTy.getLayout(), oldTy.getMemorySpace());

    rewriter.setInsertionPoint(allocOp);
    auto newAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), newTy);

    rewriter.setInsertionPoint(markOp);
    auto newMark =
        rewriter.create<annotation::MarkOp>(markOp->getLoc(), newAlloc);
    rewriter.modifyOpInPlace(newMark,
                             [&] { newMark->setAttrs(markOp->getAttrs()); });
    rewriter.setInsertionPoint(subviewOp);
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    if (tilingDim == rank - 2) {
      if (sizes[1].is<Attribute>()) {
        int64_t oldSize = cast<IntegerAttr>(sizes[1].get<Attribute>()).getInt();
        sizes[1] = rewriter.getIndexAttr(oldSize / 2);
      }
    } else {
      if (sizes[2].is<Attribute>()) {
        int64_t oldSize = cast<IntegerAttr>(sizes[2].get<Attribute>()).getInt();
        sizes[2] = rewriter.getIndexAttr(oldSize / 2);
      }
    }

    int64_t dim1 = sizes[1].is<Attribute>()
                       ? cast<IntegerAttr>(sizes[1].get<Attribute>()).getInt()
                       : ShapedType::kDynamic;
    int64_t dim2 = sizes[2].is<Attribute>()
                       ? cast<IntegerAttr>(sizes[2].get<Attribute>()).getInt()
                       : ShapedType::kDynamic;
    SmallVector<int64_t> new2DShape = {dim1, dim2};

    auto srcType = cast<MemRefType>(newAlloc.getType());
    Type elementType = srcType.getElementType();
    Attribute memorySpace = srcType.getMemorySpace();
    auto layout = StridedLayoutAttr::get(rewriter.getContext(),
                                         ShapedType::kDynamic, {dim2, 1});
    auto result2DType =
        MemRefType::get(new2DShape, elementType, layout, memorySpace);

    auto newSubview = rewriter.create<memref::SubViewOp>(
        subviewOp.getLoc(), result2DType, newAlloc, subviewOp.getMixedOffsets(),
        sizes, subviewOp.getMixedStrides());

    auto dualAttr =
        FixpipeDualDstModeAttr::get(rewriter.getContext(), splitMode);
    rewriter.setInsertionPoint(op);
    NamedAttrList attrs(op->getAttrs());
    attrs.set(op.getDualDstModeAttrName(), dualAttr);

    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        op.getLoc(), TypeRange{}, ValueRange{op.getSrc(), newSubview},
        attrs.getAttrs());

    rewriter.replaceAllUsesWith(allocVal, newAlloc.getResult());
    rewriter.replaceAllUsesWith(subviewOp.getResult(), newSubview.getResult());
    rewriter.replaceOp(op, newFixpipeOp->getResults());

    rewriter.eraseOp(subviewOp);
    rewriter.eraseOp(markOp);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

static LogicalResult runSplitFixpipe(
    func::FuncOp funcOp,
    const DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<splitFixpipe>(funcOp.getContext(),
                             tightlyCoupledBufferToTilingDim);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }
  if (funcOp
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

static LogicalResult tileAndSliceOpAIC(
    func::FuncOp func,
    const DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim) {
  return runSplitFixpipe(func, tightlyCoupledBufferToTilingDim);
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
