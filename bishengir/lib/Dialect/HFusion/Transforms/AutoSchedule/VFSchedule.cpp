//===- VFSchedule.cpp -- Vector Function Schedule -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for vector function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/VFSchedule.h"
#include "AutoScheduleAttrDefs.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"

#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [VF] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// VFScheduler
//===----------------------------------------------------------------------===//

SmallVector<int64_t>
getTileSizesOnAnchor(Value value, ArrayRef<int64_t> anchorTileSizes,
                     hfusion::detail::DimensionAnalyzer &analyzer) {
  SmallVector<int64_t> interchangeAxis =
      analyzer.getNormalizedInterchange(value);
  SmallVector<int64_t> tileSizes;
  for (int64_t axis : interchangeAxis) {
    tileSizes.push_back(anchorTileSizes[axis]);
  }
  return tileSizes;
}

ArrayAttr getTileSizesAttr(ArrayRef<int64_t> tileSizes, MLIRContext *ctx) {
  SmallVector<Attribute> tileSizesAttr;
  auto i64Type = IntegerType::get(ctx, 64);
  for (int64_t size : tileSizes) {
    tileSizesAttr.push_back(IntegerAttr::get(i64Type, size));
  }
  return ArrayAttr::get(ctx, tileSizesAttr);
}

LogicalResult VFScheduler::markTilingData() {
  KernelInfo *kernelInfo = this->getKernelInfo();
  auto *analyzer = kernelInfo->getAnalyzer();
  SmallVector<int64_t> anchorTileSizes;
  TilingInfo *tilingInfo = getTilingInfo();

  // tilingStruct = [tilingKey, tileSize0, tileSize1, ..., tileSizeN]
  SmallVector<TilingData *> tilingStruct = tilingInfo->getTilingStruct();
  for (size_t i = 0; i < analyzer->getAnchorRank(); ++i) {
    // tile size should always be constant for vectorization
    anchorTileSizes.push_back(tilingStruct[i + 1]->getConst());
  }

  auto *anyPBRInfo = llvm::cast<AnyPBRKernelInfo>(kernelInfo);
  func::FuncOp funcOp = anyPBRInfo->originalKernel;
  funcOp.walk([&](mlir::linalg::LinalgOp op) {
    if (isa<hfusion::LoadOp>(op) || isa<hfusion::StoreOp>(op)) {
      return;
    }
    // TODO: Can different result have different anchor axis
    // Currectly, Suppose all results have same anchor axis for tiling
    Value result = op->getResults()[0];
    SmallVector<int64_t> interchange =
        analyzer->getNormalizedInterchange(result);
    SmallVector<int64_t> tileSizes =
        getTileSizesOnAnchor(result, anchorTileSizes, *analyzer);
    MLIRContext *ctx = op->getContext();
    ArrayAttr arrayAttr = getTileSizesAttr(tileSizes, ctx);
    op->setAttr(kTileSizesTagName, arrayAttr);
  });

  return success();
}

TilingComputeFn VFScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            StmtExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);
    auto *anyPBRInfo = llvm::cast<AnyPBRKernelInfo>(kernelInfo);
    int64_t anchorRank =
        static_cast<int64_t>(anyPBRInfo->getAnalyzer()->getAnchorRank());
    assert(anchorRank > 0 && "anchor rank should be greater than 0");
    LDBG("Anchor Rank: " << anchorRank);
    // Calculate tiling data.
    // Only tile the last dimension on anchor
    MLIRContext *ctx = opBuilder->getContext();
    size_t numTilingData = static_cast<size_t>(anchorRank) +
                           /*numTilingKey=*/1u;
    TilingStruct s(numTilingData);
    TilingCases c;

    // always set tiling key to the last dim
    Expr tilingKey = opBuilder->createConstExpr(anchorRank - 1);
    auto tilingDataType = IntegerType::get(ctx, 64);
    s[0] = std::make_unique<TilingData>(
        TilingData(std::move(tilingKey), tilingDataType));

    // TODO: find a better place to put register size
    int64_t regSizeInBits = 256 * 8;
    int64_t lastDimTileSize =
        regSizeInBits / kernelInfo->getSmallestElementTypeBits();
    int64_t nonLastDimTileSize = 1;

    int64_t dimUpperBound = anchorRank - 1;
    for (int64_t dimIdx = 0; dimIdx < anchorRank; ++dimIdx) {
      if (failed(c.addKey(dimIdx)))
        return {};
      LDBG("Added tiling case: " << dimIdx);
      int64_t tileSize =
          (dimIdx == dimUpperBound ? lastDimTileSize : nonLastDimTileSize);
      auto tilingDataForDim =
          TilingData(opBuilder->createConstExpr(tileSize), tilingDataType);
      tilingDataForDim.setHeuristicValueForKey(dimIdx, tileSize);
      LDBG("Setting tiling data heuristic value: dimIdx="
           << dimIdx << " heuristic=" << tileSize);
      s[dimIdx + 1] = std::make_unique<TilingData>(std::move(tilingDataForDim));
    }
    return TilingFnResultTy(std::make_pair(std::move(c), std::move(s)));
  };
}

LogicalResult VFScheduler::createScheduleImpl(TilingKey key,
                                              OpBuilder &opBuilder) {

  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);
  auto *anyPBRInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getKernelInfo());
  assert(anyPBRInfo != nullptr);
  if (anyPBRInfo == nullptr) {
    return failure();
  }

  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  tileParallelAxesAndFuseProducers(key, *tilingInfo, *anyPBRInfo, opBuilder);

  if (!needToSplitReduction(key))
    return success();

  LDBG("Need to split condition is true");
  // Apply canonicalization before tiling again.
  applyCanonicalization(opBuilder);

  tileReduceAxesAndFuseProducers(key, *tilingInfo, *anyPBRInfo, opBuilder);
  return success();
}

LogicalResult VFScheduler::runPostScheduleProcedure(OpBuilder &opBuilder) {
  ModuleOp module = getModule();
  auto result = module.walk([&](mlir::func::FuncOp funcOp) {
    if (failed(applyRemoveCacheIOPass(funcOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}