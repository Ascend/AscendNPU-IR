//===- SingleCubeSchedule.cpp -- Auto-schedule fused kernels -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for single cube kernels.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/SingleCubeSchedule.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <cassert>

#define DEBUG_TYPE "hfusion-single-cube-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Single Cube] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

static constexpr int64_t kL0CSizeInBytes = 128 * 1024;
static constexpr int64_t kL1SizeInBytes = 512 * 1024;

/// Tiling Key
static constexpr int64_t kTilingCaseKeysAttched[1] = {
    /* kTilingCaseKeyBm128n256k256 */
    300,
};

struct BlockShapeTilingData {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct ProcessShapeTilingData {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct SplitKSlicesTilingData {
  int64_t k;
};

struct SwizzleDefaultTilingData {
  int64_t direction;
  int64_t offset;
};

struct ShuffleKTypeTilingData {
  int64_t type;
};

struct EpilogueTilingData {
  int64_t pTile;
};

struct SingleCubeTilingConfig {
  BlockShapeTilingData block;
  ProcessShapeTilingData process;
  SplitKSlicesTilingData splitKSlices;
  SwizzleDefaultTilingData swizzle;
  ShuffleKTypeTilingData shuffleKType;
  EpilogueTilingData epilogue;
};

/// Tiling Struct Default configs
static constexpr SingleCubeTilingConfig kSingleCubeDefaultTilingInfo[1] = {
    /* kTilingCaseKeyBm128n256k256 */
    {{128, 256, 256}, {128, 256, 64}, {1}, {0, 3}, {0}, {4}}};
} // namespace

//===----------------------------------------------------------------------===//
// SingleCubeScheduler
//===----------------------------------------------------------------------===//

void buildTilingStruct(MLIRContext *ctx, const SmallVector<Expr> &exprs,
                       TilingStruct &s) {
  auto tilingDataType = IntegerType::get(ctx, 64);
  for (Expr e : exprs) {
    TilingData d = TilingData(std::move(e), tilingDataType);
    s.push_back(std::move(d));
  }
}

TilingComputeFn SingleCubeScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            StmtExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);
    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();
    TilingCases c;
    TilingStruct s;
    assert(!kernelInfo->matmulOp2Info.empty());
    auto matmulOpInfo = kernelInfo->matmulOp2Info.begin()->second;
    auto tuningInfo = kernelInfo->cubeTilingTuning;
    for (auto [tilingKey, tilingConfig] :
         llvm::zip(kTilingCaseKeysAttched, kSingleCubeDefaultTilingInfo)) {
      // Set tiling keys.
      if (failed(c.addKey(tilingKey)))
        return {};
      // Set tiling data.
      Expr lengthM = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 0);
      Expr lengthK = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 1);
      Expr lengthN = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorBId, 1);
      if (matmulOpInfo.transposeA) {
        lengthM = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 1);
        lengthK = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 0);
      }
      if (matmulOpInfo.transposeB) {
        lengthN = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorBId, 0);
      }
      Expr c0 = opBuilder->createConstExpr(0);
      Expr c1 = opBuilder->createConstExpr(1);
      Expr c128 = opBuilder->createConstExpr(128);
      Expr c256 = opBuilder->createConstExpr(256);
      Expr tilingKeyExpr = opBuilder->createConstExpr(tilingKey);
      Expr blockTileM = opBuilder->createConstExpr(tilingConfig.block.m);
      Expr blockTileN = opBuilder->createConstExpr(tilingConfig.block.n);
      if (!matmulOpInfo.transposeA && matmulOpInfo.transposeB) {
        blockTileM = select(lengthM <= c256, c128,
                            c128 * (lengthN <= lengthM) +
                                c256 * (c1 - (lengthN <= lengthM)));
        blockTileN = select(lengthM <= c256, c256,
                            c256 * (lengthN <= lengthM) +
                                c128 * (c1 - (lengthN <= lengthM)));
      }
      Expr blockTileK = opBuilder->createConstExpr(tilingConfig.block.k);
      if (tuningInfo.size() >= 3 && tuningInfo[0] != -1) {
        blockTileM = opBuilder->createConstExpr(tuningInfo[0]);
        blockTileN = opBuilder->createConstExpr(tuningInfo[1]);
        blockTileK = opBuilder->createConstExpr(tuningInfo[2]);
        if (tuningInfo[0] * tuningInfo[1] * sizeof(Float32Type) >
            kL0CSizeInBytes) {
          kernelInfo->originalKernel->emitError(
              "BlockM * BlockN * sizeof(float) must <= 128K(L0C Cache Size)");
        }
        if ((tuningInfo[0] * tuningInfo[1] + tuningInfo[1] * tuningInfo[2]) *
                2 >
            kL1SizeInBytes) {
          kernelInfo->originalKernel->emitError(
              "(BlockM * BlockN + BlockK * BlockN) * 2 <= 512K(L1 Cache Size)");
        }
      }
      Expr processTileM = opBuilder->createConstExpr(tilingConfig.process.m);
      Expr processTileN = opBuilder->createConstExpr(tilingConfig.process.n);
      Expr processTileK = opBuilder->createConstExpr(tilingConfig.process.k);
      Expr splitKSlices =
          opBuilder->createConstExpr(tilingConfig.splitKSlices.k);
      Expr shuffleKType =
          opBuilder->createConstExpr(tilingConfig.shuffleKType.type);
      Expr swizzleOffset =
          opBuilder->createConstExpr(tilingConfig.swizzle.offset);
      Expr swizzleDirection = select(lengthN <= lengthM, c0, c1);
      if (tuningInfo.size() >= 5 && tuningInfo[3] != -1) {
        swizzleDirection = opBuilder->createConstExpr(tuningInfo[3]);
        swizzleOffset = opBuilder->createConstExpr(tuningInfo[4]);
        if (tuningInfo[3] != 0 && tuningInfo[3] != 1) {
          kernelInfo->originalKernel->emitError(
              "swizzle direction must be one or zero!");
        }
        if (tuningInfo[4] <= 0) {
          kernelInfo->originalKernel->emitError(
              "swizzle offset must be greater than zero!");
        }
      }
      Expr epiloguePTile =
          opBuilder->createConstExpr(tilingConfig.epilogue.pTile);
      // Build tiling struct.
      SmallVector<Expr> exprs = {tilingKeyExpr, blockTileM,   blockTileN,
                                 blockTileK,    processTileM, processTileN,
                                 processTileK,  splitKSlices, swizzleDirection,
                                 swizzleOffset, shuffleKType, epiloguePTile};
      buildTilingStruct(ctx, exprs, s);
    }
    return TilingFnResultTy(std::make_pair(std::move(c), std::move(s)));
  };
}

//===----------------------------------------------------------------------===//
// Implementation of SingleCubeScheduler schedule functions.
//===----------------------------------------------------------------------===//

LogicalResult SingleCubeScheduler::createScheduleImpl(TilingKey key,
                                                      OpBuilder &opBuilder) {
#ifndef NDEBUG
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);
#endif
  return success();
}

LogicalResult
SingleCubeScheduler::runPreScheduleProcedure(OpBuilder &opBuilder) {
  func::FuncOp currentFunc = getOriginalKernel();
  // 1. apply tensor result to out params
  if (failed(applyTensorResultToOutParamsPass(currentFunc)))
    return failure();

  // 2. analyze kernel
  if (failed(analyzeAndVerifyKernel()))
    return currentFunc->emitWarning("Failed to analyze and verify kernel.");
  return success();
}

LogicalResult
SingleCubeScheduler::runPostScheduleProcedure(OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  auto tilingKey2Kernel = tilingInfo->getTilingKey2KernelMap();
  assert(!tilingKey2Kernel.empty());

  // packing tiling data to tiling struct memref
  auto tilingFunc = tilingInfo->getHostTilingFunc();
  if (failed(applyPackTilingDataPass(tilingFunc)))
    return failure();

  // mark matmul op with tiling struct memref
  BlockArgument tilingStruct = nullptr;
  auto tilingCases = tilingInfo->getTilingCases();
  func::FuncOp funcOp = tilingKey2Kernel[tilingCases[0]];
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (hacc::utils::isKernelArg(funcOp, idx,
                                 hacc::KernelArgType::kTilingStruct)) {
      tilingStruct = arg;
      break;
    }
  }
  assert(tilingStruct != nullptr);
  funcOp.walk([&](Operation *op) {
    if (isMatmulOps(op)) {
      opBuilder.setInsertionPointAfter(op);
      StringAttr tilingStructAttr = opBuilder.getStringAttr(
          stringifyEnum(hacc::KernelArgType::kTilingStruct));
      SmallVector<Attribute> arrayList{tilingStructAttr};
      ArrayAttr arrayAttr = opBuilder.getArrayAttr(arrayList);
      NamedAttribute namedAttribute(tilingStructAttr, opBuilder.getUnitAttr());
      opBuilder.create<annotation::MarkOp>(op->getLoc(), op->getResult(0),
                                           ValueRange{tilingStruct}, arrayAttr);
    }
  });
  if (failed(applyCSEAndCanonicalizePass(funcOp)))
    return failure();
  return success();
}