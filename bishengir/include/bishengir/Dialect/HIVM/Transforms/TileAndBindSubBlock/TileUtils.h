//===- TileUtils.h - Tile/bind pass run pipeline helpers ------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_TILEANDBINDSUBBLOCK_RUN_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_TILEANDBINDSUBBLOCK_RUN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <string>

namespace mlir {

class Operation;

namespace hivm {

constexpr llvm::StringLiteral AICAttrTilingDim =
    "hivm.tiling_dim";
constexpr llvm::StringLiteral tilghlyCoupledBufferAttr = 
    "hivm.tightly_coupled_buffer";
/// Set on annotation.mark after AIV UB half-tile in BufferizationBubbleUpPattern;
/// used to prune `tightlyCoupledBufferToTilingDim` before AIC fixpipe split.
constexpr llvm::StringLiteral kTiledTightlyCoupledAlloc =
    "tiledAlloc";

struct InsertFixpipeDstPropagateUp : public OpRewritePattern<FixpipeOp> {
  const llvm::DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim;

  InsertFixpipeDstPropagateUp(
      MLIRContext *context,
      const llvm::DenseMap<int32_t, int64_t> &tightlyCoupledMapIn);

  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult limitUniqueSubBlockToStore(func::FuncOp funcOp);

struct FuncRollbackBackup {
  std::string originalName;
  Operation *backupOp = nullptr;
};

void createFuncBackups(ArrayRef<func::FuncOp> funcs,
                       SmallVectorImpl<FuncRollbackBackup> &backups);

void destroyFuncBackups(SmallVectorImpl<FuncRollbackBackup> &backups);

LogicalResult restoreFunctionsFromBackups(
    ModuleOp moduleOp, SmallVectorImpl<FuncRollbackBackup> &backups,
    bool limitSubBlockToStore);

void removeTilingDimMappingMarksFromModule(ModuleOp moduleOp);

void runTileAndBindSubBlockEarlyPatterns(ModuleOp moduleOp);

void collectMixAicAndAivFuncs(ModuleOp moduleOp,
                              SmallVectorImpl<func::FuncOp> &aicFunctions,
                              SmallVectorImpl<func::FuncOp> &aivFunctions);

bool hasBatchMatmulLoopInAicFuncs(ArrayRef<func::FuncOp> aicFunctions);

bool hasImplicitTransposeWithLastAxisInAiv(
    ArrayRef<func::FuncOp> aivFunctions);

LogicalResult pruneTightlyCoupledBufferToTilingDimAfterAivBubbleUp(
    func::FuncOp newFunc,
    llvm::DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim);

LogicalResult tileAicFixpipeFuncsIfNeeded(
    ArrayRef<func::FuncOp> aicFunctions,
    const llvm::DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim);

/// Resolve fixpipe dual-dst split mode and halved UB alloc shape from dma mode
/// and AIV tiling dim. Sets invalidTilingDim when tilingDim cannot map to
/// row/column split (warning emitted on op).
LogicalResult computeFixpipeSplitInfo(FixpipeOp op, int64_t tilingDim,
                                      Value allocVal,
                                      FixpipeDualDstMode &splitMode,
                                      SmallVectorImpl<int64_t> &splitShape,
                                      bool &invalidTilingDim);


} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_TILEANDBINDSUBBLOCK_RUN_H
