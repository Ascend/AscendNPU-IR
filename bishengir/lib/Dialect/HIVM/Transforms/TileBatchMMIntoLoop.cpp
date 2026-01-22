//===--------------------- TileBatchMMIntoLoop.cpp ------------------------===//
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstdint>
#include <memory>

#define DEBUG_TYPE "hivm-tile-batchmm-into-loop"

namespace mlir {
#define GEN_PASS_DEF_TILEBATCHMMINTOLOOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
static constexpr llvm::StringLiteral fixpipeAlreadyInserted =
    "fixpipe_already_inserted";
}

namespace {

LogicalResult
getBatchSingleUseChain(Operation *op, int64_t batchDim,
                       SmallVector<Operation *> &recursiveUseChain,
                       Block *fixedBlock) {
  if (op->isUsedOutsideOfBlock(fixedBlock))
    return failure();

  if (isa<hivm::FixpipeOp>(op)) {
    assert(recursiveUseChain.empty());
    recursiveUseChain.push_back(op);
    return success();
  }

  // ToDo: Generalize more states.
  // Currently here exists restriction that tile couple batch matmul and
  // fixpipe, which also requests shape operations between above have to keep
  // batch dimension
  if (isa<tensor::ExtractSliceOp>(op) && op->hasOneUse()) {
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
    if (extractSliceOp.getResultType().getRank() == 3 &&
        extractSliceOp.getStaticOffsets()[0] == 0 &&
        extractSliceOp.getStaticStrides()[0] == 1 &&
        extractSliceOp.getStaticSizes()[0] == batchDim) {
      Operation *userOp = *(op->user_begin());
      auto next = getBatchSingleUseChain(userOp, batchDim, recursiveUseChain,
                                         fixedBlock);
      if (succeeded(next)) {
        recursiveUseChain.push_back(op);
      }
      return next;
    }
  }

  return failure();
}

RankedTensorType extractNonBatchType(RankedTensorType originType) {
  if (originType.getRank() != 3)
    llvm_unreachable("current RankedTensorType must be 3D");

  // Drop batch dimension
  return RankedTensorType::get(originType.getShape().drop_front(),
                               originType.getElementType());
}

Value extractTensorValueWithoutBatch(Location loc, Value originVal,
                                     Value batchIdx,
                                     PatternRewriter &rewriter) {
  auto originType = dyn_cast<RankedTensorType>(originVal.getType());
  assert(originType);
  auto extractType = extractNonBatchType(originType);
  // extract_slice offsets
  SmallVector<OpFoldResult> offsets(originType.getRank(),
                                    rewriter.getIndexAttr(0));
  offsets[0] = batchIdx;
  // extract_slice sizes
  auto valShape = getValueFromShape(originVal, rewriter);
  assert(succeeded(valShape));
  SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1)};
  sizes.append(getAsOpFoldResult(ArrayRef<Value>(*valShape).drop_front()));
  // extract_slice strides
  SmallVector<OpFoldResult> strides(originType.getRank(),
                                    rewriter.getIndexAttr(1));

  auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, extractType, originVal, offsets, sizes, strides);

  return extractSliceOp.getResult();
}

Value insertTensorValueWithoutBatch(Location loc, Value src, Value into,
                                    Value batchIdx, PatternRewriter &rewriter) {
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  assert(srcType);

  SmallVector<OpFoldResult> offsets;
  offsets.push_back(batchIdx);
  offsets.append(srcType.getRank(), rewriter.getIndexAttr(0));

  auto valShape = getValueFromShape(src, rewriter);
  assert(succeeded(valShape));
  SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1)};
  sizes.append(getAsOpFoldResult(ArrayRef<Value>(*valShape)));
  // insert_slice strides
  SmallVector<OpFoldResult> strides(srcType.getRank() + 1,
                                    rewriter.getIndexAttr(1));

  auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      loc, src, into, offsets, sizes, strides);
  return insertSliceOp.getResult();
}

Value subviewMemrefValueWithoutBatch(Location loc, Value originVal,
                                     Value batchIdx,
                                     PatternRewriter &rewriter) {
  auto originType = dyn_cast<MemRefType>(originVal.getType());
  assert(originType && originType.getRank() == 3);
  SmallVector<OpFoldResult> subOffsets(originType.getRank(),
                                       rewriter.getIndexAttr(0));
  subOffsets[0] = batchIdx;
  auto valueOfShape = getValueFromShape(originVal, rewriter);
  assert(succeeded(valueOfShape));
  SmallVector<OpFoldResult> subSizes{rewriter.getIndexAttr(1)};
  subSizes.append(
      getAsOpFoldResult(ArrayRef<Value>(*valueOfShape).drop_front()));
  SmallVector<OpFoldResult> subStrides(originType.getRank(),
                                       rewriter.getIndexAttr(1));
  auto subviewOp = rewriter.create<memref::SubViewOp>(
      loc, originVal, subOffsets, subSizes, subStrides);

  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0, 1});
  reassociation.push_back({2});
  auto collapseOp = rewriter.create<memref::CollapseShapeOp>(
      loc, subviewOp.getResult(), reassociation);
  return collapseOp.getResult();
}

Value rewriteMatrixCShapeChange(ArrayRef<Operation *> useChain, Value matrixC,
                                PatternRewriter &rewriter) {
  Value curVal = matrixC;
  for (int i = static_cast<int>(useChain.size() - 1); i >= 0; --i) {
    if (auto originOp =
            dyn_cast_if_present<tensor::ExtractSliceOp>(useChain[i])) {
      auto originOffsets = originOp.getMixedOffsets();
      auto originSizes = originOp.getMixedSizes();
      auto originStrides = originOp.getMixedStrides();
      auto newType = extractNonBatchType(originOp.getResultType());
      auto newOp = rewriter.create<tensor::ExtractSliceOp>(
          originOp.getLoc(), newType, curVal,
          // Drop batch dimension, while all batch dimension hasn't been changed
          // after origin extract_slice which is restriction when get use chain.
          ArrayRef<OpFoldResult>(originOffsets).drop_front(),
          ArrayRef<OpFoldResult>(originSizes).drop_front(),
          ArrayRef<OpFoldResult>(originStrides).drop_front());

      curVal = newOp.getResult();
    } else {
      llvm_unreachable("unsupported operation which uses matmul's output");
    }
  }

  return curVal;
}

Value rewriteMmadThrowOutBatch(hivm::BatchMmadL1Op batchmmOp,
                               SmallVector<Value> indexes,
                               RankedTensorType matrixCType,
                               const SmallVector<Operation *> &useChain,
                               PatternRewriter &rewriter) {
  // Adjust matrix A & matrix B
  assert(indexes.size() == 1);
  Value matrixA = extractTensorValueWithoutBatch(
      batchmmOp->getLoc(), batchmmOp.getA(), indexes[0], rewriter);
  Value matrixB = extractTensorValueWithoutBatch(
      batchmmOp->getLoc(), batchmmOp.getB(), indexes[0], rewriter);
  // Get new matrix C
  SmallVector<Value> outputsDynSize;
  auto outputValShape = getValueFromShape(batchmmOp.getC(), rewriter);
  assert(succeeded(outputValShape));
  for (int i = 1; i < matrixCType.getRank(); ++i) {
    if (matrixCType.isDynamicDim(i))
      outputsDynSize.push_back((*outputValShape)[i]);
  }
  auto newOutput = rewriter.create<tensor::EmptyOp>(
      batchmmOp->getLoc(), extractNonBatchType(matrixCType), outputsDynSize);
  auto tiledMmad = rewriter.create<hivm::MmadL1Op>(
      batchmmOp.getLoc(), TypeRange{extractNonBatchType(matrixCType)}, matrixA,
      matrixB, batchmmOp.getInitCondition(), batchmmOp.getRealM(),
      batchmmOp.getRealK(), batchmmOp.getRealN(), /*C=*/newOutput,
      batchmmOp.getPerChannelBias(), batchmmOp.getATransposeAttr(),
      batchmmOp.getBTransposeAttr(), batchmmOp.getEnable_HF32Attr());
  if (batchmmOp->getAttr(fixpipeAlreadyInserted)) {
    tiledMmad->setAttr(fixpipeAlreadyInserted,
                       batchmmOp->getAttr(fixpipeAlreadyInserted));
  }
  Value matrixToStore = rewriteMatrixCShapeChange(
      /* ignore first fixpipe */ ArrayRef<Operation *>(useChain).drop_front(),
      tiledMmad.getResultTensors()[0], rewriter);
  return matrixToStore;
}

void rewriteFixpipeThrowOutBatch(Value matrixToStore,
                                 SmallVector<Value> indexes,
                                 RankedTensorType matrixCType,
                                 const SmallVector<Operation *> &useChain,
                                 ArrayRef<BlockArgument> forIterArgs,
                                 PatternRewriter &rewriter) {
  assert(indexes.size() == 1);
  auto originFixpipe = dyn_cast<hivm::FixpipeOp>(useChain.front());
  Value oriFixpipeDst = originFixpipe.getDst();
  if (isa<mlir::MemRefType>(oriFixpipeDst.getType())) {
    Value fixpipeDst = subviewMemrefValueWithoutBatch(
        originFixpipe.getLoc(), oriFixpipeDst, indexes[0], rewriter);

    rewriter.create<hivm::FixpipeOp>(
        originFixpipe.getLoc(), Type{}, /*src=*/matrixToStore,
        /*dst=*/fixpipeDst,
#if (!BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES)
        originFixpipe.getEnableNz2ndAttr(),
#else
        originFixpipe.getDmaModeAttr(),
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
        originFixpipe.getPreQuantAttr(), originFixpipe.getPreReluAttr(),
        originFixpipe.getChannelSplitAttr());
  } else if (isa<mlir::TensorType>(oriFixpipeDst.getType())) {
    assert(matrixToStore.getDefiningOp() &&
           matrixToStore.getDefiningOp()->getParentOp());
    assert(isa<scf::ForOp>(matrixToStore.getDefiningOp()->getParentOp()) &&
           forIterArgs.size() == 1);
    BlockArgument iterationArg = forIterArgs[0];
    assert(iterationArg);

    Value fixpipeDst = extractTensorValueWithoutBatch(
        originFixpipe.getLoc(), iterationArg, indexes[0], rewriter);
    Type resultType =
        extractNonBatchType(dyn_cast<RankedTensorType>(iterationArg.getType()));

    auto newfixpipe = rewriter.create<hivm::FixpipeOp>(
        originFixpipe.getLoc(), resultType, /*src=*/matrixToStore,
        /*dst=*/fixpipeDst,
#if (!BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES)
        originFixpipe.getEnableNz2ndAttr(),
#else
        originFixpipe.getDmaModeAttr(),
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
        originFixpipe.getPreQuantAttr(), originFixpipe.getPreReluAttr(),
        originFixpipe.getChannelSplitAttr());

    Value insert = insertTensorValueWithoutBatch(
        originFixpipe.getLoc(), newfixpipe.getResults()[0], iterationArg,
        indexes[0], rewriter);
    rewriter.create<scf::YieldOp>(originFixpipe.getLoc(), insert);
  }
}
/// This pattern wanna convert all hivm::BatchMmadL1Op and releated fixpipe to
/// loop of new hivm::MmadL1Op and hivm::FixpipeOp
///
/// batch_matmul
/// fixpipe
///
/// =>
///
/// for i batch {
///   %t = extract [i][1]
///   batch_matmul ins(%t)
///   fixpipe
/// }
class TileBatchMM : public OpRewritePattern<hivm::BatchMmadL1Op> {
public:
  using OpRewritePattern<hivm::BatchMmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::BatchMmadL1Op batchmmOp,
                                PatternRewriter &rewriter) const override {
    Value matrixC = batchmmOp.getResultTensors()[0];
    auto matrixCType = dyn_cast<RankedTensorType>(matrixC.getType());
#ifndef NDEBUG
    const int batchMMSize = 3;
    assert(matrixCType.getRank() == batchMMSize && batchmmOp->hasOneUse());
#endif
    SmallVector<Operation *> recursiveUseChain;
    if (failed(getBatchSingleUseChain(
            *(batchmmOp->user_begin()), matrixCType.getShape()[0],
            recursiveUseChain, batchmmOp->getBlock())))
      return batchmmOp.emitOpError(
          "unaccepted batch matmul: use chain between hivm::BatchMmadL1Op and "
          "hivm::FixpipeOp is illegal; just support tensor::extractSliceOp "
          "between above two and all exist in the same block");

    assert(isa<hivm::FixpipeOp>(recursiveUseChain.front()));
    auto originFixpipe = dyn_cast<hivm::FixpipeOp>(recursiveUseChain.front());
    // For tensor data-flow, here needs to make original fixpipe destination
    // with batch dimension as loop iteration.
    // Then use extract_slice/insert_slice to update tensor continuously.
    SmallVector<Value> forInitArgs;
    if (isa<TensorType>(originFixpipe.getDst().getType()))
      forInitArgs.push_back(originFixpipe.getDst());

    auto buildLoopBody =
        [&batchmmOp, &rewriter, &matrixCType,
         &recursiveUseChain](const SmallVector<Value> &indexes,
                             Block::BlockArgListType iterArgs) -> void {
      Value matrixToStore = rewriteMmadThrowOutBatch(
          batchmmOp, indexes, matrixCType, recursiveUseChain, rewriter);
      rewriteFixpipeThrowOutBatch(matrixToStore, indexes, matrixCType,
                                  recursiveUseChain,
                                  ArrayRef<BlockArgument>{iterArgs}, rewriter);
    };

    std::set<int> loopDims = {0}; // First dim is batch axis
    // To ensure order domination, new created op should after origin fixpipe
    rewriter.setInsertionPointAfter(originFixpipe);
    auto nestFor = createNestedLoops(rewriter, batchmmOp.getLoc(), matrixC,
                                     loopDims, buildLoopBody, 0, forInitArgs);

    assert(nestFor.size() == 1);
    if (isa<TensorType>(originFixpipe.getDst().getType())) {
      rewriter.replaceAllUsesWith(originFixpipe->getResult(0),
                                  nestFor[0]->getResult(0));
    }

    for (Operation *op : recursiveUseChain)
      rewriter.eraseOp(op);
    rewriter.eraseOp(batchmmOp);

    return success();
  }
};

} // anonymous namespace

class TileBatchMMIntoLoopPass
    : public impl::TileBatchMMIntoLoopBase<TileBatchMMIntoLoopPass> {
public:
  void runOnOperation() override;
};

void TileBatchMMIntoLoopPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<TileBatchMM>(&getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createTileBatchMMIntoLoopPass() {
  return std::make_unique<TileBatchMMIntoLoopPass>();
}
