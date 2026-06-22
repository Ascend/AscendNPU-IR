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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
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

static constexpr llvm::StringLiteral elideAfterBufferize =
    "elide_after_bufferize";
} // namespace

namespace {

LogicalResult
traceBatchChainToFixpipe(Operation *op, int64_t batchDim,
                       SmallVector<Operation *> &recursiveUseChain,
                       Block *fixedBlock) {
  if (op->getBlock() != fixedBlock)
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
      auto next = traceBatchChainToFixpipe(userOp, batchDim, recursiveUseChain,
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
    llvm::report_fatal_error("current RankedTensorType must be 3D");

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
  // TODO: temporarily ignore insert_slice op generated by batch_matmul to
  // avoid infinite loop while compiling, remove it after gather_load is
  // supported
  insertSliceOp->setAttr(elideAfterBufferize, rewriter.getUnitAttr());
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
      llvm::report_fatal_error("unsupported operation which uses matmul's output");
    }
  }

  return curVal;
}

Value createTiledMmadL1(hivm::BatchMmadL1Op batchmmOp,
                        SmallVector<Value> indexes,
                        RankedTensorType matrixCType,
                        PatternRewriter &rewriter) {
  assert(indexes.size() == 1);
  Value matrixA = extractTensorValueWithoutBatch(
      batchmmOp->getLoc(), batchmmOp.getA(), indexes[0], rewriter);
  Value matrixB = extractTensorValueWithoutBatch(
      batchmmOp->getLoc(), batchmmOp.getB(), indexes[0], rewriter);

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
      batchmmOp.getBTransposeAttr(), batchmmOp.getEnable_HF32Attr(), batchmmOp.getEnable_I4Attr());
  if (batchmmOp->getAttr(fixpipeAlreadyInserted)) {
    tiledMmad->setAttr(fixpipeAlreadyInserted,
                       batchmmOp->getAttr(fixpipeAlreadyInserted));
  }
  return tiledMmad.getResultTensors()[0];
}

FailureOr<SmallVector<SmallVector<Operation *>>>
collectUseChains(hivm::BatchMmadL1Op batchmmOp,
                 RankedTensorType matrixCType) {
  SmallVector<SmallVector<Operation *>> allUseChains;
  for (Operation *user : batchmmOp->getUsers()) {
    SmallVector<Operation *> chain;
    if (failed(traceBatchChainToFixpipe(user, matrixCType.getShape()[0], chain,
                                      batchmmOp->getBlock())))
      return batchmmOp.emitOpError(
          "unaccepted batch matmul: use chain between "
          "hivm::BatchMmadL1Op and "
          "hivm::FixpipeOp is illegal; just support "
          "tensor::extractSliceOp "
          "between above two and all exist in the same block");
    allUseChains.push_back(std::move(chain));
  }
  return allUseChains;
}

struct TensorChainInfo {
  SmallVector<int> indices;
  SmallVector<Value> initArgs;
};

TensorChainInfo collectTensorChainInitArgs(
    const SmallVector<SmallVector<Operation *>> &allUseChains) {
  TensorChainInfo info;
  for (int i = 0; i < static_cast<int>(allUseChains.size()); ++i) {
    auto fixpipe = dyn_cast<hivm::FixpipeOp>(allUseChains[i].front());
    if (isa<TensorType>(fixpipe.getDst().getType())) {
      info.indices.push_back(i);
      info.initArgs.push_back(fixpipe.getDst());
    }
  }
  return info;
}

Operation *getLastChainOp(
    const SmallVector<SmallVector<Operation *>> &allUseChains) {
  Operation *lastChainOp = nullptr;
  for (const auto &chain : allUseChains) {
    for (Operation *op : chain) {
      if (!lastChainOp || lastChainOp->isBeforeInBlock(op))
        lastChainOp = op;
    }
  }
  return lastChainOp;
}

SmallVector<Operation *>
collectDownstreamOpsToMove(
    ArrayRef<int> tensorChainIndices,
    const SmallVector<SmallVector<Operation *>> &allUseChains,
    Operation *lastChainOp) {
  SmallVector<Operation *> fixpipeOps;
  for (int idx : tensorChainIndices)
    fixpipeOps.push_back(allUseChains[idx].front());

  SmallVector<Operation *> opsToMove;
  SmallVector<Operation *> worklist;
  for (Operation *fixpipeOp : fixpipeOps) {
    for (Operation *user : fixpipeOp->getUsers())
      if (!lastChainOp->isBeforeInBlock(user))
        worklist.push_back(user);
  }

  SmallPtrSet<Operation *, 8> seen;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (seen.contains(op))
      continue;
    seen.insert(op);
    opsToMove.push_back(op);
    for (Operation *user : op->getUsers()) {
      if (!lastChainOp->isBeforeInBlock(user))
        worklist.push_back(user);
    }
  }

  llvm::sort(opsToMove, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return opsToMove;
}

LogicalResult checkOpsMoveSafety(ArrayRef<Operation *> opsToMove,
                                 hivm::BatchMmadL1Op batchmmOp) {
  for (Operation *op : opsToMove) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      continue;

    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (auto &effect : effects) {
      // Global effects (no specific Value) like debug prints are safe to move.
      // Only reject ops that have Write/Allocate effects on specific Values
      // that could conflict with the for loop body.
      if (!effect.getValue())
        continue;
      if (isa<MemoryEffects::Write>(effect.getEffect()) ||
          isa<MemoryEffects::Allocate>(effect.getEffect())) {
        return batchmmOp.emitOpError(
            "downstream user of fixpipe result has memory side effects "
            "and cannot be moved after the tiled loop");
      }
    }
  }
  return success();
}

void cloneAndReplaceDownstreamOps(
    ArrayRef<Operation *> opsToMove, scf::ForOp forOp,
    ArrayRef<int> tensorChainIndices,
    const SmallVector<SmallVector<Operation *>> &allUseChains,
    PatternRewriter &rewriter) {
  DenseMap<Value, Value> replacements;
  int resultIdx = 0;
  for (int idx : tensorChainIndices) {
    auto fixpipe = dyn_cast<hivm::FixpipeOp>(allUseChains[idx].front());
    replacements[fixpipe->getResult(0)] = forOp.getResult(resultIdx);
    resultIdx++;
  }

  rewriter.setInsertionPointAfter(forOp.getOperation());
  for (Operation *op : opsToMove) {
    IRMapping mapping;
    for (auto &[oldVal, newVal] : replacements)
      mapping.map(oldVal, newVal);

    auto *cloned = rewriter.clone(*op, mapping);
    for (unsigned r = 0; r < op->getNumResults(); ++r)
      replacements[op->getResult(r)] = cloned->getResult(r);
  }

  for (auto &[oldVal, newVal] : replacements)
    rewriter.replaceAllUsesWith(oldVal, newVal);
}

std::optional<Value>
rewriteFixpipeThrowOutBatch(Value matrixToStore, SmallVector<Value> indexes,
                            const SmallVector<Operation *> &useChain,
                            BlockArgument forIterArg,
                            PatternRewriter &rewriter) {
  assert(indexes.size() == 1);
  auto originFixpipe = dyn_cast<hivm::FixpipeOp>(useChain.front());
  Value oriFixpipeDst = originFixpipe.getDst();

  if (isa<mlir::MemRefType>(oriFixpipeDst.getType())) {
    Value fixpipeDst = subviewMemrefValueWithoutBatch(
        originFixpipe.getLoc(), oriFixpipeDst, indexes[0], rewriter);

    rewriter.create<hivm::FixpipeOp>(
        originFixpipe.getLoc(), Type{}, /*src=*/matrixToStore,
        /*dst=*/fixpipeDst, originFixpipe.getDmaModeAttr(),
        originFixpipe.getDualDstModeAttr(), originFixpipe.getPreQuantAttr(),
        originFixpipe.getPreReluAttr(), originFixpipe.getChannelSplitAttr());
    return std::nullopt;
  }

  if (isa<mlir::TensorType>(oriFixpipeDst.getType())) {
    // All tensor-type chains use iter args for accumulation.
    assert(forIterArg);
    auto iterationType =
        dyn_cast<RankedTensorType>(forIterArg.getType());
    assert(iterationType);

    Value fixpipeDst = extractTensorValueWithoutBatch(
        originFixpipe.getLoc(), forIterArg, indexes[0], rewriter);
    Type resultType = extractNonBatchType(iterationType);

    auto newfixpipe = rewriter.create<hivm::FixpipeOp>(
        originFixpipe.getLoc(), resultType, /*src=*/matrixToStore,
        /*dst=*/fixpipeDst, originFixpipe.getDmaModeAttr(),
        originFixpipe.getDualDstModeAttr(), originFixpipe.getPreQuantAttr(),
        originFixpipe.getPreReluAttr(), originFixpipe.getChannelSplitAttr());

    Value insert = insertTensorValueWithoutBatch(
        originFixpipe.getLoc(), newfixpipe.getResults()[0], forIterArg,
        indexes[0], rewriter);
    return insert;
  }

  llvm::report_fatal_error("unsupported fixpipe dst type");
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
    assert(matrixCType.getRank() == 3);

    // 1. Collect and validate use chains
    auto allUseChains = collectUseChains(batchmmOp, matrixCType);
    if (failed(allUseChains))
      return failure();

    // 2. Identify tensor-type chains for iter args
    auto tensorInfo = collectTensorChainInitArgs(*allUseChains);

    // 3. Collect downstream users and check move safety .
    Operation *lastChainOp = getLastChainOp(*allUseChains);
    auto opsToMove = collectDownstreamOpsToMove(
        tensorInfo.indices, *allUseChains, lastChainOp);
    if (failed(checkOpsMoveSafety(opsToMove, batchmmOp)))
      return failure();

    // 4. Build loop body and create for loop
    auto buildLoopBody =
        [&batchmmOp, &rewriter, &matrixCType, &allUseChains = *allUseChains](
            const SmallVector<Value> &indexes,
            Block::BlockArgListType iterArgs) -> void {
      Value tiledMmadResult =
          createTiledMmadL1(batchmmOp, indexes, matrixCType, rewriter);

      SmallVector<Value> yieldValues;
      int tensorChainCounter = 0;
      for (int i = 0; i < static_cast<int>(allUseChains.size()); ++i) {
        const auto &chain = allUseChains[i];
        Value matrixToStore = rewriteMatrixCShapeChange(
            ArrayRef<Operation *>(chain).drop_front(), tiledMmadResult,
            rewriter);

        auto originFixpipe = dyn_cast<hivm::FixpipeOp>(chain.front());
        BlockArgument iterArg;
        if (isa<TensorType>(originFixpipe.getDst().getType())) {
          iterArg = iterArgs[tensorChainCounter++];
        }

        auto result = rewriteFixpipeThrowOutBatch(
            matrixToStore, indexes, chain, iterArg, rewriter);
        if (result)
          yieldValues.push_back(*result);
      }

      if (!yieldValues.empty())
        rewriter.create<scf::YieldOp>(batchmmOp.getLoc(), yieldValues);
    };

    std::set<int> loopDims = {0};
    rewriter.setInsertionPointAfter(lastChainOp);

    auto nestFor = createNestedLoops(
        rewriter, batchmmOp.getLoc(), matrixC, loopDims, buildLoopBody, 0,
        tensorInfo.initArgs.empty()
            ? std::optional<SmallVector<Value>>{std::nullopt}
            : std::optional<SmallVector<Value>>(tensorInfo.initArgs));

    assert(nestFor.size() == 1);
    scf::ForOp forOp = nestFor[0];

    // 5. Move downstream users after the for loop
    cloneAndReplaceDownstreamOps(opsToMove, forOp, tensorInfo.indices,
                                 *allUseChains, rewriter);

    // 6. Erase old ops
    for (Operation *op : llvm::reverse(opsToMove))
      rewriter.eraseOp(op);
    for (auto &chain : *allUseChains)
      for (Operation *op : chain)
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
