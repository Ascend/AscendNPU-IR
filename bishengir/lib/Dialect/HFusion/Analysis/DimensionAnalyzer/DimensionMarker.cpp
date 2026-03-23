//===- DimensionMarker.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "dimension-analyzer-marker"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

bool DimensionAnalyzer::processOperation(Operation *op, Value current) {
  LLVM_DEBUG(llvm::dbgs() << "Processing operation: [" << *op << "]\n"
                          << "with value [" << current << "]\n");
  bool success = true;
  TypeSwitch<Operation *>(op)
      .Case<linalg::BroadcastOp>(
          [&](auto broadcastOp) { processBroadcastOp(broadcastOp); })
      .Case<linalg::ReduceOp>(
          [&](auto reduceOp) { processReduceLikeOp(reduceOp); })
      .Case<linalg::TransposeOp>(
          [&](auto transposeOp) { processTransposeOp(transposeOp); })
      .Case<linalg::MatmulOp, linalg::MatmulTransposeAOp,
            linalg::MatmulTransposeBOp>([&](auto transposeOp) {
        bool isTransposeA = isa<linalg::MatmulTransposeAOp>(op);
        bool isTransposeB = isa<linalg::MatmulTransposeBOp>(op);
        processMatmulOp(op, isTransposeA, isTransposeB);
      })
      .Case<linalg::BatchMatmulOp>(
          [&](auto batchMatmulOp) { processBatchMatmulOp(batchMatmulOp); })
      .Case<hfusion::ReduceWithIndexOp>([&](auto reduceWithIndexOp) {
        processReduceLikeOp(reduceWithIndexOp);
      })
      .Case<scf::ForOp>([&](auto forOp) { processForOp(forOp); })
      .Case<scf::YieldOp>([&](auto yieldOp) { processYieldOp(yieldOp); })
      .Case<bufferization::ToTensorOp>(
          [&](auto toTensorOp) { processToTensorOp(toTensorOp); })
      .Case<bufferization::ToMemrefOp>(
          [&](auto toMemrefOp) { processToMemrefOp(toMemrefOp); })
      .Case<memref::CastOp>([&](memref::CastOp castOp) {
        auto src = castOp.getSource();
        auto res = castOp.getResult();
        createDummyRefIfNotExist({src, res});
        processValue(src, res);
      })
      .Case<memref::SubViewOp>(
          [&](auto subviewOp) { processSubviewOp(subviewOp); })
      .Case<hfusion::GatherOp>(
          [&](auto gatherOp) { processGatherOp(gatherOp); })
      .Case<hfusion::InterleaveOp>(
          [&](auto interleaveOp) { processInterleaveOp(interleaveOp); })
      .Case<hfusion::DeinterleaveOp>(
          [&](auto deinterleaveOp) { processDeinterleaveOp(deinterleaveOp); })
      .Case<hfusion::CumsumOp>([&](auto cumsumOp) { processCumOp(cumsumOp); })
      .Case<hfusion::CumprodOp>(
          [&](auto cumprodOp) { processCumOp(cumprodOp); })
      .Case<hfusion::FlipOp>([&](auto flipOp) { processFlipOp(flipOp); })
      .Case<hfusion::SortOp>([&](auto sortOp) { processSortOp(sortOp); })
      .Case<hfusion::EmbeddingGatherOp>([&](auto embeddingGatherOp) {
        processEmbeddingGatherOp(embeddingGatherOp);
      })
      .Case<hfusion::IndirectLoadOp>(
          [&](auto indirectLoadOp) { processIndirectLoadOp(indirectLoadOp); })
      .Case<hfusion::IndirectStoreOp>([&](auto indirectStoreOp) {
        processIndirectStoreOp(indirectStoreOp);
      })
      .Default([&](Operation *op) {
        bool isParallelRegbased =
            options.registerBased &&
            isa_and_present<memref::CopyOp, hivm::CopyOp,
                            bufferization::MaterializeInDestinationOp,
                            hfusion::MulExtOp, memref::MemorySpaceCastOp,
                            annotation::MarkOp>(op);
        if (isAllParallelOp(op) || isParallelRegbased) {
          processParallelOp(op, current);
        } else {
          if (!DimensionAnalyzerBase::processOperation(op, current)) {
            LLVM_DEBUG(llvm::dbgs() << "Warning: operation is unchecked " << *op
                                    << "\n";);
            success = false;
          }
        }
      });

  return success;
}

void DimensionAnalyzer::processBroadcastOp(linalg::BroadcastOp op) {
  Value input = op.getInput();
  Value output = op.getInit();

  createDummyRefIfNotExist({output});
  assert(argumentsRefPointer_.contains(output));
  const auto &outputArgs = getArgumentRef(output);
  auto newValRef =
      processDecreasingDimensions(outputArgs, op.getDimensions(), input);
  initCollapseOrVerify(input, newValRef);

  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

template <class T, typename>
void DimensionAnalyzer::processReduceLikeOp(T reduceOp) {
  ArrayRef<int64_t> reduceDims = reduceOp.getDimensions();
  auto dpsOp = cast<DestinationStyleOpInterface>(reduceOp.getOperation());
  SmallVector<Value> inputs = dpsOp.getDpsInputs();
  SmallVector<Value> outputs = dpsOp.getDpsInits();
  auto &pivotInput = inputs[0];
  auto &pivotOutput = outputs[0];
  createDummyRefIfNotExist({pivotInput});
  assert(argumentsRefPointer_.contains(pivotInput));
  const auto refPtr = argumentsRefPointer_.at(pivotInput);
  for (Value input : inputs) {
    initCollapseOrVerify(input, refPtr);
  }

  // Connect input and output
  assert(argumentsRefPointer_.contains(pivotInput));
  auto inputArgs = getArgumentRefOrCreateDummy(pivotInput);
  auto newValRef =
      processDecreasingDimensions(inputArgs, reduceDims, pivotOutput);
  initCollapseOrVerify(pivotOutput, newValRef);

  const auto refPtrOut = argumentsRefPointer_.at(pivotOutput);
  for (Value output : outputs) {
    initCollapseOrVerify(output, refPtrOut);
  }
  for (Value result : reduceOp.getResults()) {
    processValue(result, pivotOutput);
  }
  reduceOp.walk([&](linalg::IndexOp indexOp) {
    const auto accessedIdx = indexOp.getDim();
    LDBG(accessedIdx);
    const auto &inputArgs = getArgumentRef(pivotInput);
    LDBG(inputArgs.size());
    if (accessedIdx - 1 >= 0)
      disconnect(inputArgs[accessedIdx - 1], inputArgs[accessedIdx]);
    LDBG("Disconnect with left");
    if (accessedIdx + 1 < inputArgs.size())
      disconnect(inputArgs[accessedIdx], inputArgs[accessedIdx + 1]);
    LDBG("Disconnect with right");
  });
}

void DimensionAnalyzer::processTransposeOp(linalg::TransposeOp op) {
  Value input = op.getInput();
  Value output = op.getInit();
  auto perm = op.getPermutation();
  const auto &inputArgs = getArgumentRefOrCreateDummy(input);
  auto newValRef = processPermutation(inputArgs, perm, output);
  initCollapseOrVerify(output, newValRef);
  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

template <class T>
void DimensionAnalyzer::processCumOp(T cumOp) {
  auto input = cumOp.getInput();
  auto inputRef = getArgumentRefOrCreateDummy(input);
  auto res = cumOp.getResult();
  auto resRef = getArgumentRefOrCreateDummy(res);
  auto resultShape = utils::getShape(res.getType());
  auto rank = resultShape.size();
  BitVector cumMask(rank);
  for (auto &cumDim : cumOp.getCumDims())
    cumMask.set(cumDim);
  for (unsigned i = 0; i < rank; ++i) {
    if (!cumMask[i]) {
      isConnected_[resRef[i]].elementKind =
          (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
      joinShape(resRef[i], inputRef[i]);
    } else {
      isConnected_[resRef[i]].elementKind = ElementKind::HasMutation;
      joinCollapser(resRef[i], inputRef[i]);
    }
  }
}

void DimensionAnalyzer::processGatherOp(hfusion::GatherOp gatherOp) {
  auto axis = gatherOp.getAxis();
  auto resVariadic = gatherOp.getResult();
  Value res;
  if (resVariadic.empty())
    res = gatherOp.getDpsInitOperand(0)->get();
  else
    res = gatherOp.getResult().front();

  auto resultShape = utils::getShape(res.getType());
  auto rank = resultShape.size();

  for (auto opr : gatherOp.getOperands()) {
    if (utils::getShapeRank(opr.getType()).value_or(0) != resultShape.size())
      continue;
    createDummyRefIfNotExist({opr, res});
    auto oprArgRef = getArgumentRef(opr);
    auto gatherResRef = getArgumentRef(res);
    for (unsigned i = 0; i < rank; i++) {
      if (i != axis) {
        isConnected_[gatherResRef[i]].elementKind =
            (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
        joinShape(gatherResRef[i], oprArgRef[i]);
      } else {
        isConnected_[gatherResRef[i]].elementKind = ElementKind::HasMutation;
        joinCollapser(gatherResRef[i], oprArgRef[i]);
      }
    }
  }
}

void DimensionAnalyzer::processInterleaveOp(
    hfusion::InterleaveOp interleaveOp) {
  auto res = interleaveOp.getResult();
  auto resultShape = utils::getShape(res.getType());
  const auto rank = static_cast<int>(resultShape.size());
  auto firstOperand = interleaveOp.getOperand(0);
  createDummyRefIfNotExist(SmallVector<Value>(interleaveOp->getOperands()));
  for (auto opr : interleaveOp.getOperands()) {
    if (utils::getShapeRank(opr.getType()).value_or(0) != resultShape.size())
      continue;
    auto oprRef = getArgumentRefOrCreateDummy(opr);
    auto resRef = getArgumentRefOrCreateDummy(res);
    for (int i = 0; i < rank - 1; ++i) {
      isConnected_[resRef[i]].elementKind =
          (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
      // Shape elem has type inference, can only be joined if the shape is
      // exactly the same
      joinShape(resRef[i], oprRef[i]);
    }
    // Last element is a mutation kind
    auto firstOperandRef = getArgumentRefOrCreateDummy(firstOperand);
    isConnected_[resRef[rank - 1]].elementKind = ElementKind::NoMutation;
    // Bind shape with each other
    joinShape(firstOperandRef[rank - 1], oprRef[rank - 1]);
    // Bind structure with res
    joinCollapser(resRef[rank - 1], oprRef[rank - 1]);
  }
  // No need to disconnect anything, handled by elementKind resolver
}

void DimensionAnalyzer::processDeinterleaveOp(
    hfusion::DeinterleaveOp deinterleaveOp) {
  SmallVector<Value> results = deinterleaveOp.getResults();
  createDummyRefIfNotExist(results);
  auto resultShape = utils::getShape(results[0].getType());
  const auto rank = static_cast<int>(resultShape.size());
  auto src = deinterleaveOp.getInput();
  auto oprRef = getArgumentRefOrCreateDummy(src);
  for (auto res : results) {
    auto resRef = getArgumentRefOrCreateDummy(res);
    for (int i = 0; i < rank - 1; ++i) {
      isConnected_[resRef[i]].elementKind =
          (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
      // Shape elem has type inference, can only be joined if the shape is
      // exactly the same
      joinShape(resRef[i], oprRef[i]);
    }
    // Last element is a mutation kind
    auto firstResRef = getArgumentRefOrCreateDummy(results[0]);
    isConnected_[resRef[rank - 1]].elementKind = ElementKind::NoMutation;
    // Bind shape with each other
    joinShape(firstResRef[rank - 1], resRef[rank - 1]);
    // Bind structure with opr, but not shape
    joinCollapser(resRef[rank - 1], oprRef[rank - 1]);
  }
  // No need to disconnect anything, handled by elementKind resolver
}

void DimensionAnalyzer::processYieldOp(scf::YieldOp op) {
  LDBG("Processing YieldOp " << op);
  assert(op->getParentOp() != nullptr);
  Operation *parentOp = op->getParentOp();
  if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
    for (auto [beforeArg, yieldOpResult] :
         llvm::zip_equal(whileOp.getBeforeArguments(), op.getOperands())) {

      createDummyRefIfNotExist({beforeArg, yieldOpResult});
      if (isa<ShapedType>(beforeArg.getType()))
        processValue(beforeArg, yieldOpResult);
    }
    return;
  }

  for (auto [parentOpResult, yieldOpResult] :
       llvm::zip_equal(parentOp->getResults(), op.getOperands())) {
    createDummyRefIfNotExist({parentOpResult, yieldOpResult});
    if (isa<ShapedType>(parentOpResult.getType()))
      processValue(parentOpResult, yieldOpResult);
  }
}

void DimensionAnalyzer::processForOp(scf::ForOp op) {
  LDBG("Processing ForOp " << op);
  auto yieldOp = dyn_cast<scf::YieldOp>(op.getBody()->getTerminator());
  if (yieldOp) {
    for (const auto &[regionArg, initArg, yieldVal] : zip_equal(
             op.getRegionIterArgs(), op.getInitArgs(), yieldOp.getResults())) {
      createDummyRefIfNotExist({regionArg, initArg, yieldVal});
      processValue(regionArg, initArg);
      processValue(yieldVal, initArg);
    }
  } else {
    for (const auto &[regionArg, initArg] :
         zip_equal(op.getRegionIterArgs(), op.getInitArgs())) {
      createDummyRefIfNotExist({regionArg, initArg});
      processValue(regionArg, initArg);
    }
  }
}

void DimensionAnalyzer::processToTensorOp(bufferization::ToTensorOp op) {
  LDBG("Processing bufferization::ToTensorOp " << op);
  auto opr = op.getOperand();
  auto res = op.getResult();
  createDummyRefIfNotExist({opr, res});
  processValue(opr, res);
}

void DimensionAnalyzer::processToMemrefOp(bufferization::ToMemrefOp op) {
  LDBG("Processing ToMemrefOp " << op);
  auto opr = op.getOperand();
  auto res = op.getMemref();
  createDummyRefIfNotExist({opr, res});
  processValue(opr, res);
}

void DimensionAnalyzer::processSubviewOp(memref::SubViewOp slicingOp) {
  auto src = slicingOp.getSource();
  auto res = slicingOp.getResult();
  SmallVector<OpFoldResult> srcShape;
  if (auto expandOp = src.getDefiningOp<memref::ExpandShapeOp>()) {
    srcShape =
        getMixedValues(expandOp.getStaticOutputShape(),
                       expandOp.getOutputShape(), slicingOp.getContext());
  } else {
    srcShape = llvm::map_to_vector(
        utils::getShape(src.getType()),
        [&slicingOp](int64_t elementShape) -> OpFoldResult {
          return getAsIndexOpFoldResult(slicingOp.getContext(), elementShape);
        });
  }
  auto resShape = slicingOp.getMixedSizes();
  auto droppedDims = slicingOp.getDroppedDims();
  auto rank = srcShape.size();
  auto srcRefPtr = getArgumentRefOrCreateDummy(src);
  auto resRefPtr = getArgumentRefOrCreateDummy(res);
  int resPtr = 0;
  for (unsigned i = 0; i < rank; i++) {
    if (droppedDims[i]) {
      // Dropped unit dimensions
      isConnected_[srcRefPtr[i]].elementKind = ElementKind::Unit;
      continue;
    }
    isConnected_[srcRefPtr[i]].elementKind =
        (getConstantIntValue(srcShape[i]).value_or(ShapedType::kDynamic) == 1)
            ? ElementKind::Unit
            : ElementKind::NoMutation;
    LDBG("From the extract slice: "
         << stringtifyElementKind(isConnected_[srcRefPtr[i]].elementKind));
    joinShape(srcRefPtr[i], resRefPtr[resPtr]);
    // Used the resPtr
    resPtr++;
  }
}

void DimensionAnalyzer::processFlipOp(hfusion::FlipOp op) {
  Value flipResult = op.getResult();
  createDummyRefIfNotExist({flipResult});
  for (Value v : op->getOperands()) {
    processValue(v, flipResult);
  }
  auto argRef = getArgumentRef(flipResult);
  auto rank = argRef.size();
  auto flipAxis = op.getFlipAxis();
  if (flipAxis >= 1) {
    disconnect(argRef[flipAxis - 1], argRef[flipAxis]);
  }
  if (flipAxis + 1 < rank) {
    disconnect(argRef[flipAxis], argRef[flipAxis + 1]);
  }
}

void DimensionAnalyzer::processSortOp(hfusion::SortOp op) {
  Value sortResult = op.getResult()[0];
  createDummyRefIfNotExist({sortResult});
  for (Value v : op->getOperands()) {
    processValue(v, sortResult);
  }
  auto argRef = getArgumentRef(sortResult);
  auto rank = argRef.size();
  auto sortAxis = op.getSortAxis();
  if (sortAxis >= 1) {
    disconnect(argRef[sortAxis - 1], argRef[sortAxis]);
  }
  // currently hfusion.sort only support sort on last axis,
  // but the expansion support here has no negative effects
  if (sortAxis + 1 < rank) {
    disconnect(argRef[sortAxis], argRef[sortAxis + 1]);
  }
}

void DimensionAnalyzer::processEmbeddingGatherOp(
    hfusion::EmbeddingGatherOp op) {
  auto gatherResult = op.getResult();
  auto indexTable = op.getIndex();
  auto opOut = op.getDst();
  createDummyRefIfNotExist({gatherResult, indexTable});
  LDBG("[Here embedding] " << indexTable);

  // The embedding_gather semantic: result[b][i][d] = src[index[b][i]][d]
  // So result dimensions come from:
  // - Batch/sequence dims from index tensor
  // - Embedding feature dim from src tensor

  auto indexRank = indexTable.getType().getRank();
  // 1 or 2 (sequence or batch x sequence)
  [[maybe_unused]] auto resultRank = gatherResult.getType().getRank();
  // 2 or 3 (index_dims + embedding_dim)
  assert(indexRank + 1 == resultRank);

  auto indexRefPtr = getArgumentRef(indexTable);
  auto resultRefPtr = getArgumentRef(gatherResult);
  processValue(opOut, gatherResult);
  // Bind index dimensions to corresponding result dimensions
  // index[b][i] -> result[b][i][d]
  for (unsigned i = 0; i < indexRank; i++) {
    // Index dimensions map directly to result dimensions (no mutation)
    isConnected_[resultRefPtr[i]].elementKind = ElementKind::NoMutation;
    joinShape(indexRefPtr[i], resultRefPtr[i]);
    disconnect(resultRefPtr[i], resultRefPtr[i + 1]);
  }

  // The last dimension of result is the embedding dimension from src
  // This is a "gather" dimension - it comes from src but indexed by index
  // values Disconnect batch/sequence dimensions from embedding dimension
  // TODO: adjust numels
}

void DimensionAnalyzer::processIndirectLoadOp(
    hfusion::IndirectLoadOp indirectLoadOp) {
  auto indirectLoadResult = indirectLoadOp.getResult();
  createDummyRefIfNotExist({indirectLoadResult});
  for (auto opr : indirectLoadOp.getOperands()) {
    if (isa<RankedTensorType>(opr.getType())) {
      processValue(opr, indirectLoadResult);
    }
  }
}

void DimensionAnalyzer::processIndirectStoreOp(hfusion::IndirectStoreOp op) {
  Value input = op.getSrc();
  Value offsets = op.getOffsets();
  Value mask = op.getMask();
  createDummyRefIfNotExist({input, offsets});
  processValue(offsets, input);
  if (mask) {
    createDummyRefIfNotExist({mask});
    processValue(mask, input);
  }
}

} // namespace detail
} // namespace hfusion
} // namespace mlir
