//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utils function for propagate reshape
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>

#define DEBUG_TYPE "tensor-propagate-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir {
namespace tensor {
namespace reshape_utils {

template <class OpDimTy>
void updateDimensionalOp(OpDimTy op, PatternRewriter &rewriter,
                         ArrayRef<int64_t> newDimensions) {
  rewriter.modifyOpInPlace(op, [&]() { op.setDimensions(newDimensions); });
}

void updateHIVMDimensionalOp(hivm::VBrcOp op, PatternRewriter &rewriter,
                             ArrayRef<int64_t> newDimensions) {
  rewriter.modifyOpInPlace(op, [&]() { op.setBroadcastDims(newDimensions); });
}

void updateHIVMDimensionalOp(hivm::VReduceOp op, PatternRewriter &rewriter,
                             ArrayRef<int64_t> newDimensions) {
  rewriter.modifyOpInPlace(op, [&]() { op.setReduceDims(newDimensions); });
}

void updateDefiningOp(Operation *definingOp, PatternRewriter &rewriter,
                      ArrayRef<Value> newOperands) {
  if (!isa<DestinationStyleOpInterface>(definingOp))
    updateDefiningOpNonDst(definingOp, rewriter, newOperands);
  rewriter.modifyOpInPlace(definingOp, [&]() {
    definingOp->setOperands(newOperands);
    auto dpsInits = cast<DestinationStyleOpInterface>(definingOp).getDpsInits();
    for (unsigned i = 0; i < definingOp->getNumResults(); ++i) {
      Value initOperand = dpsInits[i];
      auto collapsedType = initOperand.getType();
      definingOp->getResult(i).setType(collapsedType);
    }
  });
}

void updateDefiningOpNonDst(Operation *definingOp, PatternRewriter &rewriter,
                            ArrayRef<Value> newOperands) {
  rewriter.modifyOpInPlace(definingOp,
                           [&]() { definingOp->setOperands(newOperands); });
}

void updateDefiningOpNonDst(Operation *definingOp, PatternRewriter &rewriter,
                            ArrayRef<Value> newOperands,
                            ArrayRef<int64_t> collapsedShape) {
  rewriter.modifyOpInPlace(definingOp, [&]() {
    definingOp->setOperands(newOperands);
    for (unsigned i = 0; i < definingOp->getNumResults(); ++i) {
      auto oldType = getElementTypeOrSelf(definingOp->getResult(i));
      definingOp->getResult(i).setType(
          RankedTensorType::get(collapsedShape, oldType));
    }
  });
}

void renumberReassociation(
    SmallVector<ReassociationIndices> &newReassociation) {
  int shapeCounter = 0;
  for (auto &reassociationIndex : newReassociation) {
    for (auto &shapeIndex : reassociationIndex) {
      shapeIndex = shapeCounter++;
    }
  }
}

void renumberReassociationAndGetNewDimensions(
    SmallVector<ReassociationIndices> &newReassociation,
    SmallVector<int64_t> &newDimensions) {
  newDimensions.clear();
  int shapeCounter = 0;
  for (auto &reassociationIndex : newReassociation) {
    for (auto &shapeIndex : reassociationIndex) {
      if (shapeIndex == -1)
        newDimensions.push_back(shapeCounter);
      shapeIndex = shapeCounter++;
    }
  }
}

bool checkValueIsInit(Operation *op, Value val) {
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    auto inits = dpsOp.getDpsInits();
    return llvm::is_contained(inits, val);
  }
  return false;
}

template <class ReshapeOpTy, class BuilderTy>
Operation *createNewReshapingOp(BuilderTy &rewriter, Location loc,
                                Value operand,
                                ArrayRef<ReassociationIndices> reassociation,
                                ArrayRef<int64_t> resultShape) {
  auto resultType =
      RankedTensorType::get(resultShape, getElementTypeOrSelf(operand));
  return rewriter.template create<ReshapeOpTy>(loc, resultType, operand,
                                               reassociation);
}

void collapseAndReplace(PatternRewriter &rewriter,
                        MutableArrayRef<ReassociationIndices> reassociation,
                        const SmallVector<int64_t> &outputShape,
                        Value replacedVal, Value newUncollapsedVal,
                        Operation *userOp) {
  rewriter.setInsertionPointAfterValue(newUncollapsedVal);
  Operation *replacerOp =
      createNewReshapingOp<tensor::CollapseShapeOp, PatternRewriter>(
          rewriter, userOp->getLoc(), newUncollapsedVal, reassociation,
          outputShape);
  SmallPtrSet<Operation *, 2> excepted = {replacerOp, userOp};
  rewriter.replaceAllUsesExcept(replacedVal, replacerOp->getResult(0),
                                excepted);
}

void collapseAndReplace(PatternRewriter &rewriter,
                        MutableArrayRef<ReassociationIndices> reassociation,
                        const SmallVector<int64_t> &outputShape, Value newVal,
                        Operation *userOp) {
  rewriter.setInsertionPointAfter(userOp);
  Operation *replacerOp =
      createNewReshapingOp<tensor::CollapseShapeOp, PatternRewriter>(
          rewriter, userOp->getLoc(), newVal, reassociation, outputShape);
  SmallPtrSet<Operation *, 2> excepted = {replacerOp, userOp};
  rewriter.replaceAllUsesExcept(newVal, replacerOp->getResult(0), excepted);
}

void expandAndReplace(PatternRewriter &rewriter,
                      MutableArrayRef<ReassociationIndices> reassociation,
                      const SmallVector<int64_t> &outputShape, Value newVal,
                      Operation *userOp) {
  rewriter.setInsertionPointAfter(userOp);
  Operation *replacerOp =
      createNewReshapingOp<tensor::ExpandShapeOp, PatternRewriter>(
          rewriter, userOp->getLoc(), newVal, reassociation, outputShape);
  SmallPtrSet<Operation *, 2> excepted = {replacerOp, userOp};
  rewriter.replaceAllUsesExcept(newVal, replacerOp->getResult(0), excepted);
}

template <class ReshapeOpTy>
void collapseAndReplace(PatternRewriter &rewriter, ReshapeOpTy reshapeOp,
                        Type ty, Value newVal, Operation *definingOp) {
  auto reassociation = reshapeOp.getReassociationIndices();
  auto outputShape = utils::getShape(ty);
  collapseAndReplace(rewriter, reassociation, outputShape, newVal, definingOp);
}

void collapseAndReplace(PatternRewriter &rewriter,
                        tensor::CollapseShapeOp collapseOp, Value newVal,
                        Operation *userOp) {
  collapseAndReplace(rewriter, collapseOp, collapseOp.getResult().getType(),
                     newVal, userOp);
}

SmallVector<ReassociationIndices> getResultReassociation(Operation *op) {
  auto initVal = *cast<DestinationStyleOpInterface>(op).getDpsInits().begin();
  Operation *initOp = initVal.getDefiningOp();
  assert(initOp != nullptr);
  auto expandShape = cast<tensor::ExpandShapeOp>(initOp);
  return expandShape.getReassociationIndices();
}

// Explicit template instantiations
template Operation *
createNewReshapingOp<tensor::CollapseShapeOp, PatternRewriter>(
    PatternRewriter &rewriter, Location loc, Value operand,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<int64_t> resultShape);

template Operation *
createNewReshapingOp<tensor::ExpandShapeOp, PatternRewriter>(
    PatternRewriter &rewriter, Location loc, Value operand,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<int64_t> resultShape);

template Operation *createNewReshapingOp<tensor::CollapseShapeOp, OpBuilder>(
    OpBuilder &rewriter, Location loc, Value operand,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<int64_t> resultShape);

template Operation *createNewReshapingOp<tensor::ExpandShapeOp, OpBuilder>(
    OpBuilder &rewriter, Location loc, Value operand,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<int64_t> resultShape);

template void
updateDimensionalOp<mlir::linalg::BroadcastOp>(mlir::linalg::BroadcastOp op,
                                               PatternRewriter &rewriter,
                                               ArrayRef<int64_t> newDimensions);

template void
updateDimensionalOp<mlir::linalg::ReduceOp>(mlir::linalg::ReduceOp op,
                                            PatternRewriter &rewriter,
                                            ArrayRef<int64_t> newDimensions);

template void updateDimensionalOp<mlir::hfusion::ReduceWithIndexOp>(
    mlir::hfusion::ReduceWithIndexOp op, PatternRewriter &rewriter,
    ArrayRef<int64_t> newDimensions);

template void collapseAndReplace<mlir::tensor::ExpandShapeOp>(
    PatternRewriter &rewriter, mlir::tensor::ExpandShapeOp reshapeOp, Type ty,
    Value newVal, Operation *definingOp);

LogicalResult computeExpandPad(OpBuilder &rewriter, tensor::PadOp &padOp,
                               ArrayRef<ReassociationIndices> reassociation,
                               DenseMap<uint64_t, uint64_t> &padBodyMapping,
                               SmallVector<OpFoldResult> &newPadLow,
                               SmallVector<OpFoldResult> &newPadHigh,
                               SmallVector<OpFoldResult> &newExpandOutputShape,
                               ArrayRef<OpFoldResult> oldExpandOutputShape,
                               ArrayRef<int64_t> dimensionResult) {
  auto loc = padOp.getLoc();
  auto padSrcMixedType =
      tensor::getMixedSizes(rewriter, loc, padOp.getSource());
  auto padLow = padOp.getStaticLow();
  auto padLowMixed = padOp.getMixedLowPad();
  auto padHigh = padOp.getStaticHigh();
  auto padHighMixed = padOp.getMixedHighPad();
  for (size_t i = 0; i < reassociation.size(); i++) {
    LDBG("Pad Low index " << i);
    // Default value
    padBodyMapping[i] = reassociation[i].back();
    if (padLow[i] == 0 && padHigh[i] == 0) {
      // padLow [0, 10, 0, 0, 5]
      // padHi  [0 , 3, 2, 0, 0]
      //         ^
      // Expand [[A, B], ......]
      // Expand A B like usual from the above

      // This can be expanded as you like, newOutputShape for the topExpand
      // shall be the same
      for (auto idx : reassociation[i]) {
        newPadLow.push_back(rewriter.getI64IntegerAttr(0));
        newPadHigh.push_back(rewriter.getI64IntegerAttr(0));
        newExpandOutputShape.push_back(oldExpandOutputShape[idx]);
      }
    } else {
      assert(reassociation.size() == padSrcMixedType.size());
      // Can only be unit extent
      int64_t nonUnitCnt = 0;
      for (auto idx : reassociation[i]) {
        if (dimensionResult[idx] != 1) {
          nonUnitCnt++;
          // This is the pad body mapping, it maps the old to the new expand
          padBodyMapping[i] = static_cast<uint64_t>(idx);
          newPadLow.push_back(padLowMixed[i]);
          newPadHigh.push_back(padHighMixed[i]);
          // Same as the source, can be dynamic, so find the opfoldresult of
          // the source
          newExpandOutputShape.push_back(padSrcMixedType[i]);
        } else {
          newPadLow.push_back(rewriter.getI64IntegerAttr(0));
          newPadHigh.push_back(rewriter.getI64IntegerAttr(0));
          // is a unit, friendly 1 unit is added
          newExpandOutputShape.push_back(rewriter.getI64IntegerAttr(1));
        }
      }
      if (nonUnitCnt == 0) {
        nonUnitCnt++;
        newPadLow.back() = padLowMixed[i];
        newPadHigh.back() = padHighMixed[i];
        // Same as the source, can be dynamic, so find the opfoldresult of
        // the source
        newExpandOutputShape.back() = padSrcMixedType[i];
      }
      if (nonUnitCnt != 1)
        return failure();
    }
  }
  return success();
}

void clonePadRegion(OpBuilder &rewriter, tensor::PadOp &padOp,
                    tensor::PadOp &newPadOp,
                    DenseMap<uint64_t, uint64_t> &padBodyMapping) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (!padOp.getRegion().empty()) {
    // Clone the region including all blocks and operations
    IRMapping irMapping;
    // Now map all the mapping to the new arguments
    // Expand will increase arguments from the padOp body, each of the nonused
    // is padded with 0, the inside of padding can be filled depending on the
    // index pad

    Block *newBlock = rewriter.createBlock(&newPadOp.getRegion());
    // Create a new block with the same number of arguments as the expanded
    // dimensions Add block arguments for each dimension plus one for the
    // element type
    auto loc = newPadOp.getLoc();
    auto newPadRank = newPadOp.getResult().getType().getRank();
    for (int32_t i = 0; i < newPadRank; ++i) {
      newBlock->addArgument(rewriter.getIndexType(), loc);
    }
    auto newArguments = newPadOp.getRegion().getArguments();
    for (const auto &[i, oldArg] :
         llvm::enumerate(padOp.getRegion().getArguments())) {
      LDBG("Mapping args new Pad Op: " << oldArg << " "
                                       << newArguments[padBodyMapping[i]]);
      irMapping.map(oldArg, newArguments[padBodyMapping[i]]);
    }

    padOp.getRegion().cloneInto(&newPadOp.getRegion(), irMapping);
    auto targetBlock = newPadOp.getRegion().getBlocks().begin();
    auto clonedBlock = std::next(targetBlock);
    // padOp doesn't support cf as body, has a verification inside that
    // blocks.size() <= 1
    // Is there a better way to do this?
    // Move all operations from clonedBlock to targetBlock
    targetBlock->getOperations().splice(targetBlock->end(),
                                        clonedBlock->getOperations());
    // Remove the now-empty cloned block
    clonedBlock->erase();
  }
}

void clonePadRegion(OpBuilder &rewriter, tensor::PadOp &padOp,
                    tensor::PadOp &newPadOp) {
  DenseMap<uint64_t, uint64_t> padBodyMapping;
  for (size_t i = 0; i < padOp.getRegion().getNumArguments(); ++i) {
    padBodyMapping[i] = i;
  }
  clonePadRegion(rewriter, padOp, newPadOp, padBodyMapping);
}

tensor::ConcatOp buildNewConcat(OpBuilder &rewriter, tensor::ConcatOp &concatOp,
                                ArrayRef<ReassociationIndices> reassociation,
                                uint64_t &newConcatDim,
                                SmallVector<OpFoldResult> &newExpandOutputShape,
                                ArrayRef<OpFoldResult> operandsNewDimSize) {
  SmallVector<Value> newExpandedOperands;
  for (const auto [opIdx, opr] : llvm::enumerate(concatOp.getInputs())) {
    // asserted verification for tensor concat
    // Every inputs will be expanded
    rewriter.setInsertionPointAfterValue(opr);
    auto newOprOutputShape = newExpandOutputShape;
    newOprOutputShape[newConcatDim] = operandsNewDimSize[opIdx];

    auto staticExpandOprShape = decomposeMixedValues(newOprOutputShape).first;
    auto expandOprType = RankedTensorType::get(
        staticExpandOprShape, getElementTypeOrSelf(opr.getType()));
    auto newExpandedOperand = rewriter.create<tensor::ExpandShapeOp>(
        opr.getLoc(), expandOprType, /* src */ opr, reassociation);
    newExpandedOperands.push_back(newExpandedOperand);
  }

  auto loc = concatOp.getLoc();
  rewriter.setInsertionPointAfter(concatOp);

  auto staticOutputShape = decomposeMixedValues(newExpandOutputShape).first;
  auto concatResType = RankedTensorType::get(
      staticOutputShape, getElementTypeOrSelf(concatOp.getType()));
  auto newConcatOp = rewriter.create<tensor::ConcatOp>(
      loc, concatResType, newConcatDim, newExpandedOperands);
  return newConcatOp;
}

tensor::ExpandShapeOp
createExpand(PatternRewriter &rewriter, Location loc, Value src,
             ArrayRef<ReassociationIndices> reassociation,
             const SmallVector<OpFoldResult> &newOutputShape) {
  auto staticOutputShape = decomposeMixedValues(newOutputShape).first;
  auto expandResType = RankedTensorType::get(
      staticOutputShape, getElementTypeOrSelf(src.getType()));
  return rewriter.create<tensor::ExpandShapeOp>(loc, expandResType, src,
                                                reassociation, newOutputShape);
}

memref::ExpandShapeOp
createMemrefExpand(PatternRewriter &rewriter, Location loc, Value src,
                   ArrayRef<ReassociationIndices> reassociation,
                   const SmallVector<OpFoldResult> &newOutputShape) {
  auto staticOutputShape = decomposeMixedValues(newOutputShape).first;
  auto expandResType =
      MemRefType::get(staticOutputShape, getElementTypeOrSelf(src.getType()));
  return rewriter.create<memref::ExpandShapeOp>(loc, expandResType, src,
                                                reassociation, newOutputShape);
}

using Hyperrectangle = SmallVector<HyperrectangularSlice>;

static bool adjustSubviewExpansion(int64_t totalSrc, int64_t innermostStride,
                                   int64_t operandStride,
                                   SmallVector<int64_t> &resultShape,
                                   SmallVector<int64_t> &srcShape) {

  int64_t k = (int64_t)resultShape.size();
  if (k == 0)
    return false;

  // Step 1: Calculate the number of elements spanned by a unit increment along
  // dimension i
  SmallVector<int64_t> inner(k);
  inner[k - 1] = 1;
  for (int i = k - 2; i >= 0; --i) {
    if (resultShape[i + 1] <= 0)
      return false;
    inner[i] = inner[i + 1] * resultShape[i + 1];
  }

  // Step 2: Calculate subview result strides
  SmallVector<int64_t> resStrides(k);
  for (int i = 0; i < k; ++i) {
    resStrides[i] = operandStride * innermostStride * inner[i];
  }

  // Step 3: Calculate subview operand strides: [1, 1, ..., T]
  SmallVector<int64_t> subStrides(k, 1);
  subStrides[k - 1] = operandStride;

  // Step 4: Calculate srcMemref strides
  SmallVector<int64_t> srcStrides(k);
  for (int i = 0; i < k; ++i) {
    if (resStrides[i] % subStrides[i] != 0)
      return false;
    srcStrides[i] = resStrides[i] / subStrides[i];
  }

  if (srcStrides[k - 1] != innermostStride)
    return false;

  // Step 5: Calculate dimensions from last to first.
  srcShape.resize(k);
  srcShape[k - 1] = -1;
  for (int i = k - 2; i >= 0; --i) {
    if (srcStrides[i + 1] == 0 || srcStrides[i] % srcStrides[i + 1] != 0) {
      return false;
    }
    srcShape[i + 1] = srcStrides[i] / srcStrides[i + 1];
  }
  int64_t runningProduct = 1;
  for (int i = 1; i < k; ++i) {
    if (srcShape[i] <= 0)
      return false;
    runningProduct *= srcShape[i];
  }

  if (runningProduct == 0 || totalSrc % runningProduct != 0) {
    return false;
  }
  srcShape[0] = totalSrc / runningProduct;

  // Verify result
  int64_t total = std::accumulate(srcShape.begin(), srcShape.end(), 1LL,
                                  std::multiplies<int64_t>());
  if (total != totalSrc)
    return false;
  return true;
}

static std::optional<SmallVector<int64_t>>
getConstantStrides(MemRefType memrefType) {
  SmallVector<int64_t> strides;
  int64_t offset;
  LogicalResult hasStaticInformation =
      getStridesAndOffset(memrefType, strides, offset);
  if (failed(hasStaticInformation)) {
    return std::nullopt;
  }
  return strides;
}

// Helper function to convert OpFoldResult to constants where possible
SmallVector<int64_t> convertToConstantValues(ArrayRef<OpFoldResult> values) {
  SmallVector<int64_t> constantValues;
  for (auto val : values) {
    constantValues.push_back(
        getConstantIntValue(val).value_or(ShapedType::kDynamic));
  }
  return constantValues;
}

SmallVector<int64_t>
getRankReducingShape(const SmallVector<int64_t> &expandedShape,
                     ArrayRef<ReassociationIndices> fullReassociation,
                     const llvm::SmallBitVector &droppedDims) {

  SmallVector<int64_t> newShape;
  for (auto it : llvm::enumerate(fullReassociation)) {
    unsigned srcDimIdx = it.index();
    const ReassociationIndices &expandedGroup = it.value();
    if (droppedDims.test(srcDimIdx)) {
      continue;
    }
    for (int64_t dimIdx : expandedGroup) {
      newShape.push_back(expandedShape[dimIdx]);
    }
  }
  return newShape;
}

SmallVector<OpFoldResult>
getRankReducingShape(const SmallVector<OpFoldResult> &expandedShape,
                     ArrayRef<ReassociationIndices> reassociation,
                     const llvm::SmallBitVector &droppedDims) {

  SmallVector<OpFoldResult> newShape;
  for (auto it : llvm::enumerate(reassociation)) {
    unsigned srcDimIdx = it.index();
    const ReassociationIndices &expandedGroup = it.value();
    if (droppedDims.test(srcDimIdx)) {
      continue;
    }
    LDBG("expandedShape size: " << expandedShape.size());
    for (int64_t dimIdx : expandedGroup) {
      LDBG("dimIdx: " << dimIdx);
      newShape.push_back(expandedShape[dimIdx]);
    }
  }
  return newShape;
}

RankedTensorType
getRankReducingType(const SmallVector<OpFoldResult> &sizes,
                    ArrayRef<ReassociationIndices> reassociation,
                    const llvm::SmallBitVector &droppedDims, Type elementType) {
  auto fullShape = convertToConstantValues(sizes);
  auto newShape = getRankReducingShape(fullShape, reassociation, droppedDims);
  return RankedTensorType::get(newShape, elementType);
}

SmallVector<ReassociationIndices>
getFullReassociation(ArrayRef<ReassociationIndices> reassociation,
                     const llvm::SmallBitVector &droppedDims) {
  SmallVector<ReassociationIndices> result;
  unsigned oldIdx = 0;
  unsigned nextExpDim = 0;

  for (unsigned i = 0; i < droppedDims.size(); ++i) {
    if (droppedDims.test(i)) {
      result.push_back({nextExpDim++});
    } else {
      ReassociationIndices adjustedGroup;
      for (unsigned j = 0; j < reassociation[oldIdx].size(); ++j) {
        adjustedGroup.push_back(nextExpDim++);
      }
      result.push_back(adjustedGroup);
      oldIdx++;
    }
  }
  return result;
}

static SmallVector<ReassociationIndices>
getSubReassociation(ArrayRef<ReassociationIndices> reassociation,
                    const llvm::SmallBitVector &droppedDims) {
  SmallVector<ReassociationIndices> result;
  unsigned oldIdx = 0;
  unsigned nextExpDim = 0;

  for (unsigned i = 0; i < droppedDims.size(); ++i) {
    if (!droppedDims.test(i)) {
      ReassociationIndices adjustedGroup;
      for (unsigned j = 0; j < reassociation[oldIdx].size(); ++j) {
        adjustedGroup.push_back(nextExpDim++);
      }
      result.push_back(adjustedGroup);
    }
    oldIdx++;
  }
  return result;
}

SmallVector<int64_t>
getUndroppedSubviewShape(ArrayRef<int64_t> subview,
                         const llvm::SmallBitVector &droppedDims) {
  SmallVector<int64_t> result;
  unsigned idx = 0;
  for (unsigned i = 0; i < droppedDims.size(); ++i) {
    if (droppedDims.test(i)) {
      result.push_back(1);
    } else {
      result.push_back(subview[idx]);
      ++idx;
    }
  }
  return result;
}

SmallVector<OpFoldResult>
getUndroppedSubviewShape(PatternRewriter &rewriter,
                         ArrayRef<OpFoldResult> subview,
                         const llvm::SmallBitVector &droppedDims) {
  SmallVector<OpFoldResult> result;
  unsigned idx = 0;
  for (unsigned i = 0; i < droppedDims.size(); ++i) {
    if (droppedDims.test(i)) {
      result.push_back(rewriter.getI64IntegerAttr(1));
    } else {
      result.push_back(subview[idx]);
      ++idx;
    }
  }
  return result;
}

SmallVector<OpFoldResult>
getUndroppedExpandedSubviewShape(PatternRewriter &rewriter,
                                 ArrayRef<OpFoldResult> expandedSubview,
                                 ArrayRef<ReassociationIndices> reassociation,
                                 const llvm::SmallBitVector &droppedDims) {
  SmallVector<OpFoldResult> result;
  unsigned idx = 0;
  for (unsigned i = 0; i < droppedDims.size(); ++i) {
    if (droppedDims.test(i)) {
      result.push_back(rewriter.getI64IntegerAttr(1));
    } else {
      const ReassociationIndices &expandedGroup = reassociation[idx];
      for (int64_t dimIdx : expandedGroup) {
        result.push_back(expandedSubview[dimIdx]);
      }
      ++idx;
    }
  }
  return result;
}

// Helper function to handle dimensions with no mutation
static void handleNoMutation(PatternRewriter &rewriter,
                             ArrayRef<OpFoldResult> fullExpandedRef,
                             const ReassociationIndices &reassociationIndices,
                             SliceModifyingOpResult &result) {
  for (auto idx : reassociationIndices) {
    LDBG(fullExpandedRef[idx] << " current size for " << idx);
    result.append(rewriter.getI64IntegerAttr(0), // offset
                  fullExpandedRef[idx],          // size
                  rewriter.getI64IntegerAttr(1), // stride
                  fullExpandedRef[idx],          // subviewShape
                  fullExpandedRef[idx]           // superviewShape
    );
  }
}

static LogicalResult
handleHyperrectangleCase(PatternRewriter &rewriter, ArrayRef<int64_t> slicedRef,
                         int64_t superviewSize, int64_t staticOffset,
                         int64_t staticSize, int64_t staticStride,
                         SliceModifyingOpResult &result) {
  std::optional<Hyperrectangle> hyperrectangle =
      getExtendHyperrectangleFromArray(superviewSize, staticOffset, staticSize,
                                       staticStride, slicedRef);
  if (!hyperrectangle.has_value()) {
    LDBG("[failed] Can't compute hyperrectangle");
    return failure();
  }
  for (auto hyperslice : hyperrectangle.value()) {
    result.append(rewriter.getI64IntegerAttr(hyperslice.offset),
                  rewriter.getI64IntegerAttr(hyperslice.size),
                  rewriter.getI64IntegerAttr(
                      hyperslice.stride == 0 ? 1 : hyperslice.stride),
                  rewriter.getI64IntegerAttr(hyperslice.size),
                  rewriter.getI64IntegerAttr(slicedRef[hyperslice.dimension]));
  }
  return success();
}

// Helper function to handle the non-hyperrectangle case for dimensions with
// mutations
static LogicalResult
handleMutation(PatternRewriter &rewriter,
               const SmallVector<int64_t> &constantFullExpandedRef,
               const ReassociationIndices &reassociationIndices,
               OpFoldResult mixedOffset, OpFoldResult mixedSize,
               OpFoldResult mixedStride, Value superview,
               unsigned dimensionIndex, SliceModifyingOpResult &result) {
  int dimPushed = 0;
  // TODO: support other dynamic cases, this part assumes that it has only unit
  // mutations!
  auto start = result.size();
  for (auto idx : reassociationIndices) {
    LDBG("Iterating reassociation " << idx);
    LDBG(constantFullExpandedRef[idx]);
    if (constantFullExpandedRef[idx] != 1) {
      LDBG("find mutation here");
      auto superviewShape =
          isa<RankedTensorType>(superview.getType())
              ? tensor::getMixedSize(rewriter, superview.getLoc(), superview,
                                     dimensionIndex)
              : memref::getMixedSize(rewriter, superview.getLoc(), superview,
                                     dimensionIndex);
      result.append(mixedOffset, mixedSize, mixedStride, mixedSize,
                    superviewShape);
      dimPushed++;
    } else {
      LDBG("find normal here");
      result.append(
          rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(1));
    }
  }

  // If no dimension was pushed, use the first one as the mutation point
  if (dimPushed == 0) {
    LDBG("Dimension pushed is empty");
    dimPushed++;
    auto superviewShape =
        isa<RankedTensorType>(superview.getType())
            ? tensor::getMixedSize(rewriter, superview.getLoc(), superview,
                                   dimensionIndex)
            : memref::getMixedSize(rewriter, superview.getLoc(), superview,
                                   dimensionIndex);
    // Replace the last entries with the mutation values
    result.replaceAt(start, mixedOffset, mixedSize, mixedStride, mixedSize,
                     superviewShape);
  }

  if (dimPushed != 1) {
    return failure();
  }

  return success();
}

static LogicalResult
checkHyperRectangle(PatternRewriter &rewriter,
                    const ReassociationIndices &reassociation,
                    ArrayRef<int64_t> constantFullExpandedRef, bool isSubview,
                    SliceModifyingOpResult &result, int64_t superviewShape,
                    OpFoldResult mixedOffset, OpFoldResult mixedSize,
                    OpFoldResult mixedStride, int64_t laststride) {
  SmallVector<int64_t> resSlicedRef;
  SmallVector<int64_t> srcSlicedRef;
  int64_t reassociationSize = (int64_t)reassociation.size();
  resSlicedRef.reserve(reassociationSize);
  srcSlicedRef.reserve(reassociationSize);
  for (long j : reassociation) {
    resSlicedRef.emplace_back(constantFullExpandedRef[j]);
    if (ShapedType::isDynamic(resSlicedRef.back())) {
      LDBG("[failed] Dynamic in the sliced ref");
      return failure();
    }
  }
  auto staticStride = getConstantIntValue(mixedStride);
  if (isSubview) {
    if (!staticStride) {
      LDBG("[failed] staticStride can't be null");
      return failure();
    }
    // Compute totalSize and check if all dimensions are static
    bool adjusted =
        adjustSubviewExpansion(superviewShape, laststride, staticStride.value(),
                               resSlicedRef, srcSlicedRef);
    if (!adjusted) {
      LDBG("[failed] Hyperrectangle case can't be adjusted");
      return failure();
    }
    resSlicedRef = srcSlicedRef;
  }
  // Handle dimensions with mutation
  auto staticOffset = getConstantIntValue(mixedOffset);
  auto staticSize = getConstantIntValue(mixedSize);
  // Try to handle as hyperrectangle case if static values are available
  if (staticOffset && staticSize && staticStride) {
    if (succeeded(handleHyperrectangleCase(
            rewriter, resSlicedRef, superviewShape, staticOffset.value(),
            staticSize.value(), staticStride.value(), result))) {
      return success();
    }
  }
  return failure();
}

// Main function to handle the extraction of slice modifying operations
template <class T>
static LogicalResult
getSliceModifyingOp(PatternRewriter &rewriter, T slicingOp,
                    ArrayRef<ReassociationIndices> reassociation,
                    ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                    SliceModifyingOpResult &result) {
  bool isInsert = std::is_same_v<T, InsertSliceOp>;
  // Look at this example
  // [32, 2, 20] -> [16, 1, 10]
  // [32, 2, 20] is the superview
  // [[2, 8, 2], [2], [20]]
  // If we were to reduce the [32] -> [16], The given expansion will be
  // [2, 8, 2] Knowing this, we can use hyperrectangle algo to map the
  // [32] with known OSS (Offsets, Sizes, Strides) to a new coordinate [2, 8, 2]

  // If we were to increase [16] -> [32] (finding its superview), will try to
  // find the most possible superview shape

  SmallVector<OpFoldResult> mixedOffsets = slicingOp.getMixedOffsets();
  SmallVector<OpFoldResult> mixedSizes = slicingOp.getMixedSizes();
  SmallVector<OpFoldResult> mixedStrides = slicingOp.getMixedStrides();
  auto src = slicingOp.getSource();
  auto res = slicingOp.getResult();
  auto srcShape = utils::getShape(src.getType());
  auto resShape = utils::getShape(res.getType());
  int64_t lastStride = 1;
  rewriter.setInsertionPoint(slicingOp);
  // Convert fullExpandedRef to constants where possible
  SmallVector<int64_t> constantFullExpandedRef =
      convertToConstantValues(expandedRef);

  Value superview;
  if (auto insertOp = dyn_cast<InsertSliceOp>(slicingOp.getOperation())) {
    superview = insertOp.getDest();
  } else if (auto extractOp =
                 dyn_cast<ExtractSliceOp>(slicingOp.getOperation())) {
    superview = extractOp.getSource();
  } else if (auto subviewOp =
                 dyn_cast<memref::SubViewOp>(slicingOp.getOperation())) {
    superview = subviewOp.getSource();
  } else {
    llvm_unreachable("Matcher is neither insert or extract");
  }
  auto droppedDims = slicingOp.getDroppedDims();

  // rebuild no-drop-dims version reassociation and subview
  SmallVector<ReassociationIndices> fullReasso;
  SmallVector<ReassociationIndices> subReasso;
  // if given subveiw reassociation, return superview reassociation, else return
  // subview reassociation
  if (isSubview) {
    fullReasso = getFullReassociation(reassociation, droppedDims);
    subReasso = llvm::to_vector(reassociation);
    result.setReassociaion(fullReasso);
  } else {
    fullReasso.assign(reassociation.begin(), reassociation.end());
    subReasso = getSubReassociation(reassociation, droppedDims);
    result.setReassociaion(subReasso);
  }
  LDBG("full reassociation: " << to_string(fullReasso));
  LDBG("sub reassociation: " << to_string(subReasso));
  auto rank = fullReasso.size();

  utils::DimensionShape subviewShape;
  utils::DimensionShape superviewShape;
  if (isInsert) {
    superviewShape = resShape;
    subviewShape = srcShape;
  } else {
    superviewShape = srcShape;
    subviewShape = resShape;
  }
  subviewShape = getUndroppedSubviewShape(subviewShape, droppedDims);
  LDBG("superview shape: " << to_string(superviewShape));
  LDBG("subview shape: " << to_string(subviewShape));

  // Process each dimension
  for (unsigned i = 0; i < rank; i++) {
    if (superviewShape[i] == subviewShape[i] &&
        !ShapedType::isDynamic(superviewShape[i])) {
      LDBG("No Mutation " << i);
      // Handle dimensions with no mutation
      handleNoMutation(rewriter, expandedRef, fullReasso[i], result);
    } else {
      LDBG("Has Mutation " << i);

      // If its %A = ... -> %B = Collapse -> %C = extract
      // fullExpandRef is %A
      // %B is a reshape of the original tensor %A
      // %C is a subview of %B

      // Otherwise if its %A = ... -> %C = extract -> %B = expand
      // fullExpandRef of %A is unknown
      // %C is a subview of %A
      // %B is the reshaped version of the subview
      // We don't know the fully expanded original tensor %A

      //  %ex = tensor.extract_slice %2[%9, %6, 0] [1, %7, 16] [1, 1, 1] ->
      //  tensor<24x32x16xf32> to tensor<1x?x16xf32> %ep = tensor.expand_shape
      //  %ex
      //  [[0], [1], [2, 3]] output_shape [1, %7, 16, 1]

      //  %ep = tensor.expand_shape %2 [[0], [1], [2, 3]] output_shape [24, 32,
      //  16, 1] %ex = tensor.extract_slice %ep [%9, %6, 0, 0] [1, %7, 16, 1]
      //  [1, 1, 1, 1] -> tensor<24x32x16x1xf32> to tensor<1x?x16x1xf32>

      //  %col = tensor.collapse [[0, 1], [2], [3, 4, 5]] -> <4x6x5x2x2x3> to
      //  <24x5x12> %ex = tensor.extract_slice %col[0, 0, 0] [24, 3, 12] [1, 1,
      //  1]
      //  -> tensor<24x5x12> to tensor<24x3x12xf32>
      // Need to count how many units, fallback to non hyperrectangle for unit
      // cases
      auto unitCount = llvm::count_if(
          fullReasso[i],
          [&constantFullExpandedRef](const auto &reassoc) -> bool {
            return constantFullExpandedRef[reassoc] == 1;
          });
      bool allowHyperrectangle = true;
      if (static_cast<uint32_t>(unitCount + 1) >= fullReasso[i].size()) {
        allowHyperrectangle = false;
      }
      LDBG("Hyperrectangle case");
      if (auto subview = dyn_cast<memref::SubViewOp>(*slicingOp)) {
        auto srcMemRefType = dyn_cast<MemRefType>(src.getType());
        auto srcStride = getConstantStrides(srcMemRefType);
        assert(srcStride.has_value() && "srcStride must be present");
        lastStride = srcStride.value()[i];
      }
      if (allowHyperrectangle) {
        // auto superviewShape = isInsert ? resShape[i] : srcShape[i];
        // auto superviewShapei = superviewShape[i];
        auto isHyperRectangle = checkHyperRectangle(
            rewriter, fullReasso[i], constantFullExpandedRef, isSubview, result,
            superviewShape[i], mixedOffsets[i], mixedSizes[i], mixedStrides[i],
            lastStride);
        if (succeeded(isHyperRectangle))
          continue;
      }

      // Fall back to non-hyperrectangle case
      if (failed(handleMutation(rewriter, constantFullExpandedRef,
                                fullReasso[i], mixedOffsets[i], mixedSizes[i],
                                mixedStrides[i], superview, i, result))) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult
getSubviewModifyingOp(PatternRewriter &rewriter, memref::SubViewOp slicingOp,
                      ArrayRef<ReassociationIndices> reassociation,
                      ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                      SmallVector<OpFoldResult> &newMixedOffsets,
                      SmallVector<OpFoldResult> &newMixedSizes,
                      SmallVector<OpFoldResult> &newMixedStrides,
                      SmallVector<OpFoldResult> &expandOutputShape) {
  SliceModifyingOpResult result;
  LogicalResult res = getSliceModifyingOp(rewriter, slicingOp, reassociation,
                                          expandedRef, isSubview, result);
  if (res.failed())
    return res;

  // Copy results to output parameters
  newMixedOffsets = llvm::to_vector(result.getMixedOffsets());
  newMixedSizes = llvm::to_vector(result.getMixedSizes());
  newMixedStrides = llvm::to_vector(result.getMixedStrides());
  // Only need superview
  expandOutputShape = llvm::to_vector(result.getSuperviewOutputShape());

  return success();
}

LogicalResult
getExtractSliceModifyingOp(PatternRewriter &rewriter, ExtractSliceOp slicingOp,
                           ArrayRef<ReassociationIndices> reassociation,
                           ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                           SmallVector<OpFoldResult> &newMixedOffsets,
                           SmallVector<OpFoldResult> &newMixedSizes,
                           SmallVector<OpFoldResult> &newMixedStrides,
                           SmallVector<OpFoldResult> &expandOutputShape,
                           SmallVector<ReassociationIndices> &newReasso) {
  SliceModifyingOpResult result;
  LogicalResult res = getSliceModifyingOp(rewriter, slicingOp, reassociation,
                                          expandedRef, isSubview, result);
  if (res.failed())
    return res;

  // Copy results to output parameters
  newMixedOffsets = llvm::to_vector(result.getMixedOffsets());
  newMixedSizes = llvm::to_vector(result.getMixedSizes());
  newMixedStrides = llvm::to_vector(result.getMixedStrides());
  // Only need superview
  expandOutputShape = llvm::to_vector(result.getSuperviewOutputShape());
  newReasso = llvm::to_vector(result.getReassociation());

  return success();
}

LogicalResult
getInsertSliceModifyingOp(PatternRewriter &rewriter, InsertSliceOp slicingOp,
                          ArrayRef<ReassociationIndices> reassociation,
                          ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                          SmallVector<OpFoldResult> &newMixedOffsets,
                          SmallVector<OpFoldResult> &newMixedSizes,
                          SmallVector<OpFoldResult> &newMixedStrides,
                          SmallVector<OpFoldResult> &expandSrcOutputShape,
                          SmallVector<OpFoldResult> &expandDestOutputShape,
                          SmallVector<ReassociationIndices> &newReasso) {
  SliceModifyingOpResult result;
  LogicalResult res = getSliceModifyingOp(rewriter, slicingOp, reassociation,
                                          expandedRef, isSubview, result);
  if (res.failed())
    return res;

  // Copy results to output parameters
  newMixedOffsets = llvm::to_vector(result.getMixedOffsets());
  newMixedSizes = llvm::to_vector(result.getMixedSizes());
  newMixedStrides = llvm::to_vector(result.getMixedStrides());
  expandSrcOutputShape = llvm::to_vector(result.getSubviewOutputShape());
  expandDestOutputShape = llvm::to_vector(result.getSuperviewOutputShape());
  newReasso = llvm::to_vector(result.getReassociation());

  return success();
}

SmallVector<OpFoldResult> getMixedSizesOrOutputShape(PatternRewriter &rewriter,
                                                     Value val) {
  auto *op = val.getDefiningOp();
  auto loc = val.getLoc();
  if (auto hasSizeOp = dyn_cast_or_null<tensor::ExpandShapeOp>(op)) {
    SmallVector<OpFoldResult> outputShape(
        getMixedValues(hasSizeOp.getStaticOutputShape(),
                       hasSizeOp.getOutputShape(), rewriter));
    return outputShape;
  }
  if (auto hasSizeOp = dyn_cast_or_null<tensor::ExtractSliceOp>(op)) {
    return hasSizeOp.getMixedSizes();
  }
  if (auto hasSizeOp = dyn_cast_or_null<memref::SubViewOp>(op)) {
    return hasSizeOp.getMixedSizes();
  }
  // Consider returning reify?
  if (isa<RankedTensorType>(val.getType())) {
    auto valMixed = tensor::getMixedSizes(rewriter, loc, val);
    return valMixed;
  } else {
    auto valMixed = memref::getMixedSizes(rewriter, loc, val);
    return valMixed;
  }
}

void updateHFusionReduceWithIndexDim(
    PatternRewriter &rewriter, Operation *reduceWithIndexOp,
    const SmallVector<int64_t> &newDimensions) {
  rewriter.modifyOpInPlace(reduceWithIndexOp, [&]() {
    // assuming one region
    Region &region = reduceWithIndexOp->getRegion(0);
    // assuming one block inside the region
    Block &block = *(region.begin());
    // assuming one IndexOp
    // may not need reference (linalg::IndexOp&) here because linalg::IndexOp is
    // a pointer wrapper
    linalg::IndexOp indexOp = *(block.getOps<linalg::IndexOp>().begin());
    // currently hfusion::ReduceWithIndexOp only supports single reduction
    // dimension; if PropagateCollapseDown generates multi-reduction-dimension
    // cases, the following assertion will catch that
    assert(newDimensions.size() == 1);
    indexOp.setDim(newDimensions[0]);
  });
  // for robustness
  assert(
      succeeded(cast<hfusion::ReduceWithIndexOp>(reduceWithIndexOp).verify()));
}

bool isUnitDimReshape(ArrayRef<int64_t> expandedShape,
                      ArrayRef<mlir::ReassociationIndices> reassociation) {
  for (const auto &group : reassociation) {
    int nonUnitDimCount = 0;
    for (int64_t dimIdx : group) {
      if (expandedShape[dimIdx] != 1) {
        nonUnitDimCount++;
      }
    }
    if (nonUnitDimCount > 1) {
      return false;
    }
  }
  return true;
}

void createTransposedReassoc(
    SmallVector<ReassociationIndices, 4> &oldReassociation,
    ArrayRef<int64_t> expandedShape, ArrayRef<int64_t> permutation,
    SmallVector<int64_t, 4> &newExpandedShape,
    SmallVector<ReassociationIndices, 4> &newReassociation) {
  // Calculate tranposed reassociation indices.
  SmallVector<ReassociationIndices, 4> transposedReassociation;
  auto rank = oldReassociation.size();
  LDBG("rank is " << rank);
  for (size_t i = 0; i < rank; i++) {
    assert(permutation[i] >= 0 && static_cast<size_t>(permutation[i]) < rank);
    const ReassociationIndices &deepCopy = oldReassociation[permutation[i]];
    transposedReassociation.push_back(deepCopy);
  }
  LDBG("old reassociation " << to_string(oldReassociation));
  // flat tranposed reassociation for mapping index
  SmallVector<int64_t, 4> indexRemap;
  for (const auto &vec : transposedReassociation)
    for (auto i : vec)
      indexRemap.push_back(i);
  LDBG("index remap " << to_string(indexRemap));
  // Create new output shape
  for (size_t i : indexRemap)
    newExpandedShape.push_back(expandedShape[i]);
  LDBG("newExpandedShape " << to_string(newExpandedShape));

  // Create new reassociation
  int64_t index = 0;
  for (const auto &vec : transposedReassociation) {
    ReassociationIndices newIndices;
    for (size_t i = 0; i < vec.size(); i++) {
      newIndices.push_back(index++);
    }
    newReassociation.push_back(newIndices);
  }
}

SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> res(permutation.size());
  assert(isPermutationVector(permutation) && "should be permutation");
  for (size_t i = 0; i < permutation.size(); ++i) {
    res[permutation[i]] = static_cast<int>(i);
  }
  return res;
}

void createNewPermutation(size_t rank, ArrayRef<int64_t> permutation,
                          SmallVector<ReassociationIndices, 4> &reassociation,
                          SmallVector<int64_t, 4> &newPermutation) {
  for (size_t i = 0; i < rank; i++) {
    assert(static_cast<size_t>(permutation[i]) < reassociation.size());
    ReassociationIndices deepCopy = reassociation[permutation[i]];
    for (size_t j : deepCopy)
      newPermutation.push_back(j);
  }
}

bool isNonUnitExpandOrEmptyReassoc(
    ArrayRef<int64_t> expandedShape,
    ArrayRef<ReassociationIndices> reassociation) {
  LDBG("expanded shape " << to_string(expandedShape));
  if (reassociation.empty()) {
    return true;
  }
  if (llvm::all_of(expandedShape, [](int64_t val) { return val == 1; })) {
    return true;
  }
  for (auto &reassoc : reassociation) {
    int nonUnitShape =
        llvm::count_if(reassoc, [expandedShape](int64_t el) -> int {
          return expandedShape[el] != 1 ? 1 : 0;
        });
    if (nonUnitShape > 1)
      return true;
  }
  return false;
}

} // namespace reshape_utils
} // namespace tensor

Operation *createNewExpandOpFromExpandOp(tensor::ExpandShapeOp expandOp,
                                         PatternRewriter &rewriter,
                                         Location loc, Value operand) {
  auto reassociation = expandOp.getReassociationIndices();
  auto currentShape = utils::getShape(expandOp.getResult().getType());
  auto resultType =
      RankedTensorType::get(currentShape, getElementTypeOrSelf(operand));
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, operand,
                                                reassociation);
}

Operation *createNewExpandOpFromCollapseOp(Operation *collapseOp,
                                           PatternRewriter &rewriter,
                                           Location loc, Value operand) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(operand);
  // Extract common properties regardless of collapse op type
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<int64_t> currentShape;
  if (auto memrefCollapse = dyn_cast<memref::CollapseShapeOp>(collapseOp)) {
    reassociation = memrefCollapse.getReassociationIndices();
    currentShape = utils::getShape(memrefCollapse.getSrc().getType());
  } else if (auto tensorCollapse =
                 dyn_cast<tensor::CollapseShapeOp>(collapseOp)) {
    reassociation = tensorCollapse.getReassociationIndices();
    currentShape = utils::getShape(tensorCollapse.getSrc().getType());
  } else {
    llvm::report_fatal_error(
        "Expected memref::CollapseShapeOp or tensor::CollapseShapeOp");
  }
  // Create expand op based on operand type
  if (auto operandType = dyn_cast<MemRefType>(operand.getType())) {
    auto resultType = memref::ExpandShapeOp::computeExpandedType(
        operandType, currentShape, reassociation);
    if (failed(resultType))
      llvm::report_fatal_error("cannot create new expand from this collapse");
    return rewriter.create<memref::ExpandShapeOp>(loc, resultType.value(),
                                                  operand, reassociation);
  }
  auto resultType =
      RankedTensorType::get(currentShape, getElementTypeOrSelf(operand));
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, operand,
                                                reassociation);
}

/// @brief Attempts to expand an operand to match a target rank by creating an
/// expand operation.
///
/// This function checks if the given operand's rank matches the target rank. If
/// it does, it creates a new expand operation from the provided collapse
/// operation to expand the operand. If the ranks don't match (e.g., scalar
/// element-wise cases), the original operand is returned unchanged.
///
/// @param collapseOp The `memref::CollapseShapeOp` used as a template for
/// creating the expand operation
/// @param rewriter The pattern rewriter used to create new operations
/// @param operand The value to potentially expand
/// @param targetRank The desired rank for the expanded operand
///
/// @return The expanded operand if ranks match, otherwise the original operand
/// unchanged
Value tryExpandOperand(Operation *collapseOp, PatternRewriter &rewriter,
                       Value operand, int64_t targetRank) {
  auto shapeRank = utils::getShapeRank(operand);
  // Only expand if ranks match (skip scalar elemwise cases)
  if (!shapeRank.has_value() ||
      static_cast<int64_t>(*shapeRank) != targetRank) {
    LLVM_DEBUG(llvm::dbgs() << "Can't expand inequal rank " << shapeRank
                            << " : " << targetRank << "\n");
    return operand;
  }
  rewriter.setInsertionPointAfterValue(operand);
  Operation *expandedOp;
  if (auto memrefCollapse = dyn_cast<memref::CollapseShapeOp>(collapseOp)) {
    expandedOp = createNewExpandOpFromCollapseOp(memrefCollapse, rewriter,
                                                 operand.getLoc(), operand);
  } else if (auto tensorCollapse =
                 dyn_cast<tensor::CollapseShapeOp>(collapseOp)) {
    expandedOp = createNewExpandOpFromCollapseOp(tensorCollapse, rewriter,
                                                 operand.getLoc(), operand);
  } else {
    return operand;
  }
  return expandedOp->getResult(0);
}

/// @brief Transforms all operands of a user operation by potentially expanding
/// them to a target rank.
///
/// This function iterates through all operands of the given user operation and
/// attempts to expand each one to match the specified target rank using
/// `tryExpandOperand()`. Operands that don't match the target rank are left
/// unchanged.
///
/// @param collapseOp The `memref::CollapseShapeOp` used as a template for
/// creating expand operations
/// @param rewriter The pattern rewriter used to create new operations
/// @param userOp The operation whose operands should be processed
/// @param targetRank The desired rank for expanded operands
///
/// @return A vector containing the transformed operands (expanded or original)
SmallVector<Value> getNewOperands(Operation *collapseOp,
                                  PatternRewriter &rewriter, Operation *userOp,
                                  int64_t targetRank) {
  SmallVector<Value> newOperands;
  for (Value operand : userOp->getOperands()) {
    Value newOperand =
        tryExpandOperand(collapseOp, rewriter, operand, targetRank);
    newOperands.push_back(newOperand);
  }
  return newOperands;
}
} // namespace mlir
