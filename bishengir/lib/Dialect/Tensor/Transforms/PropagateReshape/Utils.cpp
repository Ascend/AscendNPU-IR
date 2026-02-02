//===- Utils.cpp ----------------------------------------------------------===//
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

static bool adjustSubviewExpansion(int64_t totalSize, int64_t totalSrc,
                                   MutableArrayRef<int64_t> slicedRef) {
  bool validSize = true;
  LDBG("Checking: " << to_string(slicedRef));
  for (auto it = slicedRef.rbegin(); it != slicedRef.rend(); it++) {
    if (totalSrc % (*it) == 0) {
      // Divide the dimension, this is fully covered
      totalSrc /= (*it);
      totalSize /= (*it);
      continue;
    }
    totalSize /= (*it);
    if (totalSrc % totalSize != 0) {
      validSize = false;
    } else {
      *it = totalSrc / totalSize;
      totalSrc = 1;
    }
    break;
  }
  validSize &= (totalSize == 1);
  slicedRef.front() *= totalSrc;
  LDBG("Checking: " << to_string(slicedRef) << " " << totalSrc << " "
                    << totalSize);
  return validSize;
}

// Helper function to convert OpFoldResult to constants where possible
static SmallVector<int64_t>
convertToConstantValues(ArrayRef<OpFoldResult> values) {
  SmallVector<int64_t> constantValues;
  for (auto val : values) {
    constantValues.push_back(
        getConstantIntValue(val).value_or(ShapedType::kDynamic));
  }
  return constantValues;
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
  std::optional<Hyperrectangle> hyperrectangle = getHyperrectangleFromArray(
      superviewSize, staticOffset, staticSize, staticStride, slicedRef);
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
  for (auto idx : reassociationIndices) {
    LDBG("Iterating reassociation " << idx);
    LDBG(constantFullExpandedRef[idx]);
    if (constantFullExpandedRef[idx] != 1) {
      LDBG("find mutation here");
      result.append(mixedOffset, mixedSize, mixedStride, mixedSize,
                    tensor::getMixedSize(rewriter, superview.getLoc(),
                                         superview, dimensionIndex));
      dimPushed++;
    } else {
      LDBG("find normal here");
      result.append(
          rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(1));
    }
  }

  // If no dimension was pushed, use the last one as the mutation point
  if (dimPushed == 0) {
    LDBG("Dimension pushed is empty");
    dimPushed++;
    // Replace the last entries with the mutation values
    result.replaceBack(mixedOffset, mixedSize, mixedStride, mixedSize,
                       tensor::getMixedSize(rewriter, superview.getLoc(),
                                            superview, dimensionIndex));
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
                    OpFoldResult mixedStride) {
  int64_t totalSize = 1;
  SmallVector<int64_t> slicedRef;
  for (long j : reassociation) {
    slicedRef.emplace_back(constantFullExpandedRef[j]);
    if (ShapedType::isDynamic(slicedRef.back())) {
      LDBG("[failed] Dynamic in the sliced ref");
      return failure();
    }
    totalSize *= slicedRef.back();
  }
  if (isSubview) {
    // Compute totalSize and check if all dimensions are static
    bool adjusted =
        adjustSubviewExpansion(totalSize, superviewShape, slicedRef);
    if (!adjusted) {
      LDBG("[failed] Hyperrectangle case can't be adjusted");
      return failure();
    }
  }
  // Handle dimensions with mutation
  auto staticOffset = getConstantIntValue(mixedOffset);
  auto staticSize = getConstantIntValue(mixedSize);
  auto staticStride = getConstantIntValue(mixedStride);
  // Try to handle as hyperrectangle case if static values are available
  if (staticOffset && staticSize && staticStride) {
    if (succeeded(handleHyperrectangleCase(
            rewriter, slicedRef, superviewShape, staticOffset.value(),
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

  auto rank = reassociation.size();
  SmallVector<OpFoldResult> mixedOffsets = slicingOp.getMixedOffsets();
  SmallVector<OpFoldResult> mixedSizes = slicingOp.getMixedSizes();
  SmallVector<OpFoldResult> mixedStrides = slicingOp.getMixedStrides();
  auto src = slicingOp.getSource();
  auto res = slicingOp.getResult();
  auto srcShape = utils::getShape(src.getType());
  auto resShape = utils::getShape(res.getType());
  rewriter.setInsertionPoint(slicingOp);

  // Convert fullExpandedRef to constants where possible
  SmallVector<int64_t> constantFullExpandedRef =
      convertToConstantValues(expandedRef);

  // Process each dimension
  for (unsigned i = 0; i < rank; i++) {
    if (srcShape[i] == resShape[i] && !ShapedType::isDynamic(srcShape[i])) {
      LDBG("No Mutation " << i);
      // Handle dimensions with no mutation
      handleNoMutation(rewriter, expandedRef, reassociation[i], result);
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
      LDBG("Hyperrectangle case");
      auto superviewShape = isInsert ? resShape[i] : srcShape[i];
      auto isHyperRectangle = checkHyperRectangle(
          rewriter, reassociation[i], constantFullExpandedRef, isSubview,
          result, superviewShape, mixedOffsets[i], mixedSizes[i],
          mixedStrides[i]);
      if (succeeded(isHyperRectangle))
        continue;
      Value superview;
      if (auto insertOp = dyn_cast<InsertSliceOp>(slicingOp.getOperation())) {
        superview = insertOp.getDest();
      } else if (auto extractOp =
                     dyn_cast<ExtractSliceOp>(slicingOp.getOperation())) {
        superview = extractOp.getSource();
      } else {
        llvm_unreachable("Matcher is neither insert or extract");
      }
      // Fall back to non-hyperrectangle case
      if (failed(handleMutation(rewriter, constantFullExpandedRef,
                                reassociation[i], mixedOffsets[i],
                                mixedSizes[i], mixedStrides[i], superview, i,
                                result))) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult
getExtractSliceModifyingOp(PatternRewriter &rewriter, ExtractSliceOp slicingOp,
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
getInsertSliceModifyingOp(PatternRewriter &rewriter, InsertSliceOp slicingOp,
                          ArrayRef<ReassociationIndices> reassociation,
                          ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                          SmallVector<OpFoldResult> &newMixedOffsets,
                          SmallVector<OpFoldResult> &newMixedSizes,
                          SmallVector<OpFoldResult> &newMixedStrides,
                          SmallVector<OpFoldResult> &expandSrcOutputShape,
                          SmallVector<OpFoldResult> &expandDestOutputShape) {
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
  auto valMixed = tensor::getMixedSizes(rewriter, loc, val);
  return valMixed;
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
    assert(newDimensions.size() != 0);
    indexOp.setDim(newDimensions[0]);
  });
  // for robustness
  assert(
      succeeded(cast<hfusion::ReduceWithIndexOp>(reduceWithIndexOp).verify()));
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
    for (size_t _ = 0; _ < vec.size(); ++_)
      newIndices.push_back(index++);
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
} // namespace mlir
