//===- PropagateExpandUp.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Propagate expand up will try to bubble up the expandshape operation to the
//  top
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateExpandUp.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <type_traits>

#define DEBUG_TYPE "propagate-reshape-expand-up"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir {
namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

namespace {
// %c = elemwise %a, %b
// %d = expand %c
// Folds into
//
// |
// v
//
//
//
// %newA = expand %a
// %newB = expand %b
// %newC = elemwise in(%newA), outs(%newB)
// %d = expand %newC
// %old_c = collapse %newC (replacing %c)
// %old_b = collapse %newB (replacing %b)
//
//
// %newA = expand %a
// %newB = expand %b
// %newC = elemwise in(%newA), outs(%newB)
// %old_c = collapse %newC (replacing %c)
// %old_b = collapse %newB (replacing %b)
LogicalResult handleElementwiseOp(tensor::ExpandShapeOp expandOp,
                                  PatternRewriter &rewriter,
                                  Operation *definingOp) {
  rewriter.setInsertionPointAfter(definingOp);
  auto loc = expandOp.getLoc();
  SmallVector<Value, 4> newOperands;

  LLVM_DEBUG(llvm::dbgs() << "Trying " << expandOp << " to an elemwise "
                          << *definingOp << "\n";);
  auto sourceRank = utils::getShapeRank(expandOp.getSrc());
  for (Value operand : definingOp->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand);
    // Check in case its scalar elemwise
    if (shapeRank.has_value() && shapeRank == sourceRank) {
      Operation *newReshapeOp =
          createNewExpandOpFromExpandOp(expandOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newReshapeOp << "\n";);
      newOperands.push_back(newReshapeOp->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't collapse inequal rank " << shapeRank
                              << " : " << sourceRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  auto oldShape = expandOp.getSrc().getType();
  auto oldDpsInits =
      cast<DestinationStyleOpInterface>(definingOp).getDpsInits();
  // Must do this or else it would be replaced by update Defining op
  updateDefiningOp(definingOp, rewriter, newOperands);
  collapseAndReplace(rewriter, expandOp, oldShape, definingOp->getResult(0),
                     definingOp);
  collapseAndReplace(rewriter, expandOp, oldShape, *oldDpsInits.begin(),
                     definingOp);
  rewriter.replaceOp(expandOp, definingOp->getResult(0));
  return success();
}

// %b = hfusion.flip %a, axis
// %c = expand %b
// Folds into
//
// |
// v
//
// %newA = expand %a
// %newB = hfusion.flip %newA, newAxis
// %newC = collapse %newB
// %newB replace %c
// %newC replace %b
LogicalResult handleFlipOp(tensor::ExpandShapeOp expandOp,
                           PatternRewriter &rewriter, Operation *definingOp) {
  auto flipOp = cast<hfusion::FlipOp>(definingOp);

  // Step 1: Acquire info
  // Get original flipAxis and reassociation indices
  auto flipAxis = flipOp.getFlipAxis();
  auto reassociation = expandOp.getReassociationIndices();
  auto expandInShape = expandOp.getSrcType().getShape();
  if (flipAxis >= reassociation.size() || reassociation[flipAxis].empty()) {
    return expandOp.emitOpError("invalid flipAxis for reassociation");
  }
  // if flipAxis is expanded, expandUp forbidden
  if (reassociation[flipAxis].size() > 1U) {
    return failure();
  }
  auto expandOutShape = expandOp.getResultType().getShape();
  // Step 2: Calculate the new flip axis
  uint64_t newFlipAxis = static_cast<uint64_t>(reassociation[flipAxis][0]);

  LLVM_DEBUG(llvm::dbgs() << "Try to lift " << expandOp << " before a flip "
                          << *definingOp << ";\n Flip Axis: " << flipAxis
                          << "\n";);

  // Step 3: rewriter
  // Add New Expand operation
  auto src = flipOp->getOperands()[0];
  rewriter.setInsertionPointAfterValue(src);
  auto resTy = RankedTensorType::get(expandOutShape, getElementTypeOrSelf(src));
  tensor::ExpandShapeOp newSrcOp = rewriter.create<tensor::ExpandShapeOp>(
      src.getLoc(), resTy, src, reassociation);
  // Add new flip operation
  rewriter.setInsertionPointAfter(flipOp);
  auto newFlipOp = rewriter.create<hfusion::FlipOp>(flipOp->getLoc(), resTy,
                                                    newSrcOp, newFlipAxis);
  // Add new collapse operation (old flip replaced by collapse)
  rewriter.setInsertionPointAfter(expandOp);
  resTy = RankedTensorType::get(expandInShape, getElementTypeOrSelf(expandOp));
  auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
      flipOp.getLoc(), resTy, newFlipOp->getResult(0), reassociation);
  rewriter.replaceAllUsesExcept(flipOp->getResult(0), collapseOp->getResult(0),
                                expandOp);
  // old expand replaced by new transpose
  rewriter.replaceOp(expandOp, newFlipOp);
  rewriter.eraseOp(flipOp);
  return success();
}

// %c = transpose %a into %b
// %d = expand %c
// Folds into
//
// |
// v
//
// %newA = expand %a
// %newB = expand %b
// %newC = transpose %newA into %newB
// %newD = collapse %newC
// %newC replace %d
// %newD replace %c
template <class T>
LogicalResult handleTransposeOp(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter,
                                Operation *definingOp) {
  auto expandOutShape = expandOp.getResultType().getShape();
  auto expandInShape = expandOp.getSrcType().getShape();
  auto reassociation = expandOp.getReassociationIndices();

  // operands of transposeOp should be 2
  linalg::TransposeOp transposeOp = cast<T>(definingOp);
  auto permutation = transposeOp.getPermutation();
  auto transposeRank =
      utils::getShapeRank(transposeOp->getResult(0)).value_or(0);

  // Create new Expand input shape
  SmallVector<int64_t, 4> newExpandInputShape;
  // Create new Expand reassociation
  SmallVector<ReassociationIndices, 4> newExpandReassociation;
  reshape_utils::createTransposedReassoc(
      reassociation, expandOutShape, getInversePermutation(permutation),
      newExpandInputShape, newExpandReassociation);
  // Create new tranpose permutation
  SmallVector<int64_t, 4> newPermutation;
  reshape_utils::createNewPermutation(transposeRank, permutation,
                                      newExpandReassociation, newPermutation);

  LLVM_DEBUG(llvm::dbgs() << "Try to lift " << expandOp << " before "
                          << *definingOp << "\n";);

  auto srcOp = transposeOp.getDpsInputOperand(0)->get();
  auto dstOp = transposeOp.getDpsInitOperand(0)->get();

  // Add New Expand operations
  rewriter.setInsertionPointAfterValue(srcOp);
  auto resTy =
      RankedTensorType::get(newExpandInputShape, getElementTypeOrSelf(srcOp));
  tensor::ExpandShapeOp newSrcOp = rewriter.create<tensor::ExpandShapeOp>(
      srcOp.getLoc(), resTy, srcOp, newExpandReassociation);

  rewriter.setInsertionPointAfterValue(dstOp);
  resTy = RankedTensorType::get(expandOutShape, getElementTypeOrSelf(dstOp));
  tensor::ExpandShapeOp newDstOp = rewriter.create<tensor::ExpandShapeOp>(
      dstOp.getLoc(), resTy, dstOp, reassociation);

  // Add new transpose operations
  rewriter.setInsertionPointAfter(transposeOp);
  Operation *newTransposeOp;
  if (isa<hivm::VTransposeOp>(definingOp)) {
    newTransposeOp = rewriter.create<hivm::VTransposeOp>(
        transposeOp->getLoc(), newDstOp->getResultTypes(), newSrcOp.getResult(),
        newDstOp.getResult(), rewriter.getDenseI64ArrayAttr(newPermutation));
  } else if (isa<linalg::TransposeOp>(definingOp)) {
    newTransposeOp = rewriter.create<linalg::TransposeOp>(
        transposeOp->getLoc(), newSrcOp, newDstOp, newPermutation);
  } else {
    llvm_unreachable("Transpose op unrecognized");
  }

  // Add new collapse operation (old transpose replaced by collapse)
  rewriter.setInsertionPointAfter(expandOp);
  resTy = RankedTensorType::get(expandInShape, getElementTypeOrSelf(expandOp));
  auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
      transposeOp.getLoc(), resTy, newTransposeOp->getResult(0), reassociation);
  rewriter.replaceAllUsesExcept(transposeOp->getResult(0),
                                collapseOp->getResult(0), expandOp);

  // old expand replaced by new transpose
  rewriter.replaceOp(expandOp, newTransposeOp);
  rewriter.eraseOp(transposeOp);
  return success();
}
/*
%a = deinterleave %in
%b = expand %a

|
v

%new_in = expand %in
%new_b = deinterleave %new_in
%new_a = collapse %new_b

note that [de]interleave edit shape by halving (/=2) the last dimention and in
order to propagate expand up, we should double the last dimention (*=2) to
cancel the halving.
%
*/
LogicalResult handleDeinterleaveOp(tensor::ExpandShapeOp expandOp,
                                   PatternRewriter &rewriter,
                                   Operation *definingOp) {
  auto expandOutShape = expandOp.getResultType().getShape();
  auto expandInShape = expandOp.getSrcType().getShape();
  auto reassociation = expandOp.getReassociationIndices();

  // dynamic dimentions are not supported yet
  if (ShapedType::isDynamic(expandOutShape.back()) ||
      ShapedType::isDynamic(expandInShape.back())) {
    return failure();
  }

  PatternRewriter::InsertionGuard guard(rewriter);
  hfusion::DeinterleaveOp deinterleaveOp =
      cast<hfusion::DeinterleaveOp>(definingOp);

  // e.g = deinterleave 12xf32 -> 6xf32
  // src = [A,B,A,B,A,B,A,B,A,B,A,B]
  // [A,A,A,A,A,A] (channel 0)
  // -------------------
  // [B,B,B,B,B,B] (channel 1)

  // Expand 6 ->YxZxf32
  // [A,A,A]
  // [A,A,A]
  // or
  // [A,A]
  // [A,A]
  // [A,A]
  // would change the deinterleave to
  // [Yx2Z]

  // [A,B,A,B,A,B]
  // [A,B,A,B,A,B]
  // or
  // [A,B,A,B]
  // [A,B,A,B]
  // [A,B,A,B]
  // both tensor will propagate correctly, according to interleave channel num

  SmallVector<int64_t> newExpandOutShape(expandOutShape.begin(),
                                         expandOutShape.end());
  newExpandOutShape.back() *= deinterleaveOp.getDeInterLeaveChannelNum();

  // expand the only input
  auto aOp = deinterleaveOp.getInput();
  rewriter.setInsertionPointAfterValue(aOp);
  auto aResTy =
      RankedTensorType::get(newExpandOutShape, getElementTypeOrSelf(aOp));
  auto newA = rewriter.create<tensor::ExpandShapeOp>(aOp.getLoc(), aResTy, aOp,
                                                     reassociation);

  // deinterleave the expanded input
  rewriter.setInsertionPointAfterValue(newA);
  auto dResTy =
      RankedTensorType::get(expandOutShape, getElementTypeOrSelf(aOp));
  auto newDeinterleaveOp = rewriter.create<hfusion::DeinterleaveOp>(
      newA.getLoc(), dResTy, newA, deinterleaveOp.getChannelIndex());

  for (int i = 0; i < static_cast<int>(deinterleaveOp.getNumResults()); i++) {
    // extra collaspe op in case some user wanted the pre-expanded result
    rewriter.setInsertionPointAfter(newDeinterleaveOp);
    auto cResTy =
        RankedTensorType::get(expandInShape, getElementTypeOrSelf(aOp));
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        newDeinterleaveOp.getLoc(), cResTy, newDeinterleaveOp.getResult(i),
        reassociation);

    rewriter.replaceAllUsesExcept(deinterleaveOp.getResult(i), collapseOp,
                                  expandOp);
    rewriter.replaceAllUsesWith(expandOp, newDeinterleaveOp.getResult(i));
  }

  rewriter.eraseOp(expandOp);
  rewriter.eraseOp(deinterleaveOp);

  return success();
}

/*
%a = interleave %in0 %in1
%b = expand %a

|
v

new_in0 = expand %in0
new_in1 = expand %in1
new_b = interleave %new_in0 %new_in1
new_a = collapse %new_b // incase there is a user of the old %a


note that interleave edit shape by doubling (*=2) the last dimention and in
order to propagate expand up, we should divide the last dimention (/=2) to
cancel the doubling.
*/
LogicalResult handleInterleaveOp(tensor::ExpandShapeOp expandOp,
                                 PatternRewriter &rewriter,
                                 Operation *definingOp) {
  auto expandOutShape = expandOp.getResultType().getShape();
  auto expandInShape = expandOp.getSrcType().getShape();
  auto reassociation = expandOp.getReassociationIndices();

  // dynamic dimentions are not supported yet
  if (ShapedType::isDynamic(expandOutShape.back()) ||
      ShapedType::isDynamic(expandInShape.back())) {
    return failure();
  }

  hfusion::InterleaveOp interleaveOp = cast<hfusion::InterleaveOp>(definingOp);

  // make sure we can propagate the expand without loosing the parity
  // e.g = interleave 6xf32 -> 12xf32
  // [A,A,A,A,A,A] (channel 0)
  // -------------------
  // [B,B,B,B,B,B] (channel 1)
  // [A,B,A,B,A,B,A,B,A,B,A,B] interleaved

  // expand 12xf32 -> 2x6xf32
  // [A,B,A,B,A,B]
  // [A,B,A,B,A,B]
  // Alignable with 2x3xf32

  // expand 12xf32 -> 4x3xf32
  // [A,B,A]
  // [B,A,B]
  // [A,B,A]
  // [B,A,B]
  // Unalignable

  if (expandOutShape.back() % interleaveOp.getInterLeaveChannelNums() != 0) {
    return failure();
  }

  PatternRewriter::InsertionGuard guard(rewriter);

  // setting up the new expand out shape
  SmallVector<int64_t> newExpandOutShape(expandOutShape.begin(),
                                         expandOutShape.end());
  newExpandOutShape.back() /= interleaveOp.getInterLeaveChannelNums();

  // expand each input of the interleave
  SmallVector<Value> newInOps;
  for (int i = 0; i < static_cast<int>(interleaveOp.getNumOperands()); i++) {
    auto inOp = interleaveOp->getOperand(i);
    rewriter.setInsertionPointAfterValue(inOp);
    auto aResTy =
        RankedTensorType::get(newExpandOutShape, getElementTypeOrSelf(inOp));
    auto newInOp = rewriter.create<tensor::ExpandShapeOp>(
        inOp.getLoc(), aResTy, inOp, expandOp.getReassociationIndices());
    newInOps.push_back(newInOp);
  }

  // interleave the new expanded inputs
  rewriter.setInsertionPointAfterValue(interleaveOp);
  auto iResTy =
      RankedTensorType::get(expandOutShape, getElementTypeOrSelf(expandOp));
  auto newIntervalOp = rewriter.create<hfusion::InterleaveOp>(
      interleaveOp.getLoc(), iResTy, ValueRange(newInOps));

  // extra collaspe op in case some user wanted the pre-expanded result
  rewriter.setInsertionPointAfter(newIntervalOp);
  auto cResTy =
      RankedTensorType::get(expandInShape, getElementTypeOrSelf(newIntervalOp));
  auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
      newIntervalOp.getLoc(), cResTy, newIntervalOp, reassociation);

  // order is important
  rewriter.replaceAllUsesExcept(interleaveOp, collapseOp, expandOp);
  rewriter.replaceAllUsesWith(expandOp, newIntervalOp);
  rewriter.eraseOp(expandOp);
  rewriter.eraseOp(interleaveOp);

  return success();
}

LogicalResult handleBitcastOp(tensor::ExpandShapeOp expandOp,
                              PatternRewriter &rewriter,
                              hivm::BitcastOp bitcastOp) {
  rewriter.setInsertionPointAfter(bitcastOp);
  auto loc = expandOp.getLoc();
  SmallVector<Value, 4> newOperands;
  auto bitcastOpResult = bitcastOp.getResult();
  auto bitcastOpResultType = bitcastOpResult.getType();
  LLVM_DEBUG(llvm::dbgs() << "Trying " << expandOp << " to an elemwise "
                          << *bitcastOp << "\n";);
  for (Value operand : bitcastOp->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand).value_or(0);
    auto sourceRank = utils::getShapeRank(expandOp.getSrc()).value_or(0);
    // Check in case its scalar elemwise
    if (shapeRank == sourceRank) {
      Operation *newReshapeOp =
          createNewExpandOpFromExpandOp(expandOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newReshapeOp << "\n";);
      newOperands.push_back(newReshapeOp->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't collapse inequal rank " << shapeRank
                              << " : " << sourceRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  // Must do this or else it would be replaced by update Defining op
  updateDefiningOpNonDst(bitcastOp, rewriter, newOperands,
                         utils::getShape(expandOp.getResult().getType()));
  collapseAndReplace(rewriter, expandOp, bitcastOpResultType, bitcastOpResult,
                     bitcastOp);
  return success();
}

/**********************************/
/** Utils for propagations       **/
/**********************************/

BitVector createBitVector(Value largerDimensionVal,
                          ArrayRef<int64_t> dimensions) {
  BitVector bits(utils::getShapeRank(largerDimensionVal).value_or(0), true);
  for (auto dim : dimensions) {
    bits[dim] = false;
  }
  return bits;
}

template <class LinalgBRTy>
SmallVector<Value, 4> expandPropagateCreateNewOperands(
    PatternRewriter &rewriter, tensor::ExpandShapeOp &expandOp, LinalgBRTy op,
    const SmallVector<int64_t> &newOutputShape,
    MutableArrayRef<ReassociationIndices> newReassociation) {
  auto loc = expandOp.getLoc();
  SmallVector<Value, 4> newOperands;

  LLVM_DEBUG(llvm::dbgs() << "Expand Propagate Create New Operands\n";);
  auto opOperands = op->getOpOperands();
  for (OpOperand &operand : opOperands) {
    Value operandVal = operand.get();

    LLVM_DEBUG(llvm::dbgs() << "Checking " << operandVal << "\n";);
    // Skip scalar like operand
    Operation *newExpandedOperand = nullptr;
    bool isInput = op.isDpsInput(&operand);
    bool isInit = op.isDpsInit(&operand);
    if (isa<RankedTensorType>(operandVal.getType()) && (isInput || isInit)) {
      LLVM_DEBUG(llvm::dbgs()
                     << utils::getShapeRank(operandVal).value_or(0) << "\n";);
      LLVM_DEBUG(llvm::dbgs() << operandVal << "\n";);
      // Depends which is needed to be expanded, is it input or init
      if (isInput) {
        rewriter.setInsertionPointAfterValue(operandVal);
        newExpandedOperand = createNewReshapingOp<tensor::ExpandShapeOp>(
            rewriter, loc, operandVal, newReassociation, newOutputShape);
      } else {
        rewriter.setInsertionPointAfterValue(operandVal);
        newExpandedOperand =
            createNewExpandOpFromExpandOp(expandOp, rewriter, loc, operandVal);
      }
    }

    if (newExpandedOperand != nullptr)
      newOperands.push_back(newExpandedOperand->getResult(0));
    else
      newOperands.push_back(operandVal);
  }
  return newOperands;
}

/**********************************/
/** Broadcast propagations logic **/
/**********************************/

// %oldD = broadcast in(%a) out(%b)
// %d = expand %oldD
//
// Folds into
//
// %newA = expand(%a)
// %newB = expand(%b)
// %d = broadcast(%newA) out(%newB)
// %oldD = collapsed(%d) // for replacing the oldC
// %collapsed_b = collapsed(%newB)
//
// Broadcast <AxBxCxD> -> <AxBxExCxDxF>
// would be marked as      1 1 0 1 1 0
// This bitvector will be used to adjust new Dimensions and new
// Reassociation Indices
// When expanding shape
// <AxBxExCxDxF> -> <AxB1xB2xE1xE2xE3xCxDxF>
// [[0], [1, 2], [3, 4, 5], [6], [7], [8]]
//  1    1       0          1    1    0
//
// [3, 4, 5] and [8] is removed because it's not part of bit vector
// E and F is ignored because its from a broadcasted dimensions
//
// First expand will be transformed into
//
// <AxBxCxD> -> <AxB1xB2xCxD>
// [[0], [1, 2], [6], [7]]
//
// |
// | Renumbering
// v
//
// [[0], [1, 2], [3], [4]]
//
// Which will be the new reassociation for the propagation
//
// <AxB1xB2xCxD> -> <AxB1xB2xE1xE2xE3xCxDxF>
//                   0 0  0  1  1  1  0 0 1
//                   0 1  2  3  4  5  6 7 8
//
//                   We have new dimension is 3, 4, 5, 8
//
// Which is the reassociation indices over the BitVector

void obtainUpDimIncReassocs(const BitVector &broadcastBits,
                            ArrayRef<ReassociationIndices> reassociationIndices,
                            ArrayRef<int64_t> outputShape,
                            ArrayRef<int64_t> dimensions,
                            SmallVector<ReassociationIndices> &newReassociation,
                            SmallVector<int64_t> &newOutputShape,
                            SmallVector<int64_t> &newDimensions) {
  LLVM_DEBUG(llvm::dbgs() << to_string(broadcastBits)
                          << " -- broadcastBits\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reassociationIndices)
                          << " -- reassociationIndices\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(outputShape) << " -- outputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(dimensions) << " -- dimensions\n";);
  for (size_t i = 0; i < broadcastBits.size(); i++) {
    if (!broadcastBits[i])
      continue;
    newReassociation.push_back(reassociationIndices[i]);
    for (auto shapeIndex : reassociationIndices[i]) {
      newOutputShape.push_back(outputShape[shapeIndex]);
    }
  }
  for (auto dim : dimensions)
    for (auto expandDimension : reassociationIndices[dim])
      newDimensions.push_back(expandDimension);

  LLVM_DEBUG(llvm::dbgs() << to_string(newReassociation)
                          << " -- newReassociation\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(newOutputShape)
                          << " -- newOutputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(newDimensions)
                          << " -- newDimensions\n";);
  renumberReassociation(newReassociation);
}

void obtainHIVMUpDimIncReassocs(
    const BitVector &broadcastBits,
    ArrayRef<ReassociationIndices> reassociationIndices,
    ArrayRef<int64_t> outputShape, ArrayRef<int64_t> dimensions,
    SmallVector<int64_t> &newOutputShape, SmallVector<int64_t> &newDimensions) {
  LLVM_DEBUG(llvm::dbgs() << to_string(broadcastBits)
                          << " -- broadcastBits\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reassociationIndices)
                          << " -- reassociationIndices\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(outputShape) << " -- outputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(dimensions) << " -- dimensions\n";);
  for (size_t i = 0; i < broadcastBits.size(); i++) {
    for (auto shapeIndex : reassociationIndices[i]) {
      if (broadcastBits[i])
        newOutputShape.push_back(outputShape[shapeIndex]);
      else
        newOutputShape.push_back(1);
    }
  }
  // This is the same
  for (auto dim : dimensions)
    for (auto expandDimension : reassociationIndices[dim])
      newDimensions.push_back(expandDimension);
  LLVM_DEBUG(llvm::dbgs() << to_string(newOutputShape)
                          << " -- newOutputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(newDimensions)
                          << " -- newDimensions\n";);
}

LogicalResult handleBroadcastOp(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter,
                                Operation *definingOp) {
  auto broadcastOp = cast<linalg::BroadcastOp>(definingOp);
  ArrayRef<int64_t> dimensions = broadcastOp.getDimensions();
  BitVector broadcastBits =
      createBitVector(broadcastOp->getResult(0), dimensions);
  SmallVector<ReassociationIndices> reassociationIndices =
      expandOp.getReassociationIndices();
  SmallVector<ReassociationIndices> newReassociation;
  SmallVector<int64_t> newOutputShape;
  SmallVector<int64_t> newDimensions;

  // Step 2: Get reassocs
  obtainUpDimIncReassocs(broadcastBits, reassociationIndices,
                         utils::getShape(expandOp.getResult().getType()),
                         dimensions, newReassociation, newOutputShape,
                         newDimensions);

  // Step 3: update everything
  auto newOperands = expandPropagateCreateNewOperands(
      rewriter, expandOp, broadcastOp, newOutputShape, newReassociation);

  LLVM_DEBUG(llvm::dbgs() << "Got new Operands: " << newOperands.size()
                          << "\n";);
  updateDimensionalOp(broadcastOp, rewriter, newDimensions);
  auto oldShape = expandOp.getSrc().getType();
  auto oldDpsInits =
      cast<DestinationStyleOpInterface>(definingOp).getDpsInits();
  // Must do this or else it would be replaced by update Defining op
  updateDefiningOp(definingOp, rewriter, newOperands);
  collapseAndReplace(rewriter, expandOp, oldShape, definingOp->getResult(0),
                     definingOp);
  collapseAndReplace(rewriter, expandOp, oldShape, *oldDpsInits.begin(),
                     definingOp);
  rewriter.replaceOp(expandOp, definingOp->getResult(0));
  return success();
}

/**********************************/
/** Reduce propagations logic    **/
/**********************************/

// %oldD = reduce in(%a) out(%b)
// %d = expand %oldD
//
// Folds into
//
// %newA = expand(%a)
// %newB = expand(%b)
// %d = reduce(%newA) out(%newB)
// %oldD = collapsed(%d)
// %collapsed_b = collapsed(%newB)
//
// Reduce   `          <AxBxExCxDxF> -> <AxBxCxD>
// would be marked as   1 1 0 1 1 0
// Old dimension is [2, 5]
// This bitvector will be used to adjust new Dimensions and new
// Reassociation Indices

// When expanding shape
// <AxBxCxD> -> <AxB1xB2xCxD>
// [[0], [1, 2], [3], [4]]
//
// will increase new
//
// <AxBxExCxDxF> -> <AxB1xB2xExCxDxF>
//  1    1       0    1    1    0
// [[0], [1, 2], [0], [3], [4], [0]]
//
// |
// | Renumbering
// v
//
// [[0], [1, 2], [3], [4], [5], [6]]
//
// Then, cloning reduce, changing the input
// <AxB1xB2xExCxDxF> -> <AxB1xB2xCxD>
//  1 1  1  0 1 1 0
//
// We have new dimension is [3, 6], got from the new renumbering
//
// Which is the reassociation indices over the BitVector
//
//

void obtainUpDimDecReassocs(const BitVector &reduceBits,
                            ArrayRef<ReassociationIndices> reassociationIndices,
                            ArrayRef<int64_t> reduceInputShape,
                            ArrayRef<int64_t> outputShape,
                            ArrayRef<int64_t> dimensions,
                            SmallVector<ReassociationIndices> &newReassociation,
                            SmallVector<int64_t> &newOutputShape,
                            SmallVector<int64_t> &newDimensions) {
  int reassocPtr = 0;

  LLVM_DEBUG(llvm::dbgs() << to_string(reduceBits) << " -- reduceBits\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reassociationIndices)
                          << " -- reassociationIndices\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reduceInputShape)
                          << " -- reduceInputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(outputShape) << " -- outputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(dimensions) << " -- dimensions\n";);
  for (size_t i = 0; i < reduceBits.size(); i++) {
    if (!reduceBits[i]) {
      newReassociation.push_back({-1});
      newOutputShape.push_back(reduceInputShape[i]);
    } else {
      newReassociation.push_back(reassociationIndices[reassocPtr]);
      for (auto shapeIndex : reassociationIndices[reassocPtr]) {
        newOutputShape.push_back(outputShape[shapeIndex]);
      }
      reassocPtr++;
    }
  }
  renumberReassociation(newReassociation);
  for (auto dim : dimensions) {
    for (auto expandDimension : newReassociation[dim]) {
      newDimensions.push_back(expandDimension);
    }
  }
  // Handle reducing tensor.empty, we need to adjust according to size of the
  // expand
  if (reassociationIndices.empty()) {
    std::reverse(newOutputShape.begin(), newOutputShape.end());
    std::reverse(newReassociation.front().begin(),
                 newReassociation.front().end());
    for (size_t i = 0; i < outputShape.size(); i++) {
      newReassociation.front().push_back(-1);
      newOutputShape.push_back(1);
    }
    for (auto &dim : newDimensions) {
      dim += static_cast<int64_t>(outputShape.size());
    }
    std::reverse(newOutputShape.begin(), newOutputShape.end());
    std::reverse(newReassociation.front().begin(),
                 newReassociation.front().end());
    renumberReassociation(newReassociation);
  }
  LLDBG(to_string(newOutputShape));
  LLDBG(to_string(newReassociation));
}

void obtainHIVMUpDimDecReassocs(
    const BitVector &reduceBits,
    ArrayRef<ReassociationIndices> reassociationIndices,
    ArrayRef<int64_t> reduceInputShape, ArrayRef<int64_t> outputShape,
    SmallVector<int64_t> &newOutputShape, SmallVector<int64_t> &newDimensions) {
  LLVM_DEBUG(llvm::dbgs() << to_string(reduceBits) << " -- reduceBits\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reassociationIndices)
                          << " -- reassociationIndices\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reduceInputShape)
                          << " -- reduceInputShape\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(outputShape) << " -- outputShape\n";);
  for (size_t i = 0; i < reduceBits.size(); i++) {
    if (!reduceBits[i]) {
      newDimensions.push_back(newOutputShape.size());
      newOutputShape.push_back(reduceInputShape[i]);
      for (size_t j = 1; j < reassociationIndices[i].size(); ++j) {
        newDimensions.push_back(newOutputShape.size());
        newOutputShape.push_back(1);
      }
    } else {
      for (auto shapeIndex : reassociationIndices[i]) {
        newOutputShape.push_back(outputShape[shapeIndex]);
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << to_string(newOutputShape) << "\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(newDimensions) << "\n";);
}

template <typename OpType>
LogicalResult handleReduceLikeOp(tensor::ExpandShapeOp expandOp,
                                 PatternRewriter &rewriter,
                                 Operation *definingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Ok reduce writing\n";);
  auto reduceOp = cast<OpType>(definingOp);

  auto inputsReduce = reduceOp.getInputs();
  LLVM_DEBUG(llvm::dbgs() << "Finding input " << inputsReduce[0] << "\n";);

  ArrayRef<int64_t> dimensions = reduceOp.getDimensions();

  BitVector reduceBits = createBitVector(inputsReduce[0], dimensions);

  SmallVector<ReassociationIndices> reassociationIndices =
      expandOp.getReassociationIndices();
  SmallVector<ReassociationIndices> newReassociation;
  SmallVector<int64_t> newOutputShape; // output: the expanded input of reduce
  SmallVector<int64_t> newDimensions;
  auto expandedShape =
      getMixedSizesOrOutputShape(rewriter, expandOp.getResult());
  auto expandInputRank = expandOp.getSrcType().getRank();
  // skip add unitdim to rank 0 output of reduce
  if (isUnitDimReshape(convertToConstantValues(expandedShape),
                       reassociationIndices) &&
      expandInputRank == 0) {
    return failure();
  }
  // Step 2: obtains dimensions and reassociations
  auto opResult = cast<OpResult>(expandOp.getSrc());
  auto oldReplaceIndex = opResult.getResultNumber();

  obtainUpDimDecReassocs(reduceBits, reassociationIndices,
                         utils::getShape(inputsReduce[0].getType()),
                         utils::getShape(expandOp.getResult().getType()),
                         dimensions, newReassociation, newOutputShape,
                         newDimensions);

  // Step 3: Change everything
  auto oldDpsInits =
      cast<DestinationStyleOpInterface>(definingOp).getDpsInits();
  auto newOperands = expandPropagateCreateNewOperands(
      rewriter, expandOp, reduceOp, newOutputShape, newReassociation);

  auto oldShape = expandOp.getSrc().getType();

  updateDimensionalOp(reduceOp, rewriter, newDimensions);
  updateDefiningOp(reduceOp, rewriter, newOperands);

  LLVM_DEBUG(llvm::dbgs() << "Old shape is " << oldShape << "\n";);
  for (auto res : definingOp->getResults()) {
    LDBG("Collapsing results");
    collapseAndReplace(rewriter, expandOp, oldShape, res, definingOp);
  }
  for (auto oldInit : oldDpsInits) {
    LDBG("Collapsing inits");
    collapseAndReplace(rewriter, expandOp, oldShape, oldInit, definingOp);
  }
  rewriter.replaceOp(expandOp, definingOp->getResult(oldReplaceIndex));
  if constexpr (std::is_same_v<OpType, hfusion::ReduceWithIndexOp>) {
    if (reduceOp.getInputs().size() == 1) {
      // reduce_with_index without the index input (getInputs().size() == 1)
      // has a linalg::IndexOp operation inside the region,
      // whose dimension needs to be updated accordingly
      updateHFusionReduceWithIndexDim(rewriter, definingOp, newDimensions);
    }
  }
  return success();
}

LogicalResult handleHIVMBroadcastOp(tensor::ExpandShapeOp expandOp,
                                    PatternRewriter &rewriter,
                                    Operation *definingOp) {
  auto broadcastOp = cast<hivm::VBrcOp>(definingOp);
  ArrayRef<int64_t> dimensions = broadcastOp.getBroadcastDims();
  BitVector broadcastBits =
      createBitVector(broadcastOp->getResult(0), dimensions);
  SmallVector<ReassociationIndices> reassociationIndices =
      expandOp.getReassociationIndices();
  SmallVector<int64_t> newOutputShape;
  SmallVector<int64_t> newDimensions;

  // Step 2: Get reassocs
  obtainHIVMUpDimIncReassocs(broadcastBits, reassociationIndices,
                             utils::getShape(expandOp.getResult().getType()),
                             dimensions, newOutputShape, newDimensions);

  // Step 3: update everything
  auto newOperands = expandPropagateCreateNewOperands(
      rewriter, expandOp, broadcastOp, newOutputShape, reassociationIndices);

  LLVM_DEBUG(llvm::dbgs() << "Got new Operands: " << newOperands.size()
                          << "\n";);
  updateHIVMDimensionalOp(broadcastOp, rewriter, newDimensions);
  auto oldShape = expandOp.getSrc().getType();
  auto oldDpsInits =
      cast<DestinationStyleOpInterface>(definingOp).getDpsInits();
  // Must do this or else it would be replaced by update Defining op
  updateDefiningOp(definingOp, rewriter, newOperands);
  collapseAndReplace(rewriter, expandOp, oldShape, definingOp->getResult(0),
                     definingOp);
  collapseAndReplace(rewriter, expandOp, oldShape, *oldDpsInits.begin(),
                     definingOp);
  rewriter.replaceOp(expandOp, definingOp->getResult(0));
  return success();
}

LogicalResult handleHIVMReduceOp(tensor::ExpandShapeOp expandOp,
                                 PatternRewriter &rewriter,
                                 Operation *definingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Ok reduce writing\n";);
  auto reduceOp = cast<hivm::VReduceOp>(definingOp);

  auto inputsReduce = reduceOp.getDpsInputs();
  LLVM_DEBUG(llvm::dbgs() << "Finding input " << inputsReduce[0] << "\n";);

  auto opResult = cast<OpResult>(expandOp.getSrc());
  auto resultIdx = opResult.getResultNumber();

  ArrayRef<int64_t> dimensions = reduceOp.getReduceDims();

  BitVector reduceBits = createBitVector(inputsReduce[0], dimensions);

  SmallVector<ReassociationIndices> reassociationIndices =
      expandOp.getReassociationIndices();
  SmallVector<int64_t> newOutputShape;
  SmallVector<int64_t> newDimensions;
  // Step 2: obtains dimensions and reassociations

  obtainHIVMUpDimDecReassocs(reduceBits, reassociationIndices,
                             utils::getShape(inputsReduce[0].getType()),
                             utils::getShape(expandOp.getResult().getType()),
                             newOutputShape, newDimensions);

  // Step 3: Change everything
  auto oldDpsInits =
      cast<DestinationStyleOpInterface>(definingOp).getDpsInits();
  auto newOperands = expandPropagateCreateNewOperands(
      rewriter, expandOp, reduceOp, newOutputShape, reassociationIndices);

  auto oldShape = expandOp.getSrc().getType();

  updateHIVMDimensionalOp(reduceOp, rewriter, newDimensions);
  updateDefiningOp(reduceOp, rewriter, newOperands);

  LLVM_DEBUG(llvm::dbgs() << "Old shape is " << oldShape << "\n";);
  for (auto res : definingOp->getResults()) {
    collapseAndReplace(rewriter, expandOp, oldShape, res, definingOp);
  }
  for (auto oldInit : oldDpsInits) {
    collapseAndReplace(rewriter, expandOp, oldShape, oldInit, definingOp);
  }
  rewriter.replaceOp(expandOp, definingOp->getResult(resultIdx));
  return success();
}

LogicalResult handleExtractOp(tensor::ExpandShapeOp expandOp,
                              PatternRewriter &rewriter,
                              Operation *definingOp) {
  auto extractOp = dyn_cast<tensor::ExtractOp>(definingOp);
  if (!extractOp)
    return failure();

  auto expandedIndices = extractOp.getIndices();
  auto curShape = expandOp.getResultType().getShape();

  // Prepare dimension sizes as Values.
  SmallVector<Value> dimensionSizes;
  Location loc = expandOp.getLoc();
  rewriter.setInsertionPoint(expandOp);
  for (int64_t dim : curShape) {
    Value dimSizeValue;
    if (ShapedType::isDynamic(dim))
      dimSizeValue = rewriter.create<tensor::DimOp>(loc, expandOp.getResult(),
                                                    dimensionSizes.size());
    else
      dimSizeValue = rewriter.create<arith::ConstantIndexOp>(loc, dim);
    dimensionSizes.push_back(dimSizeValue);
  }

  // Define getDimSize lambda function.
  auto getDimSize = [&](int idx) -> Value { return dimensionSizes[idx]; };

  // Compute collapsed indices using the utility function.
  auto collapsedIndices = hfusion::computeExtractCollapsedIndices(
      expandOp.getReassociationIndices(), expandedIndices, getDimSize, rewriter,
      loc);

  rewriter.setInsertionPointAfter(expandOp);
  if (utils::getShapeRank(expandOp.getSrc()).value_or(0) == 1) {
    if (collapsedIndices.empty()) {
      collapsedIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
  }
  auto newExtract = rewriter.create<tensor::ExtractOp>(loc, expandOp.getSrc(),
                                                       collapsedIndices);
  rewriter.replaceOp(extractOp, newExtract.getResult());
  return success();
}

LogicalResult handleConcatOp(tensor::ExpandShapeOp expandOp,
                             PatternRewriter &rewriter, Operation *definingOp) {
  auto reassociation = expandOp.getReassociationIndices();
  auto dimensionResult = utils::getShape(expandOp.getResultType());
  auto concatOp = dyn_cast<tensor::ConcatOp>(definingOp);
  auto mixedDimensionResult =
      getMixedValues(dimensionResult, expandOp.getOutputShape(), rewriter);
  uint64_t newConcatDim = 0;
  assert(ssize_t(mixedDimensionResult.size()) ==
         reassociation.back().back() + 1);
  assert(ssize_t(reassociation.size()) == concatOp.getType().getRank());
  SmallVector<OpFoldResult> newExpandOutputShape;
  SmallVector<OpFoldResult> operandsNewDimSize;
  auto res = computeExpandConcat(rewriter, concatOp, reassociation,
                                 mixedDimensionResult, newConcatDim,
                                 newExpandOutputShape, operandsNewDimSize);
  if (res.failed())
    return failure();
  rewriter.setInsertionPointAfter(concatOp);
  // Need to expand all of the operands
  auto newConcatOp =
      buildNewConcat(rewriter, concatOp, reassociation, newConcatDim,
                     newExpandOutputShape, operandsNewDimSize);
  // oldA = concatOp(%arg0)
  // oldB = expand(oldA)
  // now oldA might be used elsse where
  // newB = expand(%arg0)
  // newA = concatOp(newB)
  // extraC = collapse(newA)
  // so we will collapse newA first, and replace all usage of oldA with extraC
  // after that, replace all usage of oldB with newA
  auto oldShape = expandOp.getSrc().getType();
  collapseAndReplace(rewriter, reassociation, utils::getShape(oldShape),
                     concatOp.getResult(), newConcatOp.getResult(), definingOp);
  rewriter.replaceOp(expandOp, newConcatOp.getResult());
  return success();
}

LogicalResult handlePadOp(tensor::ExpandShapeOp expandOp,
                          PatternRewriter &rewriter, Operation *definingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Handling pad here\n";);
  auto reassociation = expandOp.getReassociationIndices();
  auto dimensionResult = utils::getShape(expandOp.getResultType());
  auto padOp = dyn_cast<tensor::PadOp>(definingOp);
  SmallVector<OpFoldResult> newPadLow;
  SmallVector<OpFoldResult> newPadHigh;
  SmallVector<OpFoldResult> newExpandOutputShape;
  // Expand first
  auto padSrc = padOp.getSource();
  auto loc = padOp.getLoc();
  auto oldExpandOutputShape = getMixedValues(
      expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter);
  DenseMap<uint64_t, uint64_t> padBodyMapping;
  auto result = computeExpandPad(rewriter, padOp, reassociation, padBodyMapping,
                                 newPadLow, newPadHigh, newExpandOutputShape,
                                 oldExpandOutputShape, dimensionResult);
  if (result.failed())
    return failure();
  auto staticOutputShape = decomposeMixedValues(newExpandOutputShape).first;
  auto expandResType = RankedTensorType::get(
      staticOutputShape, getElementTypeOrSelf(padSrc.getType()));
  // Pad is special case, manually create expandShape
  rewriter.setInsertionPointAfter(padOp);
  auto newExpandedOperand = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandResType, /* src */ padSrc, reassociation,
      newExpandOutputShape);

  auto newStaticLow = decomposeMixedValues(newPadLow).first;
  auto newStaticHigh = decomposeMixedValues(newPadHigh).first;
  Type padType = tensor::PadOp::inferResultType(expandResType, newStaticLow,
                                                newStaticHigh);
  auto newPadOp = rewriter.create<tensor::PadOp>(
      loc, padType, newExpandedOperand.getResult(), newPadLow, newPadHigh);
  clonePadRegion(rewriter, padOp, newPadOp, padBodyMapping);
  auto oldShape = expandOp.getSrc().getType();
  // oldA = padOp(%arg0)
  // oldB = expand(oldA)
  // now oldA might be used elsse where
  // newB = expand(%arg0)
  // newA = padOp(newB)
  // extraC = collapse(newA)
  // so we will collapse newA first, and replace all usage of oldA with extraC
  // after that, replace all usage of oldB with newA
  rewriter.setInsertionPointAfter(newPadOp);
  collapseAndReplace(rewriter, reassociation, utils::getShape(oldShape),
                     padOp.getResult(), newPadOp.getResult(), definingOp);
  rewriter.replaceOp(expandOp, newPadOp.getResult());
  return success();
}

LogicalResult handleExtractSliceOp(tensor::ExpandShapeOp expandOp,
                                   PatternRewriter &rewriter,
                                   Operation *definingOp) {
  auto reassociation = expandOp.getReassociationIndices();
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(definingOp);
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  SmallVector<OpFoldResult> expandOutputShape;
  SmallVector<ReassociationIndices> newReasso;

  auto dropDims = extractSliceOp.getDroppedDims();
  auto expandedShape =
      getMixedSizesOrOutputShape(rewriter, expandOp.getResult());
  if (isUnitDimReshape(convertToConstantValues(expandedShape), reassociation)) {
    // skip collapseShape that only add unit dims
    return failure();
  }
  expandedShape = getUndroppedExpandedSubviewShape(rewriter, expandedShape,
                                                   reassociation, dropDims);
  auto res = getExtractSliceModifyingOp(
      rewriter, cast<ExtractSliceOp>(definingOp), reassociation, expandedShape,
      /* subview */ true, newMixedOffsets, newMixedSizes, newMixedStrides,
      expandOutputShape, newReasso);
  if (res.failed())
    return failure();
  auto loc = definingOp->getLoc();
  auto expandedNewSrc = createExpand(rewriter, loc, extractSliceOp.getSource(),
                                     newReasso, expandOutputShape);
  auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, cast<RankedTensorType>(expandOp.getResult().getType()),
      expandedNewSrc.getResult(), newMixedOffsets, newMixedSizes,
      newMixedStrides);
  auto extractSliceOpResultShape =
      utils::getShape(extractSliceOp.getResult().getType());
  rewriter.setInsertionPointAfter(newExtractSliceOp);
  collapseAndReplace(rewriter, reassociation, extractSliceOpResultShape,
                     extractSliceOp.getResult(), newExtractSliceOp.getResult(),
                     definingOp);
  rewriter.replaceOp(expandOp, newExtractSliceOp.getResult());
  rewriter.eraseOp(extractSliceOp);
  return success();
}
LogicalResult handleInsertSliceOp(tensor::ExpandShapeOp expandOp,
                                  PatternRewriter &rewriter,
                                  Operation *definingOp) {
  auto reassociation = expandOp.getReassociationIndices();
  auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(definingOp);
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  SmallVector<OpFoldResult> expandSrcOutputShape;
  SmallVector<OpFoldResult> expandDestOutputShape;
  SmallVector<ReassociationIndices> newReasso;

  // The source must be the result of the insert slice, meaning its always
  // the superview
  auto res = getInsertSliceModifyingOp(
      rewriter, cast<InsertSliceOp>(definingOp), reassociation,
      getMixedSizesOrOutputShape(rewriter, expandOp.getResult()), false,
      newMixedOffsets, newMixedSizes, newMixedStrides, expandSrcOutputShape,
      expandDestOutputShape, newReasso);
  if (res.failed())
    return failure();

  // get new src shape
  auto rankReducedShape = getRankReducingShape(
      expandSrcOutputShape, reassociation, insertSliceOp.getDroppedDims());

  auto loc = definingOp->getLoc();
  // Gonna collapse the destination as well?
  // Expand result or expand source
  auto expandedNewSrc = createExpand(rewriter, loc, insertSliceOp.getSource(),
                                     newReasso, rankReducedShape);
  auto expandedNewDest = createExpand(rewriter, loc, insertSliceOp.getDest(),
                                      reassociation, expandDestOutputShape);
  auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      loc, expandedNewSrc, expandedNewDest, newMixedOffsets, newMixedSizes,
      newMixedStrides);
  auto insertSliceOpResultShape =
      utils::getShape(insertSliceOp.getResult().getType());
  rewriter.setInsertionPointAfter(newInsertSliceOp);
  collapseAndReplace(rewriter, reassociation, insertSliceOpResultShape,
                     insertSliceOp.getResult(), newInsertSliceOp.getResult(),
                     definingOp);
  rewriter.replaceOp(expandOp, newInsertSliceOp.getResult());
  rewriter.eraseOp(insertSliceOp);
  return success();
}

LogicalResult handleBufferizationToTensor(tensor::ExpandShapeOp expandOp,
                                          PatternRewriter &rewriter,
                                          Operation *definingOp) {
  bufferization::ToTensorOp toTensorOp =
      cast<bufferization::ToTensorOp>(definingOp);
  auto oldResultType = toTensorOp.getType();
  auto reassociation = expandOp.getReassociationIndices();
  auto memrefSrc = toTensorOp.getMemref();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(toTensorOp);
  auto expandOutputShape = getMixedValues(expandOp.getStaticOutputShape(),
                                          expandOp.getOutputShape(), rewriter);
  auto newExpandMemref =
      createMemrefExpand(rewriter, toTensorOp.getLoc(), memrefSrc,
                         reassociation, expandOutputShape);
  rewriter.modifyOpInPlace(toTensorOp, [&]() {
    toTensorOp.setOperand(newExpandMemref.getResult());
    toTensorOp.getResult().setType(expandOp.getType());
  });
  rewriter.setInsertionPointAfter(toTensorOp);
  auto collapseOldResult = rewriter.create<tensor::CollapseShapeOp>(
      toTensorOp.getLoc(), oldResultType, toTensorOp.getResult(),
      reassociation);
  SmallPtrSet<Operation *, 2> exceptedOp = {expandOp, collapseOldResult};
  rewriter.replaceAllUsesExcept(toTensorOp, collapseOldResult, exceptedOp);
  rewriter.replaceOp(expandOp, toTensorOp);

  return success();
}

LogicalResult handleArangeOp(tensor::ExpandShapeOp expandOp,
                             PatternRewriter &rewriter, Operation *definingOp) {
  auto reassociation = expandOp.getReassociationIndices();

  // To deal with the init, make it get its shape expanded instead
  auto arangeOp = cast<tensor::ArangeOp>(definingOp);
  auto expandedArangeInit = rewriter.create<tensor::ExpandShapeOp>(
      definingOp->getLoc(), expandOp.getResultType(), arangeOp.getInit(),
      expandOp.getReassociationAttr(), expandOp.getOutputShape(),
      expandOp.getStaticOutputShape());

  // Get the new strides.
  SmallVector<Value> expandedArangeStrides;
  auto originalStrides = arangeOp.getStrides();
  // Go through each reassociation group in order.
  for (size_t reassociationIndexId = 0;
       reassociationIndexId < reassociation.size(); reassociationIndexId++) {
    // Reading the strides from right to left in this group:
    //    S0 is always the original stride.
    //    S1 = outputDim(reassociationInidices[num_indices - 1]) * S0
    //    S2 = outputDim(reassociationInidices[num_indices - 2]) * S1
    //    ...
    // This will allow the expanded dimension to be strided correctly based on
    // the original stride and the new shape we are creating.
    const auto &reassociationInidices = reassociation[reassociationIndexId];
    Value newStride = originalStrides[reassociationIndexId];
    SmallVector<Value> reassociationStrides;
    // Add the original stride
    reassociationStrides.push_back(newStride);
    // Get the expanded dimension strides. We need to start at the end of the
    // list and go to the front, stopping before the first element.
    for (int i = (int)reassociationInidices.size() - 1; i > 0; i--) {
      // Get the size of the dimension at this reassociation index location.
      auto dim = rewriter.create<tensor::DimOp>(
          expandOp.getLoc(), expandedArangeInit, reassociationInidices[i]);
      // Multiply the dim from the last stride.
      newStride =
          rewriter.create<arith::MulIOp>(expandOp.getLoc(), newStride, dim);
      // Insert at the front to reverse the strides to left to right.
      reassociationStrides.insert(reassociationStrides.begin(), newStride);
    }
    // Insert this group of strides into the overall strides.
    expandedArangeStrides.append(reassociationStrides.begin(),
                                 reassociationStrides.end());
  }

  // Create the new arange op.
  // Since offset is optional, need to check if it is there.
  Value offset = arangeOp.getOffset();
  if (offset) {
    rewriter.replaceOpWithNewOp<tensor::ArangeOp>(
        expandOp, offset, expandedArangeStrides,
        expandedArangeInit.getResult());
  } else {
    rewriter.replaceOpWithNewOp<tensor::ArangeOp>(
        expandOp, expandedArangeStrides, expandedArangeInit.getResult());
  }

  return success();
}

} // namespace

LogicalResult
PropagateExpandUp::matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                   PatternRewriter &rewriter) const {
  Value source = expandOp.getSrc();
  Operation *definingOp = source.getDefiningOp();
  if (!definingOp)
    return rewriter.notifyMatchFailure(expandOp.getOperation(),
                                       "Defining op doesn't exist");
  if (isStopPropagatable(definingOp))
    return rewriter.notifyMatchFailure(
        definingOp, "Propagation stopped because of defining type");
  auto oneUse = definingOp->hasOneUse();
  if ((definingOp->getParentOp() != expandOp->getParentOp()) && (!oneUse))
    return rewriter.notifyMatchFailure(
        definingOp->getParentOp(), "expandOp is NOT in the same region as "
                                   "srcDefOp and srcDefOp has multiple users");
  if (llvm::all_of(expandOp->getUsers(),
                   [&](Operation *op) { return isOutOp(op); })) {
    return rewriter.notifyMatchFailure(expandOp, "All user of expand is out");
  }
  if (isa<bufferization::ToTensorOp>(definingOp)) {
    return handleBufferizationToTensor(expandOp, rewriter, definingOp);
  }
  if (options.forHIVM &&
      !isNonUnitExpandOrEmptyReassoc(expandOp.getResultType().getShape(),
                                     expandOp.getReassociationIndices()))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << (definingOp->getParentOp() ? *(definingOp->getParentOp()) : *definingOp) << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "-- Found definingOp: " << *definingOp << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "Ok rewriting\n";);
  if (auto bitcastOp = dyn_cast<hivm::BitcastOp>(definingOp)) {
    return handleBitcastOp(expandOp, rewriter, bitcastOp);
  }
  if (isa<hfusion::MulExtOp>(definingOp)) {
    PropagatableMulExt propagater;
    return propagater.matchAndRewriteExpand(rewriter, definingOp, expandOp);
  }
  if (isa<scf::ForOp>(definingOp)) {
    PropagatableScfFor propagater;
    return propagater.matchAndRewriteExpand(rewriter, definingOp, expandOp);
  }
  if (isa<tensor::ConcatOp>(definingOp)) {
    return handleConcatOp(expandOp, rewriter, definingOp);
  }
  if (isa<tensor::PadOp>(definingOp)) {
    return handlePadOp(expandOp, rewriter, definingOp);
  }
  if (isa<linalg::BroadcastOp>(definingOp)) {
    return handleBroadcastOp(expandOp, rewriter, definingOp);
  }
  if (isa<hfusion::ReduceWithIndexOp>(definingOp)) {
    return handleReduceLikeOp<hfusion::ReduceWithIndexOp>(expandOp, rewriter,
                                                          definingOp);
  }
  if (isa<linalg::ReduceOp>(definingOp)) {
    return handleReduceLikeOp<linalg::ReduceOp>(expandOp, rewriter, definingOp);
  }
  if (isa<hfusion::FlipOp>(definingOp)) {
    return handleFlipOp(expandOp, rewriter, definingOp);
  }
  if (isa<linalg::TransposeOp>(definingOp)) {
    return handleTransposeOp<linalg::TransposeOp>(expandOp, rewriter,
                                                  definingOp);
  }
  if (isa<hivm::VTransposeOp>(definingOp)) {
    // TODO: handle VTransposeOp to support more than one dimension transpose
    return failure();
  }
  if (isa<hfusion::InterleaveOp>(definingOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Propagate expand up - Interleave\n";);
    return handleInterleaveOp(expandOp, rewriter, definingOp);
  }
  if (isa<hfusion::DeinterleaveOp>(definingOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Propagate expand up - Deinterleave\n";);
    return handleDeinterleaveOp(expandOp, rewriter, definingOp);
  }
  if (!options.forHIVM && isa<tensor::ExtractSliceOp>(definingOp)) {
    return handleExtractSliceOp(expandOp, rewriter, definingOp);
  }
  if (!options.forHIVM && isa<tensor::InsertSliceOp>(definingOp)) {
    return handleInsertSliceOp(expandOp, rewriter, definingOp);
  }
  if (isMarkedAsElementwiseOp(definingOp)) {
    return handleElementwiseOp(expandOp, rewriter, definingOp);
  }
  if (mlir::hivm::detail::isElemwiseNaryOpImpl(definingOp) ||
      isa<hivm::CopyOp>(definingOp) || isa<hivm::StoreOp>(definingOp) ||
      isa<hivm::LoadOp>(definingOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Propagate expand up - HIVM Elemwise\n";);
    return handleElementwiseOp(expandOp, rewriter, definingOp);
  }
  if (isa<hivm::VBrcOp>(definingOp)) {
    return handleHIVMBroadcastOp(expandOp, rewriter, definingOp);
  }
  if (isa<hivm::VReduceOp>(definingOp)) {
    return handleHIVMReduceOp(expandOp, rewriter, definingOp);
  }
  if (isa<tensor::ExtractOp>(definingOp)) {
    return handleExtractOp(expandOp, rewriter, definingOp);
  }
  if (isa<tensor::ArangeOp>(definingOp)) {
    return handleArangeOp(expandOp, rewriter, definingOp);
  }
  return failure();
}

} // namespace tensor
} // namespace mlir
