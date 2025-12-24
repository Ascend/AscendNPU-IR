//===- PropagateCollapseDown.cpp ------------------------------------------===//
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
// Propagate collapse down will try to bring the collapse shape operation to the
// end
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateCollapseDown.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <type_traits>

#define DEBUG_TYPE "propagate-reshape-collapse-down"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

namespace {

bool isStrided(Value operand) {
  MemRefType memrefType = dyn_cast<MemRefType>(operand.getType());
  return memrefType && isa<StridedLayoutAttr>(memrefType.getLayout());
}

int64_t getOffset(Value operand) {
  if (!isStrided(operand))
    return ShapedType::kDynamic;
  return dyn_cast<StridedLayoutAttr>(
             dyn_cast<MemRefType>(operand.getType()).getLayout())
      .getOffset();
}

StridedLayoutAttr calcStridedLayout(MLIRContext *context,
                                    SmallVector<int64_t> shape,
                                    int64_t offset) {
  SmallVector<int64_t> outputStrides(shape.size(), ShapedType::kDynamic);
  int64_t stride = 1;
  for (int i = (int)(shape.size()) - 1; i >= 0; --i) {
    outputStrides[i] = stride;
    stride *= shape[i];
    if (shape[i] == ShapedType::kDynamic) {
      break;
    }
  }
  return StridedLayoutAttr::get(context, offset, outputStrides);
}

Operation *createNewExpandOpFromCollapseOp(tensor::CollapseShapeOp &collapseOp,
                                           PatternRewriter &rewriter,
                                           Location loc, Value operand) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(operand);
  auto reassociation = collapseOp.getReassociationIndices();
  auto currentShape = utils::getShape(collapseOp.getSrc().getType());
  if (isa<MemRefType>(operand.getType())) {
    MemRefType resultType;
    if (!isStrided(operand)) {
      resultType = MemRefType::get(currentShape, getElementTypeOrSelf(operand));
    } else {
      resultType =
          MemRefType::get(currentShape, getElementTypeOrSelf(operand),
                          calcStridedLayout(rewriter.getContext(), currentShape,
                                            getOffset(operand)));
    }
    return rewriter.create<memref::ExpandShapeOp>(loc, resultType, operand,
                                                  reassociation);
  }
  auto resultType =
      RankedTensorType::get(currentShape, getElementTypeOrSelf(operand));
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, operand,
                                                reassociation);
}

// %a = collapse_shape %arg0
// %b = elemwise_binary(%a, %arg1), outs(%c);
//
// |
// v
//
//
// %arg1_expand = expandOp(%arg1)          (new)
// %c_expand = expandOp(%c)                (new)
// %b = elemwise_binary(%arg0, %arg1_expand), outs(%c_expand); (preserve)
// %c_replacer = collapse_shape %c_expand  (new replaces C)
// %b_replacer = collapse_shape %b         (new replaces B)
//
// |
// v
//
// %arg1_expand = expandOp(%arg1)          (new)
// %c_expand = expandOp(%c)                (new)
// %b = elemwise_binary(%arg0, %arg1_expand), outs(%c_expand); (preserve)
// %c_replacer = collapse_shape %c_expand  (new replaces C)
// %b_replacer = collapse_shape %b         (new replaces B)
LogicalResult handleElementwiseOp(tensor::CollapseShapeOp collapseOp,
                                  PatternRewriter &rewriter,
                                  Operation *userOp) {
  rewriter.setInsertionPointAfter(userOp);
  auto loc = collapseOp.getLoc();

  LLVM_DEBUG(llvm::dbgs() << "Trying " << collapseOp << " to an elemwise "
                          << *userOp << "\n";);

  auto resultRank = utils::getShapeRank(collapseOp.getResult()).value_or(0);
  SmallVector<Value> newOperands;
  auto oldDpsInits = cast<DestinationStyleOpInterface>(userOp).getDpsInits();

  for (Value operand : userOp->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand);
    // Check in case its scalar elemwise
    if (shapeRank.has_value() && shapeRank.value() == resultRank) {
      Operation *newExpandedOperand =
          createNewExpandOpFromCollapseOp(collapseOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newExpandedOperand << "\n";);
      newOperands.push_back(newExpandedOperand->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't expand inequal rank " << shapeRank
                              << " : " << resultRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  updateDefiningOp(userOp, rewriter, newOperands);
  collapseAndReplace(rewriter, collapseOp, userOp->getResult(0), userOp);
  collapseAndReplace(rewriter, collapseOp, *oldDpsInits.begin(), userOp);
  return success();
}

// %a = collapse_shape %arg0
// %b = transpose %a [permutaiton]
//
// |
// v
//
// %a = collapse_shape %arg0 (preserved, will be propagated later if match
// others) %newA = transpose %arg0 [new permutation] %newB = collapse_shape
// %newA %newB replace all uses of %b
template <class T>
LogicalResult handleTransposeOp(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter, Operation *userOp) {
  auto collapseInShape = collapseOp.getSrcType().getShape();
  linalg::TransposeOp transposeOp = cast<linalg::TransposeOp>(userOp);
  auto permutation = transposeOp.getPermutation();
  auto reassociation = collapseOp.getReassociationIndices();
  auto invPermutation = reshape_utils::getInversePermutation(permutation);
  bool isInit = checkValueIsInit(userOp, collapseOp);

  // Create new Expand input shape
  SmallVector<int64_t, 4> newTransposeOutputShape;
  // Create new Expand reassociation
  SmallVector<ReassociationIndices, 4> newCollapseReassociation;
  reshape_utils::createTransposedReassoc(
      reassociation, collapseInShape, (isInit ? invPermutation : permutation),
      newTransposeOutputShape, newCollapseReassociation);

  // Create new tranpose permutation
  SmallVector<int64_t, 4> newPermutation;
  auto transposeRank =
      utils::getShapeRank(transposeOp->getResult(0)).value_or(0);
  reshape_utils::createNewPermutation(
      transposeRank, permutation,
      (isInit ? newCollapseReassociation : reassociation), newPermutation);

  LLVM_DEBUG(llvm::dbgs() << "Try to push " << collapseOp << " after "
                          << *userOp << "\n";);

  auto collapseSrcOp = collapseOp.getSrc();
  auto transposeSrcOp = transposeOp.getDpsInputOperand(0)->get();
  rewriter.setInsertionPointAfterValue(collapseSrcOp);

  auto expandedTy = collapseOp.getSrcType().clone(newTransposeOutputShape);
  linalg::TransposeOp newTransposeOp;

  if (isInit) {
    auto expandArg = rewriter.create<tensor::ExpandShapeOp>(
        collapseOp->getLoc(), expandedTy, transposeSrcOp,
        newCollapseReassociation);
    newTransposeOp = rewriter.create<linalg::TransposeOp>(
        transposeOp->getLoc(), expandArg, collapseSrcOp, newPermutation);
  } else {
    tensor::EmptyOp dstOp = rewriter.create<tensor::EmptyOp>(
        transposeOp.getLoc(), expandedTy, ValueRange());
    // add new transpose operation
    newTransposeOp = rewriter.create<linalg::TransposeOp>(
        transposeOp->getLoc(), collapseSrcOp, dstOp, newPermutation);
  }

  // add new collapse operation
  auto newCollapseOutType = transposeOp->getResultTypes()[0];
  auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      collapseOp.getLoc(), newCollapseOutType, newTransposeOp->getResult(0),
      (isInit ? reassociation : newCollapseReassociation));

  // old transpose replaced by new collapse
  rewriter.replaceOp(transposeOp, newCollapseOp);
  return success();
}
/*

%a = collapse %input
%b = deinterleave %a

|
v

%new_in = deinterleave %input
%new_b = collapse %new_in

note that we are not going to erase the (%a = collapse %input)
in case it has any other users.
*/
LogicalResult handleDeinterleaveOp(tensor::CollapseShapeOp collapseOp,
                                   PatternRewriter &rewriter,
                                   Operation *userOp) {
  auto collapseOutShape = collapseOp.getResultType().getShape();
  auto collapseInShape = collapseOp.getSrcType().getShape();
  auto reassociation = collapseOp.getReassociationIndices();

  // dynamic dimentions are not supported yet
  if (ShapedType::isDynamic(collapseOutShape.back()) ||
      ShapedType::isDynamic(collapseInShape.back())) {
    return failure();
  }

  hfusion::DeinterleaveOp deinterleaveOp =
      cast<hfusion::DeinterleaveOp>(userOp);

  // we only support the case when the last dimention of the input is a multiple
  // of 2, because other wise, we need an extra collapse operation.
  // example:
  // collapse 2x3xf32 -> 6xf32
  // [A,B,A] -> [A,B,A,B,A,B]
  // [B,A,B]
  // e.g = deinterleave 6xf32 -> 3xf32
  // [A,A,A] (channel 0)
  // -------------------
  // [B,B,B] (channel 1)
  // deinterleave can't be altered because it does not align
  if (collapseInShape.back() % deinterleaveOp.getDeInterLeaveChannelNum() !=
      0) {
    return failure();
  }

  PatternRewriter::InsertionGuard guard(rewriter);

  // setting up the new deinterleave out shape
  SmallVector<int64_t> newDeinterleaveOutShape(collapseInShape.begin(),
                                               collapseInShape.end());
  newDeinterleaveOutShape.back() /= deinterleaveOp.getDeInterLeaveChannelNum();

  // create the new deinterleave op, directly connected to the input
  auto aOp = collapseOp.getSrc();
  rewriter.setInsertionPointAfterValue(aOp);
  auto diResTy =
      RankedTensorType::get(newDeinterleaveOutShape, getElementTypeOrSelf(aOp));
  auto newDeinterleaveOp = rewriter.create<hfusion::DeinterleaveOp>(
      aOp.getLoc(), diResTy, aOp, deinterleaveOp.getChannelIndex());

  // collapse the new deinterleave-ed input
  rewriter.setInsertionPointAfter(newDeinterleaveOp);
  for (int i = 0; i < static_cast<int>(deinterleaveOp->getNumResults()); i++) {
    auto cResTy = RankedTensorType::get(
        utils::getShape(deinterleaveOp->getResult(i).getType()),
        getElementTypeOrSelf(aOp));
    auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
        newDeinterleaveOp.getLoc(), cResTy, newDeinterleaveOp->getResult(i),
        reassociation);

    rewriter.replaceAllUsesWith(deinterleaveOp->getResult(i), newCollapseOp);
  }

  rewriter.eraseOp(deinterleaveOp);

  return success();
}

/*
%collapsed_a = collapse %a
%collapsed_b = %b
%res = interleave %collapsed_a, %collapsed_b

|
v

%expanded_a = %a
%expanded_b = expand %b
%foo = interleave %expanded_a, %expanded_b
%res = collapse %foo

note that we are not going to erase the (%collapsed_a = collapse %a)
in case it has any other users.
*/
LogicalResult handleInterleaveOp(tensor::CollapseShapeOp collapseOp,
                                 PatternRewriter &rewriter, Operation *userOp) {
  auto collapseOutShape = collapseOp.getResultType().getShape();
  auto collapseInShape = collapseOp.getSrcType().getShape();
  auto reassociation = collapseOp.getReassociationIndices();

  // dynamic dimensions are not supported yet
  // collapse 2x3xf32 -> 6xf32
  // [A,A,A] -> [A,A,A,A,A,A]
  // [A,A,A]
  // e.g = interleave 6xf32 -> 12xf32
  // [A,A,A,A,A,A] (channel 0)
  // -------------------
  // [B,B,B,B,B,B] (channel 1)
  // [A,B,A,B,A,B,A,B,A,B,A,B] interleaved
  // interleaving 2x3xf32 would result in 2x6xf32
  // [A,B,A,B,A,B]
  // [A,B,A,B,A,B]
  if (ShapedType::isDynamic(collapseOutShape.back()) ||
      ShapedType::isDynamic(collapseInShape.back())) {
    return failure();
  }

  PatternRewriter::InsertionGuard guard(rewriter);
  hfusion::InterleaveOp interleaveOp = cast<hfusion::InterleaveOp>(userOp);

  SmallVector<Value> newInOps;
  for (int i = 0; i < static_cast<int>(interleaveOp.getNumOperands()); i++) {
    auto inOp = interleaveOp->getOperand(i);
    rewriter.setInsertionPointAfterValue(inOp);
    auto eResTy =
        RankedTensorType::get(collapseInShape, getElementTypeOrSelf(inOp));
    auto newInOp = rewriter.create<tensor::ExpandShapeOp>(inOp.getLoc(), eResTy,
                                                          inOp, reassociation);
    newInOps.push_back(newInOp);
  }

  // removing the effect of interleave on the shapes
  SmallVector<int64_t> newInterleaveOutShape(collapseInShape.begin(),
                                             collapseInShape.end());
  newInterleaveOutShape.back() *= interleaveOp.getInterLeaveChannelNums();
  SmallVector<int64_t> newCollapseOutShape(collapseOutShape.begin(),
                                           collapseOutShape.end());
  newCollapseOutShape.back() *= interleaveOp.getInterLeaveChannelNums();

  // interleave the new expanded inputs
  rewriter.setInsertionPointAfterValue(interleaveOp);
  auto iResTy = RankedTensorType::get(newInterleaveOutShape,
                                      getElementTypeOrSelf(interleaveOp));
  auto newIntervalOp = rewriter.create<hfusion::InterleaveOp>(
      interleaveOp.getLoc(), iResTy, ValueRange(newInOps));

  // collapsing the new interleave
  rewriter.setInsertionPointAfter(newIntervalOp);
  auto cResTy = RankedTensorType::get(newCollapseOutShape,
                                      getElementTypeOrSelf(newIntervalOp));
  auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      newIntervalOp.getLoc(), cResTy, newIntervalOp, reassociation);

  rewriter.replaceAllUsesWith(interleaveOp, newCollapseOp);
  rewriter.eraseOp(interleaveOp);

  return success();
}

/// %collapse = collapse %arg0
/// %arange = arange outs(%collapse)
/// |
/// v
/// %collapse = collapse %arg0 (unchanged)
/// %arange = arange outs(%arg0) // original shape
/// %new_collapse = collapse %arange // (new)
LogicalResult handleArangeOp(tensor::CollapseShapeOp collapseOp,
                             PatternRewriter &rewriter, ArangeOp arange) {
  llvm::SmallVector<ReassociationIndices, 4> reassociation =
      collapseOp.getReassociationIndices();

  rewriter.setInsertionPoint(arange);
  Location loc = arange->getLoc();
  Value initVal = collapseOp.getSrc();

  // Calculate the new strides
  SmallVector<Value> expandedStrides;
  ValueRange originalStrides = arange.getStrides();

  for (unsigned dim = 0; dim < reassociation.size(); ++dim) {
    unsigned reassSize = reassociation[dim].size();
    if (reassSize == 1) {
      expandedStrides.push_back(originalStrides[dim]);
      continue;
    }
    // Break apart the dimensions:
    // Original stride only has one value due to being collapsed, now we break
    // that into multiple from the size we will generate: stride[dim] =
    // stride[dim+1]*shape[dim+1], while stride of the innermost dimension stays
    // the same. Since we have to populate this list from back to front, we will
    // need a temporary working set before inserting into the expandedStrides.
    SmallVector<Value> curExpandedStrides(reassSize, originalStrides[dim]);
    // Skip the last dim since we filled working list with the original stride
    for (int i = static_cast<int>(reassSize) - 2; i >= 0; --i) {
      curExpandedStrides[i] = rewriter.createOrFold<arith::MulIOp>(
          loc, curExpandedStrides[i + 1],
          rewriter.createOrFold<tensor::DimOp>(loc, initVal,
                                               reassociation[dim][i + 1]));
    }
    expandedStrides.append(curExpandedStrides);
  }
  auto expandedArange =
      rewriter.create<ArangeOp>(loc, expandedStrides, initVal);
  if (arange.getOffset())
    expandedArange.getOffsetMutable().append(arange.getOffset());

  Value result = (arange.hasPureTensorSemantics())
                     ? expandedArange.getResultTensor()
                     : expandedArange.getInit();
  rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(arange, result,
                                                       reassociation);
  return success();
}

LogicalResult handleBitcastOp(tensor::CollapseShapeOp collapseOp,
                              PatternRewriter &rewriter,
                              hivm::BitcastOp bitcastOp) {
  rewriter.setInsertionPointAfter(bitcastOp);
  auto loc = collapseOp.getLoc();
  auto resultRank = utils::getShapeRank(collapseOp.getResult()).value_or(0);
  SmallVector<Value> newOperands;
  auto opTmp = bitcastOp->getOperands();
  SmallVector<Value> oldOperands(opTmp.begin(), opTmp.end());
  auto bitcastOpResult = bitcastOp.getResult();
  auto bitcastOpResultType = bitcastOpResult.getType();

  for (Value operand : bitcastOp->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand).value_or(0);
    // Check in case its scalar elemwise
    if (shapeRank == resultRank) {
      Operation *newExpandedOperand =
          createNewExpandOpFromCollapseOp(collapseOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newExpandedOperand << "\n";);
      newOperands.push_back(newExpandedOperand->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't expand inequal rank " << shapeRank
                              << " : " << resultRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  updateDefiningOpNonDst(bitcastOp, rewriter, newOperands,
                         utils::getShape(collapseOp.getSrc().getType()));
  collapseAndReplace(rewriter, collapseOp, bitcastOpResultType, bitcastOpResult,
                     bitcastOp);
  return success();
}

template <class LinalgBRTy>
SmallVector<Value, 4> collapsePropagateCreateNewOperands(
    PatternRewriter &rewriter, tensor::CollapseShapeOp &collapseOp,
    LinalgBRTy op, const SmallVector<int64_t> &newOutputShape,
    const SmallVector<ReassociationIndices> &newReassociation,
    bool isAsInit = false) {
  auto loc = collapseOp.getLoc();
  SmallVector<Value, 4> newOperands;
  auto opOperands = op->getOpOperands();
  for (OpOperand &operand : opOperands) {
    Value operandVal = operand.get();
    // Skip scalar like operand
    Operation *newExpandedOperand = nullptr;
    bool isInput = op.isDpsInput(&operand);
    bool isInit = op.isDpsInit(&operand);
    if (isa<RankedTensorType>(operandVal.getType()) && (isInput || isInit)) {
      // Depends which is needed to be expanded, is it input or init
      if (isInput ^ isAsInit) {
        newExpandedOperand = createNewExpandOpFromCollapseOp(
            collapseOp, rewriter, loc, operandVal);
      } else {
        rewriter.setInsertionPointAfterValue(operandVal);
        auto expandedType = RankedTensorType::get(
            newOutputShape, getElementTypeOrSelf(operandVal));
        newExpandedOperand = rewriter.create<tensor::ExpandShapeOp>(
            loc, expandedType, operandVal, newReassociation);
      }
    }
    if (newExpandedOperand != nullptr)
      newOperands.push_back(newExpandedOperand->getResult(0));
    else
      newOperands.push_back(operandVal);
  }
  return newOperands;
}

void obtainDownDimIncReassocs(
    ArrayRef<int64_t> incDimOpShape, ArrayRef<int64_t> collapseSourceShape,
    ArrayRef<int64_t> dimensions, ArrayRef<ReassociationIndices> reassociation,
    SmallVector<ReassociationIndices> &newReassociation,
    SmallVector<int64_t> &newOutputShape, SmallVector<int64_t> &newDimensions) {
  // Generate new Reassociation and new Output Shapes
  // Create new reassociation indices and dimensions

  // Output Shape
  newReassociation.reserve(incDimOpShape.size());

  // Run two pointer algorithm to generate the new Reassociation
  const auto *dimPtr = dimensions.begin();
  const auto *reassocPtr = reassociation.begin();
  LLVM_DEBUG(llvm::dbgs() << "Here ok obtain " << reassociation.size()
                          << "\n";);
  for (uint32_t i = 0; i < incDimOpShape.size(); ++i) {
    if (dimPtr != dimensions.end() && *dimPtr == i) {
      // This is increased dimensions
      LLVM_DEBUG(llvm::dbgs()
                     << "New reassociation adding " << *dimPtr << "\n";);
      newReassociation.push_back({-1}); // Mark for broadcast dimension
      newOutputShape.push_back(incDimOpShape[*dimPtr]);
      dimPtr++;
    } else {
      newReassociation.push_back(*reassocPtr);
      for (auto el : *reassocPtr) {
        newOutputShape.push_back(collapseSourceShape[el]);
      }
      reassocPtr++;
    }
  }
  LLDBG("Collapse source shape: " << to_string(collapseSourceShape));
  // Handle reducing tensor.empty, only appears on this case! dimDec rank
  // is guaranteed to be non 0
  if (reassociation.empty()) {
    std::reverse(newOutputShape.begin(), newOutputShape.end());
    std::reverse(newReassociation.front().begin(),
                 newReassociation.front().end());
    for (size_t i = 0; i < collapseSourceShape.size(); ++i) {
      newReassociation.front().push_back(1);
      newOutputShape.push_back(1);
    }
    std::reverse(newReassociation.front().begin(),
                 newReassociation.front().end());
    std::reverse(newOutputShape.begin(), newOutputShape.end());
  }
  renumberReassociationAndGetNewDimensions(newReassociation, newDimensions);
  LLVM_DEBUG(llvm::dbgs() << newReassociation.size() << "\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(newReassociation) << "\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(newDimensions) << "\n";);
}

// One weird case is incDimOpShape = [A, 1, B, C]
// dimptr = [0, 2, 3]
// but reassociation is empty because expand comes from an empty dimension
void obtainHIVMDownDimIncReassocs(ArrayRef<int64_t> incDimOpShape,
                                  ArrayRef<int64_t> collapseSourceShape,
                                  ArrayRef<int64_t> dimensions,
                                  ArrayRef<ReassociationIndices> reassociation,
                                  SmallVector<int64_t> &newOutputShape,
                                  SmallVector<int64_t> &newDimensions) {
  const auto *dimPtr = dimensions.begin();
  // In broadcast, incDimOpShape is the broadcast result, which is the higher
  // dimension
  for (size_t i = 0; i < incDimOpShape.size(); ++i) {
    if (dimPtr == dimensions.end() || static_cast<size_t>(*dimPtr) != i) {
      for (auto el : reassociation[i]) {
        newOutputShape.push_back(collapseSourceShape[el]);
      }
      continue;
    }
    // This is increased dimensions, in hivm the size is the same
    newOutputShape.push_back(incDimOpShape[i]);
    for (size_t j = 0; j < reassociation[i].size(); ++j) {
      if (j >= 1)
        newOutputShape.push_back(1);
      newDimensions.push_back(reassociation[i][j]);
    }
    dimPtr++;
  }
}

void obtainDownDimDecReassocs(
    ArrayRef<int64_t> collapseSourceShape, ArrayRef<int64_t> dimensions,
    ArrayRef<ReassociationIndices> reassociation,
    SmallVector<ReassociationIndices> &newReassociation,
    SmallVector<int64_t> &newOutputShape, SmallVector<int64_t> &newDimensions) {
  LLVM_DEBUG(llvm::dbgs() << "Doing decreasing\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reassociation) << "\n";);
  const auto *dimPtr = dimensions.begin();
  for (uint32_t i = 0; i < reassociation.size(); ++i) {
    if (dimPtr != dimensions.end() && *dimPtr == i) {
      for (auto el : reassociation[i]) {
        newDimensions.push_back(el);
      }
      dimPtr++;
    } else {
      newReassociation.push_back(reassociation[i]);
      for (auto el : reassociation[i]) {
        newOutputShape.push_back(collapseSourceShape[el]);
      }
    }
  }

  renumberReassociation(newReassociation);
}

void obtainHIVMDownDimDecReassocs(ArrayRef<int64_t> collapseSourceShape,
                                  ArrayRef<int64_t> dimensions,
                                  ArrayRef<ReassociationIndices> reassociation,
                                  SmallVector<int64_t> &newOutputShape,
                                  SmallVector<int64_t> &newDimensions) {
  LLVM_DEBUG(llvm::dbgs() << "Doing decreasing\n";);
  LLVM_DEBUG(llvm::dbgs() << to_string(reassociation) << "\n";);
  const auto *dimPtr = dimensions.begin();
  for (uint32_t i = 0; i < reassociation.size(); ++i) {
    if (dimPtr != dimensions.end() && *dimPtr == i) {
      for (auto el : reassociation[i]) {
        newDimensions.push_back(el);
        newOutputShape.push_back(1);
      }
      dimPtr++;
    } else {
      for (auto el : reassociation[i]) {
        newOutputShape.push_back(collapseSourceShape[el]);
      }
    }
  }
}

// Case #1 ------------------------------------------
//
// CollapsePropagating for Broadcast as input
//
// <AxBxCxDxExF> -> <AxBCxDEF>
// Collapse reassociation: [[0], [1, 2], [3, 4, 5]]
// Broadcast
//                0 1  2 3   4
// <AxBCxDEF> -> <AxBCxGxDEFxH>
// Dimensions = [2, 4]
// ----------------------
// Will be changed to
//                 0 1 2 3 4 5 6 7
// <AxBxCxDxE> -> <AxBxCxGxDxExFxH>
// newDimensions = [3, 7]
//
// Collapse
// <AxBxCxDxE> -> <AxBCxGxDEFxH>
//
//
// inserting dimensions at : [2, 4] (old one)
// New Dimensions can be get by: checking the replaced number of -1
//
//                   [[0], [1, 2], [-1], [3, 4, 5], [-1]]
//                              |
//                              | Renumbering and get newDimensions
//                              v
// newReassociation: [[0], [1, 2], [3], [4, 5, 6], [7]]
//
// %a = collapse_shape %arg0
// %b = broadcast(%a), outs(%c);
//
// |
// v
//
// %c_expand = expandOp(%c)                (new)
// %b = broadcast(%arg0), outs(%c_expand); (preserve)
// %c_replacer = collapse_shape %c_expand  (new replaces C)
// %b_replacer = collapse_shape %b         (new replaces B)
//
// Case #2 ------------------------------------------
//
// CollapsePropagating for Broadcast as output
// Basically the same as reducing
//
LogicalResult handleBroadcastOp(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter, Operation *userOp,
                                bool isAsInit) {
  auto broadcastOp = cast<linalg::BroadcastOp>(userOp);

  // Step 1: Acquire info
  // Get original dimensions and reassociation indices
  ArrayRef<int64_t> dimensions = broadcastOp.getDimensions();
  SmallVector<ReassociationIndices> reassociation =
      collapseOp.getReassociationIndices();

  auto collapseSourceShape = utils::getShape(collapseOp.getSrc().getType());

  SmallVector<ReassociationIndices> newReassociation;
  SmallVector<int64_t> newOutputShape;
  SmallVector<int64_t> newDimensions;
  LLVM_DEBUG(llvm::dbgs() << "Ok acquired here\n";);
  auto resultShape = utils::getShape(broadcastOp->getResult(0).getType());
  if (!isAsInit) {
    auto alterDimOpShape = utils::getShape(broadcastOp->getResult(0).getType());
    LLVM_DEBUG(llvm::dbgs() << "Checking here reassoc\n";);
    obtainDownDimIncReassocs(alterDimOpShape, collapseSourceShape, dimensions,
                             reassociation, newReassociation, newOutputShape,
                             newDimensions);
  } else {
    obtainDownDimDecReassocs(collapseSourceShape, dimensions, reassociation,
                             newReassociation, newOutputShape, newDimensions);
  }

  // Step 3: Changes and updates
  auto newOperands = collapsePropagateCreateNewOperands(
      rewriter, collapseOp, broadcastOp, newOutputShape, newReassociation,
      isAsInit);

#ifndef NDEBUG
  for (auto &op : newOperands) {
    LLVM_DEBUG(llvm::dbgs() << op << "\n";);
  }
#endif
  auto oldDpsInits = cast<DestinationStyleOpInterface>(userOp).getDpsInits();
  updateDefiningOp(userOp, rewriter, newOperands);

  updateDimensionalOp(broadcastOp, rewriter, newDimensions);
  auto resultReassociation = getResultReassociation(broadcastOp);
  assert(oldDpsInits.size() == 1);
  collapseAndReplace(rewriter, resultReassociation, resultShape,
                     userOp->getResult(0), userOp);
  collapseAndReplace(rewriter, resultReassociation, resultShape,
                     *oldDpsInits.begin(), userOp);
  return success();
}

LogicalResult handleHIVMBroadcastOp(tensor::CollapseShapeOp collapseOp,
                                    PatternRewriter &rewriter,
                                    Operation *userOp, bool isAsInit) {
  auto broadcastOp = cast<hivm::VBrcOp>(userOp);

  // Step 1: Acquire info
  // Get original dimensions and reassociation indices
  ArrayRef<int64_t> dimensions = broadcastOp.getBroadcastDims();
  SmallVector<ReassociationIndices> reassociation =
      collapseOp.getReassociationIndices();

  auto collapseSourceShape = utils::getShape(collapseOp.getSrc().getType());

  SmallVector<ReassociationIndices> newReassociation;
  SmallVector<int64_t> newOutputShape;
  SmallVector<int64_t> newDimensions;

  auto resultShape = utils::getShape(broadcastOp->getResult(0).getType());
  if (!isAsInit) {
    auto alterDimOpShape = utils::getShape(broadcastOp->getResult(0).getType());
    obtainHIVMDownDimIncReassocs(alterDimOpShape, collapseSourceShape,
                                 dimensions, reassociation, newOutputShape,
                                 newDimensions);
  } else {
    obtainHIVMDownDimDecReassocs(collapseSourceShape, dimensions, reassociation,
                                 newOutputShape, newDimensions);
  }

  // Step 3: Changes and updates
  auto newOperands = collapsePropagateCreateNewOperands(
      rewriter, collapseOp, broadcastOp, newOutputShape, reassociation,
      isAsInit);

  auto oldDpsInits = cast<DestinationStyleOpInterface>(userOp).getDpsInits();
  updateDefiningOp(userOp, rewriter, newOperands);

  updateHIVMDimensionalOp(broadcastOp, rewriter, newDimensions);
  auto resultReassociation = getResultReassociation(broadcastOp);
  assert(oldDpsInits.size() == 1);
  collapseAndReplace(rewriter, resultReassociation, resultShape,
                     userOp->getResult(0), userOp);
  collapseAndReplace(rewriter, resultReassociation, resultShape,
                     *oldDpsInits.begin(), userOp);
  return success();
}

// Collapse Propagating for Reduce
// <AxBxCxDxExFxGxHxI> -> <AxBCxDxEFGxHxI>
// Collapse reassociation: [[0], [1, 2], [3], [4, 5, 6], [7], [8]]
// Reduce
// <AxBCxDxEFGxHxI> -> <AxDxEFGxI>
// Dimensions = [1, 4]
//
// Will be changed to
//  0 1 2 3 4 5 6 7 8
// <AxBxCxDxExFxGxHxI> -> <AxDxExFxGxI>
// newDimensions = [1, 2, 7]
//
// <AxDxExFxGxI> -> <AxDxEFGxI>
//
// New Dimensions can be get by: checking the removed elements
//
//                   [[0], [3], [4, 5, 6], [8]]
//                              |
//                              | Renumbering
//                              v
// newReassociation: [[0], [1], [2, 3, 4], [5]]
//
// %a = collapse_shape %arg0
// %b = reduce(%a), outs(%c);
//
// |
// v
//
// %c_expand = expandOp(%c)                (new)
// %b = reduce(%arg0), outs(%c_expand); (preserve)
// %c_replacer = collapse_shape %c_expand  (new replaces C)
// %b_replacer = collapse_shape %b         (new replaces B)

template <typename OpType>
LogicalResult handleReduceLikeOp(tensor::CollapseShapeOp collapseOp,
                                 PatternRewriter &rewriter, Operation *userOp,
                                 bool isAsInit) {
  auto reduceOp = cast<OpType>(userOp);

  // Step 1: Acquire Info
  // Get original dimensions and reassociation indices
  ArrayRef<int64_t> dimensions = reduceOp.getDimensions();
  SmallVector<ReassociationIndices> reassociation =
      collapseOp.getReassociationIndices();

  // Step 2: Create new reassociation indices and dimensions
  SmallVector<ReassociationIndices> newReassociation;
  SmallVector<int64_t> newDimensions;
  SmallVector<int64_t> newOutputShape; // output: the expanded input of reduce

  auto collapseSourceShape = utils::getShape(collapseOp.getSrc().getType());

  auto resultShape = utils::getShape(reduceOp->getResult(0).getType());
  if (!isAsInit) {
    obtainDownDimDecReassocs(collapseSourceShape, dimensions, reassociation,
                             newReassociation, newOutputShape, newDimensions);
  } else {
    auto alterDimOpShape = utils::getShape(reduceOp.getInputs()[0].getType());
    obtainDownDimIncReassocs(alterDimOpShape, collapseSourceShape, dimensions,
                             reassociation, newReassociation, newOutputShape,
                             newDimensions);
  }

  LLVM_DEBUG(llvm::dbgs() << to_string(newReassociation) << "\n";);
  // Step 3: Change and update
  auto newOperands = collapsePropagateCreateNewOperands(
      rewriter, collapseOp, reduceOp, newOutputShape, newReassociation,
      isAsInit);

  auto oldDpsInits = cast<DestinationStyleOpInterface>(userOp).getDpsInits();
  updateDefiningOp(userOp, rewriter, newOperands);
  updateDimensionalOp(reduceOp, rewriter, newDimensions);

  auto resultReassociation = getResultReassociation(reduceOp);
  for (auto res : userOp->getResults()) {
    collapseAndReplace(rewriter, resultReassociation, resultShape, res, userOp);
  }
  for (auto oldInit : oldDpsInits) {
    collapseAndReplace(rewriter, resultReassociation, resultShape, oldInit,
                       userOp);
  }
  if constexpr (std::is_same_v<OpType, hfusion::ReduceWithIndexOp>) {
    if (reduceOp.getInputs().size() == 1) {
      // reduce_with_index without the index input (getInputs().size() == 1)
      // has a linalg::IndexOp operation inside the region,
      // whose dimension needs to be updated accordingly
      updateHFusionReduceWithIndexDim(rewriter, userOp, newDimensions);
    }
  }
  return success();
}

LogicalResult handleHIVMReduceOp(tensor::CollapseShapeOp collapseOp,
                                 PatternRewriter &rewriter, Operation *userOp,
                                 bool isAsInit) {
  auto reduceOp = cast<hivm::VReduceOp>(userOp);

  // Step 1: Acquire Info
  // Get original dimensions and reassociation indices
  ArrayRef<int64_t> dimensions = reduceOp.getReduceDims();
  SmallVector<ReassociationIndices> reassociation =
      collapseOp.getReassociationIndices();

  // Step 2: Create new reassociation indices and dimensions
  SmallVector<int64_t> newDimensions;
  SmallVector<int64_t> newOutputShape;

  auto collapseSourceShape = utils::getShape(collapseOp.getSrc().getType());

  auto resultShape = utils::getShape(reduceOp->getResult(0).getType());
  if (!isAsInit) {
    obtainHIVMDownDimDecReassocs(collapseSourceShape, dimensions, reassociation,
                                 newOutputShape, newDimensions);
  } else {
    auto alterDimOpShape =
        utils::getShape(reduceOp.getDpsInputs()[0].getType());
    obtainHIVMDownDimIncReassocs(alterDimOpShape, collapseSourceShape,
                                 dimensions, reassociation, newOutputShape,
                                 newDimensions);
  }

  // Step 3: Change and update
  auto newOperands = collapsePropagateCreateNewOperands(
      rewriter, collapseOp, reduceOp, newOutputShape, reassociation, isAsInit);

  auto oldDpsInits = cast<DestinationStyleOpInterface>(userOp).getDpsInits();
  updateDefiningOp(userOp, rewriter, newOperands);
  updateHIVMDimensionalOp(reduceOp, rewriter, newDimensions);

  auto resultReassociation = getResultReassociation(reduceOp);
  for (auto res : userOp->getResults()) {
    collapseAndReplace(rewriter, resultReassociation, resultShape, res, userOp);
  }
  for (auto oldInit : oldDpsInits) {
    collapseAndReplace(rewriter, resultReassociation, resultShape, oldInit,
                       userOp);
  }
  return success();
}

LogicalResult handleExtractOp(tensor::CollapseShapeOp collapseOp,
                              PatternRewriter &rewriter, Operation *userOp) {
  auto extractOp = dyn_cast<tensor::ExtractOp>(userOp);
  if (!extractOp)
    return failure();

  // Get collapsed indices and source tensor
  auto collapsedIndices = extractOp.getIndices();
  auto src = collapseOp.getSrc();
  auto srcShape = utils::getShape(src.getType());
  auto reassociation = collapseOp.getReassociationIndices();
  SmallVector<Value> expandedIndices;
  int collapsedIdx = 0;
  Location loc = extractOp.getLoc();
  rewriter.setInsertionPoint(extractOp);

  for (const auto &group : reassociation) {
    if (group.size() == 1) {
      expandedIndices.push_back(collapsedIndices[collapsedIdx]);
      collapsedIdx++;
      continue;
    }
    Value collapsedIndex = collapsedIndices[collapsedIdx];
    for (int i = static_cast<int>(group.size()) - 1; i > 0; --i) {
      int dimIdx = group[i];
      Value dimSize;
      if (srcShape[dimIdx] == ShapedType::kDynamic) {
        dimSize = rewriter.create<tensor::DimOp>(loc, src, dimIdx);
      } else {
        dimSize =
            rewriter.create<arith::ConstantIndexOp>(loc, srcShape[dimIdx]);
      }
      Value remainder =
          rewriter.create<arith::RemUIOp>(loc, collapsedIndex, dimSize);
      expandedIndices.push_back(remainder);
      collapsedIndex =
          rewriter.create<arith::DivUIOp>(loc, collapsedIndex, dimSize);
    }
    expandedIndices.push_back(collapsedIndex);
    std::reverse(expandedIndices.end() - group.size(), expandedIndices.end());
    collapsedIdx++;
  }
  if (llvm::all_of(srcShape, [](uint64_t dim) { return dim == 1; })) {
    if (expandedIndices.empty()) {
      expandedIndices.assign(srcShape.size(),
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
  }
  auto newExtract =
      rewriter.create<tensor::ExtractOp>(loc, src, expandedIndices);
  rewriter.replaceOp(extractOp, newExtract.getResult());
  return success();
}

LogicalResult handleConcatOp(tensor::CollapseShapeOp collapseOp,
                             PatternRewriter &rewriter, Operation *userOp) {
  auto reassociation = collapseOp.getReassociationIndices();
  auto concatOp = dyn_cast<tensor::ConcatOp>(userOp);
  uint64_t newConcatDim = 0;
  auto collapseSrc = collapseOp.getSrc();
  auto dimensionResult = utils::getShape(collapseSrc.getType());

  auto mixedDimensionResult =
      getMixedSizes(rewriter, collapseOp.getLoc(), collapseSrc);
  SmallVector<OpFoldResult> newExpandOutputShape;
  SmallVector<OpFoldResult> operandsNewDimSize;
  LogicalResult res = computeExpandConcat(
      rewriter, concatOp, reassociation, mixedDimensionResult, newConcatDim,
      newExpandOutputShape, operandsNewDimSize);
  // Return value used is newExpandOutputShape, and new concat dim
  if (res.failed())
    return failure();

  rewriter.setInsertionPointAfter(concatOp);
  // Need to expand all of the operands
  auto newConcatOp =
      buildNewConcat(rewriter, concatOp, reassociation, newConcatDim,
                     newExpandOutputShape, operandsNewDimSize);

  auto concatOpResultShape = utils::getShape(concatOp.getResult().getType());
  // concatOp is collapsed, newConcatOp is the expanded, so to replace the old
  // one, collapse the expanded
  collapseAndReplace(rewriter, reassociation,
                     /*Old concat result shape*/ concatOpResultShape,
                     concatOp.getResult(), newConcatOp.getResult(), userOp);
  rewriter.eraseOp(concatOp);
  return success();
}

// Supports dynamic dimension for padding
// Will ignore all ambiguous padding
LogicalResult handlePadOp(tensor::CollapseShapeOp collapseOp,
                          PatternRewriter &rewriter, Operation *userOp) {
  // There are 2 cases on this one
  // E.g collapse [A, B][C][D, E, F] -> G, H, I
  // Define the low and high pair as {Low, High}
  // Padding a single dimension works iff a single dimension pad is
  // - {0, 0}
  // - {?, ?}, and the reassoc for that only has one main dimension (ignore
  //   unit), like [1, 24, 1]
  auto reassociation = collapseOp.getReassociationIndices();
  // Check for case
  auto padOp = dyn_cast<tensor::PadOp>(userOp);
  auto collapseSrc = collapseOp.getSrc();
  // Rank and reassociation shall be the same
  assert(reassociation.size() == padOp.getSource().getType().getRank());
  // We need to expand them first, and then collapse them
  SmallVector<OpFoldResult> newPadLow;
  SmallVector<OpFoldResult> newPadHigh;
  SmallVector<OpFoldResult> newExpandOutputShape;
  // %a = Collapse %src
  // %b = pad %a
  auto dimensionResult = utils::getShape(collapseSrc.getType());
  assert(dimensionResult.size() == reassociation.back().back() + 1);
  auto oldExpandOutputShape =
      getMixedSizes(rewriter, collapseOp.getLoc(), collapseSrc);

  // Removing collapse is an easy thing because we can always do expand it back
  // to its normal version
  // insert an inverse expand, but for pad, everything is single source, so
  // %newB = pad %src

  DenseMap<uint64_t, uint64_t> padBodyMapping;

  auto result = computeExpandPad(rewriter, padOp, reassociation, padBodyMapping,
                                 newPadLow, newPadHigh, newExpandOutputShape,
                                 oldExpandOutputShape, dimensionResult);
  if (!result.succeeded())
    return failure();
  auto loc = padOp.getLoc();
  // Now create another padOp that uses the src
  auto tensorTy = cast<RankedTensorType>(collapseSrc.getType());
  auto newStaticLow = decomposeMixedValues(newPadLow).first;
  auto newStaticHigh = decomposeMixedValues(newPadHigh).first;
  Type padType =
      tensor::PadOp::inferResultType(tensorTy, newStaticLow, newStaticHigh);
  // Mapping and cloning

  LDBG("Creating pad Op");
  LDBG(to_string(newStaticHigh));
  LDBG(to_string(newStaticLow));
  auto newPadOp = rewriter.create<tensor::PadOp>(loc, padType, collapseSrc,
                                                 newPadLow, newPadHigh);

  clonePadRegion(rewriter, padOp, newPadOp, padBodyMapping);
  // collapse this newPadOp
  auto padOpResultShape = utils::getShape(padOp.getResult().getType());
  collapseAndReplace(rewriter, reassociation, padOpResultShape,
                     padOp.getResult(), newPadOp.getResult(), userOp);
  // padOp is now dead and not being used anywhere
  rewriter.eraseOp(padOp);
  return success();
}

LogicalResult handleExtractSliceOp(tensor::CollapseShapeOp collapseOp,
                                   PatternRewriter &rewriter,
                                   Operation *userOp) {
  auto reassociation = collapseOp.getReassociationIndices();
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(userOp);
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  SmallVector<OpFoldResult> dummyExpand;
  auto res = getExtractSliceModifyingOp(
      rewriter, cast<ExtractSliceOp>(userOp), reassociation,
      getMixedSizesOrOutputShape(rewriter, collapseOp.getSrc()),
      /*superview*/ false, newMixedOffsets, newMixedSizes, newMixedStrides,
      dummyExpand);
  if (res.failed())
    return failure();
  auto loc = userOp->getLoc();
  auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, collapseOp.getSrc(), newMixedOffsets, newMixedSizes,
      newMixedStrides);
  auto extractSliceOpResultShape =
      utils::getShape(extractSliceOp.getResult().getType());
  collapseAndReplace(rewriter, reassociation, extractSliceOpResultShape,
                     extractSliceOp.getResult(), newExtractSliceOp.getResult(),
                     userOp);
  rewriter.eraseOp(extractSliceOp);
  return success();
}

LogicalResult handleInsertSliceOp(tensor::CollapseShapeOp collapseOp,
                                  PatternRewriter &rewriter,
                                  Operation *userOp) {
  LDBG("Handle dropping insert slice here");
  auto reassociation = collapseOp.getReassociationIndices();
  auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(userOp);
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  SmallVector<OpFoldResult> expandSrcOutputShape;
  SmallVector<OpFoldResult> expandDestOutputShape;
  // will expand this both
  // Collapse <AxBxCxf32>
  //          <D  xCxf32>
  bool isSrcCollapsed = insertSliceOp.getSource() == collapseOp.getResult();
  // If the collapsed one is the source, it means it is a subview,
  // inserting [16] (source) -> [24] (dest)
  LDBG("What " << collapseOp.getSrc());
  auto res = getInsertSliceModifyingOp(
      rewriter, cast<InsertSliceOp>(userOp), reassociation,
      getMixedSizesOrOutputShape(rewriter, collapseOp.getSrc()), isSrcCollapsed,
      newMixedOffsets, newMixedSizes, newMixedStrides, expandSrcOutputShape,
      expandDestOutputShape);
  if (res.failed())
    return failure();

  auto loc = userOp->getLoc();
  // Gonna collapse the destination as well?
  // Expand result or expand source
  auto expandedNewSrc = createExpand(rewriter, loc, insertSliceOp.getSource(),
                                     reassociation, expandSrcOutputShape);
  auto expandedNewDest = createExpand(rewriter, loc, insertSliceOp.getDest(),
                                      reassociation, expandDestOutputShape);
  auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      loc, expandedNewSrc, expandedNewDest, newMixedOffsets, newMixedSizes,
      newMixedStrides);
  auto insertSliceOpResultShape =
      utils::getShape(insertSliceOp.getResult().getType());
  collapseAndReplace(rewriter, reassociation, insertSliceOpResultShape,
                     insertSliceOp.getResult(), newInsertSliceOp.getResult(),
                     userOp);
  rewriter.eraseOp(insertSliceOp);
  return success();
}
LogicalResult handleHIVMStoreOp(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter, Operation *userOp) {
  auto storeOp = cast<hivm::StoreOp>(userOp);
  auto resultRank = storeOp.getDstOperandType().getRank();
  SmallVector<Value> newOperands;
  auto loc = storeOp.getLoc();
  if (storeOp.hasPureTensorSemantics())
    return failure();
  for (Value operand : userOp->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand);
    // Check in case its scalar elemwise
    if (shapeRank.has_value() &&
        static_cast<int>(shapeRank.value()) == resultRank) {
      Operation *newExpandedOperand =
          createNewExpandOpFromCollapseOp(collapseOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newExpandedOperand << "\n";);
      newOperands.push_back(newExpandedOperand->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't expand inequal rank " << shapeRank
                              << " : " << resultRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  updateDefiningOp(userOp, rewriter, newOperands);
  return success();
}

} // namespace

LogicalResult
PropagateCollapseDown::matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                       PatternRewriter &rewriter) const {
  Value result = collapseOp.getResult();
  auto userRange = result.getUsers();
  SmallVector<Operation *> users(userRange.begin(), userRange.end());
  // Propagate one by one, to be safe
  auto *src = collapseOp.getSrc().getDefiningOp();
  if (!src || isStopPropagatable(src))
    return failure();
  if (options.forHIVM &&
      !isNonUnitExpandOrEmptyReassoc(collapseOp.getSrcType().getShape(),
                                     collapseOp.getReassociationIndices()))
    return failure();
  for (Operation *userOp : users) {
    LDBG(*userOp);
    if (collapseOp->getParentOp() != userOp->getParentOp())
      continue;
    if (isa<hivm::StoreOp>(userOp)) {
      return handleHIVMStoreOp(collapseOp, rewriter, userOp);
    }
    auto dsiOp = dyn_cast<DestinationStyleOpInterface>(userOp);
    if (dsiOp && !dsiOp.hasPureTensorSemantics()) {
      continue;
    }
    if (auto arange = dyn_cast<ArangeOp>(userOp)) {
      return handleArangeOp(collapseOp, rewriter, arange);
    }
    if (auto bitcastOp = dyn_cast<hivm::BitcastOp>(userOp)) {
      return handleBitcastOp(collapseOp, rewriter, bitcastOp);
    }
    if (isa<tensor::ConcatOp>(userOp)) {
      return handleConcatOp(collapseOp, rewriter, userOp);
    }
    if (isa<tensor::PadOp>(userOp)) {
      return handlePadOp(collapseOp, rewriter, userOp);
    }
    if (isa<linalg::BroadcastOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate collapse down - Broadcast\n";);
      return handleBroadcastOp(collapseOp, rewriter, userOp,
                               checkValueIsInit(userOp, result));
    }
    if (isa<hfusion::ReduceWithIndexOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Propagate collapse down - ReduceWithIndex\n";);
      return handleReduceLikeOp<hfusion::ReduceWithIndexOp>(
          collapseOp, rewriter, userOp, checkValueIsInit(userOp, result));
    }
    if (isa<linalg::ReduceOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate collapse down - Reduce\n";);
      return handleReduceLikeOp<linalg::ReduceOp>(
          collapseOp, rewriter, userOp, checkValueIsInit(userOp, result));
    }
    if (!options.forHIVM && isa<tensor::ExtractSliceOp>(userOp)) {
      return handleExtractSliceOp(collapseOp, rewriter, userOp);
    }
    if (!options.forHIVM && isa<tensor::InsertSliceOp>(userOp)) {
      return handleInsertSliceOp(collapseOp, rewriter, userOp);
    }
    if (isMarkedAsElementwiseOp(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate collapse down - Elemwise\n";);
      return handleElementwiseOp(collapseOp, rewriter, userOp);
    }
    if (isa<linalg::TransposeOp>(userOp)) {
      return handleTransposeOp<linalg::TransposeOp>(collapseOp, rewriter,
                                                    userOp);
    }
    if (isa<hivm::VTransposeOp>(userOp)) {
      // TODO: handle VTransposeOp to support more than one dimension transpose
      continue;
    }
    if (isa<hfusion::InterleaveOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate collapse down - Interleave\n";);
      return handleInterleaveOp(collapseOp, rewriter, userOp);
    }
    if (isa<hfusion::DeinterleaveOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate collapse down - Deinterleave\n";);
      return handleDeinterleaveOp(collapseOp, rewriter, userOp);
    }
    if (mlir::hivm::detail::isElemwiseNaryOpImpl(userOp) ||
        isa<hivm::CopyOp>(userOp) || isa<hivm::StoreOp>(userOp) ||
        isa<hivm::LoadOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate collapse down - HIVM Elemwise\n";);
      return handleElementwiseOp(collapseOp, rewriter, userOp);
    }
    if (isa<hivm::VBrcOp>(userOp)) {
      return handleHIVMBroadcastOp(collapseOp, rewriter, userOp,
                                   checkValueIsInit(userOp, result));
    }
    if (isa<hivm::VReduceOp>(userOp)) {
      return handleHIVMReduceOp(collapseOp, rewriter, userOp,
                                checkValueIsInit(userOp, result));
    }
    if (isa<tensor::ExtractOp>(userOp)) {
      return handleExtractOp(collapseOp, rewriter, userOp);
    }
  }
  return failure();
}

LogicalResult PropagateCollapseDownToI1Cast::matchAndRewrite(
    tensor::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const {
  Value result = collapseOp.getResult();

  if (!result.hasOneUse())
    return failure();

  Operation *userOp = *result.getUsers().begin();
  if (!userOp)
    return failure();

  if (collapseOp->getParentOp() != userOp->getParentOp())
    return failure();

  if (!mlir::isa<hfusion::CastOp>(userOp))
    return failure();

  auto srcTy = mlir::dyn_cast<RankedTensorType>(collapseOp.getSrc().getType());
  auto resTy =
      mlir::dyn_cast<RankedTensorType>(collapseOp.getResult().getType());
  if (!srcTy || !resTy)
    return failure();

  if (!srcTy.getElementType().isInteger(1) ||
      !resTy.getElementType().isInteger(1))
    return failure();

  auto *src = collapseOp.getSrc().getDefiningOp();
  if (!src || isStopPropagatable(src))
    return failure();

  LDBG(*userOp);
  LLVM_DEBUG(
      llvm::dbgs()
          << "Propagate collapse down, single use hfusion.cast on i1\n";);

  return handleElementwiseOp(collapseOp, rewriter, userOp);
}

} // namespace tensor
} // namespace mlir
