//===- Collapser.cpp ------------------------------------------------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Flattener/Flattener.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "flattener-collapser"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

using CollapseGroup = SmallVector<ReassociationIndices>;
// Collapse Group Utils

bool Flattener::hasCollapseGroup(Value res) const {
  return argumentsRefPointer_.count(res);
}

// given a value, if the collapse mapping exist
// Will return the collapse group:
// 1024x1024x1024
// ->
// 1048576x1024
//
// will be [[0, 1], [2]]
CollapseGroup Flattener::getCollapseGroup(Value res) const {
  assert(hasCollapseGroup(res) && "collapse group doesn't exist");
  const auto args = getArgumentRef(res);
  if (args.empty())
    return {};
  CollapseGroup indices;
  ReassociationIndices currentIndices;
  currentIndices.push_back(0);
  for (size_t i = 1; i < args.size(); ++i) {
    if (isConnected_[args[i - 1]].rightConnected &&
        isConnected_[args[i]].leftConnected) {
      currentIndices.push_back(i);
    } else {
      indices.push_back(currentIndices);
      currentIndices.clear();
      currentIndices.push_back(i);
    }
  }
  assert(!currentIndices.empty());
  indices.push_back(currentIndices);

  LLVM_DEBUG(for (auto a
                  : indices) {
    for (auto b : a) {
      llvm::dbgs() << b << " ";
    }
    llvm::dbgs() << "\n";
  });
  return indices;
}

// This function receive a value and return the expanded shape mixed size
// output shape conclusion from the solverShapeElem_
SmallVector<OpFoldResult> Flattener::getFlattenMixedSizes(Value res) const {
  OpBuilder builder(op_);
  if (!argumentsRefPointer_.count(res))
    return {};
  const auto args = getArgumentRef(res);
  if (args.empty())
    return {};
  SmallVector<OpFoldResult> outputShape;

  LLVM_DEBUG(llvm::dbgs() << "Getting mixed size " << res << "\n";);
  for (const auto &[idx, elem] : llvm::enumerate(args)) {
    auto [minParent, shape] = solverShapeElem_->getMinParentAndShapePair(elem);

    LLVM_DEBUG(llvm::dbgs() << minParent.first << " " << minParent.second << " "
                            << shape << "\n";);
    if (shape != ShapedType::kDynamic) {
      setInsertionPointBeforeValue(builder, res);
      outputShape.push_back(builder.getIndexAttr(shape));
    } else {
      // Get the minParent
      if (minParent.first < static_cast<int64_t>(argumentList_.size())) {
        auto &inferrableArg = argumentList_[minParent.first];
        LLVM_DEBUG(llvm::dbgs() << inferrableArg << "\n";);
        builder.setInsertionPointAfterValue(inferrableArg);

        outputShape.push_back(tensor::getMixedSize(
            builder, inferrableArg.getLoc(), inferrableArg, minParent.second));
      } else {
        builder.setInsertionPointAfterValue(res);
        outputShape.push_back(
            tensor::getMixedSize(builder, res.getLoc(), res, idx));
      }
    }
  }
  return outputShape;
}

void Flattener::adjustOperations() {
  OpBuilder builder(op_);
  LLVM_DEBUG(llvm::dbgs() << "Adjusting block arguments\n");
  for (auto &arg : argumentList_) {
    collapserForArg(arg, builder);
  }
  LLVM_DEBUG(dumpModuleOP(););

  LLVM_DEBUG(llvm::dbgs() << "Adjusting other operations\n");
  flattenerWorkList.clear();
  op_->walk<WalkOrder::PostOrder>(
      [&](Operation *op) { flattenerWorkList.insert(op); });
  while (!flattenerWorkList.empty()) {
    Operation *op = flattenerWorkList.front();
    flattenerWorkList.erase(flattenerWorkList.begin());
    if (!op || isSkippableOp(op))
      continue;
    // Whitelist check
    bool allowed = isExplicitlyAllowedCollapseOp(op);
    if (!allowed) {
      if (op->getNumResults() == 0)
        continue;
      const auto result = op->getResult(0);
      if (!hasCollapseGroup(result)) {
        continue;
      }
    }
    // Check if empty collapse group result
    LLVM_DEBUG(llvm::dbgs() << "Checking " << *op << "\n";);
    [[maybe_unused]] auto collapseResult = collapser(op, builder);
    assert(succeeded(collapseResult));
  }

  LLVM_DEBUG(dumpModuleOP(););
  // FLATTEN-OUT
  LLVM_DEBUG(llvm::dbgs() << "Expanding return values\n");

  for (Operation *op : outList_) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      adjustReturnOp(returnOp, builder);
    } else if (auto materializeOp =
                   dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      adjustTensorOutOpSource<bufferization::MaterializeInDestinationOp>(
          materializeOp, builder);
    } else if (auto reshapeOp = dyn_cast<tensor::ReshapeOp>(op)) {
      adjustTensorOutOpSource<tensor::ReshapeOp>(reshapeOp, builder);
    } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      adjustTensorOutOpSrc<tensor::ExpandShapeOp>(expandShapeOp, builder);
    } else if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      adjustTensorOutOpSrc<tensor::CollapseShapeOp>(collapseShapeOp, builder);
    }
    LLVM_DEBUG(dumpModuleOP(););
  };
}

// Adjust block arguments
void Flattener::collapserForArg(Value &arg, OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "\nTrying " << arg << "\n";);
  CollapseGroup collapseGroup = getCollapseGroup(arg);
  assert(previousType_.count(arg));
  auto argType = previousType_[arg];

  LLVM_DEBUG(llvm::dbgs() << argType << "\n";);
  if (utils::getShapeRank(argType).value_or(0) == 0 || collapseGroup.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Collapse group for " << arg << " is scalar\n");
    return;
  }

  // No collapse needed
  if (collapseGroup.size() == utils::getShapeRank(argType).value_or(0)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "No collapse needed because rank is the same\n";);
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Previous type " << argType << "\n";);
  auto collapsedType =
      tensor::CollapseShapeOp::inferCollapsedType(argType, collapseGroup);

  builder.setInsertionPointAfterValue(arg);

  LLVM_DEBUG(llvm::dbgs() << "Collapsing arg " << arg << "\n";);

  tensor::CollapseShapeOp collapseOp = builder.create<tensor::CollapseShapeOp>(
      arg.getLoc(), collapsedType, arg, collapseGroup);

  LLVM_DEBUG(llvm::dbgs() << "into " << collapseOp << "\n\n";);
  updatePreviousType(collapseOp.getResult(), argType);
  collapsePropagateOrVerify(collapseOp.getResult(), arg);
  LLVM_DEBUG(llvm::dbgs() << "Will replace " << arg << " with "
                          << collapseOp.getResult() << "\n";);
  arg.replaceUsesWithIf(collapseOp.getResult(), [&](OpOperand &use) {
    Operation *defOp = use.getOwner();
    if (!defOp)
      return true;
    LLVM_DEBUG(llvm::dbgs() << "checking *defOp: " << *defOp << "\n";);
    if (defOp == collapseOp)
      return false;
    if (isa<tensor::DimOp>(defOp))
      return false;
    return true;
  });
}

template <class T>
SmallVector<int64_t>
Flattener::adjustCollapseDimensions(T op, CollapseGroup indices) const {
  SmallVector<int64_t> newDimensions;
  int newDimPtr = 0;
  //             0          1       2    3
  // e.g: ref = [[0, 1, 2], [3, 4], [5], [6, 7, 8]]

  // if dimension is 3, 4, 5
  // it will be [1, 2]

  LLVM_DEBUG(llvm::dbgs() << "\nGetting new dimensions: ";);
  for (const auto dim : op.getDimensions()) {
    while (indices[newDimPtr].back() < dim)
      newDimPtr++;
    if (newDimensions.empty() || newDimensions.back() != newDimPtr) {
      LLVM_DEBUG(llvm::dbgs() << newDimPtr << " ";);
      newDimensions.push_back(newDimPtr);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\n";);
  return newDimensions;
}

void Flattener::adjustExtractOpIndices(tensor::ExtractOp extractOp,
                                       OpBuilder &builder) {
  auto extractIndices = extractOp.getIndices();
  auto src = extractOp.getTensor();
  auto collapseGroups = getCollapseGroup(src);

  SmallVector<OpFoldResult> oldMixedSize = getFlattenMixedSizes(src);
  builder.setInsertionPoint(extractOp);
  Location loc = extractOp.getLoc();
  // Define getDimSize lambda function.
  auto getDimSize = [&](int idx) -> Value {
    auto currentIntValue = getConstantIntValue(oldMixedSize[idx]);
    if (currentIntValue.has_value())
      return builder.create<arith::ConstantIndexOp>(loc,
                                                    currentIntValue.value());
    return cast<Value>(oldMixedSize[idx]);
  };

  // Compute new indices using the utility function.
  auto newIndices = computeExtractCollapsedIndices(
      collapseGroups, extractIndices, getDimSize, builder, loc);

  // Prepare the new operands.
  SmallVector<Value> extractOpNewOperands;
  extractOpNewOperands.reserve(newIndices.size() + 1);
  extractOpNewOperands.push_back(src);
  extractOpNewOperands.append(newIndices);

  updatePreviousType(extractOp.getResult());
  builder.setInsertionPoint(extractOp);
  extractOp->setOperands(extractOpNewOperands);
  LLVM_DEBUG(llvm::dbgs() << "Ok extractOp done";);
  LLVM_DEBUG(llvm::dbgs() << *extractOp->getParentOfType<func::FuncOp>(););
}

void Flattener::adjustPadOp(tensor::PadOp padOp, OpBuilder &builder) {
  auto src = padOp.getSource();
  auto collapseGroups = getCollapseGroup(src);
  auto &refPtr = argumentsRef_[argumentsRefPointer_[src]];
  auto paddedDim = padOp.getPaddedDims();
  auto staticLowPad = padOp.getStaticLow();
  auto staticHighPad = padOp.getStaticHigh();
  auto mixLowPad = padOp.getMixedLowPad();
  auto mixHighPad = padOp.getMixedHighPad();
  // sum would do, if sum wouldn't do, then collapse is invalid
  SmallVector<OpFoldResult> newMixLowPad;
  SmallVector<OpFoldResult> newMixHighPad;
  // get which one to collapse together
  LLVM_DEBUG(llvm::dbgs() << *padOp->getParentOp(););
  DenseMap<uint64_t, uint64_t> padBodyMapping;
  for (unsigned i = 0; i < collapseGroups.size(); i++) {
    int dimPushed = 0;
    for (auto idx : collapseGroups[i]) {
      padBodyMapping[idx] = i;
      if (isConnected_[refPtr[idx]].elementKind == ElementKind::HasMutation) {
        dimPushed++;
        newMixLowPad.push_back(mixLowPad[idx]);
        newMixHighPad.push_back(mixHighPad[idx]);
      }
    }
    if (dimPushed == 0) {
      dimPushed++;
      newMixLowPad.push_back(builder.getI64IntegerAttr(0));
      newMixHighPad.push_back(builder.getI64IntegerAttr(0));
    }
    assert(dimPushed == 1);
  }
  builder.setInsertionPointAfter(padOp);
  Type padType = tensor::PadOp::inferResultType(src.getType(), staticLowPad,
                                                staticHighPad);
  auto newPadOp = builder.create<tensor::PadOp>(padOp.getLoc(), padType, src,
                                                newMixLowPad, newMixHighPad);

  tensor::reshape_utils::clonePadRegion(builder, padOp, newPadOp,
                                        padBodyMapping);
  collapsePropagateOrVerify(newPadOp.getResult(), padOp.getResult());
  LLVM_DEBUG(llvm::dbgs() << "Setting pre type to "
                          << padOp.getResult().getType() << "\n";);
  updatePreviousType(newPadOp.getResult(), padOp.getResult().getType());
  padOp->replaceAllUsesWith(newPadOp);
  eraseOp(padOp);
  LLVM_DEBUG(llvm::dbgs() << "Okay all here " << *newPadOp->getParentOp()
                          << "\n";);
}

void Flattener::adjustGatherOp(hfusion::GatherOp gatherOp, OpBuilder &builder) {
  auto collapseGroups = getCollapseGroup(gatherOp.getODSOperands(0)[0]);
  auto tmpAxis = gatherOp.getAxis();
  int64_t newAxis = 0;
  for (size_t i = 0; i < collapseGroups.size(); ++i) {
    if (tmpAxis < collapseGroups[i].size()) {
      newAxis = static_cast<int64_t>(i);
      break;
    }
    tmpAxis -= collapseGroups[i].size();
  }
  builder.setInsertionPointAfter(gatherOp);
  Location loc = gatherOp.getLoc();
  Value src = gatherOp.getSrc();
  Value idx = gatherOp.getIndex();
  Value init = gatherOp.getInit();
  auto newGatherOp = builder.create<GatherOp>(loc, src, idx, init, newAxis);
  auto resVariadic = gatherOp.getResult();
  if (!resVariadic.empty()) {
    auto resType = cast<TypedValue<RankedTensorType>>(*resVariadic.begin());
    auto newResType =
        cast<TypedValue<RankedTensorType>>(*newGatherOp.getResult().begin());
    collapsePropagateOrVerify(newResType, resType);
    updatePreviousType(newResType, resType.getType());
  }
  gatherOp.replaceAllUsesWith(newGatherOp);
  eraseOp(gatherOp);
}

void Flattener::adjustConcatOp(tensor::ConcatOp concatOp) {
  auto collapseGroups = getCollapseGroup(concatOp.getInputs()[0]);
  auto tmpDim = concatOp.getDim();
  int64_t newDim = 0;
  for (size_t i = 0; i < collapseGroups.size(); ++i) {
    if (tmpDim < collapseGroups[i].size()) {
      newDim = static_cast<int64_t>(i);
      break;
    }
    tmpDim -= collapseGroups[i].size();
  }
  concatOp.setDim(newDim);
  updatePreviousType(concatOp.getResult());
  auto resType = concatOp.inferResultType(newDim, concatOp->getOperandTypes());
  concatOp.getResult().setType(resType);
}

void Flattener::adjustInterleaveOp(hfusion::InterleaveOp interleaveOp) {
  updatePreviousType(interleaveOp.getResult());
  auto input0Type = interleaveOp.getInput()[0].getType();
  auto staticShape = utils::getShape(input0Type);
  if (!ShapedType::isDynamic(staticShape.back())) {
    staticShape.back() *= interleaveOp.getInterLeaveChannelNums();
  }
  interleaveOp.getResult().setType(
      RankedTensorType::get(staticShape, getElementTypeOrSelf(input0Type)));
}

void Flattener::adjustDeinterleaveOp(hfusion::DeinterleaveOp deinterleaveOp) {
  auto inputType = deinterleaveOp.getInput().getType();
  auto staticShape = utils::getShape(inputType);
  if (!ShapedType::isDynamic(staticShape.back())) {
    staticShape.back() /= deinterleaveOp.getDeInterLeaveChannelNum();
  }
  for (auto res : deinterleaveOp->getResults()) {
    updatePreviousType(res);
    res.setType(
        RankedTensorType::get(staticShape, getElementTypeOrSelf(inputType)));
  }
}

LogicalResult Flattener::VerifyCollapsedOperand(Operation *op) const {
  for (Value operand : op->getOperands()) {
    if (!argumentsRefPointer_.contains(operand)) {
      if (utils::getShapeRank(operand).value_or(0) == 0) {
        continue;
      }
      llvm::errs() << "Error: Not all operands are collapsed for op: " << *op
                   << "\n";
      llvm::errs() << operand << "\n";
      return failure();
    }
  }
  return success();
}

void Flattener::adjustResultType(DestinationStyleOpInterface dpsLikeOp) {
  for (unsigned i = 0; i < dpsLikeOp->getNumResults(); ++i) {
    // Get the collapsed type from the corresponding init operand
    auto collapsedType = dpsLikeOp.getDpsInitOperand(i)->get().getType();

    // Modify the existing operation's result type
    auto currentRes = dpsLikeOp->getResult(i);
    // assign for collapsed
    updatePreviousType(currentRes);
    currentRes.setType(collapsedType);
  }
}

void Flattener::adjustBroadcastOp(linalg::BroadcastOp broadcastOp,
                                  OpBuilder &builder) {
  auto newDimensions = adjustCollapseDimensions(
      broadcastOp, getCollapseGroup(broadcastOp->getResult(0)));
  broadcastOp.setDimensionsAttr(builder.getDenseI64ArrayAttr(newDimensions));
}

void Flattener::adjustTransposeOp(linalg::TransposeOp transposeOp,
                                  OpBuilder &builder) const {
  LLVM_DEBUG(llvm::dbgs() << "Adjusting transpose \n";);
  auto oldPerm = transposeOp.getPermutation();
  auto mapping = utils::getReassociationMapping(
      getCollapseGroup(transposeOp->getResult(0)));
  auto newPerm = utils::getNewIndexingFullPermutation(oldPerm, mapping);
  transposeOp.setPermutationAttr(builder.getDenseI64ArrayAttr(newPerm));
}

template <class T>
void Flattener::adjustReduceLikeOpBody(T reduceOp) const {
  auto collapseGroup = getCollapseGroup(reduceOp.getDpsInputs()[0]);
  auto newDimensions = adjustCollapseDimensions(reduceOp, collapseGroup);
  reduceOp.setDimensionsAttr(
      DenseI64ArrayAttr::get(reduceOp.getContext(), newDimensions));
  DenseMap<uint64_t, uint64_t> targetLinalgIndex;
  for (size_t i = 0; i < collapseGroup.size(); ++i) {
    for (auto idx : collapseGroup[i]) {
      targetLinalgIndex[idx] = i;
      LDBG(idx << " " << i);
    }
  }
  reduceOp.walk([&](linalg::IndexOp indexOp) {
    const auto accessedIdx = indexOp.getDim();
    indexOp.setDim(targetLinalgIndex[accessedIdx]);
  });
}

template <class T>
void Flattener::adjustCumOp(T cumOp, OpBuilder &builder) {
  auto collapseGroups = getCollapseGroup(cumOp.getODSOperands(0)[0]);
  int64_t tmpCumDim = cumOp.getCumDims()[0];
  int64_t newCumDim = 0;

  // Given group reassociation, find the new cumulative dimension by tracking
  // the collapsed groups
  // For example
  //  0    1          2    3
  // [[A], [B, C, D], [E], [F, G]]
  // if the old dimension is 4
  // then the new Dimension is 2
  // Take a look at adjustCollapseDimensions in the future if cumDim is more
  // than 1
  for (size_t i = 0; i < collapseGroups.size(); ++i) {
    auto groupSize = static_cast<int64_t>(collapseGroups[i].size());
    if (tmpCumDim < groupSize) {
      newCumDim = static_cast<int64_t>(i);
      break;
    }
    tmpCumDim -= groupSize;
  }

  // change new cum dims
  llvm::SmallVector<int64_t> newCumDims = {newCumDim};
  cumOp.setCumDims(newCumDims);

  // the output type should be the same with input
  auto inputTy = cumOp.getInput().getType();
  auto res = cumOp.getResult();
  updatePreviousType(res);
  res.setType(inputTy);
}

template <class T>
void Flattener::computeNewSlicingOperands(
    T slicingOp, SmallVector<OpFoldResult> &newMixedOffsets,
    SmallVector<OpFoldResult> &newMixedSizes,
    SmallVector<OpFoldResult> &newMixedStrides, OpBuilder &builder) {
  LDBG(*slicingOp.getOperation()->getParentOp());
  OpBuilder::InsertionGuard guard(builder);
  auto src = slicingOp.getSource();
  auto &refPtr = argumentsRef_[argumentsRefPointer_[src]];
  auto collapseGroups = getCollapseGroup(src);
  auto loc = slicingOp.getLoc();
  auto mixedOffsets = slicingOp.getMixedOffsets();
  auto mixedSizes = slicingOp.getMixedSizes();
  auto mixedStrides = slicingOp.getMixedStrides();
  for (auto &collapseGroup : collapseGroups) {
    LDBG("Computing collapse: " << to_string(collapseGroup));
    int dimPushed = 0;
    for (auto idx : collapseGroup) {
      if (isConnected_[refPtr[idx]].elementKind == ElementKind::HasMutation) {
        dimPushed++;
        newMixedOffsets.push_back(mixedOffsets[idx]);
        newMixedSizes.push_back(mixedSizes[idx]);
        newMixedStrides.push_back(mixedStrides[idx]);
      }
    }
    if (dimPushed == 0) {
      dimPushed++;
      newMixedOffsets.push_back(builder.getI64IntegerAttr(0));
      newMixedStrides.push_back(builder.getI64IntegerAttr(1));
      builder.setInsertionPointAfter(slicingOp);

      // Multiply all the extract slice here, dynamic support
      int64_t realVal = 1;
      bool isStatic = true;
      for (auto idx : collapseGroup) {
        auto sizeInt = getConstantIntValue(mixedSizes[idx]);
        if (!sizeInt.has_value()) {
          isStatic = false;
          continue;
        }
        realVal *= sizeInt.value();
      }
      if (!isStatic) {
        builder.setInsertionPointToStart(slicingOp.getOperation()->getBlock());
        Value multiplier = builder.create<arith::ConstantIndexOp>(loc, realVal);
        SmallVector<Value> valueMixed;
        for (auto idx : collapseGroup) {
          if (mixedSizes[idx].template is<Value>()) {
            valueMixed.push_back(mixedSizes[idx].template get<Value>());
            // To avoid this kind of dependency issue:
            // %accumulator
            // %value2 = ...
            // %mult2 = mult %mult1, % value2
            // %value1 = ...
            // %mult1 = mult %accumulator, %value1
          }
        }
        std::sort(valueMixed.begin(), valueMixed.end(),
                  [](const Value &valueA, const Value &valueB) -> bool {
                    if (isa<BlockArgument>(valueB))
                      return false;
                    if (isa<BlockArgument>(valueA))
                      return true;
                    return (valueA.getDefiningOp()->isBeforeInBlock(
                        valueB.getDefiningOp()));
                  });
        for (auto val : valueMixed) {
          builder.setInsertionPointAfterValue(val);
          multiplier = builder.create<arith::MulIOp>(loc, multiplier, val);
        }
        newMixedSizes.push_back(multiplier);
      } else {
        newMixedSizes.push_back(builder.getI64IntegerAttr(realVal));
      }
    }
    assert(dimPushed == 1);
  }
}

void Flattener::adjustExtractSliceOp(tensor::ExtractSliceOp extractSliceOp,
                                     mlir::OpBuilder &builder) {
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  computeNewSlicingOperands(extractSliceOp, newMixedOffsets, newMixedSizes,
                            newMixedStrides, builder);
  // get which one to collapse together
  builder.setInsertionPoint(extractSliceOp);
  auto newExtractSliceOp = builder.create<tensor::ExtractSliceOp>(
      extractSliceOp.getLoc(), extractSliceOp.getSource(), newMixedOffsets,
      newMixedSizes, newMixedStrides);

  collapsePropagateOrVerify(newExtractSliceOp.getResult(),
                            extractSliceOp.getResult());
  updatePreviousType(newExtractSliceOp.getResult(),
                     extractSliceOp.getResult().getType());
  extractSliceOp->replaceAllUsesWith(newExtractSliceOp);
  eraseOp(extractSliceOp);
}

void Flattener::adjustInsertSliceOp(tensor::InsertSliceOp insertSliceOp,
                                    mlir::OpBuilder &builder) {
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  computeNewSlicingOperands(insertSliceOp, newMixedOffsets, newMixedSizes,
                            newMixedStrides, builder);
  builder.setInsertionPoint(insertSliceOp);
  auto newInsertSliceOp = builder.create<tensor::InsertSliceOp>(
      insertSliceOp.getLoc(), insertSliceOp.getSource(),
      insertSliceOp.getDest(), newMixedOffsets, newMixedSizes, newMixedStrides);

  collapsePropagateOrVerify(newInsertSliceOp.getResult(),
                            insertSliceOp.getResult());
  updatePreviousType(newInsertSliceOp.getResult(),
                     insertSliceOp.getResult().getType());
  insertSliceOp->replaceAllUsesWith(newInsertSliceOp);
  eraseOp(insertSliceOp);
}

LogicalResult Flattener::collapser(Operation *op, OpBuilder &builder) {
  // Check if the operation is skippable
  if (isSkippableOp(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Skipping OP " << *op << "\n";);
    return success();
  }

  // Check if all operands are collapsed before
  if (!isa<linalg::FillOp>(op) && failed(VerifyCollapsedOperand(op)))
    return failure();

  // This is manual check for non linalg operations, dont forget to set the
  // previous types
  if (auto cumsumOp = dyn_cast<hfusion::CumsumOp>(op)) {
    adjustCumOp(cumsumOp, builder);
    return success();
  }

  if (auto cumprodOp = dyn_cast<hfusion::CumprodOp>(op)) {
    adjustCumOp(cumprodOp, builder);
    return success();
  }

  if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
    adjustPadOp(padOp, builder);
    return success();
  }

  if (auto gatherOp = dyn_cast<hfusion::GatherOp>(op)) {
    adjustGatherOp(gatherOp, builder);
    return success();
  }

  if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
    adjustConcatOp(concatOp);
    return success();
  }

  if (auto interleaveOp = dyn_cast<hfusion::InterleaveOp>(op)) {
    adjustInterleaveOp(interleaveOp);
    return success();
  }

  if (auto deinterleaveOp = dyn_cast<hfusion::DeinterleaveOp>(op)) {
    adjustDeinterleaveOp(deinterleaveOp);
    return success();
  }

  if (auto tensorExtractOp = dyn_cast<tensor::ExtractOp>(op)) {
    adjustExtractOpIndices(tensorExtractOp, builder);
    return success();
  }

  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    adjustExtractSliceOp(extractSliceOp, builder);
    return success();
  }

  if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    adjustInsertSliceOp(insertSliceOp, builder);
    return success();
  }

  linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // If the operation is not a linalg op, we can skip the rest of the checks
  if (!linalgOp)
    return success();

  // Check if op has only one init operand (for linalg ops)
  if (op->getNumResults() != linalgOp.getNumDpsInits()) {
    llvm::errs()
        << "Warning: Op should have exactly same number of result as init "
        << *op << "\n";
    return failure();
  }

  // Check if the operation is "legal" (i.e., supported for collapsing)
  if (!isLegalOp(op)) {
    llvm::errs() << "Warning: Unsupported operation for collapsing: " << *op
                 << "\n";
    return failure();
  }

  // Get the collapsed types from the init operands and modify the existing
  // operation's result types
  if (auto dpsLikeOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    adjustResultType(dpsLikeOp);
  }

  // Handle Reduce and Broadcast ops
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    adjustReduceLikeOpBody(reduceOp);
  } else if (auto reduceWithIndexOp =
                 dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    adjustReduceLikeOpBody(reduceWithIndexOp);
  } else if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    adjustBroadcastOp(broadcastOp, builder);
  } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    adjustTransposeOp(transposeOp, builder);
  }
  return success();
}

void Flattener::adjustReturnOp(Operation *op, OpBuilder &builder) const {
  SmallVector<Value> newOperands;
  bool needsUpdate = false;
  const auto &funcResults =
      op->getParentOfType<func::FuncOp>().getFunctionType().getResults();

  for (const auto &[idx, operand] : llvm::enumerate(op->getOperands())) {
    if (!argumentsRefPointer_.contains(operand)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Operand is not expandable " << operand << "\n";);
      newOperands.push_back(operand);
      continue;
    }
    CollapseGroup collapseGroup = getCollapseGroup(operand);
    if (!collapseGroup.empty() &&
        collapseGroup.size() <
            utils::getShapeRank(funcResults[idx]).value_or(0)) {
      builder.setInsertionPoint(op);
      // Use the function's return type instead of computing it
      auto expandedType = cast<RankedTensorType>(funcResults[idx]);
      auto mixedSize = getFlattenMixedSizes(operand);
      auto expandOp = builder.create<tensor::ExpandShapeOp>(
          op->getLoc(), expandedType, operand, collapseGroup, mixedSize);
      newOperands.push_back(expandOp.getResult());
      needsUpdate = true;
    } else {
      newOperands.push_back(operand);
    }
  }
  if (needsUpdate) {
    op->setOperands(newOperands);
  }
}

template <typename OpTy>
FailureOr<tensor::ExpandShapeOp> Flattener::collapseForOut(OpTy &tensorOutOp,
                                                           Value &collapsedVal,
                                                           OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "\nCollapsing " << collapsedVal << "\n";);
  if (!previousType_.contains(collapsedVal)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Source is not expandable, previousType_ not found"
               << "\n");
    return failure();
  }

  CollapseGroup collapseGroup = getCollapseGroup(collapsedVal);
  if (collapseGroup.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Collapse group invalid"
                            << "\n");
    return failure();
  }
  auto expandedType = cast<RankedTensorType>(previousType_[collapsedVal]);
  builder.setInsertionPoint(tensorOutOp);

  auto mixedSize = getFlattenMixedSizes(collapsedVal);
  auto expandOp = builder.create<tensor::ExpandShapeOp>(
      tensorOutOp.getLoc(), expandedType, collapsedVal, collapseGroup,
      mixedSize);
  return expandOp;
}

// TODO: Check whether they have common interfaces
template <typename OpTy>
void Flattener::adjustTensorOutOpSource(OpTy tensorOutOp, OpBuilder &builder) {
  Value source = tensorOutOp.getSource();
  if (!utils::getShapeRank(source).value_or(0))
    return;
  auto expandOp = collapseForOut(tensorOutOp, source, builder);
  if (succeeded(expandOp)) {
    tensorOutOp.getSourceMutable().assign(expandOp.value().getResult());
    return;
  }
  llvm::report_fatal_error("Expand op collapse failed");
}

template <typename OpTy>
void Flattener::adjustTensorOutOpDest(OpTy tensorOutOp, OpBuilder &builder) {
  Value dest = tensorOutOp.getDest();
  if (!utils::getShapeRank(dest).value_or(0))
    return;
  auto expandOp = collapseForOut(tensorOutOp, dest, builder);
  if (!failed(expandOp)) {
    tensorOutOp.getDestMutable().assign(expandOp.value().getResult());
    return;
  }
  llvm::report_fatal_error("Expand op collapse failed");
}

template <typename OpTy>
void Flattener::adjustTensorOutOpSrc(OpTy tensorOutOp, OpBuilder &builder) {
  Value source = tensorOutOp.getSrc();
  if (!utils::getShapeRank(source).value_or(0))
    return;
  auto expandOp = collapseForOut(tensorOutOp, source, builder);
  if (!failed(expandOp)) {
    tensorOutOp.getSrcMutable().assign(expandOp.value().getResult());
    return;
  }
  llvm::report_fatal_error("Expand op collapse failed");
}

void Flattener::eraseOp(mlir::Operation *op) {
  const auto *pos = llvm::find(flattenerWorkList, op);
  if (pos != flattenerWorkList.end()) {
    flattenerWorkList.erase(pos);
  }
  for (auto res : op->getResults()) {
    argumentsRefPointer_.erase(res);
    previousType_.erase(res);
  }
  op->erase();
}
} // namespace detail
} // namespace hfusion
} // namespace mlir
