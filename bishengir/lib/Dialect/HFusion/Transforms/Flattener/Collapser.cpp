//===- Collapser.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Flattener/Flattener.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
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

  LDBG("[Mixed size] " << res);
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
        LDBG("[Inferrable arg] " << inferrableArg);
        builder.setInsertionPointAfterValue(inferrableArg);
        if (isa<RankedTensorType>(inferrableArg.getType())) {
          outputShape.push_back(
              tensor::getMixedSize(builder, inferrableArg.getLoc(),
                                   inferrableArg, minParent.second));
        } else {
          assert(isa<MemRefType>(inferrableArg.getType()));
          outputShape.push_back(
              memref::getMixedSize(builder, inferrableArg.getLoc(),
                                   inferrableArg, minParent.second));
        }
      } else {
        assert(valueReplacement.count(res));
        OpResult replacementValue = cast<OpResult>(valueReplacement.at(res));
        if (auto subviewOp =
                replacementValue.getDefiningOp<memref::SubViewOp>()) {
          return subviewOp.getMixedSizes();
        }
        LDBG("[Parent not part of args] " << res << " " << idx);
        ReifiedRankedShapedTypeDims reifiedDims;
        auto reifyRes = reifyResultShapes(builder, replacementValue.getOwner(),
                                          reifiedDims);
        if (failed(reifyRes)) {
          report_fatal_error("Cant get shape of mixed size");
        }
        auto reifiedDim = reifiedDims[replacementValue.getResultNumber()][idx];
        outputShape.push_back(reifiedDim);
      }
    }
  }
  return outputShape;
}

void Flattener::collectOperations(Operation *op,
                                  SetVector<Operation *> &operations) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (!ifOp.getThenRegion().empty()) {
      for (Operation &blockOp : *ifOp.thenBlock()) {
        collectOperations(&blockOp, operations);
      }
    }
    if (ifOp.elseBlock()) {
      for (Operation &blockOp : *ifOp.elseBlock()) {
        collectOperations(&blockOp, operations);
      }
    }
    operations.insert(ifOp);
    return;
  }

  // default: pre-order
  operations.insert(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &blockOp : block) {
        collectOperations(&blockOp, operations);
      }
    }
  }
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
  collectOperations(op_, flattenerWorkList);
  while (!flattenerWorkList.empty()) {
    Operation *op = flattenerWorkList.front();
    LDBG("[Iterating collapser] " << *op);
    flattenerWorkList.erase(flattenerWorkList.begin());
    if (!op || isHeadOperation(op) || isTailOperation(op) ||
        isUnsupportedOp(op))
      continue;
    // Whitelist check
    bool allowed = isExplicitlyAllowedCollapseOp(op);
    // Allowed for regbase only
    allowed |= options.registerBased &&
               isa_and_present<memref::StoreOp, memref::LoadOp>(op);
    if (!allowed) {
      if (op->getNumResults() == 0)
        continue;
      const auto result = op->getResult(0);
      if (!hasCollapseGroup(result)) {
        continue;
      }
    }
    // Check if empty collapse group result
    LDBG("[Iterating collapser] " << *op);
    [[maybe_unused]] auto collapseResult = collapser(op, builder);
    assert(succeeded(collapseResult));
  }

  LLVM_DEBUG(dumpModuleOP(););
  // FLATTEN-OUT
  LLVM_DEBUG(llvm::dbgs() << "Expanding return values\n");

  for (Operation *op : outList_) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      adjustReturnOp(returnOp, builder);
    } else if (auto reshapeOp = dyn_cast<tensor::ReshapeOp>(op)) {
      adjustTensorOutOpSource<tensor::ReshapeOp>(reshapeOp, builder);
    } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      adjustTensorOutOpSrc<tensor::ExpandShapeOp>(expandShapeOp, builder);
    } else if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      adjustTensorOutOpSrc<tensor::CollapseShapeOp>(collapseShapeOp, builder);
    } else if (auto memrefExpand = dyn_cast<memref::ExpandShapeOp>(op)) {
      adjustMemrefOutOpSrc<memref::ExpandShapeOp>(memrefExpand, builder);
    } else if (auto memrefCollapse = dyn_cast<memref::CollapseShapeOp>(op)) {
      adjustMemrefOutOpSrc<memref::CollapseShapeOp>(memrefCollapse, builder);
    } else if (auto materializeOp =
                   dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      // For register based, the inputs and outputs will stay collapsed
      if (!options.registerBased) {
        adjustTensorOutOpSource<bufferization::MaterializeInDestinationOp>(
            materializeOp, builder);
      }
    }
    LLVM_DEBUG(dumpModuleOP(););
  };
  for (auto toErase : markToDelete) {
    LDBG("[Erase] " << *toErase);
    toErase->erase();
  }
  LDBG("[Finish]" << *op_);
}

void Flattener::collapseMemrefArg(Value arg, OpBuilder &builder) {
  LDBG("Collapsing Memref Arg\n");
  auto argType = cast<MemRefType>(previousType_.at(arg));
  CollapseGroup collapseGroup = getCollapseGroup(arg);

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
      memref::CollapseShapeOp::computeCollapsedType(argType, collapseGroup);

  builder.setInsertionPointAfterValue(arg);

  LLVM_DEBUG(llvm::dbgs() << "Collapsing arg " << arg << "\n";);

  auto collapseOp = builder.create<memref::CollapseShapeOp>(
      arg.getLoc(), collapsedType, arg, collapseGroup);

  LLVM_DEBUG(llvm::dbgs() << "into " << collapseOp << "\n\n";);
  updatePreviousType(collapseOp.getResult(), argType);
  collapsePropagateOrVerify(collapseOp.getResult(), arg);
  LLVM_DEBUG(llvm::dbgs() << "Will replace " << arg << " with "
                          << collapseOp.getResult() << "\n";);
  newlyAddedCollapse.insert(collapseOp.getOperation());
  arg.replaceUsesWithIf(collapseOp.getResult(), [&](OpOperand &use) {
    Operation *defOp = use.getOwner();
    LLVM_DEBUG(llvm::dbgs() << "checking *defOp: " << *defOp << "\n";);
    if (defOp == collapseOp)
      return false;
    if (isa<memref::DimOp>(defOp))
      return false;
    // If this arg is directly returned, no need to reshape
    if (isTailOperation(defOp) || isHeadOperation(defOp)) {
      LDBG("Yea here");
      return false;
    }
    return true;
  });
}

void Flattener::collapseTensorArg(Value arg, OpBuilder &builder) {
  LDBG("Collapsing Tensor Arg\n");
  auto argType = cast<RankedTensorType>(previousType_.at(arg));
  CollapseGroup collapseGroup = getCollapseGroup(arg);

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
    // If this arg is directly returned, no need to reshape
    if (isTailOperation(defOp) || isHeadOperation(defOp))
      return false;
    return true;
  });
}

// Adjust block arguments
void Flattener::collapserForArg(Value arg, OpBuilder &builder) {
  LDBG("[collapserForArg] Collapsing arg here " << arg);
  if (isa<RankedTensorType>(previousType_.at(arg))) {
    collapseTensorArg(arg, builder);
    return;
  }
  collapseMemrefArg(arg, builder);
}

template <class T>
SmallVector<int64_t>
Flattener::adjustCollapseDimensions(T op, CollapseGroup indices) const {
  return adjustCollapseDimensionsArray(llvm::to_vector(op.getDimensions()),
                                       indices);
}

SmallVector<int64_t>
Flattener::adjustCollapseDimensionsArray(SmallVector<int64_t> dimensions,
                                         CollapseGroup indices) const {
  SmallVector<int64_t> newDimensions;
  int newDimPtr = 0;
  //             0          1       2    3
  // e.g: ref = [[0, 1, 2], [3, 4], [5], [6, 7, 8]]

  // if dimension is 3, 4, 5
  // it will be [1, 2]

  LLVM_DEBUG(llvm::dbgs() << "\nGetting new dimensions: ";);
  for (const auto dim : dimensions) {
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
  LLVM_DEBUG(llvm::dbgs() << (padOp->getParentOp() ? *(padOp->getParentOp()) : *padOp););
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
  replaceOpUsage(padOp, newPadOp);

  eraseOp(padOp);
  LLVM_DEBUG(llvm::dbgs() << "newPadOp parent: "
                          << (newPadOp->getParentOp() ? *(newPadOp->getParentOp()) : *newPadOp)
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
  replaceOpUsage(gatherOp, newGatherOp);
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
    if (isa<TensorType>(operand.getType()) &&
        !argumentsRefPointer_.contains(operand)) {
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

void Flattener::adjustResultTypeFromOperand(Operation *op) {
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    auto currentRes = op->getResult(i);
    // assign for collapsed
    updatePreviousType(currentRes);
    currentRes.setType(op->getOperands().front().getType());
  }
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

std::optional<SmallVector<int64_t>>
Flattener::getConstantStrides(MemRefType memrefType) {
  SmallVector<int64_t> strides;
  int64_t offset;
  LogicalResult hasStaticInformation =
      getStridesAndOffset(memrefType, strides, offset);
  if (failed(hasStaticInformation)) {
    return std::nullopt;
  }
  return strides;
}

template <class T>
void Flattener::calculateOffsets(T slicingOp,
                                 ReassociationIndices &collapseGroup,
                                 SmallVector<OpFoldResult> &newMixedOffsets,
                                 OpBuilder &builder) {
  ShapedType srcMemRefType;
  if (std::is_same_v<T, tensor::InsertSliceOp>) {
    srcMemRefType = dyn_cast<ShapedType>(slicingOp.getResult().getType());
  } else {
    srcMemRefType = dyn_cast<ShapedType>(previousType_[slicingOp.getSource()]);
  }
  if (!srcMemRefType)
    llvm_unreachable("srcMemRefType is NULL");
  ArrayRef<int64_t> srcShape = srcMemRefType.getShape();
  auto loc = slicingOp.getLoc();
  int64_t groupSize = (int64_t)collapseGroup.size();
  auto mixedOffsets = slicingOp.getMixedOffsets();
  SmallVector<int64_t> cumSize(groupSize);
  int curSize = 1;
  for (int64_t i = groupSize - 1; i >= 0; --i) {
    auto idx = collapseGroup[i];
    cumSize[i] = curSize;
    curSize *= srcShape[idx];
  }
  int64_t realVal = 0;
  bool isStatic = true;
  for (int64_t i = 0; i < groupSize; ++i) {
    auto idx = collapseGroup[i];
    OpFoldResult ofr = mixedOffsets[idx];
    if (isa<Value>(ofr)) {
      isStatic = false;
      continue;
    }
    if (isa<Attribute>(ofr)) {
      auto offsetInt = getConstantIntValue(ofr);
      if (!offsetInt.has_value())
        continue;
      realVal += offsetInt.value() * cumSize[i];
    }
  }
  if (!isStatic) {
    builder.setInsertionPoint(slicingOp);
    Value adder = builder.create<arith::ConstantIndexOp>(loc, realVal);
    for (int64_t i = 0; i < groupSize; ++i) {
      auto idx = collapseGroup[i];
      if (mixedOffsets[idx].template is<Value>()) {
        auto val = mixedOffsets[idx].template get<Value>();
        Value constVal =
            builder.create<arith::ConstantIndexOp>(loc, cumSize[i]);
        Value product = builder.create<arith::MulIOp>(loc, val, constVal);
        adder = builder.create<arith::AddIOp>(loc, adder, product);
      }
    }
    newMixedOffsets.push_back(adder);
  } else {
    newMixedOffsets.push_back(builder.getI64IntegerAttr(realVal));
  }
}

void Flattener::calculateStrides(memref::SubViewOp slicingOp,
                                 ReassociationIndices &collapseGroup,
                                 SmallVector<OpFoldResult> &newMixedStrides,
                                 OpBuilder &builder) {
  auto src = slicingOp.getSource();
  auto res = slicingOp.getResult();
  auto resMemRefType = res.getType();
  auto srcMemRefType = dyn_cast<MemRefType>(previousType_[src]);
  LDBG("srcMemRefType " << srcMemRefType);
  auto mixedSizes = slicingOp.getMixedSizes();
  auto mixedStrides = slicingOp.getMixedStrides();
  auto srcStride = Flattener::getConstantStrides(srcMemRefType);
  if (!srcStride.has_value())
    llvm_unreachable("srcStride is expected to have a value here");
  auto contiguousMask =
      mlir::hivm::detail::getContiguousAxesImpl({resMemRefType});
  bool isContinuous = true;
  int64_t lastGroupIdx = (int64_t)collapseGroup.size() - 1;

  ArrayRef<int64_t> srcShape = srcMemRefType.getShape();
  int64_t groupSize = (int64_t)collapseGroup.size();
  SmallVector<int64_t> cumSize(groupSize);
  int64_t curSize = 1;
  for (int64_t j = groupSize - 1; j >= 0; --j) {
    cumSize[j] = curSize;
    curSize *= srcShape[collapseGroup[j]];
  }

  // Traverse collapseGroup from back to front.
  for (int64_t i = lastGroupIdx; i >= 0; --i) {
    auto idx = collapseGroup[i];
    auto sizeInt = getConstantIntValue(mixedSizes[idx]);
    auto strideInt = getConstantIntValue(mixedStrides[idx]);

    // Check if it's a non-contiguous unit dim merge.
    if (sizeInt.has_value() && sizeInt.value() == 1 && i != 0) {
      if (!contiguousMask[idx]) {
        isContinuous = false;
      }
      continue;
    }
    // Process cases with dynamic values
    // size of a collapse group for the dynamical stride  must be 1
    // so just pass the old mixedStrides values.
    if (!strideInt.has_value() ||
        ShapedType::isDynamic(srcStride.value()[idx])) {
      newMixedStrides.push_back(mixedStrides[idx]);
      return;
    }

    // Non-contiguous dim merging: use cumSize within the group rather than
    // the full srcStride, because after collapsing, the stride of the
    // collapsed dim only covers dims inside this group.
    if (i != lastGroupIdx && !isContinuous) {
      newMixedStrides.push_back(builder.getI64IntegerAttr(
          strideInt.value() * cumSize[i]));
      return;
    }

    // Contiguous dim merging, add the mixedStrides trailing-axis parameter.
    auto trailingStride =
        getConstantIntValue(mixedStrides[collapseGroup[lastGroupIdx]]);
    newMixedStrides.push_back(
        builder.getI64IntegerAttr(trailingStride.value()));
    return;
  }
}

template <class T>
void Flattener::computeNewSlicingOperands(
    T slicingOp, SmallVector<OpFoldResult> &newMixedOffsets,
    SmallVector<OpFoldResult> &newMixedSizes,
    SmallVector<OpFoldResult> &newMixedStrides, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  Value src;
  if (std::is_same_v<T, tensor::InsertSliceOp>) {
    src = slicingOp.getResult();
  } else {
    src = slicingOp.getSource();
  }
  auto refPtr = getArgumentRef(src);
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
      if (auto subview = dyn_cast<memref::SubViewOp>(*slicingOp)) {
        calculateStrides(subview, collapseGroup, newMixedStrides, builder);
      } else {
        auto realStridedValue =
            getConstantIntValue(mixedStrides[collapseGroup.back()]);
        assert(realStridedValue.has_value());
        auto realStrided = realStridedValue.value();
        newMixedStrides.push_back(builder.getI64IntegerAttr(realStrided));
      }
      calculateOffsets(slicingOp, collapseGroup, newMixedOffsets, builder);
      builder.setInsertionPointAfter(slicingOp);

      // Multiply all the extract slice here, dynamic support
      int64_t realVal = 1;
      bool isStatic = true;
      for (auto idx : collapseGroup) {
        OpFoldResult ofr = mixedSizes[idx];
        if (isa<Value>(ofr)) {
          isStatic = false;
          continue;
        }
        if (isa<Attribute>(ofr)) {
          auto sizeInt = getConstantIntValue(ofr);
          if (!sizeInt.has_value())
            continue;
          realVal *= sizeInt.value();
        }
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
          builder.setInsertionPoint(slicingOp);
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
  auto sourceRankedTensorType =
      llvm::cast<RankedTensorType>(extractSliceOp.getSource().getType());
  auto collapseGroups = getCollapseGroup(extractSliceOp.getResult());
  auto resultSize = collapseGroups.size();
  RankedTensorType resultType = llvm::cast<RankedTensorType>(
      tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
          resultSize, sourceRankedTensorType, newMixedOffsets, newMixedSizes,
          newMixedStrides));
  // get which one to collapse together
  builder.setInsertionPoint(extractSliceOp);
  auto newExtractSliceOp = builder.create<tensor::ExtractSliceOp>(
      extractSliceOp.getLoc(), resultType, extractSliceOp.getSource(),
      newMixedOffsets, newMixedSizes, newMixedStrides);
  collapsePropagateOrVerify(newExtractSliceOp.getResult(),
                            extractSliceOp.getResult());
  updatePreviousType(newExtractSliceOp.getResult(),
                     extractSliceOp.getResult().getType());
  replaceOpUsage(extractSliceOp, newExtractSliceOp);
  eraseOp(extractSliceOp);
}

void Flattener::adjustInsertOpIndices(tensor::InsertOp insertOp,
                                      mlir::OpBuilder &builder) {
  auto insertIndices = insertOp.getIndices();
  auto dest = insertOp.getDest();
  auto collapseGroups = getCollapseGroup(dest);

  SmallVector<OpFoldResult> oldMixedSize = getFlattenMixedSizes(dest);
  builder.setInsertionPoint(insertOp);
  Location loc = insertOp.getLoc();
  // Define getDimSize lambda function.
  auto getDimSize = [&](int idx) -> Value {
    return getValueOrCreateConstantIndexOp(builder, loc, oldMixedSize[idx]);
  };
  // Compute new indices using the utility function.
  auto newIndices = computeExtractCollapsedIndices(
      collapseGroups, insertIndices, getDimSize, builder, loc);
  // Prepare the new operands.
  SmallVector<Value> insertOpNewOperands;
  insertOpNewOperands.reserve(newIndices.size() + 2);
  insertOpNewOperands.push_back(insertOp.getScalar());
  insertOpNewOperands.push_back(dest);
  insertOpNewOperands.append(newIndices);
  updatePreviousType(insertOp.getResult());
  builder.setInsertionPoint(insertOp);
  insertOp->setOperands(insertOpNewOperands);
  insertOp.getResult().setType(dest.getType());
  LLVM_DEBUG(llvm::dbgs() << "Ok insertOp done";);
  LLVM_DEBUG(llvm::dbgs() << *insertOp->getParentOfType<func::FuncOp>(););
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
  replaceOpUsage(insertSliceOp, newInsertSliceOp);
  eraseOp(insertSliceOp);
}

MemRefType Flattener::inferResultMemRefType(memref::SubViewOp subviewOp) {
  auto res = subviewOp.getResult();
  auto resMemRefType = dyn_cast<MemRefType>(res.getType());
  auto collapseGroups = getCollapseGroup(res);
  auto collapsedType = memref::CollapseShapeOp::computeCollapsedType(
      resMemRefType, collapseGroups);
  return collapsedType;
}

void Flattener::adjustSubviewOp(memref::SubViewOp subviewOp,
                                mlir::OpBuilder &builder) {
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  auto resultMemRefType = inferResultMemRefType(subviewOp);
  LDBG("[here-resultMemref]" << resultMemRefType);
  computeNewSlicingOperands(subviewOp, newMixedOffsets, newMixedSizes,
                            newMixedStrides, builder);
  // get which one to collapse together
  builder.setInsertionPoint(subviewOp);
  auto newSubviewOp = builder.create<memref::SubViewOp>(
      subviewOp.getLoc(), resultMemRefType, subviewOp.getSource(),
      newMixedOffsets, newMixedSizes, newMixedStrides);

  collapsePropagateOrVerify(newSubviewOp.getResult(), subviewOp.getResult());
  updatePreviousType(newSubviewOp.getResult(), subviewOp.getResult().getType());
  replaceOpUsage(subviewOp, newSubviewOp);
  LDBG("Ok deleting subview");
  LDBG("[here-subview]" << subviewOp);
  eraseOp(subviewOp);
  LDBG("[new-subview]" << newSubviewOp);
}

void Flattener::adjustEmbeddingGatherOp(
    hfusion::EmbeddingGatherOp embeddingGatherOp, mlir::OpBuilder &builder) {
  // TODO: adjust numels and offsets
  embeddingGatherOp.getResult().setType(embeddingGatherOp.getDst().getType());
}

void Flattener::adjustToTensorOp(bufferization::ToTensorOp op,
                                 mlir::OpBuilder &builder) {
  auto newType = op.getResult().getType().clone(
      utils::getShape(op.getOperand().getType()));
  LDBG("operation result " << op << " " << newType);
  op.getResult().setType(newType);
}

void Flattener::adjustToMemrefOp(bufferization::ToMemrefOp op,
                                 mlir::OpBuilder &builder) {
  auto oldType = cast<MemRefType>(op.getMemref().getType());
  // TODO: adjust the layout instead of using identity
  MemRefType newType = MemRefType::Builder(oldType)
                           .setShape(utils::getShape(op.getOperand().getType()))
                           .setLayout(MemRefLayoutAttrInterface());
  LDBG("operation result " << op << " " << newType << "\n");
  op.getMemref().setType(newType);
}

void Flattener::adjustCastOp(memref::CastOp castOp, mlir::OpBuilder &builder) {
  auto srcType = cast<MemRefType>(castOp.getSource().getType());
  auto oldResType = cast<MemRefType>(castOp.getResult().getType());

  if (srcType == oldResType ||
      memref::CastOp::areCastCompatible(srcType, oldResType)) {
    updatePreviousType(castOp.getResult());
    return;
  }

  LDBG("[adjustCastOp] source type changed, repairing: " << *castOp);

  SmallVector<int64_t> srcStrides, oldResStrides;
  int64_t srcOffset, oldResOffset;

  MemRefType newResType;
  if (succeeded(getStridesAndOffset(srcType, srcStrides, srcOffset)) &&
      succeeded(getStridesAndOffset(oldResType, oldResStrides, oldResOffset))) {
    auto newLayout = StridedLayoutAttr::get(
        castOp->getContext(), oldResOffset, srcStrides);
    newResType =
        MemRefType::get(srcType.getShape(), srcType.getElementType(),
                        newLayout, srcType.getMemorySpace());
  } else {
    newResType = srcType;
  }

  updatePreviousType(castOp.getResult());
  castOp.getResult().setType(newResType);
  LDBG("[adjustCastOp] result: " << *castOp);
}

void Flattener::adjustArangeOp(hfusion::ArangeOp arangeOp,
                               mlir::OpBuilder &builder) {
  auto init = arangeOp.getInit();
  auto collapseGroup = this->getCollapseGroup(init);
  // currently, only handle unit dimension flattened. which implies the stride
  // of the axis can be ignored
  auto newType = init.getType();
  assert(isOnlyUnitDimFlattened(arangeOp.getResultTensor().getType().getShape(),
                                newType.getShape()) &&
         "only support flattened unit dimension");
  LDBG("operation result " << arangeOp << " " << newType);
  auto result = arangeOp.getResultTensor();
  if (result == nullptr)
    return;
  result.setType(newType);
  SmallVector<Value> newStrides;
  const SmallVector<Value> oldStrides = arangeOp.getStrides();
  for (const auto &group : collapseGroup) {
    // We should use the last stride of this group under the assumption
    // that this multi-dim arange is contiguous.
    newStrides.push_back(oldStrides[group.back()]);
  }

  builder.setInsertionPoint(arangeOp);
  auto newArangeOp = builder.create<hfusion::ArangeOp>(
      arangeOp->getLoc(), arangeOp.getOffset(), newStrides, arangeOp.getInit());
  collapsePropagateOrVerify(newArangeOp.getResult(0), result);
  updatePreviousType(newArangeOp.getResult(0), result.getType());
  replaceOpUsage(arangeOp, newArangeOp);
  eraseOp(arangeOp);
}

void Flattener::adjustIfOp(scf::IfOp ifOp, OpBuilder &builder) {
  LDBG("[adjustIfOp] Adjusting ifOp of " << *ifOp);
  if (!ifOp.elseBlock()) {
    for (auto [yieldOpr, resultOpr] :
         llvm::zip_equal(ifOp.elseYield()->getOperands(), ifOp->getResults())) {
      updatePreviousType(resultOpr);
      collapsePropagateOrVerify(resultOpr, yieldOpr);
    }
  }

  for (auto [yieldOpr, resultOpr] :
       llvm::zip_equal(ifOp.thenYield()->getOperands(), ifOp->getResults())) {
    updatePreviousType(resultOpr);
    collapsePropagateOrVerify(resultOpr, yieldOpr);
    resultOpr.setType(yieldOpr.getType());
  }
}

void Flattener::adjustFlipOp(hfusion::FlipOp flipOp, mlir::OpBuilder &builder) {
  SmallVector<int64_t> flipAxis = {static_cast<int64_t>(flipOp.getFlipAxis())};
  auto collapseGroups = getCollapseGroup(flipOp.getInput());
  auto adjustedAxis = adjustCollapseDimensionsArray(flipAxis, collapseGroups);
  flipOp.setFlipAxis(adjustedAxis.front());
  for (unsigned i = 0; i < flipOp->getNumResults(); ++i) {
    auto currentRes = flipOp->getResult(i);
    // assign for collapsed
    updatePreviousType(currentRes);
    currentRes.setType(flipOp.getInput().getType());
  }
}

void Flattener::adjustSortOp(hfusion::SortOp sortOp, mlir::OpBuilder &builder) {
  SmallVector<int64_t> sortAxis = {static_cast<int64_t>(sortOp.getSortAxis())};
  auto collapseGroups = getCollapseGroup(sortOp.getOperand());
  auto adjustedAxis = adjustCollapseDimensionsArray(sortAxis, collapseGroups);
  sortOp.setSortAxis(adjustedAxis.front());
  for (unsigned i = 0; i < sortOp->getNumResults(); ++i) {
    auto currentRes = sortOp->getResult(i);
    // assign for collapsed
    updatePreviousType(currentRes);
    currentRes.setType(sortOp.getOperand().getType());
  }
}

void Flattener::adjustForOp(scf::ForOp forOp, OpBuilder &builder) {
  LDBG("[adjustForOp] Adjusting for loop of " << *forOp);
  auto initArgs = forOp.getInitArgs();
  auto iterArgs = forOp.getRegionIterArgs();
  auto results = forOp.getResults();
  for (auto [idx, init] : llvm::enumerate(initArgs)) {
    Value iterArg = iterArgs[idx];
    Value result = results[idx];
    if (!hasCollapseGroup(init))
      continue;
    LDBG("  Processing iter_arg " << idx << ": " << iterArg << " become "
                                  << init.getType());
    updatePreviousType(iterArg);
    collapsePropagateOrVerify(iterArg, init);
    iterArg.setType(init.getType());
    updatePreviousType(result);
    collapsePropagateOrVerify(result, init);
    result.setType(init.getType());
  }
}

void Flattener::adjustYieldOp(scf::YieldOp yieldOp, OpBuilder &builder) {
  LDBG("[adjustYieldOp] Adjusting yield operation");
  for (Value operand : yieldOp.getOperands()) {
    if (!previousType_.contains(operand)) {
      updatePreviousType(operand);
    }
  }
}

LogicalResult Flattener::collapser(Operation *op, OpBuilder &builder) {
  // Check if the operation is skippable
  if (isHeadOperation(op) || isTailOperation(op) || isUnsupportedOp(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Skipping OP " << *op << "\n";);
    return success();
  }

  // Check if all operands are collapsed before
  if (!isa<linalg::FillOp>(op) && failed(VerifyCollapsedOperand(op)))
    return failure();

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    adjustForOp(forOp, builder);
    return success();
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    adjustIfOp(ifOp, builder);
    return success();
  }

  if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    adjustYieldOp(yieldOp, builder);
    return success();
  }

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

  if (auto tensorInsertOp = dyn_cast<tensor::InsertOp>(op)) {
    adjustInsertOpIndices(tensorInsertOp, builder);
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

  if (auto embeddingGatherOp = dyn_cast<hfusion::EmbeddingGatherOp>(op)) {
    adjustEmbeddingGatherOp(embeddingGatherOp, builder);
    return success();
  }

  if (auto indirectLoadOp = dyn_cast<hfusion::IndirectLoadOp>(op)) {
    adjustIndirectLoadOp(indirectLoadOp, builder);
    return success();
  }

  if (auto memrefLoadOp = dyn_cast<memref::LoadOp>(op)) {
    adjustMemrefLoadOp(memrefLoadOp, builder);
    return success();
  }

  if (auto memrefStoreOp = dyn_cast<memref::StoreOp>(op)) {
    adjustMemrefStoreOp(memrefStoreOp, builder);
    return success();
  }

  if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
    adjustSubviewOp(subviewOp, builder);
    return success();
  }

  if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    adjustToTensorOp(toTensorOp, builder);
    return success();
  }

  if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op)) {
    adjustToMemrefOp(toMemrefOp, builder);
    return success();
  }

  if (auto arangeOp = dyn_cast<hfusion::ArangeOp>(op)) {
    adjustArangeOp(arangeOp, builder);
    return success();
  }

  if (auto flipOp = dyn_cast<hfusion::FlipOp>(op)) {
    adjustFlipOp(flipOp, builder);
    return success();
  }

  if (auto sortOp = dyn_cast<hfusion::SortOp>(op)) {
    adjustSortOp(sortOp, builder);
    return success();
  }

  if (isa_and_present<hfusion::MulExtOp>(op)) {
    adjustResultTypeFromOperand(op);
    return success();
  }

  if (auto memrefCastOp = dyn_cast<memref::CastOp>(op)) {
    adjustCastOp(memrefCastOp, builder);
    return success();
  }

  if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(op)) {
    LDBG("Before adjustResultTypeFromOperand: castOp = " << *castOp);
    auto dst = castOp.getDest();
    auto maybeDstOrigAddrSpace =
        hivm::getOptionalHIVMAddressSpace(dst.getType());
    adjustResultTypeFromOperand(op);
    // fix AddressSpace
    auto updatedDst = castOp.getDest();
    if (maybeDstOrigAddrSpace.has_value()) {
      auto dstOrigAddrSpaceAttr = hivm::AddressSpaceAttr::get(
          op->getContext(), maybeDstOrigAddrSpace.value());
      hivm::modifyBaseMemRefTypeScope(updatedDst, dstOrigAddrSpaceAttr);
    } else {
      // use the default address space
      auto dstOldType = dyn_cast<BaseMemRefType>(updatedDst.getType());
      auto dstNewType = hivm::getBaseMemRefTypeWithNewScope(dstOldType, 0);
      updatedDst.setType(dstNewType);
    }
    LDBG("After  adjustResultTypeFromOperand: castOp = " << *castOp);
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
    bool skipExpand = isValueFromHead(operand);
    skipExpand |= !argumentsRefPointer_.contains(operand);
    if (skipExpand) {
      LLVM_DEBUG(llvm::dbgs() << "Operand can be skipped " << operand << "\n";);
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
FailureOr<Operation *> Flattener::expandForTail(OpTy &tensorOutOp,
                                                Value &collapsedVal,
                                                OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "\nCollapsing " << collapsedVal << "\n";);
  if (!previousType_.contains(collapsedVal)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Source is not expandable, previousType_ not found" << "\n");
    return failure();
  }

  CollapseGroup collapseGroup = getCollapseGroup(collapsedVal);
  if (collapseGroup.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Collapse group invalid" << "\n");
    return failure();
  }

  auto expandedType = previousType_.at(collapsedVal);
  builder.setInsertionPoint(tensorOutOp);

  auto mixedSize = getFlattenMixedSizes(collapsedVal);
  if (isa<RankedTensorType>(expandedType)) {
    auto expandOp = builder.create<tensor::ExpandShapeOp>(
        tensorOutOp.getLoc(), expandedType, collapsedVal, collapseGroup,
        mixedSize);
    return expandOp.getOperation();
  }
  auto expandedMemrefType = memref::ExpandShapeOp::computeExpandedType(
      cast<MemRefType>(collapsedVal.getType()), expandedType.getShape(),
      collapseGroup);
  if (failed(expandedMemrefType))
    return failure();
  auto expandOp = builder.create<memref::ExpandShapeOp>(
      tensorOutOp.getLoc(), expandedMemrefType.value(), collapsedVal,
      collapseGroup, mixedSize);
  if (expandedMemrefType.value() != expandedType) {
    LDBG(
        "[ExpansionForTail] expected type is different with the expanded type: "
        << expandedMemrefType.value() << " " << expandedType);
    if (memref::CastOp::areCastCompatible(expandedMemrefType.value(),
                                          cast<MemRefType>(expandedType))) {
      auto castOp = builder.create<memref::CastOp>(
          tensorOutOp.getLoc(), expandedType, expandOp.getResult());
      return castOp.getOperation();
    }
    SmallVector<int64_t> targetStrides;
    int64_t targetOffset;
    if (failed(getStridesAndOffset(cast<MemRefType>(expandedType),
                                   targetStrides, targetOffset))) {
      return failure();
    }
    SmallVector<OpFoldResult> targetSizes(mixedSize.begin(), mixedSize.end());
    SmallVector<OpFoldResult> targetMixedStrides;
    for (int64_t stride : targetStrides) {
      if (ShapedType::isDynamic(stride))
        return failure();
      targetMixedStrides.push_back(builder.getIndexAttr(stride));
    }
    OpFoldResult offsetFold;
    if (ShapedType::isDynamic(targetOffset)) {
      auto extractOp = builder.create<memref::ExtractStridedMetadataOp>(
          tensorOutOp.getLoc(), expandOp.getResult());
      offsetFold = extractOp.getOffset();
    } else {
      offsetFold = builder.getIndexAttr(targetOffset);
    }
    auto reinterpreted = builder.create<memref::ReinterpretCastOp>(
        tensorOutOp.getLoc(), cast<MemRefType>(expandedType),
        expandOp.getResult(), offsetFold, targetSizes,
        targetMixedStrides);
    return reinterpreted.getOperation();
  }
  return expandOp.getOperation();
}

// TODO: Check whether they have common interfaces
template <typename OpTy>
void Flattener::adjustTensorOutOpSource(OpTy tensorOutOp, OpBuilder &builder) {
  Value source = tensorOutOp.getSource();
  if (!utils::getShapeRank(source).value_or(0))
    return;
  if (isValueFromHead(source))
    return;
  auto expandOp = expandForTail(tensorOutOp, source, builder);
  if (succeeded(expandOp)) {
    tensorOutOp.getSourceMutable().assign(expandOp.value()->getResult(0));
    return;
  }
  llvm_unreachable("tensor expand op collapse out op source failed");
}

template <typename OpTy>
void Flattener::adjustTensorOutOpDest(OpTy tensorOutOp, OpBuilder &builder) {
  Value dest = tensorOutOp.getDest();
  if (!utils::getShapeRank(dest).value_or(0))
    return;
  if (isValueFromHead(dest))
    return;
  auto expandOp = expandForTail(tensorOutOp, dest, builder);
  if (!failed(expandOp)) {
    tensorOutOp.getDestMutable().assign(expandOp.value()->getResult(0));
    return;
  }
  llvm_unreachable("tensor expand op collapse out op dest failed");
}

template <typename OpTy>
void Flattener::adjustTensorOutOpSrc(OpTy tensorOutOp, OpBuilder &builder) {
  Value source = tensorOutOp.getSrc();
  if (isValueFromHead(source))
    return;
  if (!utils::getShapeRank(source).value_or(0))
    return;
  auto expandOp = expandForTail(tensorOutOp, source, builder);
  if (!failed(expandOp)) {
    tensorOutOp.getSrcMutable().assign(expandOp.value()->getResult(0));
    return;
  }
  llvm_unreachable("tensor expand op collapse out op src failed");
}

template <typename OpTy>
void Flattener::adjustMemrefOutOpSrc(OpTy memrefOutOp, OpBuilder &builder) {
  Value source = memrefOutOp.getSrc();
  if (isValueFromHead(source))
    return;
  if (!utils::getShapeRank(source).value_or(0))
    return;
  auto expandOp = expandForTail(memrefOutOp, source, builder);
  if (!failed(expandOp)) {
    memrefOutOp.getSrcMutable().assign(expandOp.value()->getResult(0));
    return;
  }
  llvm_unreachable("memref expand op collapse failed");
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
  markToDelete.insert(op);
}

void Flattener::replaceOpUsage(Operation *oldOp, Operation *newOp) {
  oldOp->replaceAllUsesWith(newOp);
  for (auto [newValue, oldValue] :
       llvm::zip_equal(newOp->getResults(), oldOp->getResults())) {
    valueReplacement[newValue] = oldValue;
  }
}
} // namespace detail
} // namespace hfusion
} // namespace mlir
