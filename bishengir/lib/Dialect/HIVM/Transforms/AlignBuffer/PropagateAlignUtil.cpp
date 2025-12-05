//===------------------------- Util.cpp -----------------------------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"

#define DEBUG_TYPE "hivm-align-buffer-util"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using UnrealizedCastOpVec = SmallVector<UnrealizedConversionCastOp>;
using FailureOrCastVec = FailureOr<UnrealizedCastOpVec>;

namespace {
int getPrevAlignDimBeforeDrop(int postDim,
                              const llvm::SmallBitVector &droppedDims) {
  int prevDim = droppedDims.find_first_unset();
  for (int i = 1; i <= postDim; ++i) {
    if (prevDim == -1) {
      return -1;
    }
    prevDim = droppedDims.find_next_unset(prevDim);
  }
  return prevDim;
}

int getPostAlignDimAfterDrop(int prevDim,
                             const llvm::SmallBitVector &droppedDims) {
  int postDim = -1;
  int curDim = droppedDims.find_first_unset();
  while (curDim <= prevDim && curDim != -1) {
    postDim++;
    curDim = droppedDims.find_next_unset(curDim);
  };
  return postDim;
}

int countNotOneDim(const ReassociationIndices &reassociation,
                   const ArrayRef<int64_t> shape,
                   std::optional<int> &lastDimOfNotOne) {
  int counterOfNotOne = 0;
  for (auto it = reassociation.rbegin(); it < reassociation.rend(); it++) {
    if (shape[*it] == 1) {
      continue;
    }
    counterOfNotOne++;
    if (!lastDimOfNotOne.has_value()) {
      lastDimOfNotOne = *it;
    }
  }
  return counterOfNotOne;
}

std::optional<int> getExpandedDim(
    int dim, const ArrayRef<ReassociationIndices> &expandReassociations,
    const ArrayRef<int64_t> expandShape, std::string alignDimAttrName) {
  std::optional<int> lastDimOfNotOne = std::nullopt;
  assert(dim >= 0 && dim < static_cast<int>(expandReassociations.size()));
  int counterOfNotOne =
      countNotOneDim(expandReassociations[dim], expandShape, lastDimOfNotOne);

  if (alignDimAttrName == hivm::AllocAlignDimsAttr::name.str()) {
    if (counterOfNotOne > 1) {
      return std::nullopt;
    }
    return lastDimOfNotOne.value_or(expandReassociations[dim].back());
  }

  if (alignDimAttrName == hivm::StrideAlignDimsAttr::name.str()) {
    return lastDimOfNotOne.value_or(expandReassociations[dim].back());
  }
  llvm_unreachable("unsupport propagation mode");
}

std::optional<int> getCollapsedDim(
    int dim, const ArrayRef<ReassociationIndices> &collapseReassociations,
    ArrayRef<int64_t> collapseSrcShapes, std::string alignDimAttrName) {
  for (size_t i = 0; i < collapseReassociations.size(); i++) {
    const auto &group = collapseReassociations[i];
    if (group.back() < dim) {
      continue;
    }

    std::optional<int> lastDimOfNotOne = std::nullopt;
    int counterOfNotOne =
        countNotOneDim(group, collapseSrcShapes, lastDimOfNotOne);

    if (alignDimAttrName == hivm::StrideAlignDimsAttr::name.str()) {
      if (!lastDimOfNotOne.has_value()) {
        return i;
      } else if (lastDimOfNotOne.value() <= dim) {
        return i;
      } else if (lastDimOfNotOne.value() > dim) {
        return std::nullopt;
      }
    }

    if (alignDimAttrName == hivm::AllocAlignDimsAttr::name.str()) {
      if (counterOfNotOne > 1) {
        return std::nullopt;
      } else {
        return i;
      }
    }
  }
  llvm_unreachable("unsupported propagation mode");
}

template <typename OP, typename = std::enable_if_t<
                           std::is_same_v<OP, memref::CollapseShapeOp> ||
                           std::is_same_v<OP, memref::ExpandShapeOp> ||
                           std::is_same_v<OP, memref::ReshapeOp>>>
LogicalResult propagateAlignInfoByCollapse(
    PatternRewriter &rewriter, OP op,
    const ArrayRef<ReassociationIndices> &reassociations,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str(),
    bool isPropagateUp = true) {
  if constexpr (std::is_same_v<OP, memref::ReshapeOp>) {
    auto srcType = op.getSource().getType();
    auto resultType = op.getResult().getType();
    if ((!isPropagateUp && srcType.getRank() < resultType.getRank()) ||
        (isPropagateUp && srcType.getRank() > resultType.getRank())) {
      return op->emitError() << "should be collapse behavior when propagete"
                             << (isPropagateUp ? "up" : "down");
    }
  }

  auto propagateFromValue = isPropagateUp ? op.getResult() : op.getViewSource();
  auto propagateToValue = isPropagateUp ? op.getViewSource() : op.getResult();

  LDBG("op propagate " << (isPropagateUp ? "up" : "down") << ": " << op);
  LLVM_DEBUG(dump(alignDims, alignBytes, DEBUG_TYPE));

  auto propagateFromShapes =
      cast<MemRefType>(propagateFromValue.getType()).getShape();
  llvm::SmallVector<int32_t> mappedAlignDims(alignDims.size());
  for (size_t i = 0; i < alignDims.size(); ++i) {
    auto mappedDim = getCollapsedDim(alignDims[i], reassociations,
                                     propagateFromShapes, alignDimAttrName);
    if (!mappedDim.has_value()) {
      return op.emitError() << "cannot align " << alignDims[i] << " axis for "
                            << propagateFromValue;
    }
    mappedAlignDims[i] = mappedDim.value();
  }

  LDBG("after propagation");
  LLVM_DEBUG(dump(mappedAlignDims, alignBytes, DEBUG_TYPE));
  auto [adjustAlignDims, adjustAlignBytes] =
      adjustAlignInfo(op, propagateToValue, mappedAlignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, op.getLoc(), propagateToValue, adjustAlignDims,
                    adjustAlignBytes, alignDimAttrName, alignBytesAttrName);
  return success();
}

template <typename OP, typename = std::enable_if_t<
                           std::is_same_v<OP, memref::CollapseShapeOp> ||
                           std::is_same_v<OP, memref::ExpandShapeOp> ||
                           std::is_same_v<OP, memref::ReshapeOp>>>
LogicalResult propagateAlignInfoByExpand(
    PatternRewriter &rewriter, OP op,
    const ArrayRef<ReassociationIndices> &reassociations,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str(),
    bool isPropagateUp = true) {
  if constexpr (std::is_same_v<OP, memref::ReshapeOp>) {
    auto srcType = op.getSource().getType();
    auto resultType = op.getResult().getType();
    if ((isPropagateUp && srcType.getRank() < resultType.getRank()) ||
        (!isPropagateUp && srcType.getRank() > resultType.getRank())) {
      return op->emitError() << "should be expand behavior when propagete"
                             << (isPropagateUp ? "up" : "down");
    }
  }

  LDBG("op propagate " << (isPropagateUp ? "up" : "down") << ": " << op);
  LLVM_DEBUG(dump(alignDims, alignBytes, DEBUG_TYPE));

  auto propagateFromValue = isPropagateUp ? op.getResult() : op.getViewSource();
  auto propagateToValue = isPropagateUp ? op.getViewSource() : op.getResult();
  auto propagateToValueShape =
      cast<ShapedType>(propagateToValue.getType()).getShape();
  llvm::SmallVector<int32_t> mappedAlignDims(alignDims.size());
  for (size_t i = 0; i < alignDims.size(); ++i) {
    auto mappedDim = getExpandedDim(alignDims[i], reassociations,
                                    propagateToValueShape, alignDimAttrName);
    if (!mappedDim.has_value()) {
      return op.emitError() << "cannot align " << alignDims[i] << " axis for "
                            << propagateFromValue;
    }
    mappedAlignDims[i] = mappedDim.value();
  }
  LDBG("after propagation");
  LLVM_DEBUG(dump(mappedAlignDims, alignBytes, DEBUG_TYPE));

  auto [adjustAlignDims, adjustAlignBytes] =
      adjustAlignInfo(op, propagateToValue, mappedAlignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, op.getLoc(), propagateToValue, adjustAlignDims,
                    adjustAlignBytes, alignDimAttrName, alignBytesAttrName);

  return success();
}

//===----------------------------------------------------------------------===//
// Propagate align downs
//===----------------------------------------------------------------------===//

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, memref::CastOp castOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto maybeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      castOp.getResult(), hivm::StrideAlignDimsAttr::name);
  if (maybeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  if (alignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, castOp.getLoc(), castOp.getResult(), alignDims,
                    alignBytes, alignDimAttrName, alignBytesAttrName);
  return success();
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, memref::SubViewOp subviewOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto maybeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      subviewOp.getResult(), hivm::StrideAlignDimsAttr::name);
  if (maybeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  llvm::SmallBitVector droppedDims = subviewOp.getDroppedDims();
  llvm::SmallVector<int32_t> mappedAlignDims(alignDims.size());
  for (size_t i = 0; i < alignDims.size(); ++i) {
    mappedAlignDims[i] = getPostAlignDimAfterDrop(alignDims[i], droppedDims);
  }
  if (mappedAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, subviewOp.getLoc(), subviewOp.getResult(),
                    mappedAlignDims, alignBytes, alignDimAttrName,
                    alignBytesAttrName);
  return success();
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, memref::CollapseShapeOp collapseOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto maybeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      collapseOp.getResult(), hivm::StrideAlignDimsAttr::name);
  if (maybeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  auto reassociations = collapseOp.getReassociationIndices();
  return propagateAlignInfoByCollapse<memref::CollapseShapeOp>(
      rewriter, collapseOp, reassociations, alignDims, alignBytes,
      alignDimAttrName, alignBytesAttrName, false /*isPropagateUp*/);
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, memref::ExpandShapeOp expandOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto maybeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      expandOp.getResult(), hivm::StrideAlignDimsAttr::name);
  if (maybeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  auto reassociations = expandOp.getReassociationIndices();
  return propagateAlignInfoByExpand<memref::ExpandShapeOp>(
      rewriter, expandOp, reassociations, alignDims, alignBytes,
      alignDimAttrName, alignBytesAttrName, false /*isPropagateUp*/);
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, memref::ReshapeOp reshapeOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto maybeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      reshapeOp.getResult(), hivm::StrideAlignDimsAttr::name);
  if (maybeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  auto srcType = reshapeOp.getSource().getType();
  auto resultType = reshapeOp.getResult().getType();
  auto mayReassociations =
      getReassociationIndicesForReshape(srcType, resultType);
  if (!mayReassociations.has_value()) {
    reshapeOp.emitError("cannot be inference alignment");
    return failure();
  }

  if (srcType.getRank() > resultType.getRank()) {
    // Treat as memref.collapse_shape
    return propagateAlignInfoByCollapse<memref::ReshapeOp>(
        rewriter, reshapeOp, mayReassociations.value(), alignDims, alignBytes,
        alignDimAttrName, alignBytesAttrName, false /*isPropagateUp*/);
  } else {
    // Treat as memref.expand_shape
    return propagateAlignInfoByExpand<memref::ReshapeOp>(
        rewriter, reshapeOp, mayReassociations.value(), alignDims, alignBytes,
        alignDimAttrName, alignBytesAttrName, false /*isPropagateUp*/);
  }
  llvm_unreachable("");
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, hivm::BitcastOp bitcastOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto maybeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      bitcastOp.getResult(), hivm::StrideAlignDimsAttr::name);
  if (maybeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  if (alignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, bitcastOp.getLoc(), bitcastOp.getResult(),
                    alignDims, alignBytes, alignDimAttrName,
                    alignBytesAttrName);
  return success();
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, scf::ForOp scfForOp, Value v,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto inits = scfForOp.getInitArgs();
  auto it = std::find(inits.begin(), inits.end(), v);
  assert(it != inits.end() && "only support scf for init as users");
  unsigned initIndx = static_cast<unsigned>(it - inits.begin());
  auto mayResultBeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      scfForOp.getResult(initIndx), hivm::StrideAlignDimsAttr::name);
  auto mayIterArgBeAnnotateOpWithAttr = utils::getAnnotateOpWithAttr(
      scfForOp.getRegionIterArgs()[initIndx], hivm::StrideAlignDimsAttr::name);
  if (mayResultBeAnnotateOpWithAttr.has_value() &&
      mayIterArgBeAnnotateOpWithAttr.has_value()) {
    // already propagate down
    return failure();
  }
  if (alignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, scfForOp.getLoc(), scfForOp.getResult(initIndx),
                    alignDims, alignBytes, alignDimAttrName,
                    alignBytesAttrName);
  createAlignMarkOp(rewriter, scfForOp.getLoc(),
                    scfForOp.getRegionIterArgs()[initIndx], alignDims,
                    alignBytes, alignDimAttrName, alignBytesAttrName);
  return success();
}

LogicalResult propagateAlignDown(
    PatternRewriter &rewriter, scf::YieldOp scfYieldOp, Value v,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {

  auto parentOp = scfYieldOp->getParentOp();
  if (!isa<scf::IfOp>(parentOp)) {
    return failure();
  }
  auto ifOp = cast<scf::IfOp>(parentOp);
  auto yieldOperands = scfYieldOp->getOperands();
  auto it = std::find(yieldOperands.begin(), yieldOperands.end(), v);
  unsigned yieldIndx = static_cast<unsigned>(it - yieldOperands.begin());
  auto result = ifOp->getResult(yieldIndx);
  auto mayAnnotateOp =
      utils::getAnnotateOpWithAttr(result, hivm::StrideAlignDimsAttr::name);
  if (!mayAnnotateOp.has_value()) {
    createAlignMarkOp(rewriter, ifOp.getLoc(), ifOp.getResult(yieldIndx),
                      alignDims, alignBytes, alignDimAttrName,
                      alignBytesAttrName);
    return success();
  }

  auto alreadyAnnotateOp = mayAnnotateOp.value();
  auto alreadyAlignDims =
      alreadyAnnotateOp->getAttrOfType<DenseI32ArrayAttr>(alignDimAttrName);
  auto alreadyAlignBytes =
      alreadyAnnotateOp->getAttrOfType<DenseI32ArrayAttr>(alignBytesAttrName);
  auto [unionAlignDims, unionAlignBytes] = unionAlignInfo(
      alreadyAlignDims, alreadyAlignBytes, alignDims, alignBytes);

  if (util::AlignInfo(alignDims, alignBytes) == util::AlignInfo(unionAlignDims, unionAlignBytes)) {
    return failure();
  }
  rewriter.modifyOpInPlace(alreadyAnnotateOp, [&rewriter, &alreadyAnnotateOp,
                                               &alignDimAttrName,
                                               &alignBytesAttrName,
                                               unionAlignDims = unionAlignDims,
                                               unionAlignBytes = unionAlignBytes]() {
    alreadyAnnotateOp->setAttr(
        alignDimAttrName, DenseI32ArrayAttr::get(rewriter.getContext(),
                                                 ArrayRef(unionAlignDims)));
    alreadyAnnotateOp->setAttr(
        alignBytesAttrName, DenseI32ArrayAttr::get(rewriter.getContext(),
                                                   ArrayRef(unionAlignBytes)));
  });
  return success();
}

/// Propagate the align info down to leaf operand. It return failure if
/// propagate fails or already propagated without modification
LogicalResult propagateDownAlignInfo(
    PatternRewriter &rewriter, Value v, ArrayRef<int32_t> alignDims,
    ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  bool isAlreadyPropagated = true;
  for (Operation *user : v.getUsers()) {
    auto result =
        TypeSwitch<Operation *, LogicalResult>(user)
            .Case([&rewriter, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](memref::CastOp castOp) {
              return propagateAlignDown(rewriter, castOp, alignDims, alignBytes,
                                        alignDimAttrName, alignBytesAttrName);
            })
            .Case([&rewriter, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](memref::SubViewOp subviewOp) {
              return propagateAlignDown(rewriter, subviewOp, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Case([&rewriter, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](memref::CollapseShapeOp collapseOp) {
              return propagateAlignDown(rewriter, collapseOp, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Case([&rewriter, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](memref::ExpandShapeOp expandOp) {
              return propagateAlignDown(rewriter, expandOp, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Case([&rewriter, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](memref::ReshapeOp reshapeOp) {
              return propagateAlignDown(rewriter, reshapeOp, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Case([&rewriter, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](hivm::BitcastOp bitcastOp) {
              return propagateAlignDown(rewriter, bitcastOp, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Case([&rewriter, &v, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](scf::ForOp scfForOp) {
              return propagateAlignDown(rewriter, scfForOp, v, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Case([&rewriter, &v, &alignDims, &alignBytes, &alignDimAttrName,
                   &alignBytesAttrName](scf::YieldOp scfYieldOp) {
              return propagateAlignDown(rewriter, scfYieldOp, v, alignDims,
                                        alignBytes, alignDimAttrName,
                                        alignBytesAttrName);
            })
            .Default([](Operation *) { return failure(); });
    if (succeeded(result)) {
      isAlreadyPropagated = false;
    }
  }
  return isAlreadyPropagated ? failure() : success();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, memref::CastOp castOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto [adjustAlignDims, adjustAlignBytes] =
      adjustAlignInfo(castOp, castOp.getViewSource(), alignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, castOp.getLoc(), castOp.getViewSource(),
                    adjustAlignDims, adjustAlignBytes, alignDimAttrName,
                    alignBytesAttrName);
  return success();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, memref::SubViewOp subviewOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  SmallVector<int32_t> mappedAlignDims(alignDims.size());
  llvm::SmallBitVector droppedDims = subviewOp.getDroppedDims();
  for (size_t i = 0; i < alignDims.size(); ++i) {
    mappedAlignDims[i] = getPrevAlignDimBeforeDrop(alignDims[i], droppedDims);
  }
  auto [adjustAlignDims, adjustAlignBytes] = adjustAlignInfo(
      subviewOp, subviewOp.getViewSource(), mappedAlignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, subviewOp.getLoc(), subviewOp.getViewSource(),
                    adjustAlignDims, adjustAlignBytes, alignDimAttrName,
                    alignBytesAttrName);
  return success();
}

mlir::LogicalResult
propagateAlignUp([[maybe_unused]] mlir::PatternRewriter &rewriter,
                 annotation::MarkOp markOp, memref::ViewOp viewOp,
                 ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes) {
  auto viewType = viewOp.getType();
  for (size_t i = 0; i < alignDims.size(); ++i) {
    int64_t innerStaticShape = 1;
    for (int j = alignDims[i] + 1; j < viewType.getRank(); ++j)
      if (viewType.getShape()[j] != ShapedType::kDynamic)
        innerStaticShape *= viewType.getShape()[j];
    if (innerStaticShape % alignBytes[i] != 0) {
      markOp.emitError() << "Alignment cannot be satisfied for "
                         << markOp.getSrc();
      return failure();
    }
  }
  markOp.emitRemark() << "Alignemnt already satisfied for " << markOp.getSrc();
  return success();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, memref::ReshapeOp reshapeOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto prevType = reshapeOp.getSource().getType();
  auto postType = reshapeOp.getResult().getType();
  auto mayReassocIdx = getReassociationIndicesForReshape(prevType, postType);
  if (mayReassocIdx.has_value()) {
    if (prevType.getRank() > postType.getRank()) {
      // Treat as memref.collapse_shape
      return propagateAlignInfoByExpand<memref::ReshapeOp>(
          rewriter, reshapeOp, mayReassocIdx.value(), alignDims, alignBytes,
          alignDimAttrName, alignBytesAttrName, true /*isPropagateUp*/);
    } else {
      // Treat as memref.expand_shape
      return propagateAlignInfoByCollapse<memref::ReshapeOp>(
          rewriter, reshapeOp, mayReassocIdx.value(), alignDims, alignBytes,
          alignDimAttrName, alignBytesAttrName, true /*isPropagateUp*/);
    }
  }
  LDBG("Cannot infer alignment before " << reshapeOp.getResult());
  return failure();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, hivm::BitcastOp bitcastOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto [adjustAlignDims, adjustAlignBytes] =
      adjustAlignInfo(bitcastOp, bitcastOp.getSrc(), alignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, bitcastOp.getLoc(), bitcastOp.getSrc(),
                    adjustAlignDims, adjustAlignBytes, alignDimAttrName,
                    alignBytesAttrName);
  return success();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, scf::ForOp scfForOp, OpResult result,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto resultIndx = result.getResultNumber();
  assert(scfForOp.getTiedLoopInit(result) != nullptr);
  auto tiedIterArg = scfForOp.getTiedLoopInit(result)->get();
  auto tiedYieldArg = scfForOp.getYieldedValues()[resultIndx];
  auto [adjustAlignDims, adjustAlignBytes] = adjustAlignInfo(
      scfForOp.getOperation(), tiedIterArg, alignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, scfForOp.getLoc(), tiedIterArg, adjustAlignDims,
                    adjustAlignBytes, alignDimAttrName, alignBytesAttrName);
  createAlignMarkOp(rewriter, scfForOp.getLoc(), tiedYieldArg, adjustAlignDims,
                    adjustAlignBytes, alignDimAttrName, alignBytesAttrName);
  return success();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, scf::ForOp scfForOp,
    BlockArgument blockArgument, ArrayRef<int32_t> alignDims,
    ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto tiedInitArg = scfForOp.getTiedLoopInit(blockArgument)->get();
  auto [adjustAlignDims, adjustAlignBytes] = adjustAlignInfo(
      scfForOp.getOperation(), tiedInitArg, alignDims, alignBytes);
  if (adjustAlignDims.empty()) {
    return failure();
  }
  createAlignMarkOp(rewriter, scfForOp.getLoc(), tiedInitArg, adjustAlignDims,
                    adjustAlignBytes, alignDimAttrName, alignBytesAttrName);
  return success();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, scf::IfOp scfIfOp, OpResult result,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto resultIndx = result.getResultNumber();
  bool propagateAlign = false;
  if (auto thenYield = scfIfOp.thenYield()) {
    auto tiedYieldArg = thenYield->getOperand(resultIndx);
    auto [adjustAlignDims, adjustAlignBytes] = adjustAlignInfo(
        scfIfOp.getOperation(), tiedYieldArg, alignDims, alignBytes);
    if (!adjustAlignDims.empty()) {
      propagateAlign = true;
      createAlignMarkOp(rewriter, scfIfOp.getLoc(), tiedYieldArg,
                        adjustAlignDims, adjustAlignBytes, alignDimAttrName,
                        alignBytesAttrName);
    }
  }

  if (auto elseYield = scfIfOp.elseYield()) {
    auto tiedYieldArg = elseYield->getOperand(resultIndx);
    auto [adjustAlignDims, adjustAlignBytes] = adjustAlignInfo(
        scfIfOp.getOperation(), tiedYieldArg, alignDims, alignBytes);
    if (!adjustAlignDims.empty()) {
      propagateAlign = true;
      createAlignMarkOp(rewriter, scfIfOp.getLoc(), tiedYieldArg,
                        adjustAlignDims, adjustAlignBytes, alignDimAttrName,
                        alignBytesAttrName);
    }
  }
  return propagateAlign ? success() : failure();
}

mlir::LogicalResult propagateAlignUp(
    mlir::PatternRewriter &rewriter, Operation *markedOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  if (utils::isAllocLikeOp(markedOp)) {
    rewriter.modifyOpInPlace(markedOp, [&]() {
      if (!markedOp->hasAttr(alignDimAttrName)) {
        markedOp->setAttr(
            alignDimAttrName,
            DenseI32ArrayAttr::get(rewriter.getContext(), alignDims));
        markedOp->setAttr(
            alignBytesAttrName,
            DenseI32ArrayAttr::get(rewriter.getContext(), alignBytes));
        return;
      }

      auto prevAlignDims =
          markedOp->getAttrOfType<DenseI32ArrayAttr>(alignDimAttrName);
      auto prevAlignBytes =
          markedOp->getAttrOfType<DenseI32ArrayAttr>(alignBytesAttrName);
      assert(prevAlignDims != nullptr);
      assert(prevAlignBytes != nullptr);

      auto [unionAlignDims, unionAlignBytes] =
          unionAlignInfo(prevAlignDims, prevAlignBytes, alignDims, alignBytes);

      markedOp->setAttr(
          alignDimAttrName,
          DenseI32ArrayAttr::get(rewriter.getContext(), unionAlignDims));
      markedOp->setAttr(
          alignBytesAttrName,
          DenseI32ArrayAttr::get(rewriter.getContext(), unionAlignBytes));
    });
    return success();
  }
  markedOp->emitWarning("Align mark on unsupported op");
  return failure();
}

/// Push down an UnrealizedConversionCastOp past a SubViewOp.
UnrealizedConversionCastOp
propagateSubViewOp(RewriterBase &rewriter,
                   UnrealizedConversionCastOp conversionOp,
                   memref::SubViewOp op) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto newSrcMemRefTy = cast<MemRefType>(conversionOp.getOperand(0).getType());
  auto newResultType =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          op.getType().getShape(), newSrcMemRefTy, op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides()));
  Value newSubview = rewriter.create<memref::SubViewOp>(
      op.getLoc(), newResultType, conversionOp.getOperand(0),
      op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides());
  auto newConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), op.getType(), newSubview);
  rewriter.replaceOp(op, newConversionOp);
  return newConversionOp;
}

/// Push down an UnrealizedConversionCastOp past a CollapseShapeOp.
UnrealizedConversionCastOp
propagateCollapseShapeOp(RewriterBase &rewriter,
                         UnrealizedConversionCastOp conversionOp,
                         memref::CollapseShapeOp op) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  Value newCollapse = rewriter.create<memref::CollapseShapeOp>(
      op.getLoc(), conversionOp.getOperand(0), op.getReassociationIndices());
  auto newConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), op.getType(), newCollapse);
  rewriter.replaceOp(op, newConversionOp);
  return newConversionOp;
}

/// Push down an UnrealizedConversionCastOp past a ExpandShapeOp.
UnrealizedConversionCastOp
propagateExpandShapeOp(RewriterBase &rewriter,
                       UnrealizedConversionCastOp conversionOp,
                       memref::ExpandShapeOp op) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  Value newExpand = rewriter.create<memref::ExpandShapeOp>(
      op.getLoc(), op.getResultType().getShape(), conversionOp.getOperand(0),
      op.getReassociationIndices());
  auto newConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), op.getType(), newExpand);
  rewriter.replaceOp(op, newConversionOp);
  return newConversionOp;
}

/// Push down an UnrealizedConversionCastOp past a ReShapeOp.
FailureOrCastVec propagateReshapeOp(RewriterBase &rewriter,
                                    UnrealizedConversionCastOp conversionOp,
                                    memref::ReshapeOp op) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto origSrcTy = op.getSource().getType();
  auto origDstTy = op.getResult().getType();
  auto mayReassocIdx = getReassociationIndicesForReshape(origSrcTy, origDstTy);
  FailureOr<Value> newReshape;
  // Note: reshape can only be applied to Identity-Layout memref
  //       change to collapse_shape / expand_shape
  if (mayReassocIdx.has_value()) {
    if (origSrcTy.getRank() > origDstTy.getRank()) {
      // Treat as memref.collapse_shape
      Value collapse = rewriter.create<memref::CollapseShapeOp>(
          op.getLoc(), conversionOp.getOperand(0), mayReassocIdx.value());
      newReshape = collapse;
    } else {
      // Treat as memref.expand_shape
      Value expand = rewriter.create<memref::ExpandShapeOp>(
          op.getLoc(), origDstTy.getShape(), conversionOp.getOperand(0),
          mayReassocIdx.value());
      newReshape = expand;
    }
  }
  if (failed(newReshape)) {
    LDBG("Failed to propagate aligned memref through reshape");
    return failure();
  }
  auto newConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), op.getType(), newReshape.value());
  rewriter.replaceOp(op, newConversionOp);
  return SmallVector<UnrealizedConversionCastOp>{newConversionOp};
}

/// Push down an UnrealizedConversionCastOp past a CastOp.
FailureOrCastVec propagateCastOp(RewriterBase &rewriter,
                                 UnrealizedConversionCastOp conversionOp,
                                 memref::CastOp castOp) {
  if (llvm::all_of(castOp.getResult().getUsers(),
                   llvm::IsaPred<annotation::MarkOp>)) {
    return FailureOrCastVec(UnrealizedCastOpVec{conversionOp});
  }
  Value newSrc = conversionOp.getOperand(0);
  Type dstType = castOp.getDest().getType();
  if (!memref::CastOp::areCastCompatible({newSrc.getType()}, {dstType})) {
    LDBG("Cannot cast aligned memref " << newSrc << " to " << dstType);
    return FailureOrCastVec{};
  }
  rewriter.setInsertionPoint(castOp);
  auto newCast =
      rewriter.create<memref::CastOp>(castOp.getLoc(), dstType, newSrc);
  rewriter.replaceAllUsesWith(castOp.getResult(), newCast.getResult());
  return FailureOrCastVec(UnrealizedCastOpVec{conversionOp});
}

/// Push down an UnrealizedConversionCastOp past a BitcastOp.
FailureOrCastVec propagateBitcastOp(RewriterBase &rewriter,
                                    UnrealizedConversionCastOp conversionOp,
                                    hivm::BitcastOp bitcastOp) {
  // generate new bitcast op according to conversion src operand
  auto conversionSrcMemType =
      cast<MemRefType>(conversionOp.getOperand(0).getType());
  auto bitcastResElemType =
      getElementTypeOrSelf(bitcastOp.getResult().getType());
  auto newBitcastResType = conversionSrcMemType.clone(bitcastResElemType);

  rewriter.setInsertionPoint(bitcastOp);
  auto newBitcastOp = rewriter.create<hivm::BitcastOp>(
      bitcastOp.getLoc(), newBitcastResType, conversionOp.getOperand(0));

  // add conversion op to replace
  auto newConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      newBitcastOp.getLoc(), newBitcastOp.getType(), newBitcastOp.getResult());
  rewriter.replaceOp(bitcastOp, newConversionOp);

  return SmallVector<UnrealizedConversionCastOp>({newConversionOp});
}

/// Push down an UnrealizedConversionCastOp past a scf ForOp.
FailureOrCastVec propagateScfForOp(RewriterBase &rewriter,
                                   UnrealizedConversionCastOp conversionOp,
                                   scf::ForOp op, unsigned int initIndx) {
  UnrealizedCastOpVec newConversionOps;
  OpBuilder::InsertionGuard g(rewriter);

  // set init value, update iter_arg type and result type
  op.getInitArgsMutable()[initIndx].assign(conversionOp.getOperand(0));
  auto origType = conversionOp.getResult(0).getType();
  auto newType = conversionOp.getOperand(0).getType();
  op.getRegionIterArg(initIndx).setType(newType);
  op.getResult(initIndx).setType(newType);

  // insert unrealized conversion cast after op result, before yield op and the
  // begin of region for region iter_arg
  rewriter.setInsertionPointAfter(op);
  auto resultConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), origType, op.getResult(initIndx));
  newConversionOps.push_back(resultConversionOp);
  rewriter.replaceAllUsesExcept(op.getResult(initIndx),
                                resultConversionOp.getResult(0),
                                resultConversionOp);

  // insert unrealized conversion cast the begin of region for region iter_arg
  rewriter.setInsertionPointToStart(op.getBody());
  auto iterArgConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), origType, op.getRegionIterArg(initIndx));
  newConversionOps.push_back(iterArgConversionOp);
  rewriter.replaceAllUsesExcept(op.getRegionIterArg(initIndx),
                                iterArgConversionOp.getResult(0),
                                iterArgConversionOp);

  // insert unrealized conversion cast before yield op
  rewriter.setInsertionPoint(op.getBody()->getTerminator());
  auto yieldValues = op.getYieldedValues();
  auto yieldValueConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), newType, yieldValues[initIndx]);
  auto mutableYieldValues = op.getYieldedValuesMutable();
  assert(mutableYieldValues.has_value());
  mutableYieldValues.value()[initIndx].assign(
      yieldValueConversionOp.getResult(0));

  return newConversionOps;
}

std::optional<Value> getConversionSrc(Value v) {
  auto conversionOp = v.getDefiningOp<UnrealizedConversionCastOp>();
  if (conversionOp == nullptr) {
    return std::nullopt;
  }

  return conversionOp->getOperand(0);
}

FailureOrCastVec propagateScfIfOp(RewriterBase &rewriter, scf::IfOp op,
                                  unsigned int yieldIndx) {
  auto thenYieldOp = op.thenYield();
  auto elseYieldOp = op.elseYield();
  assert(thenYieldOp && elseYieldOp);

  auto thenYieldValue = thenYieldOp->getOperand(yieldIndx);
  auto elseYieldValue = elseYieldOp->getOperand(yieldIndx);
  auto thenYieldConversionSrc = getConversionSrc(thenYieldValue);
  auto elseYieldConversionSrc = getConversionSrc(elseYieldValue);
  if (!thenYieldConversionSrc.has_value() ||
      !elseYieldConversionSrc.has_value()) {
    return UnrealizedCastOpVec{};
  }
  if (thenYieldConversionSrc.value().getType() !=
      elseYieldConversionSrc.value().getType()) {
    return UnrealizedCastOpVec{};
  }

  // replace yield value by conversion src and modify if result type
  rewriter.modifyOpInPlace(thenYieldOp, [&]() {
    thenYieldOp.setOperand(yieldIndx, thenYieldConversionSrc.value());
  });
  rewriter.modifyOpInPlace(elseYieldOp, [&]() {
    elseYieldOp.setOperand(yieldIndx, elseYieldConversionSrc.value());
  });

  auto ifResult = op->getResult(yieldIndx);
  auto ifResultOrigType = ifResult.getType();
  auto ifResultNewType = thenYieldConversionSrc.value().getType();
  ifResult.setType(ifResultNewType);

  // insert conversion of if result from modified type to original type
  // and replace original if result by conversion result
  rewriter.setInsertionPointAfter(op);
  auto resultConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), ifResultOrigType, ifResult);
  rewriter.replaceAllUsesExcept(ifResult, resultConversionOp.getResult(0),
                                resultConversionOp);
  return UnrealizedCastOpVec{resultConversionOp};
}

/// Push down an UnrealizedConversionCastOp past a CastOp.
FailureOrCastVec propagateYieldOp(RewriterBase &rewriter, scf::YieldOp yieldOp,
                                  unsigned int yieldIndex) {
  auto *yieldParentOp = yieldOp->getBlock()->getParentOp();
  if (auto ifOp = dyn_cast_if_present<scf::IfOp>(yieldParentOp)) {
    return propagateScfIfOp(rewriter, ifOp, yieldIndex);
  }
  return UnrealizedCastOpVec{};
}

/// Push down an UnrealizedConversionCastOp past a UnrealizedConversionCastOp.
FailureOrCastVec
propagateUnrealizedConversionCastOp(RewriterBase &rewriter,
                                    UnrealizedConversionCastOp producer,
                                    UnrealizedConversionCastOp consumer) {
  LDBG("propagate unrealized conversion cast down unrealized conversion cast "
       "op : "
       << *consumer);
  // Producer: %cast  = unrealized_conversion %src  : A to B
  // Consumer: %dst   = unrealized_conversion %cast : B to A
  // Directly replace the use of `%dst` with `%src`
  if (producer.getInputs().getType() == consumer.getOutputs().getType()) {
    for (auto [src, dst] :
         llvm::zip_equal(producer.getInputs(), consumer.getOutputs()))
      rewriter.replaceAllUsesWith(dst, src);
    return UnrealizedCastOpVec{};
  }
  return FailureOrCastVec{};
}

FailureOrCastVec propagateDefaultOp(RewriterBase &rewriter,
                                    UnrealizedConversionCastOp conversionOp,
                                    Operation *op, Operation *user) {
  LDBG("propagate unrealized conversion cast down default op : " << *op);
  // Skip any ops that produce MemRef result or have MemRef
  // region block arguments. These may need special handling
  // (e.g., scf.for).
  if (llvm::any_of(user->getResultTypes(),
                   [](Type t) { return isa<MemRefType>(t); })) {
    LDBG("Cannot replace uses of memref for memref-producing "
         "operation");
    return FailureOrCastVec{};
  }
  if (llvm::any_of(user->getRegions(), [](Region &r) {
        return llvm::any_of(r.getArguments(), [](BlockArgument bbArg) {
          return isa<MemRefType>(bbArg.getType());
        });
      })) {
    LDBG("Cannot replace uses of memref as block arguments");
    return FailureOrCastVec{};
  }
  rewriter.modifyOpInPlace(op, [&]() {
    op->replaceUsesOfWith(conversionOp.getOutputs()[0],
                          conversionOp.getInputs()[0]);
  });
  // No new conversion, return the original conversion
  return UnrealizedCastOpVec{conversionOp};
}

mlir::LogicalResult propagateAlignUpFromResult(
    mlir::PatternRewriter &rewriter, OpResult result, annotation::MarkOp markOp,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName = hivm::StrideAlignDimsAttr::name.str(),
    std::string alignBytesAttrName =
        hivm::StrideAlignValueInByteAttr::name.str()) {
  auto *defOp = result.getDefiningOp();
  assert(defOp);
  return TypeSwitch<Operation *, LogicalResult>(defOp)
      .Case([&](memref::CastOp castOp) {
        return propagateAlignUp(rewriter, castOp, alignDims, alignBytes,
                                alignDimAttrName, alignBytesAttrName);
      })
      .Case([&](memref::SubViewOp subviewOp) {
        return propagateAlignUp(rewriter, subviewOp, alignDims, alignBytes,
                                alignDimAttrName, alignBytesAttrName);
      })
      .Case([&](memref::ViewOp viewOp) {
        return propagateAlignUp(rewriter, markOp, viewOp, alignDims,
                                alignBytes);
      })
      .Case([&](memref::CollapseShapeOp collapseOp) {
        return propagateAlignInfoByExpand<memref::CollapseShapeOp>(
            rewriter, collapseOp, collapseOp.getReassociationIndices(),
            alignDims, alignBytes, alignDimAttrName, alignBytesAttrName,
            true /*isPropagateUp*/);
      })
      .Case([&](memref::ExpandShapeOp expandOp) {
        return propagateAlignInfoByCollapse<memref::ExpandShapeOp>(
            rewriter, expandOp, expandOp.getReassociationIndices(), alignDims,
            alignBytes, alignDimAttrName, alignBytesAttrName,
            true /*isPropagateUp*/);
      })
      .Case([&](memref::ReshapeOp reshapeOp) {
        return propagateAlignUp(rewriter, reshapeOp, alignDims, alignBytes,
                                alignDimAttrName, alignBytesAttrName);
      })
      .Case([&](hivm::BitcastOp bitcastOp) {
        return propagateAlignUp(rewriter, bitcastOp, alignDims, alignBytes,
                                alignDimAttrName, alignBytesAttrName);
      })
      .Case([&](scf::ForOp scfForOp) {
        return propagateAlignUp(rewriter, scfForOp, result, alignDims,
                                alignBytes, alignDimAttrName,
                                alignBytesAttrName);
      })
      .Case([&](scf::IfOp scfIfOp) {
        return propagateAlignUp(rewriter, scfIfOp, result, alignDims,
                                alignBytes, alignDimAttrName,
                                alignBytesAttrName);
      })
      .Default([&](Operation *markedOp) {
        return propagateAlignUp(rewriter, markedOp, alignDims, alignBytes,
                                alignDimAttrName, alignBytesAttrName);
      });
}

} // namespace

void util::AlignInfo::dump() { hivm::dump(alignDims, alignBytes, DEBUG_TYPE); }

bool util::AlignInfo::operator==(const AlignInfo &other) {
  if (this->alignDims.size() != other.alignDims.size()) {
    return false;
  }

  for (size_t i = 0; i < this->alignDims.size(); i++) {
    if (this->alignDims[i] != other.alignDims[i]) {
      return false;
    }
    if (this->alignBytes[i] != other.alignBytes[i]) {
      return false;
    }
  }

  return true;
}

bool util::AlignInfo::operator!=(const AlignInfo &other) { return !(*this == other); }

void mlir::hivm::populatePropagateAlignUpToRootAllocationPattern(
    RewritePatternSet &patterns, std::string alignDimAttrName,
    std::string alignBytesAttrName) {
  patterns.add<PropagateAlignUpToRootAllocationPattern>(
      patterns.getContext(), alignDimAttrName, alignBytesAttrName);
}

std::pair<llvm::SmallVector<int32_t>, llvm::SmallVector<int32_t>>
mlir::hivm::unionAlignInfo(const ArrayRef<int32_t> &alignDims,
                           const ArrayRef<int32_t> &alignBytes,
                           const ArrayRef<int32_t> &otherAlignDims,
                           const ArrayRef<int32_t> &otherAlignBytes,
                           bool isSorted) {
  // sort according to dimensions
  std::vector<std::pair<int32_t, int32_t>> sortedAlignInfos;
  std::vector<std::pair<int32_t, int32_t>> sortedOtherAlignInfos;
  if (isSorted) {
    for (auto [dim, byte] : llvm::zip(alignDims, alignBytes)) {
      sortedAlignInfos.emplace_back(dim, byte);
    }
    for (auto [dim, byte] : llvm::zip(otherAlignDims, otherAlignBytes)) {
      sortedOtherAlignInfos.emplace_back(dim, byte);
    }
  } else {
    sortedAlignInfos = sortAlignInfo(alignDims, alignBytes);
    sortedOtherAlignInfos = sortAlignInfo(otherAlignDims, otherAlignBytes);
  }

  llvm::SmallVector<int32_t> unionAlignDims;
  llvm::SmallVector<int32_t> unionAlignBytes;

  auto thisIt = sortedAlignInfos.begin();
  auto thatIt = sortedOtherAlignInfos.begin();
  while (thisIt != sortedAlignInfos.end() &&
         thatIt != sortedOtherAlignInfos.end()) {
    if (thisIt->first < thatIt->first) {
      unionAlignDims.push_back(thisIt->first);
      unionAlignBytes.push_back(thisIt->second);
      thisIt++;
    } else if (thisIt->first > thatIt->first) {
      unionAlignDims.push_back(thatIt->first);
      unionAlignBytes.push_back(thatIt->second);
      thatIt++;
    } else {
      assert(thisIt->first == thatIt->first);
      unionAlignDims.push_back(thisIt->first);
      unionAlignBytes.push_back(std::lcm(thisIt->second, thatIt->second));
      thisIt++;
      thatIt++;
    }
  }

  auto leftIt = thisIt == sortedAlignInfos.end() ? thatIt : thisIt;
  const auto &leftAlignInfos = thisIt == sortedAlignInfos.end()
                                   ? sortedOtherAlignInfos
                                   : sortedAlignInfos;
  for (auto it = leftIt; it < leftAlignInfos.end(); it++) {
    unionAlignDims.push_back(it->first);
    unionAlignBytes.push_back(it->second);
  }
  return std::make_pair(unionAlignDims, unionAlignBytes);
}

std::optional<annotation::MarkOp> mlir::hivm::createAlignMarkOp(
    OpBuilder &builder, const Location loc, Value markedVal,
    ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
    std::string alignDimAttrName, std::string alignBytesAttrName) {
  if (alignDims.empty())
    return std::nullopt;
  auto point = builder.saveInsertionPoint();
  builder.setInsertionPointAfterValue(markedVal);
  auto markOp = builder.create<annotation::MarkOp>(loc, markedVal);
  markOp->setAttr(alignDimAttrName, builder.getDenseI32ArrayAttr(alignDims));
  markOp->setAttr(alignBytesAttrName, builder.getDenseI32ArrayAttr(alignBytes));
  builder.restoreInsertionPoint(point);
  return markOp;
}

OpFoldResult mlir::hivm::AlignUpOFR(OpBuilder &b, const Location loc,
                                    OpFoldResult lenOFR, uint64_t alignUnit) {
  assert(alignUnit != 0);
  auto lenConst = getConstantIntValue(lenOFR);
  if (lenConst.has_value())
    return b.getIndexAttr(AlignUp(lenConst.value(), alignUnit));
  if (alignUnit == 1)
    return lenOFR;
  Value lenVal = lenOFR.get<Value>();
  Value padVal =
      b.create<arith::ConstantOp>(loc, b.getIndexAttr(alignUnit - 1));
  Value ceilVal = b.create<arith::AddIOp>(loc, lenVal, padVal);
  Value unitVal = b.create<arith::ConstantOp>(loc, b.getIndexAttr(alignUnit));
  Value remVal = b.create<arith::RemSIOp>(loc, ceilVal, unitVal);
  Value alignVal = b.create<arith::SubIOp>(loc, ceilVal, remVal);
  return alignVal;
}

std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
mlir::hivm::calculateAlignedShape(OpBuilder &b, const Location loc,
                                  const SmallVector<OpFoldResult> &shape,
                                  const SmallVector<int> &alignUnits) {
  // expand shape (s0, s1, ... sn) to (s0, s1, ... sn, 1) for processing
  // storage alignment of last dimenstion with stride
  SmallVector<OpFoldResult> subShape(shape);
  subShape.push_back(b.getIndexAttr(1));
  SmallVector<OpFoldResult> alignedShape(subShape.size());
  assert(alignUnits.size() <= alignedShape.size());
  for (size_t dim = 0; dim < alignUnits.size(); ++dim) {
    alignedShape[dim] = AlignUpOFR(b, loc, subShape[dim], alignUnits[dim]);
  }
  return std::make_pair(alignedShape, subShape);
}

/// Replace all uses of %from are replaced with %to. For view-like ops (e.g.,
/// memref.subview), the result type may depend on the operand type, so we
/// cannot just replace all uses.
/// This implementation is based on the (static) function `void
/// replaceAndPropagateMemRefType()` in:
/// mlir/lib/Dialect/MemRef/Transforms/IndependenceTransforms.cpp
///
/// Handled memref ops: cast/subview/collapse_shape/expand_shape/reshape.
/// Note: memref.view cannot be aligned, because it assumes empty layout
LogicalResult mlir::hivm::replaceAndPropagateMemRefType(RewriterBase &rewriter,
                                                        const Location loc,
                                                        Value from, Value to) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(to);

  // Wrap new results in unrealized_conversion_cast and replace all uses of
  // the original op.
  UnrealizedCastOpVec unrealizedConversions;
  auto initConversion =
      rewriter.create<UnrealizedConversionCastOp>(loc, from.getType(), to);
  unrealizedConversions.push_back(initConversion);
  rewriter.replaceAllUsesWith(from, initConversion.getResult(0));

  // Push unrealized_conversion_cast ops further down in the IR. I.e., try to
  // wrap results instead of operands in a cast.
  for (size_t i = 0; i < unrealizedConversions.size(); ++i) {
    UnrealizedConversionCastOp conversion = unrealizedConversions[i];
    // Warning: must store users in a vector because rewriting modifies
    // def-use chain
    SmallVector<OpOperand *> uses = llvm::map_to_vector(
        conversion->getUses(), [](OpOperand &operand) { return &operand; });
    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      // Handle common memref dialect ops that produce new memrefs and must
      // be recreated with the new result type.
      auto res =
          TypeSwitch<Operation *, FailureOrCastVec>(user)
              .Case([&rewriter, &conversion](memref::SubViewOp subviewOp) {
                auto res = propagateSubViewOp(rewriter, conversion, subviewOp);
                return UnrealizedCastOpVec{res};
              })
              .Case([&rewriter,
                     &conversion](memref::CollapseShapeOp collapseOp) {
                auto res =
                    propagateCollapseShapeOp(rewriter, conversion, collapseOp);
                return UnrealizedCastOpVec{res};
              })
              .Case([&rewriter, &conversion](memref::ExpandShapeOp expandOp) {
                auto res =
                    propagateExpandShapeOp(rewriter, conversion, expandOp);
                return UnrealizedCastOpVec{res};
              })
              .Case([&rewriter, &conversion](memref::ReshapeOp reshapeOp) {
                return propagateReshapeOp(rewriter, conversion, reshapeOp);
              })
              .Case([&rewriter, &conversion](memref::CastOp castOp) {
                return propagateCastOp(rewriter, conversion, castOp);
              })
              .Case([&rewriter, &conversion](hivm::BitcastOp bitcastOp) {
                return propagateBitcastOp(rewriter, conversion, bitcastOp);
              })
              .Case([&rewriter, &conversion, &use](scf::ForOp forOp) {
                unsigned int initIndx =
                    forOp.getTiedLoopResult(use).getResultNumber();
                return propagateScfForOp(rewriter, conversion, forOp, initIndx);
              })
              .Case([&rewriter, &use](scf::YieldOp yieldOp) {
                return propagateYieldOp(rewriter, yieldOp,
                                        use->getOperandNumber());
              })
              .Case([&rewriter,
                     &conversion](UnrealizedConversionCastOp conversionOp) {
                return propagateUnrealizedConversionCastOp(rewriter, conversion,
                                                           conversionOp);
              })
              .Default([&rewriter, &conversion, &user](Operation *op) {
                return propagateDefaultOp(rewriter, conversion, op, user);
              });
      if (failed(res)) {
        LDBG("unexpected failure");
        return failure();
      }

      for (auto newOp : res.value()) {
        if (newOp != conversion) {
          unrealizedConversions.push_back(newOp);
        }
      }
    }
  }

  // Erase all unrealized_conversion_cast ops without uses.
  for (auto op : unrealizedConversions)
    if (op->getUses().empty())
      rewriter.eraseOp(op);

  return success();
}

mlir::LogicalResult PropagateAlignUpToRootAllocationPattern::matchAndRewrite(
    annotation::MarkOp markOp, mlir::PatternRewriter &rewriter) const {
  auto alignDims = markOp->getAttrOfType<DenseI32ArrayAttr>(alignDimAttrName_);
  auto alignBytes =
      markOp->getAttrOfType<DenseI32ArrayAttr>(alignBytesAttrName_);
  if (alignDims == nullptr || alignBytes == nullptr)
    return failure();
  if (alignDims.size() != alignBytes.size())
    return markOp.emitError() << "Mismatched storage align marks";
  auto markSrc = markOp.getSrc();
  if (isa<BlockArgument>(markSrc) &&
      !isa<scf::ForOp>(cast<BlockArgument>(markSrc).getOwner()->getParentOp()))
    return markOp.emitError()
           << "Cannot align " << markSrc << " across blocks.";
  if (isa<UnrankedMemRefType>(markSrc.getType()))
    return markOp.emitError() << "Cannot align unranked memref " << markSrc;

  LogicalResult result = success();
  if (auto defOp = markSrc.getDefiningOp()) {
    result = propagateAlignUpFromResult(rewriter, cast<OpResult>(markSrc),
                                        markOp, alignDims, alignBytes,
                                        alignDimAttrName_, alignBytesAttrName_);
  } else if (isa<BlockArgument>(markSrc) &&
             isa<scf::ForOp>(
                 cast<BlockArgument>(markSrc).getOwner()->getParentOp())) {
    auto blockArgument = cast<BlockArgument>(markSrc);
    auto scfForOp = cast<scf::ForOp>(blockArgument.getOwner()->getParentOp());
    result = propagateAlignUp(rewriter, scfForOp, blockArgument, alignDims,
                              alignBytes);
  }

  rewriter.modifyOpInPlace(markOp, [&]() {
    removeMarkOpAttr(markOp, llvm::StringRef(alignDimAttrName_), rewriter,
                     false /*removeOp*/);
    removeMarkOpAttr(markOp, llvm::StringRef(alignBytesAttrName_), rewriter,
                     false /*removeOp*/);
  });
  if (markOp->getAttrDictionary().empty()) {
    rewriter.eraseOp(markOp);
  }
  return result;
}

LogicalResult PropagateAlignDownToLeafOperandsPattern::matchAndRewrite(
    annotation::MarkOp markOp, mlir::PatternRewriter &rewriter) const {
  if (!utils::isAnnotationWithAttr(markOp, StrideAlignDimsAttr::name)) {
    return failure();
  }

  auto alignDims =
      markOp->getAttrOfType<DenseI32ArrayAttr>(hivm::StrideAlignDimsAttr::name);
  auto alignBytes = markOp->getAttrOfType<DenseI32ArrayAttr>(
      hivm::StrideAlignValueInByteAttr::name);
  assert(alignDims != nullptr);
  assert(alignBytes != nullptr);

  return propagateDownAlignInfo(rewriter, markOp.getSrc(), alignDims,
                                alignBytes);
}
