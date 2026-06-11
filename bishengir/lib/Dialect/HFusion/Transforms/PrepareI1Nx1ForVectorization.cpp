//===- PrepareI1Nx1ForVectorization.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_PREPAREI1NX1FORVECTORIZATION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static constexpr int64_t kNx1Rank = 2;
static constexpr int64_t kOuterDim = 0;
static constexpr int64_t kUnitDim = 1;

SmallVector<ReassociationIndices> getNx1Reassociation() { return {{0, 1}}; }

bool isStaticNx1Tensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == kNx1Rank &&
         tensorType.hasStaticShape() && tensorType.getDimSize(kUnitDim) == 1 &&
         !tensorType.getEncoding();
}

RankedTensorType getCollapsedNx1Type(RankedTensorType type) {
  return RankedTensorType::get({type.getDimSize(kOuterDim)},
                               type.getElementType(), type.getEncoding());
}

bool isScalarType(Type type) { return !isa<ShapedType>(type); }

AffineMap getRank1IdentityMap(MLIRContext *context) {
  return AffineMap::getMultiDimIdentityMap(/*rank=*/1, context);
}

AffineMap getRank1ScalarMap(MLIRContext *context) {
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, context);
}

bool isRank1ToNx1I1Expand(tensor::ExpandShapeOp expandOp,
                          RankedTensorType expectedResultType) {
  auto srcType = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
  if (!srcType || expandOp.getResultType() != expectedResultType ||
      srcType.getRank() != 1 || !srcType.hasStaticShape() ||
      !srcType.getElementType().isInteger(1) ||
      !expectedResultType.getElementType().isInteger(1))
    return false;

  if (srcType.getDimSize(0) != expectedResultType.getDimSize(kOuterDim))
    return false;

  SmallVector<ReassociationIndices> reassociation =
      expandOp.getReassociationIndices();
  return reassociation.size() == 1 && reassociation[0].size() == 2 &&
         reassociation[0][0] == 0 && reassociation[0][1] == 1;
}

bool isSupportedNx1Generic(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op.getNumLoops() != kNx1Rank)
    return false;

  bool hasUnsupportedIndexOp = false;
  op.walk([&](linalg::IndexOp indexOp) {
    if (indexOp.getDim() != kOuterDim)
      hasUnsupportedIndexOp = true;
  });
  if (hasUnsupportedIndexOp)
    return false;

  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType iter) {
        return iter == utils::IteratorType::parallel;
      }))
    return false;

  auto isRank2IdentityMap = [&](AffineMap map) {
    return map == AffineMap::getMultiDimIdentityMap(kNx1Rank, op.getContext());
  };
  auto isRank2ScalarMap = [](AffineMap map) {
    return map.getNumDims() == kNx1Rank && map.getNumResults() == 0;
  };

  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  size_t numDpsInputs = static_cast<size_t>(op.getNumDpsInputs());
  size_t numDpsInits = static_cast<size_t>(op.getNumDpsInits());
  if (indexingMaps.size() != numDpsInputs + numDpsInits)
    return false;

  std::optional<int64_t> outerDim;
  auto checkTensor = [&](Value value) {
    if (!isStaticNx1Tensor(value.getType()))
      return false;
    auto tensorType = cast<RankedTensorType>(value.getType());
    if (!outerDim) {
      outerDim = tensorType.getDimSize(kOuterDim);
      return true;
    }
    return *outerDim == tensorType.getDimSize(kOuterDim);
  };

  for (auto [idx, input] : llvm::enumerate(op.getDpsInputs())) {
    Type inputType = input.getType();
    if (isScalarType(inputType)) {
      if (!isRank2ScalarMap(indexingMaps[idx]))
        return false;
      continue;
    }
    if (!checkTensor(input) || !isRank2IdentityMap(indexingMaps[idx]))
      return false;
  }
  for (auto [idx, init] : llvm::enumerate(op.getDpsInits())) {
    auto mapIdx = numDpsInputs + idx;
    if (!checkTensor(init) || !isRank2IdentityMap(indexingMaps[mapIdx]))
      return false;
  }

  bool hasExpandedI1Input = false;
  for (Value input : op.getDpsInputs()) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      continue;
    auto expandOp = input.getDefiningOp<tensor::ExpandShapeOp>();
    if (expandOp && isRank1ToNx1I1Expand(expandOp, inputType)) {
      hasExpandedI1Input = true;
      break;
    }
  }

  return hasExpandedI1Input;
}

Value collapseNx1Tensor(Value value, PatternRewriter &rewriter, Location loc) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  return rewriter.create<tensor::CollapseShapeOp>(
      loc, getCollapsedNx1Type(tensorType), value, getNx1Reassociation());
}

Value collapseNx1TensorOrKeepScalar(Value value, PatternRewriter &rewriter,
                                    Location loc) {
  if (isScalarType(value.getType()))
    return value;
  return collapseNx1Tensor(value, rewriter, loc);
}

struct CollapseI1Nx1GenericPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isSupportedNx1Generic(op))
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value> newInputs;
    SmallVector<Value> newOutputs;
    SmallVector<Type> newResultTypes;

    for (Value input : op.getDpsInputs())
      newInputs.push_back(collapseNx1TensorOrKeepScalar(input, rewriter, loc));

    for (Value init : op.getDpsInits()) {
      Value collapsedInit = collapseNx1Tensor(init, rewriter, loc);
      newOutputs.push_back(collapsedInit);
      newResultTypes.push_back(collapsedInit.getType());
    }

    SmallVector<AffineMap> newIndexingMaps;
    for (Value input : newInputs) {
      newIndexingMaps.push_back(isScalarType(input.getType())
                                    ? getRank1ScalarMap(op.getContext())
                                    : getRank1IdentityMap(op.getContext()));
    }
    newIndexingMaps.append(newOutputs.size(),
                           getRank1IdentityMap(op.getContext()));
    SmallVector<utils::IteratorType> newIteratorTypes{
        utils::IteratorType::parallel};

    auto newOp = rewriter.create<linalg::GenericOp>(
        loc, newResultTypes, newInputs, newOutputs, newIndexingMaps,
        newIteratorTypes);
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());

    SmallVector<Value> expandedResults;
    rewriter.setInsertionPointAfter(newOp);
    for (auto [idx, oldResult] : llvm::enumerate(op->getResults())) {
      auto oldResultType = cast<RankedTensorType>(oldResult.getType());
      Value expanded = rewriter.create<tensor::ExpandShapeOp>(
          loc, oldResultType, newOp.getResult(idx), getNx1Reassociation());
      expandedResults.push_back(expanded);
    }

    rewriter.replaceOp(op, expandedResults);
    return success();
  }
};

struct PrepareI1Nx1ForVectorizationPass
    : public impl::PrepareI1Nx1ForVectorizationBase<
          PrepareI1Nx1ForVectorizationPass> {
  using PrepareI1Nx1ForVectorizationBase<
      PrepareI1Nx1ForVectorizationPass>::PrepareI1Nx1ForVectorizationBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapseI1Nx1GenericPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPrepareI1Nx1ForVectorizationPass() {
  return std::make_unique<PrepareI1Nx1ForVectorizationPass>();
}
