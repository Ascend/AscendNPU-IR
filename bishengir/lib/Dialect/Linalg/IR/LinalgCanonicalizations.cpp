//===- LinalgCanonicalizations.cpp - Linalg Canonicalization impl ---------===//
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

#include "bishengir/Dialect/Linalg/IR/LinalgCanonicalizations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

static SmallVector<SmallVector<int64_t, 2>>
getReAssociation(ArrayRef<int64_t> expandDims, int64_t outRank) {
  std::set<int> expandDimsSet;
  expandDimsSet.insert(expandDims.begin(), expandDims.end());

  SmallVector<SmallVector<int64_t, 2>> retVecVec;
  SmallVector<int64_t, 2> vec;

  // push contiguous expand dims in the head of seq into vec
  int i = 0;
  for (; i < outRank; i++) {
    bool isExpandDim = expandDimsSet.count(i);
    if (isExpandDim) {
      vec.push_back(i);
    } else {
      break;
    }
  }

  // cut the vec if next is unexpand dim or unexisted
  for (; i < outRank; ++i) {
    vec.push_back(i);

    bool nextIsUnExpand = !expandDimsSet.count(i + 1);
    if (nextIsUnExpand) {
      // unexpanded dim
      retVecVec.push_back(vec);
      vec.clear();
    }
  }

  if (!vec.empty()) {
    retVecVec.push_back(vec);
  }
  return retVecVec;
}

/// Pattern to fold transpose into expand shape.
///
/// Before:
/// tensor.expand_shape + linalg.transpose
///
/// After:
/// tensor.expand_shape
///
/// Restrictions:
/// Only support the expand op expands extra 1 dim, like unsqueeze,
/// and the expanded dim is permuted by the transpose op.
/// In these case, pattern will fail:
/// (1) the number of expand 1 dim > 1
/// (2) the expand dim is not be permuted
/// (3) contains dynamic dim
/// (4) transpose op cannot be elimated after adjusting the expand op
struct FoldTransposeWithExpand : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  bool hasDynamicDim(tensor::ExpandShapeOp defExpandOp) const {
    auto defStaticOutputShape = defExpandOp.getStaticOutputShape();
    for (auto shape : defStaticOutputShape) {
      if (shape == ShapedType::kDynamic) {
        return true;
      }
    }
    return false;
  }

  bool isOnlyExpandUnitDims(tensor::ExpandShapeOp defExpandOp) const {
    auto ressociations = defExpandOp.getReassociationIndices();
    auto defStaticOutputShape = defExpandOp.getStaticOutputShape();
    for (const auto &ressociation : ressociations) {
      if (ressociation.size() == 1) {
        continue;
      }
      unsigned long unitNum = 0;
      for (const auto &dim : ressociation) {
        if (defStaticOutputShape[dim] == 1) {
          ++unitNum;
        }
      }
      if (unitNum < ressociation.size() - 1) {
        return false;
      }
    }
    return true;
  }

  SmallVector<int64_t>
  getExpandUnitDims(tensor::ExpandShapeOp defExpandOp) const {
    auto reassociations = defExpandOp.getReassociationIndices();
    auto defStaticOutputShape = defExpandOp.getStaticOutputShape();
    auto inputTy =
        llvm::cast<RankedTensorType>(defExpandOp->getOperand(0).getType());
    auto inputShape = inputTy.getShape();
    SmallVector<int64_t> expandShapes;
    for (size_t i = 0; i < inputShape.size(); ++i) {
      auto reassociation = reassociations[i];
      if (reassociation.size() == 1) {
        continue;
      }
      // If expand like [1] -> [1, 1], we cannot easily detect which dim should
      // be chosen to push_back expandShapes, so here just drop the last.
      if (inputShape[i] == 1) {
        expandShapes.append(reassociation.begin(), reassociation.end() - 1);
        continue;
      }
      for (const auto &dim : reassociation) {
        if (defStaticOutputShape[dim] == 1) {
          expandShapes.push_back(dim);
        }
      }
    }
    return expandShapes;
  }

  int64_t getIdxAfterPerm(int64_t expandDim,
                          const ArrayRef<int64_t> &perms) const {
    int64_t idxAfterPerm = -1;
    for (size_t i = 0; i < perms.size(); ++i) {
      if (perms[i] == expandDim) {
        idxAfterPerm = i;
      }
    }
    return idxAfterPerm;
  }

  bool canFold(tensor::ExpandShapeOp defExpandOp,
               linalg::TransposeOp transposeOp) const {
    // Not support dynamic now.
    if (hasDynamicDim(defExpandOp)) {
      return false;
    }

    if (!isOnlyExpandUnitDims(defExpandOp)) {
      return false;
    }

    auto expandUnitDims = getExpandUnitDims(defExpandOp);
    // Not support more than one expand dim now
    if (expandUnitDims.size() > 1) {
      return false;
    }

    auto expandDim = expandUnitDims[0];
    ArrayRef<int64_t> perms = transposeOp.getPermutation();

    int64_t idxAfterPerm = getIdxAfterPerm(expandDim, perms);
    if (idxAfterPerm == -1) {
      return false;
    }

    if (!isOnlyTransposeUnitDims(idxAfterPerm, perms)) {
      return false;
    }

    return true;
  }

  bool isOnlyTransposeUnitDims(int64_t idxAfterPerm,
                               const ArrayRef<int64_t> &perms) const {
    // An easy way to check if transpose can be elimated after insert the new
    // expand op: find the expand dim in perms and erase it, then check if the
    // remained dims are ordered.
    SmallVector<int64_t> tmpPerms(perms);
    tmpPerms.erase(tmpPerms.begin() + idxAfterPerm);
    return std::is_sorted(tmpPerms.begin(), tmpPerms.end());
  }

  SmallVector<int64_t> getExpandShape(tensor::ExpandShapeOp defExpandOp,
                                      int64_t expandDim,
                                      int64_t idxAfterPerm) const {
    SmallVector<int64_t> outputShape(defExpandOp.getStaticOutputShape());
    outputShape.erase(outputShape.begin() + expandDim);
    outputShape.insert(outputShape.begin() + idxAfterPerm, 1);
    return outputShape;
  }

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto defExpandOp =
        transposeOp.getInput().getDefiningOp<tensor::ExpandShapeOp>();
    if (!defExpandOp)
      return failure();
    auto inputTy =
        llvm::cast<RankedTensorType>(defExpandOp->getOperand(0).getType());

    if (!canFold(defExpandOp, transposeOp)) {
      return rewriter.notifyMatchFailure(transposeOp, "cannot fold");
    }

    auto expandDim = getExpandUnitDims(defExpandOp)[0];
    auto idxAfterPerm =
        getIdxAfterPerm(expandDim, transposeOp.getPermutation());

    auto newExpandShape = getExpandShape(defExpandOp, expandDim, idxAfterPerm);
    auto newExpandTy =
        RankedTensorType::get(newExpandShape, inputTy.getElementType());
    auto newReassociation =
        getReAssociation(SmallVector<int64_t>{idxAfterPerm},
                         defExpandOp.getStaticOutputShape().size());
    Value result = rewriter.create<tensor::ExpandShapeOp>(
        defExpandOp.getLoc(), newExpandTy, defExpandOp->getOperand(0),
        newReassociation);
    rewriter.replaceOp(transposeOp, result);
    return success();
  }
};

struct MergeConsecutiveReduceOp : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only support second reduce op with one input");
    }
    Value input = op.getDpsInputs().front();
    if (!input.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "not support first reduce op result with multiple users");
    }
    Operation *inputOp = input.getDefiningOp();
    auto prevReduce = dyn_cast_or_null<linalg::ReduceOp>(inputOp);
    if (!prevReduce) {
      return rewriter.notifyMatchFailure(op, "not find consecutive reduces");
    }
    if (!isReduceWithSameRegion(op, prevReduce)) {
      return rewriter.notifyMatchFailure(
          op, "not support reduce with different region");
    }
    SmallVector<unsigned> dims0;
    prevReduce.getReductionDims(dims0);
    SmallVector<unsigned> dims1;
    op.getReductionDims(dims1);
    unsigned maxRank = prevReduce.getRank(prevReduce.getDpsInputOperand(0));

    SmallVector<int64_t> dims =
        mergeConsecutiveReduceDims(dims0, dims1, maxRank);
    rewriter.setInsertionPointAfter(op);
    auto newReduce = rewriter.create<linalg::ReduceOp>(
        op->getLoc(), TypeRange(op->getResults()), prevReduce.getInputs(),
        op.getInits(), dims);
    Region &newRegion = newReduce.getRegion();
    IRMapping mapping;
    op.getRegion().cloneInto(&newRegion, newRegion.begin(), mapping);

    rewriter.replaceOp(op, newReduce);
    rewriter.eraseOp(prevReduce);
    return success();
  }

  // merge two reduce dims of consecutive reduce ops, return the merged dims
  // that work on the origin reduce input.
  // example 1:
  //   dims0: [0, 1]
  //   dims1: [0, 1]
  //   merge result dims: [0, 1, 2, 3]
  // example 2:
  //   dims0: [0, 2]
  //   dims1: [0, 1]
  //   merge result dims: [0, 1, 2, 3]
  // example 3:
  //   dims0: [0, 4]
  //   dims1: [0, 1]
  //   merge result dims: [0, 1, 2, 4]
  SmallVector<int64_t>
  mergeConsecutiveReduceDims(const SmallVector<unsigned> &dims0,
                             const SmallVector<unsigned> &dims1,
                             unsigned maxRank) const {
    BitVector availableMask(maxRank, true);
    for (unsigned dim : dims0)
      availableMask[dim] = false;
    SmallVector<int64_t> available;
    for (unsigned i = 0; i < maxRank; i++)
      if (availableMask[i])
        available.push_back(i);
    SmallVector<int64_t> newDims;
    for (unsigned dim : dims0)
      newDims.push_back(dim);
    for (unsigned dim : dims1)
      newDims.push_back(available[dim]);
    std::sort(newDims.begin(), newDims.end());
    return newDims;
  }

  bool isReduceWithSameRegion(linalg::ReduceOp op1,
                              linalg::ReduceOp op2) const {
    return OperationEquivalence::isRegionEquivalentTo(
        &op1.getRegion(), &op2.getRegion(),
        OperationEquivalence::Flags::IgnoreLocations);
  }
};

class NormalizeBroadcastDenseSplatToFillConstant
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  NormalizeBroadcastDenseSplatToFillConstant(MLIRContext *context)
      : OpRewritePattern<linalg::BroadcastOp>(context) {}

  LogicalResult matchAndRewrite(linalg::BroadcastOp brcOp,
                                PatternRewriter &rewriter) const final {
    if (!brcOp.hasPureTensorSemantics()) {
      return failure();
    }
    Value input = brcOp.getInput();
    if (!linalg::isSplatDense(input)) {
      return rewriter.notifyMatchFailure(
          brcOp, "only support broadcast from splat dense.");
    }
    Operation *inputOp = input.getDefiningOp();
    auto constantOp = cast<arith::ConstantOp>(inputOp);
    auto scalarMaybe =
        linalg::createConstantFromDenseSplat(constantOp, rewriter);
    if (!scalarMaybe.has_value()) {
      return rewriter.notifyMatchFailure(constantOp,
                                         "failed to extract dense constant.");
    }
    Value scalar = scalarMaybe.value();

    Location loc = brcOp.getLoc();
    Value brcInit = brcOp.getInit();
    auto fillOp = rewriter.create<linalg::FillOp>(
        loc, TypeRange{brcOp->getResults()}, /*inputs=*/ValueRange{scalar},
        /*outputs=*/ValueRange{brcInit});
    rewriter.replaceOp(brcOp, fillOp);
    return success();
  }
};

struct FoldBroadcastFill : public OpRewritePattern<linalg::BroadcastOp> {
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = broadcastOp.getInput().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    rewriter.replaceOpWithNewOp<linalg::FillOp>(broadcastOp, fillOp.getInputs(),
                                                broadcastOp.getInit());

    return success();
  }
};

template <typename MMOP>
struct FuseMatmulAddPattern : public OpRewritePattern<MMOP> {
  inline bool isTensorEmptyOrFilledWithZero(Value outValue) const {
    auto definingOp = outValue.getDefiningOp();
    if (!definingOp)
      return false;
    if (isa<tensor::EmptyOp>(definingOp))
      return true;
    if (auto fillOp = dyn_cast<linalg::FillOp>(definingOp)) {
      Value valueOperand = fillOp.getInputs()[0];
      if (auto constantOp =
              dyn_cast<arith::ConstantOp>(valueOperand.getDefiningOp())) {
        Attribute valueAttr = constantOp.getValue();
        if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
          return floatAttr.getValueAsDouble() == 0.0;
        }
        if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
          return intAttr.getValue() == 0;
        }
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
          if (denseAttr.isSplat()) {
            if (auto splatValue = denseAttr.getSplatValue<FloatAttr>()) {
              return splatValue.getValueAsDouble() == 0.0;
            }
            if (auto splatValue = denseAttr.getSplatValue<IntegerAttr>()) {
              return splatValue.getValue() == 0;
            }
          }
        }
      }
    }
    return false;
  }

public:
  using OpRewritePattern<MMOP>::OpRewritePattern;
  LogicalResult matchAndRewrite(MMOP matmulOp,
                                PatternRewriter &rewriter) const final {
    if (!matmulOp.hasPureTensorSemantics()) {
      return failure();
    }
    if (isOpTriviallyDead(matmulOp)) {
      rewriter.eraseOp(matmulOp);
      return success();
    }

    auto matmulResult = matmulOp->getResult(0);
    if (!matmulResult.hasOneUse()) {
      return failure();
    }
    Value out = matmulOp.getOutputs()[0];
    if (!isTensorEmptyOrFilledWithZero(out))
      return failure();
    auto addOp = dyn_cast<arith::AddFOp>(*matmulResult.getUsers().begin());
    if (!addOp) {
      return failure();
    }
    Value lhs = matmulOp.getInputs()[0];
    Value rhs = matmulOp.getInputs()[1];
    Value bias =
        (matmulResult == addOp.getLhs()) ? addOp.getRhs() : addOp.getLhs();
    Location location = matmulOp.getLoc();
    auto newMatmulOp =
        rewriter.create<MMOP>(location, ValueRange{lhs, rhs}, ValueRange{bias});
    std::string inputPrecisionStr{"input_precision"};
    if (auto inputPrecisionAttr = matmulOp->getAttr(inputPrecisionStr)) {
      newMatmulOp->setAttr(inputPrecisionStr, inputPrecisionAttr);
    }
    rewriter.replaceOp(addOp, newMatmulOp->getResult(0));
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

void linalg::getExtendedCanonicalizationPatterns(RewritePatternSet &results) {
  auto *context = results.getContext();
  results.add<NormalizeBroadcastDenseSplatToFillConstant, FoldBroadcastFill>(
      context);
  results.add<MergeConsecutiveReduceOp>(context);
  results.add<linalg::RefactorRedundantReduceLikeOp<mlir::linalg::ReduceOp>>(
      context);
  results.add<FoldTransposeWithExpand>(context);
  results.add<linalg::SimplifySplatDenseForBinary<linalg::ElemwiseBinaryOp>,
              linalg::InlineDenseSplatToGenericRegion<linalg::ElemwiseBinaryOp>,
              linalg::InlineDenseSplatToGenericRegion<linalg::ElemwiseUnaryOp>>(
      context);
  results.add<FuseMatmulAddPattern<linalg::MatmulOp>>(context);
  results.add<FuseMatmulAddPattern<linalg::BatchMatmulOp>>(context);
}
