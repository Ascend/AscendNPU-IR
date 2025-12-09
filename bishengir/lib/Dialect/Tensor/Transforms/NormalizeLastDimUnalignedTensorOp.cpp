//===- NormalizeLastDimUnalignedTensorOp.cpp ------------------------------===//
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

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

#define DEBUG_TYPE "normalize-last-dim-unaligned-tensor-op"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZELASTDIMUNALIGNEDTENSOROP
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

using namespace mlir;

TensorType transposeTensorType(TensorType oldType, ArrayRef<int64_t> perm) {
  llvm::ArrayRef<int64_t> oldShape = oldType.getShape();
  SmallVector<int64_t> newShape(oldShape.size());
  assert(newShape.size() == perm.size());
  for (size_t i = 0; i < newShape.size(); i++)
    newShape[i] = oldShape[perm[i]];
  return oldType.clone(newShape);
}

SmallVector<OpFoldResult> transposeMixedSizes(ArrayRef<OpFoldResult> oldSizes,
                                              ArrayRef<int64_t> perm) {
  int64_t rank = static_cast<int64_t>(oldSizes.size());
  SmallVector<OpFoldResult> newSizes(rank);
  assert(newSizes.size() == perm.size());
  for (int64_t i = 0; i < rank; i++)
    newSizes[i] = oldSizes[perm[i]];
  return newSizes;
}

// Transpose the given array according to the given permutation.
SmallVector<int64_t> transposeArray(llvm::ArrayRef<int64_t> array,
                                    const SmallVector<int64_t> &permutation) {
  assert(array.size() == permutation.size());
  SmallVector<int64_t> transposedArray(array.size());
  for (size_t i = 0; i < transposedArray.size(); i++) {
    transposedArray[i] = array[permutation[i]];
  }
  return transposedArray;
}

// extend first dim by broadcast with new dim size.
linalg::BroadcastOp extendFirstDim(Value src, OpBuilder &builder, Location loc,
                                   int64_t newDimSize = 1) {
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(builder, loc, src);
  SmallVector<OpFoldResult> dirtySizes = {builder.getIndexAttr(newDimSize),
                                          sizes[0]};
  TensorType srcType = cast<TensorType>(src.getType());
  auto emptyOp = builder.create<tensor::EmptyOp>(loc, dirtySizes,
                                                 srcType.getElementType());
  SmallVector<int64_t> brcDim = {0};
  return builder.create<linalg::BroadcastOp>(loc, src, emptyOp, brcDim);
}

// drop first dim by rank-reduce extract_slice
tensor::ExtractSliceOp dropFirstDim(Value src, OpBuilder &builder,
                                    Location loc) {
  auto srcType = cast<ShapedType>(src.getType());
  auto srcShape = srcType.getShape();
  auto sliceType = srcType.clone({srcShape[1]});
  SmallVector<int64_t> offsets = {0, 0};
  SmallVector<int64_t> strides = {1, 1};
  SmallVector<int64_t> sizes = {1, srcShape[1]};
  return builder.create<tensor::ExtractSliceOp>(
      loc, sliceType, src, ValueRange{}, ValueRange{}, ValueRange{}, offsets,
      sizes, strides);
}

/// replace 1-D padOp tensor<?xElemType> by 2-rank padOp with type
/// tensor<1x?xElemType> e.g.
///   %res = tensor.pad %src low[?] high[?]
/// is changed to
///   %0 = tensor.empty() : tensor<1x?xElemType>
///   %1 = linalg.broadcast %src to %0 dimensions = [0]
///   %2 = tensor.pad %src's' tensor<1x?xElemType> low[0,?] high[0,?]
///   %3 = tensor.extract_slice %2 : tensor<?xElemType>
///   replace %res by %3
///  Here use 1 as extended dim size for more optimization opportunity
tensor::PadOp extendOneDimPad(tensor::PadOp padOp, PatternRewriter &rewriter) {
  // broadcast origin 1-D tensor to 2-D with dirty data
  auto brcOp = extendFirstDim(padOp.getSource(), rewriter, padOp.getLoc());

  // create 2-D pad Op to replace old 1-D pad Op
  RankedTensorType srcType = padOp.getSourceType();
  llvm::ArrayRef<int64_t> srcShape = srcType.getShape();
  int64_t padLow = padOp.getStaticLow()[0];
  int64_t padHigh = padOp.getStaticHigh()[0];
  SmallVector<int64_t, 2> padShape = {1, srcShape[0] + padLow + padHigh};
  RankedTensorType padDirtyTy = srcType.clone(padShape);
  auto newPadOp = rewriter.create<tensor::PadOp>(
      padOp.getLoc(), padDirtyTy, brcOp->getResult(0),
      llvm::ArrayRef<int64_t>({0, padLow}),
      llvm::ArrayRef<int64_t>({0, padHigh}), ValueRange(), ValueRange());
  DenseMap<uint64_t, uint64_t> padBodyMapping;
  for (size_t i = 0; i < 2; i++) {
    padBodyMapping[i] = i;
  }
  tensor::reshape_utils::clonePadRegion(rewriter, padOp, newPadOp,
                                        padBodyMapping);

  // extract slice with rank-reduced sizes to restore the origin 1-D vector
  auto sliceOp =
      dropFirstDim(newPadOp.getResult(), rewriter, newPadOp->getLoc());
  rewriter.replaceOp(padOp, sliceOp);
  return newPadOp;
}

// Get permutation with two dims exchanged: `dim` and `0`
// TODO: if it is better to exchange `dim` and `dim - 1`
SmallVector<int64_t> getPermutation(size_t rank, size_t dim) {
  SmallVector<int64_t> perm;
  perm.reserve(rank);
  for (size_t i = 0; i < rank; i++)
    perm.push_back(i);
  perm[dim] = 0;
  perm[0] = static_cast<int64_t>(dim);
  return perm;
}

LogicalResult normalizeUnalignLastDimPad(tensor::PadOp padOp,
                                         PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(padOp);

  // Calculate the new shape of the padded tensor.
  llvm::ArrayRef<int64_t> low = padOp.getStaticLow();
  llvm::ArrayRef<int64_t> high = padOp.getStaticHigh();
  TensorType padOpResTy = padOp.getResult().getType();
  llvm::ArrayRef<int64_t> padResultShape = padOpResTy.getShape();

  size_t resultRank = padResultShape.size();
  // Create the transpose permutation
  //    [0, 1, ..., dim, ..., R - 1]
  SmallVector<int64_t> permutation;
  permutation.reserve(resultRank);
  for (size_t i = 0; i < resultRank; i++)
    permutation.push_back(i);
  //    [dim, 1, ..., 0, ..., R - 1]
  permutation[resultRank - 1] = 0;
  permutation[0] = static_cast<int64_t>(resultRank - 1);

  // Transpose the source
  TensorType transposedSourceTy =
      transposeTensorType(padOp.getSourceType(), permutation);
  auto transposedSourceInit = rewriter.create<tensor::EmptyOp>(
      padOp.getLoc(), transposedSourceTy, ValueRange());
  auto transposedSource = rewriter.create<linalg::TransposeOp>(
      padOp.getLoc(), padOp.getSource(), transposedSourceInit, permutation);

  // Permute the static low and high. This would be slightly more complicated
  // if the padding was dynamic.
  SmallVector<int64_t> permutedLow = transposeArray(low, permutation);
  SmallVector<int64_t> permutedHigh = transposeArray(high, permutation);

  // Create the new pad op. This would slightly change if the padding was
  // dynamic.
  TensorType transposedResTy = transposeTensorType(padOpResTy, permutation);
  auto newPadOp = rewriter.create<tensor::PadOp>(
      padOp.getLoc(), transposedResTy, transposedSource.getResult()[0],
      ArrayRef(permutedLow), ArrayRef(permutedHigh), ValueRange(),
      ValueRange());
  // Clone the pad region into the new pad op, permuting the arguments to the
  // padBody.
  DenseMap<uint64_t, uint64_t> padBodyMapping;
  for (size_t i = 0; i < permutation.size(); i++) {
    padBodyMapping[i] = static_cast<uint64_t>(permutation[i]);
  }
  tensor::reshape_utils::clonePadRegion(rewriter, padOp, newPadOp,
                                        padBodyMapping);

  // Transpose the result of the new pad op and replace the old pad op with
  // it.
  auto transposedResultInit = rewriter.create<tensor::EmptyOp>(
      padOp.getLoc(), padOpResTy, ValueRange());
  rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
      padOp, newPadOp.getResult(), transposedResultInit, permutation);

  // Match and rewrite successful.
  return success();
}

class NormalizeConcatOp : public mlir::OpRewritePattern<tensor::ConcatOp> {
public:
  explicit NormalizeConcatOp(MLIRContext *context)
      : OpRewritePattern<tensor::ConcatOp>(context) {}

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    // Check that the concat axis is the last dim, if not skip this op.
    auto dim = static_cast<int>(concatOp.getDim());
    int rank = concatOp.getResultType().getRank();
    if (dim != rank - 1) {
      return failure();
    }

    if (isLastDimAligned(concatOp)) {
      return rewriter.notifyMatchFailure(concatOp, "last dim is aligned");
    }

    // for unaligned last dim concat, handware cannot handle unaligned memory
    // acess. So replace old concat by: transpose + concat + transpose
    if (rank == 1) {
      // for 1-rank concat, extend dim by brc and then drop dim by slice:
      // - brc + transpose + concat + transpose + extract_slice
      concatOp = extendOneDimConcat(concatOp, rewriter);
      dim = concatOp.getDim();
      rank = concatOp.getResultType().getRank();
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(concatOp);

    // Transpose concat inputs
    SmallVector<int64_t> perm = getPermutation(rank, dim);
    SmallVector<Value> newInputs = transposeInputs(concatOp, perm, rewriter);

    // Create new concat op with concat dim being 0.
    auto oldSizes = tensor::getMixedSizes(rewriter, concatOp.getLoc(),
                                          concatOp->getResult(0));
    auto newSizes = transposeMixedSizes(oldSizes, perm);
    auto oldConcatType = concatOp.getResultType();
    auto newConcatType = transposeTensorType(oldConcatType, perm);
    auto newConcatOp = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), newConcatType, 0 /*dim*/, newInputs);

    // Transpose back new concat op
    auto newOuput = transposeOutput(newConcatOp, perm, rewriter);
    rewriter.replaceOp(concatOp, newOuput);
    return success();
  }

  bool isLastDimAligned(tensor::ConcatOp concatOp) const {
    SmallVector<int64_t> lastDimSizes;
    SmallVector<Value> inputs = concatOp.getInputs();
    // no need to check if offset is aligned for the last input
    for (auto input : llvm::drop_end(concatOp.getInputs())) {
      auto type = cast<ShapedType>(input.getType());
      auto size = type.getShape().back();
      lastDimSizes.push_back(size);
    }
    return utils::areShapesAligned(lastDimSizes, 32);
  }

  SmallVector<Value> transposeInputs(tensor::ConcatOp concatOp,
                                     ArrayRef<int64_t> perm,
                                     PatternRewriter &rewriter) const {
    Type elemType = concatOp.getResultType().getElementType();
    SmallVector<Value> result;
    for (Value operand : concatOp.getInputs()) {
      auto oldSizes =
          tensor::getMixedSizes(rewriter, concatOp.getLoc(), operand);
      auto newSizes = transposeMixedSizes(oldSizes, perm);
      auto newInit = rewriter.create<tensor::EmptyOp>(concatOp.getLoc(),
                                                      newSizes, elemType);
      auto newInput = rewriter.create<linalg::TransposeOp>(
          concatOp.getLoc(), operand, newInit, perm);
      result.push_back(newInput->getResult(0));
    }
    return result;
  }

  linalg::TransposeOp transposeOutput(tensor::ConcatOp concatOp,
                                      ArrayRef<int64_t> perm,
                                      PatternRewriter &rewriter) const {
    Type elemType = concatOp.getResultType().getElementType();
    Location loc = concatOp.getLoc();
    auto oldSizes = tensor::getMixedSizes(rewriter, loc, concatOp.getResult());
    auto newSizes = transposeMixedSizes(oldSizes, perm);
    auto resultInit = rewriter.create<tensor::EmptyOp>(loc, newSizes, elemType);
    return rewriter.create<linalg::TransposeOp>(loc, concatOp.getResult(),
                                                resultInit, perm);
  }

  /// replace 1-D concatOp by 2-D concatOp with type tensor<1x?xElemType>
  /// e.g.
  ///   %res = concat %src tensor<?xElemType> dim=[0]
  /// is changed to
  ///   %0 = tensor.empty() : tensor<1x?xElemType>
  ///   %1 = linalg.broadcast %src to %0 dimensions = [0]
  ///   %2 = concat %src's' tensor<1x?xElemType> dim=[1]
  ///   %3 = tensor.extract_slice %2 : tensor<?xElemType>
  ///   replace %res by %3
  ///  Here use 1 as extended dim size for more optimization opportunity
  tensor::ConcatOp extendOneDimConcat(tensor::ConcatOp concatOp,
                                      PatternRewriter &rewriter) const {
    SmallVector<int64_t> brcDim = {0};
    SmallVector<Value> newInputs;
    for (Value input : concatOp.getInputs()) {
      // broadcast origin 1-D tensor to 2-D with dirty data
      auto brcOp = extendFirstDim(input, rewriter, concatOp->getLoc());
      newInputs.push_back(brcOp->getResult(0));
    }

    // concat broadcasted 2-D tensors
    TensorType concatType = concatOp.getResultType();
    llvm::ArrayRef<int64_t> concatShape = concatType.getShape();
    SmallVector<int64_t, 2> newConcatShape = {1, concatShape[0]};
    RankedTensorType newConcatType = concatType.clone(newConcatShape);
    auto newConcatOp = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), newConcatType, 1 /*dim*/, newInputs);

    // extract slice with rank-reduced sizes to restore the origin 1-D vector
    auto sliceOp =
        dropFirstDim(newConcatOp.getResult(), rewriter, concatOp->getLoc());
    rewriter.replaceOp(concatOp, sliceOp);
    return newConcatOp;
  }
};

class NormalizePadOp : public mlir::OpRewritePattern<tensor::PadOp> {
public:
  explicit NormalizePadOp(MLIRContext *context)
      : OpRewritePattern<tensor::PadOp>(context) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Ignore dynamic padding for now
    if (!padOp.getLow().empty() || !padOp.getHigh().empty())
      return failure();
    // Check if the last dim is being padded, ignore if not.
    auto low = padOp.getStaticLow();
    llvm::ArrayRef<int64_t> high = padOp.getStaticHigh();
    TensorType padOpResTy = padOp.getResult().getType();
    llvm::ArrayRef<int64_t> padResultShape = padOpResTy.getShape();
    size_t resultRank = padResultShape.size();
    if (low[resultRank - 1] == 0 && high[resultRank - 1] == 0)
      return failure();

    if (utils::areShapesAligned({low.back()}, 32))
      return failure();

    // last dim of pad op is unaligned and hardware cannot process it
    // it need be replaced by unlast dim pad op, namely transpose + new pad +
    // transpose
    if (resultRank == 1) {
      // only one rank, must extend dimension of pad so that transpose can be
      // inserted before and after
      padOp = extendOneDimPad(padOp, rewriter);
    }
    return normalizeUnalignLastDimPad(padOp, rewriter);
  }
};

} // namespace

namespace mlir {
namespace tensor {

namespace {
class NormalizeLastDimUnalignedTensorOpPass
    : public impl::NormalizeLastDimUnalignedTensorOpBase<
          NormalizeLastDimUnalignedTensorOpPass> {
  using Base::Base;
  void runOnOperation() final;
};

void NormalizeLastDimUnalignedTensorOpPass::runOnOperation() {
  func::FuncOp f = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<NormalizePadOp, NormalizeConcatOp>(context);
  if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace

std::unique_ptr<Pass> createNormalizeLastDimUnalignedTensorOpPass() {
  return std::make_unique<NormalizeLastDimUnalignedTensorOpPass>();
}

} // namespace tensor
} // namespace mlir
