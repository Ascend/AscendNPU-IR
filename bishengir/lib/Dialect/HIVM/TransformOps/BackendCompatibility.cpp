//===- BackendCompatibility.cpp - vectorizer backend preparation
//--------------------------===//
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
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeResult.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct ScalarI64Select : OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(64))
      return failure();

    // A6 SIMD scalar registers have no i64 select. Use the condition once
    // so an undef condition chooses one arm for the entire value. Freeze the
    // arms because the bitwise operations also evaluate the unselected arm.
    auto loc = op.getLoc();
    auto left = rewriter.create<LLVM::FreezeOp>(loc, op.getTrueValue());
    auto right = rewriter.create<LLVM::FreezeOp>(loc, op.getFalseValue());
    auto mask = rewriter.createOrFold<arith::ExtSIOp>(loc, op.getType(),
                                                      op.getCondition());
    auto diff = rewriter.createOrFold<arith::XOrIOp>(loc, left, right);
    auto selected = rewriter.createOrFold<arith::AndIOp>(loc, mask, diff);
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, right, selected);
    return success();
  }
};

struct PredicateExtension : RewritePattern {
  explicit PredicateExtension(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {
    if (!isa<arith::ExtUIOp, arith::ExtSIOp>(operation))
      return failure();
    auto input = dyn_cast<VectorType>(operation->getOperand(0).getType());
    auto output = dyn_cast<VectorType>(operation->getResult(0).getType());
    if (!input || !output || !input.getElementType().isInteger(1))
      return failure();
    auto loc = operation->getLoc();
    auto signedExtension = isa<arith::ExtSIOp>(operation);
    auto selectedType =
        output.getElementType().isInteger(64)
            ? VectorType::get(output.getShape(), rewriter.getI32Type())
            : output;
    auto yes = rewriter.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(
            selectedType, rewriter.getIntegerAttr(selectedType.getElementType(),
                                                  signedExtension ? -1 : 1)));
    auto no = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(selectedType));
    Value result = rewriter.create<arith::SelectOp>(
        loc, operation->getOperand(0), yes, no);
    if (selectedType != output) {
      if (signedExtension)
        result = rewriter.create<arith::ExtSIOp>(loc, output, result);
      else
        result = rewriter.create<arith::ExtUIOp>(loc, output, result);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

struct OrderPredicate : OpRewritePattern<arith::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<VectorType>(op.getLhs().getType()) ||
        (op.getPredicate() != arith::CmpFPredicate::ORD &&
         op.getPredicate() != arith::CmpFPredicate::UNO))
      return failure();
    auto sourceType = cast<VectorType>(op.getLhs().getType());
    auto width =
        hivmave::symbolizeMaskWidth(sourceType.getElementTypeBitWidth());
    if (!width)
      return failure();
    auto left = rewriter.create<arith::CmpFOp>(
        op.getLoc(), arith::CmpFPredicate::OEQ, op.getLhs(), op.getLhs());
    auto right = rewriter.create<arith::CmpFOp>(
        op.getLoc(), arith::CmpFPredicate::OEQ, op.getRhs(), op.getRhs());
    auto widthAttr = hivmave::MaskWidthAttr::get(op.getContext(), *width);
    auto mask = rewriter.create<hivmave::VFPgeOp>(op.getLoc(), op.getType(),
                                                  hivmave::PgePattern::ALL);
    Value result = rewriter.create<hivmave::PregAndOp>(
        op.getLoc(), op.getType(), widthAttr, left, right, mask);
    if (op.getPredicate() == arith::CmpFPredicate::UNO) {
      result = rewriter.create<hivmave::PregNotOp>(op.getLoc(), op.getType(),
                                                   widthAttr, result, mask);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SingletonBroadcast : OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto cast = op.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast || cast.getInputs().size() != 1)
      return failure();
    auto source = cast.getInputs().front();
    auto type = dyn_cast<VectorType>(source.getType());
    if (!type || type.getNumElements() != 1)
      return failure();
    if (op.getType().getNumElements() == 1)
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(),
                                                       source);
    else
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, op.getType(),
                                                       source);
    return success();
  }
};

FailureOr<hivmave::CombiningKind>
getNativeReductionKind(vector::CombiningKind kind) {
  switch (kind) {
  case vector::CombiningKind::ADD:
    return hivmave::CombiningKind::ADD;
  case vector::CombiningKind::MAXSI:
  case vector::CombiningKind::MAXIMUMF:
  case vector::CombiningKind::MAXNUMF:
    return hivmave::CombiningKind::MAX;
  case vector::CombiningKind::MINSI:
  case vector::CombiningKind::MINIMUMF:
  case vector::CombiningKind::MINNUMF:
    return hivmave::CombiningKind::MIN;
  case vector::CombiningKind::MAXUI:
    return hivmave::CombiningKind::UMAX;
  case vector::CombiningKind::MINUI:
    return hivmave::CombiningKind::UMIN;
  case vector::CombiningKind::XOR:
    return hivmave::CombiningKind::XORI;
  default:
    return failure();
  }
}

struct NativeReduction : OpRewritePattern<vector::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    auto kind = getNativeReductionKind(op.getKind());
    auto type = op.getSourceVectorType();
    if (failed(kind) || type.getRank() != 1)
      return failure();
    auto maskable = cast<vector::MaskableOpInterface>(op.getOperation());
    Operation *root = op;
    Value mask;
    if (maskable.isMasked()) {
      auto wrapper = maskable.getMaskingOp();
      root = wrapper;
      mask = wrapper.getMask();
    }
    rewriter.setInsertionPoint(root);
    if (!mask)
      mask = rewriter.create<vector::ConstantMaskOp>(
          op.getLoc(), VectorType::get(type.getShape(), rewriter.getI1Type()),
          rewriter.getI64ArrayAttr(type.getShape()));
    Value result = rewriter.create<hivmave::ReductionOp>(
        op.getLoc(), type,
        hivmave::CombiningKindAttr::get(op.getContext(), *kind), op.getVector(),
        mask);
    if (op.getAcc()) {
      auto accumulator =
          rewriter.create<vector::BroadcastOp>(op.getLoc(), type, op.getAcc());
      result = vector::makeArithReduction(rewriter, op.getLoc(), op.getKind(),
                                          accumulator, result);
    }
    auto singletonType = VectorType::get({1}, type.getElementType());
    auto singleton = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), singletonType, result);
    auto scalar = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), singleton.getResult(0));
    rewriter.replaceOp(root, scalar.getResult(0));
    return success();
  }
};

struct SingletonExtractPattern : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto type = cast<VectorType>(op.getVector().getType());
    if (auto resultType = dyn_cast<VectorType>(op.getType())) {
      if (type.getNumElements() != resultType.getNumElements() ||
          !op.getDynamicPosition().empty() ||
          !llvm::all_of(op.getStaticPosition(),
                        [](int64_t position) { return position == 0; }))
        return failure();
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resultType,
                                                       op.getVector());
      return success();
    }
    if (type.getNumElements() != 1 || isa<VectorType>(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            op.getVector());
    return success();
  }
};

struct SingletonExtractElementPattern
    : public OpRewritePattern<vector::ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractElementOp op,
                                PatternRewriter &rewriter) const override {
    auto type = cast<VectorType>(op.getVector().getType());
    if (type.getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            op.getVector());
    return success();
  }
};

struct SingletonInsertElementPattern
    : public OpRewritePattern<vector::InsertElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType().getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            op.getSource());
    return success();
  }
};

struct UnitBroadcastTransferRead : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  Value expandMask(vector::TransferReadOp op, AffineMap map,
                   PatternRewriter &rewriter) const {
    auto mask = op.getMask();
    if (!mask)
      return {};
    auto maskType = vector::inferTransferOpMaskType(op.getVectorType(), map);
    if (maskType == mask.getType())
      return mask;
    if (auto constant = mask.getDefiningOp<vector::ConstantMaskOp>()) {
      SmallVector<int64_t> bounds(map.getNumDims(), 1);
      SmallVector<bool> used(map.getNumDims(), false);
      for (auto expr : op.getPermutationMap().getResults())
        if (auto dimension = dyn_cast<AffineDimExpr>(expr))
          used[dimension.getPosition()] = true;
      auto originalBounds = constant.getMaskDimSizes();
      unsigned index = 0;
      for (auto dimension : llvm::seq<unsigned>(0, used.size()))
        if (used[dimension])
          bounds[dimension] =
              cast<IntegerAttr>(originalBounds[index++]).getInt();
      return rewriter.create<vector::ConstantMaskOp>(
          op.getLoc(), maskType, rewriter.getI64ArrayAttr(bounds));
    }
    return rewriter.create<vector::ShapeCastOp>(op.getLoc(), maskType, mask);
  }

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    auto source = dyn_cast<MemRefType>(op.getSource().getType());
    auto type = op.getVectorType();
    auto map = op.getPermutationMap();
    if (!source || source.getRank() != type.getRank() ||
        op.hasOutOfBoundsDim() || !map.isProjectedPermutation(true) ||
        map.isPermutation())
      return failure();
    SmallVector<bool> used(source.getRank(), false);
    for (auto expr : map.getResults())
      if (auto dim = dyn_cast<AffineDimExpr>(expr))
        used[dim.getPosition()] = true;
    SmallVector<AffineExpr> results(map.getResults());
    for (auto axis : llvm::seq<unsigned>(0, results.size())) {
      if (!isa<AffineConstantExpr>(results[axis]))
        continue;
      if (type.getDimSize(axis) != 1)
        return failure();
      auto free = llvm::find(used, false);
      if (free == used.end())
        return failure();
      auto dimension = std::distance(used.begin(), free);
      if (source.getDimSize(dimension) != 1)
        return failure();
      results[axis] = rewriter.getAffineDimExpr(dimension);
      used[dimension] = true;
    }
    auto replacementMap =
        AffineMap::get(source.getRank(), 0, results, op.getContext());
    auto mask = expandMask(op, replacementMap, rewriter);
    rewriter.modifyOpInPlace(op, [&] {
      op.setPermutationMapAttr(AffineMapAttr::get(replacementMap));
      if (mask)
        op.getMaskMutable().assign(mask);
    });
    return success();
  }
};

} // namespace

LogicalResult
mlir::hivm::normalizeVectorizerBackendForms(func::FuncOp function) {
  auto module = function->getParentOfType<ModuleOp>();
  if (!hacc::utils::isRegBasedArch(module))
    return success();
  function.getContext()->getOrLoadDialect<hivmave::AVEDialect>();
  RewritePatternSet patterns(function.getContext());
  if (hacc::utils::isAscend960(module))
    patterns.add<ScalarI64Select>(function.getContext());
  patterns.add<PredicateExtension, OrderPredicate, NativeReduction,
               SingletonBroadcast, SingletonExtractPattern,
               SingletonExtractElementPattern, SingletonInsertElementPattern,
               UnitBroadcastTransferRead>(function.getContext());
  return applyPatternsGreedily(function, std::move(patterns));
}
