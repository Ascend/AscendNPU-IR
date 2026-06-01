//===- ScalarMathPromotion.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::hfusion {

/// Promote scalar math.fma to tensor form so that the MathToHFusion conversion
/// pass can convert it to linalg/hfusion ops.
///
/// Before:
///   %res = math.fma %a, %b, %c : f32
/// After:
///   %ta = tensor.from_elements %a : tensor<1xf32>
///   %tb = tensor.from_elements %b : tensor<1xf32>
///   %tc = tensor.from_elements %c : tensor<1xf32>
///   %tr = math.fma %ta, %tb, %tc : tensor<1xf32>
///   %res = tensor.extract %tr[%c0] : tensor<1xf32>
struct ScalarMathFmaPromotion : public OpRewritePattern<math::FmaOp> {
  using OpRewritePattern<math::FmaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = op.getType();
    // Skip shaped types (already in tensor/vector form).
    if (isa<ShapedType>(resultType)) {
      return failure();
    }

    // Do not promote if the operation is already inside a LinalgOp.
    // This prevents infinite rewrite loops when MathToHFusion lowers
    // tensor math ops back into linalg ops with scalar math regions.
    if (op->template getParentOfType<linalg::LinalgOp>()) {
      return failure();
    }

    // Skip types that the chip natively supports for fma —
    // no scalar-to-tensor promotion needed.
    // To add hardware support for a new type, simply append it to the isa<>
    // template parameter list (e.g., isa<Float32Type, Float16Type>).
    if (isa<Float32Type>(resultType)) {
      return failure();
    }

    auto loc = op.getLoc();
    auto tensorType = RankedTensorType::get({1}, resultType);

    Value tensorA = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, ValueRange{op.getOperand(0)});
    Value tensorB = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, ValueRange{op.getOperand(1)});
    Value tensorC = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, ValueRange{op.getOperand(2)});

    Value tensorFma = rewriter.create<math::FmaOp>(loc, tensorType, tensorA,
                                                   tensorB, tensorC);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, tensorFma, ValueRange{c0});

    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

/// Promote scalar math unary operations to tensor form so that the
/// MathToHFusion conversion pass can convert it to linalg/hfusion ops. This
/// pattern is used for ops that are NOT natively supported by the hardware in
/// scalar form, so they must always be promoted to a tensor.
template <typename MathOpTy, bool ScalarBf16Fallback = false,
          typename... NativeTypes>
struct ScalarMathUnaryPromotion : public OpRewritePattern<MathOpTy> {
  using OpRewritePattern<MathOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOpTy op,
                                PatternRewriter &rewriter) const final {
    auto resultType = op.getType();
    // Skip shaped types (already in tensor/vector form).
    if (isa<ShapedType>(resultType)) {
      return failure();
    }

    // Do not promote if the operation is already inside a LinalgOp.
    // This prevents infinite rewrite loops.
    if (op->template getParentOfType<linalg::LinalgOp>()) {
      return failure();
    }

    // Skip types that the chip natively supports —
    // no scalar-to-tensor promotion needed.
    if constexpr (sizeof...(NativeTypes) > 0) {
      if (isa<NativeTypes...>(resultType)) {
        return failure();
      }
    }

    auto loc = op.getLoc();

    // Some ops (e.g., absf, rsqrt) use a pure scalar fallback for bf16:
    //   extf bf16→f32 → scalar math op → truncf f32→bf16
    // This avoids the tensor round-trip entirely.
    if constexpr (ScalarBf16Fallback) {
      if (isa<BFloat16Type>(resultType)) {
        auto f32Type = rewriter.getF32Type();
        Value extF32 =
            rewriter.create<arith::ExtFOp>(loc, f32Type, op.getOperand());
        Value scalarResult =
            rewriter.create<MathOpTy>(loc, f32Type, extF32);
        Value truncBf16 =
            rewriter.create<arith::TruncFOp>(loc, resultType, scalarResult);
        rewriter.replaceOp(op, truncBf16);
        return success();
      }
    }

    Value operand = op.getOperand();
    Type computeType = resultType;

    // LegalizeBF16Pass has issues handling 1D bf16 tensors correctly in some
    // linalg.elemwise_unary operations. We proactively promote to f32 tensor
    // to avoid the crash.
    bool isBf16 = isa<BFloat16Type>(resultType);
    if (isBf16) {
      computeType = rewriter.getF32Type();
      operand = rewriter.create<arith::ExtFOp>(loc, computeType, operand);
    }

    auto tensorType = RankedTensorType::get({1}, computeType);

    Value tensorOperand = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, ValueRange{operand});

    Value tensorMathOp =
        rewriter.create<MathOpTy>(loc, tensorType, tensorOperand);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value extractOp =
        rewriter.create<tensor::ExtractOp>(loc, tensorMathOp, ValueRange{c0});

    if (isBf16) {
      extractOp = rewriter.create<arith::TruncFOp>(loc, resultType, extractOp);
    }

    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

void populateScalarMathPromotionPatterns(RewritePatternSet &patterns) {
  patterns.add<ScalarMathFmaPromotion,
               ScalarMathUnaryPromotion<math::AbsFOp, true, Float16Type,
                                        Float32Type>,
               ScalarMathUnaryPromotion<math::RsqrtOp, true, Float32Type>,
               ScalarMathUnaryPromotion<math::CeilOp>,
               ScalarMathUnaryPromotion<math::FloorOp>,
               ScalarMathUnaryPromotion<math::CosOp>,
               ScalarMathUnaryPromotion<math::SinOp>,
               ScalarMathUnaryPromotion<math::ErfOp>,
               ScalarMathUnaryPromotion<math::ExpOp>,
               ScalarMathUnaryPromotion<math::Exp2Op>,
               ScalarMathUnaryPromotion<math::LogOp>,
               ScalarMathUnaryPromotion<math::Log2Op>>(patterns.getContext());
}

} // namespace mlir::hfusion
