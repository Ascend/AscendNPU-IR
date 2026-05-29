//===- NormalizeTrig.cpp -----------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Transforms/Normalize/NormalizeTrigTemplate.h"

namespace mlir {
struct HFusionNormalizeSinTraits : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeSin(hfusion::ElemwiseUnaryOp op) {
    if (!op.hasPureTensorSemantics() || op.getFun() != hfusion::UnaryFn::sin) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HFusionNormalizeCosTraits : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeCos(hfusion::ElemwiseUnaryOp op) {
    if (!op.hasPureTensorSemantics() || op.getFun() != hfusion::UnaryFn::cos) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HFusionNormalizeAtanTraits : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeAtan(hfusion::ElemwiseUnaryOp op) {
    if (!op.hasPureTensorSemantics() ||
        op.getFun() != hfusion::UnaryFn::atan) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HFusionNormalizeTanTraits : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeTan(hfusion::ElemwiseUnaryOp op) {
    if (!op.hasPureTensorSemantics() || op.getFun() != hfusion::UnaryFn::tan) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HFusionNormalizeTanhTraits : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeTanh(hfusion::ElemwiseUnaryOp op) {
    if (!op.hasPureTensorSemantics() || op.getFun() != hfusion::UnaryFn::tanh) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

using NormalizeSinOp =
    NormalizeSinOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeSinTraits>;
using NormalizeCosOp =
    NormalizeCosOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeCosTraits>;
using NormalizeAtanOp = NormalizeAtanOpTemplate<hfusion::ElemwiseUnaryOp,
                                                HFusionNormalizeAtanTraits>;
using NormalizeTanOp =
    NormalizeTanOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeTanTraits>;
using NormalizeTanhOp = NormalizeTanhOpTemplate<hfusion::ElemwiseUnaryOp,
                                                HFusionNormalizeTanhTraits>;
} // namespace mlir

namespace mlir::hfusion {
static Value rebuildHighPrecisionRsqrtForTrig(PatternRewriter &rewriter,
                                              Operation *trigOp, Value input) {
  auto rsqrtOp = input.getDefiningOp<hfusion::ElemwiseUnaryOp>();
  if (!rsqrtOp || rsqrtOp.getFun() != hfusion::UnaryFn::rsqrt ||
      !input.hasOneUse()) {
    return {};
  }

  auto srcElemType = getElementTypeOrSelf(rsqrtOp.getInputs()[0].getType());
  if (!srcElemType.isF16())
    return {};

  Location loc = trigOp->getLoc();
  Value fp32Input =
      hfusion::castTo(rewriter, rsqrtOp.getInputs()[0], rewriter.getF32Type(),
                      hfusion::RoundMode::ROUND);
  Value sqrtInit = utils::createEmptyOpWithTargetElemType(
      rewriter, loc, fp32Input, rewriter.getF32Type());
  Value sqrtValue = NormalizeTraitsBase::createUnaryOp(
      rewriter, loc, fp32Input, sqrtInit, UnaryKind::Sqrt);
  Value recInit = utils::createEmptyOpWithTargetElemType(
      rewriter, loc, sqrtValue, rewriter.getF32Type());
  return NormalizeTraitsBase::createUnaryOp(rewriter, loc, sqrtValue, recInit,
                                            UnaryKind::Rec);
}

static Value getPreferredHighPrecisionTrigInput(PatternRewriter &rewriter,
                                                Operation *trigOp,
                                                Value input) {
  if (Value rebuiltRsqrt =
          rebuildHighPrecisionRsqrtForTrig(rewriter, trigOp, input)) {
    return rebuiltRsqrt;
  }

  auto castOp = input.getDefiningOp<hfusion::CastOp>();
  if (!castOp)
    return {};

  Type srcType = getElementTypeOrSelf(castOp.getInputs()[0].getType());
  Type dstType = getElementTypeOrSelf(castOp.getOutputs()[0].getType());
  if (!srcType.isF32() || !dstType.isF16())
    return {};

  auto unaryOp = castOp.getInputs()[0].getDefiningOp<hfusion::ElemwiseUnaryOp>();
  if (!unaryOp)
    return {};

  auto fun = unaryOp.getFun();
  if (fun != hfusion::UnaryFn::rec && fun != hfusion::UnaryFn::rsqrt &&
      fun != hfusion::UnaryFn::sqrt) {
    return {};
  }

  // Reuse the existing f32 producer when trig consumes the result of another
  // high-precision unary op. This avoids a transient f32->f16->f32 roundtrip
  // before lowering sin/cos.
  return castOp.getInputs()[0];
}

struct HighPrecisionNormalizeSinOp
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::sin) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      if (Value preferredInput =
              getPreferredHighPrecisionTrigInput(rewriter, op, input)) {
        input = preferredInput;
      } else {
        // for high precision, cast src to fp32 and compute and then cast it
        // back
        // TODO: remove cast after enable automatical high precision computing
        input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                                hfusion::RoundMode::ROUND);
      }
    }
    auto resOr = buildSinOrCos(rewriter, op, input, CalcMode::SIN);
    if (failed(resOr))
      return failure();
    Value res = *resOr;

    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct HighPrecisionNormalizeCosOp
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::cos) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");
    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      if (Value preferredInput =
              getPreferredHighPrecisionTrigInput(rewriter, op, input)) {
        input = preferredInput;
      } else {
        // for high precision, cast src to fp32 and compute and then cast it
        // back
        input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                                hfusion::RoundMode::ROUND);
      }
    }

    auto resOr = buildSinOrCos(rewriter, op, input, CalcMode::COS);
    if (failed(resOr))
      return failure();
    Value res = *resOr;

    if (inType.isF16()) {
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

void populateNormalizeTrigPatterns(RewritePatternSet &patterns,
                                   bool enableHighPrecision) {
  MLIRContext *ctx = patterns.getContext();
  if (enableHighPrecision) {
    patterns.add<HighPrecisionNormalizeSinOp>(ctx);
    patterns.add<HighPrecisionNormalizeCosOp>(ctx);
  } else {
    patterns.add<NormalizeSinOp>(ctx);
    patterns.add<NormalizeCosOp>(ctx);
  }
  patterns.add<NormalizeAtanOp>(ctx);
  patterns.add<NormalizeTanOp>(ctx);
  patterns.add<NormalizeTanhOp>(ctx);
}
} // namespace mlir::hfusion
