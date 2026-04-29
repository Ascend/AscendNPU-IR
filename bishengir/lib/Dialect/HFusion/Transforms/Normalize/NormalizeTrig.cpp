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
      // for high precision, cast src to fp32 and compute and then cast it back
      // TODO: remove cast after enable automatical high precision computing
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
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
      // for high precision, cast src to fp32 and compute and then cast it back
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
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
