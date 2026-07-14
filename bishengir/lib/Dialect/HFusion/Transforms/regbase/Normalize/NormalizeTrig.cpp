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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeTrigTemplate.h"

namespace mlir {
struct HFusionNormalizeHighPrecisionTrigTraitsBase : public hfusion::NormalizeTraitsBase {
  static std::optional<Value> getCastInput(Value input) {
    auto castOp = input.getDefiningOp<hfusion::CastOp>();
    if (!castOp)
      return std::nullopt;
    return castOp.getInputs()[0];
  }

  static Value getUnaryInput(Operation *op) {
    return cast<hfusion::ElemwiseUnaryOp>(op).getInputs()[0];
  }

  static bool isHighPrecisionProducer(Operation *op) {
    auto unaryOp = dyn_cast_or_null<hfusion::ElemwiseUnaryOp>(op);
    if (!unaryOp)
      return false;

    auto fun = unaryOp.getFun();
    return fun == hfusion::UnaryFn::rec || fun == hfusion::UnaryFn::rsqrt ||
           fun == hfusion::UnaryFn::sqrt;
  }

  static bool isHighPrecisionRsqrt(Operation *op) {
    auto unaryOp = dyn_cast_or_null<hfusion::ElemwiseUnaryOp>(op);
    return unaryOp && unaryOp.getFun() == hfusion::UnaryFn::rsqrt;
  }

  static FailureOr<Value>
  buildHighPrecisionSinOrCos(PatternRewriter &rewriter,
                             hfusion::ElemwiseUnaryOp op, Value input,
                             HighPrecisionTrigMode mode) {
    return hfusion::buildSinOrCos(
        rewriter, op, input,
        mode == HighPrecisionTrigMode::Sin ? hfusion::CalcMode::SIN
                                           : hfusion::CalcMode::COS);
  }
};

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

struct HFusionNormalizeHighPrecisionSinTraits
    : public HFusionNormalizeHighPrecisionTrigTraitsBase {
public:
  static bool shouldNormalizeHighPrecisionSin(hfusion::ElemwiseUnaryOp op) {
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

struct HFusionNormalizeHighPrecisionCosTraits
    : public HFusionNormalizeHighPrecisionTrigTraitsBase {
public:
  static bool shouldNormalizeHighPrecisionCos(hfusion::ElemwiseUnaryOp op) {
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

using NormalizeSinOpRegBase =
    NormalizeSinOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeSinTraits>;
using NormalizeCosOpRegBase =
    NormalizeCosOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeCosTraits>;
using HighPrecisionNormalizeSinOpRegBase =
    HighPrecisionNormalizeSinOpTemplate<hfusion::ElemwiseUnaryOp,
                                        HFusionNormalizeHighPrecisionSinTraits>;
using HighPrecisionNormalizeCosOpRegBase =
    HighPrecisionNormalizeCosOpTemplate<hfusion::ElemwiseUnaryOp,
                                        HFusionNormalizeHighPrecisionCosTraits>;
using NormalizeAtanOp = NormalizeAtanOpTemplate<hfusion::ElemwiseUnaryOp,
                                                HFusionNormalizeAtanTraits>;
using NormalizeTanOpRegBase =
    NormalizeTanOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeTanTraits>;
using NormalizeTanhOpRegBase = NormalizeTanhOpTemplate<hfusion::ElemwiseUnaryOp,
                                                HFusionNormalizeTanhTraits>;
} // namespace mlir

namespace mlir::hfusion {
void populateNormalizeTrigPatterns(RewritePatternSet &patterns,
                                   bool enableHighPrecision) {
  MLIRContext *ctx = patterns.getContext();
  if (enableHighPrecision) {
    patterns.add<HighPrecisionNormalizeSinOpRegBase>(ctx);
    patterns.add<HighPrecisionNormalizeCosOpRegBase>(ctx);
  } else {
    patterns.add<NormalizeSinOpRegBase>(ctx);
    patterns.add<NormalizeCosOpRegBase>(ctx);
  }
  patterns.add<NormalizeAtanOp>(ctx);
  patterns.add<NormalizeTanOpRegBase>(ctx);
  patterns.add<NormalizeTanhOpRegBase>(ctx);
}
} // namespace mlir::hfusion
