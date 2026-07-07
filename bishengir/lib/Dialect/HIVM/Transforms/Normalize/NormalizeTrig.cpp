//===-------- NormalizeTrig.cpp --------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/Normalize/NormalizeTrigTemplate.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
struct HIVMNormalizeHighPrecisionTrigTraitsBase : public hivm::NormalizeTraitsBase {
  static std::optional<Value> getCastInput(Value input) {
    auto castOp = input.getDefiningOp<hivm::VCastOp>();
    if (!castOp || !castOp.hasPureTensorSemantics() ||
        !castOp.getBroadcast().empty() || !castOp.getTranspose().empty()) {
      return std::nullopt;
    }
    return castOp.getSingleSrc();
  }

  static Value getUnaryInput(Operation *op) {
    if (auto recOp = dyn_cast<hivm::VRecOp>(op))
      return recOp.getSrc()[0];
    if (auto rsqrtOp = dyn_cast<hivm::VRsqrtOp>(op))
      return rsqrtOp.getSrc()[0];
    if (auto sqrtOp = dyn_cast<hivm::VSqrtOp>(op))
      return sqrtOp.getSrc()[0];
    llvm_unreachable("unsupported high-precision trig producer");
  }

  static bool hasSupportedHighPrecisionProducerSemantics(Operation *op) {
    if (auto recOp = dyn_cast_or_null<hivm::VRecOp>(op)) {
      return recOp.hasPureTensorSemantics() && recOp.getBroadcast().empty() &&
             recOp.getTranspose().empty();
    }
    if (auto rsqrtOp = dyn_cast_or_null<hivm::VRsqrtOp>(op)) {
      return rsqrtOp.hasPureTensorSemantics() &&
             rsqrtOp.getBroadcast().empty() && rsqrtOp.getTranspose().empty();
    }
    if (auto sqrtOp = dyn_cast_or_null<hivm::VSqrtOp>(op)) {
      return sqrtOp.hasPureTensorSemantics() && sqrtOp.getBroadcast().empty() &&
             sqrtOp.getTranspose().empty();
    }
    return false;
  }

  static bool isHighPrecisionProducer(Operation *op) {
    return hasSupportedHighPrecisionProducerSemantics(op);
  }

  static bool isHighPrecisionRsqrt(Operation *op) {
    return hasSupportedHighPrecisionProducerSemantics(op) &&
           isa<hivm::VRsqrtOp>(op);
  }
};

struct HIVMNormalizeSinTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeSin(hivm::VSinOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HIVMNormalizeHighPrecisionSinTraits
    : public HIVMNormalizeHighPrecisionTrigTraitsBase {
public:
  static bool shouldNormalizeHighPrecisionSin(hivm::VSinOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }

  static FailureOr<Value>
  buildHighPrecisionSinOrCos(PatternRewriter &rewriter, hivm::VSinOp op,
                             Value input, HighPrecisionTrigMode mode) {
    return mlir::buildHighPrecisionSinOrCos<HIVMNormalizeHighPrecisionSinTraits>(
        rewriter, op, input, mode);
  }
};

struct HIVMNormalizeCosTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeCos(hivm::VCosOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HIVMNormalizeHighPrecisionCosTraits
    : public HIVMNormalizeHighPrecisionTrigTraitsBase {
public:
  static bool shouldNormalizeHighPrecisionCos(hivm::VCosOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }

  static FailureOr<Value>
  buildHighPrecisionSinOrCos(PatternRewriter &rewriter, hivm::VCosOp op,
                             Value input, HighPrecisionTrigMode mode) {
    return mlir::buildHighPrecisionSinOrCos<HIVMNormalizeHighPrecisionCosTraits>(
        rewriter, op, input, mode);
  }
};

struct HIVMNormalizeAtanTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeAtan(hivm::VAtanOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HIVMNormalizeTanTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeTan(hivm::VTanOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

struct HIVMNormalizeTanhTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeTanh(hivm::VTanhOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }
    Type inputType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    return inputType.isF16() || inputType.isF32();
  }
};

using NormalizeSinOp =
    NormalizeSinOpTemplate<hivm::VSinOp, HIVMNormalizeSinTraits>;
using NormalizeCosOp =
    NormalizeCosOpTemplate<hivm::VCosOp, HIVMNormalizeCosTraits>;
using HighPrecisionNormalizeSinOp =
    HighPrecisionNormalizeSinOpTemplate<hivm::VSinOp, HIVMNormalizeHighPrecisionSinTraits>;
using HighPrecisionNormalizeCosOp =
    HighPrecisionNormalizeCosOpTemplate<hivm::VCosOp, HIVMNormalizeHighPrecisionCosTraits>;
using NormalizeAtanOp =
    NormalizeAtanOpTemplate<hivm::VAtanOp, HIVMNormalizeAtanTraits>;
using NormalizeTanOp =
    NormalizeTanOpTemplate<hivm::VTanOp, HIVMNormalizeTanTraits>;
using NormalizeTanhOp =
    NormalizeTanhOpTemplate<hivm::VTanhOp, HIVMNormalizeTanhTraits>;
} // namespace mlir

void mlir::hivm::populateNormalizeTrigPatterns(RewritePatternSet &patterns,
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
