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
using NormalizeAtanOp =
    NormalizeAtanOpTemplate<hivm::VAtanOp, HIVMNormalizeAtanTraits>;
using NormalizeTanOp =
    NormalizeTanOpTemplate<hivm::VTanOp, HIVMNormalizeTanTraits>;
using NormalizeTanhOp =
    NormalizeTanhOpTemplate<hivm::VTanhOp, HIVMNormalizeTanhTraits>;
} // namespace mlir

void mlir::hivm::populateNormalizeTrigPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeSinOp>(patterns.getContext());
  patterns.add<NormalizeCosOp>(patterns.getContext());
  patterns.add<NormalizeAtanOp>(patterns.getContext());
  patterns.add<NormalizeTanOp>(patterns.getContext());
  patterns.add<NormalizeTanhOp>(patterns.getContext());
}
