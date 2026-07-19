//===- NormalizeAtomic.cpp -------------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeAtomicTemplate.h"

namespace mlir::hivm {
namespace NormalizeAtomicOps {
struct HIVMAtomicTraits : public NormalizeTraitsBase {
  static bool isHardwareSupported(Type dtype) {
    Type elemType = getElementTypeOrSelf(dtype);
    if (isa<Float16Type, Float32Type, BFloat16Type>(elemType))
      return true;

    auto intType = dyn_cast<IntegerType>(elemType);
    if (!intType)
      return false;

    unsigned width = intType.getWidth();
    return width == 8 || width == 16 || width == 32;
  }

  static bool shouldDecomposeStore(hivm::StoreOp op) {
    return getStoreDecompositionBinaryKind(op).has_value();
  }

  static std::optional<BinaryKind>
  getStoreDecompositionBinaryKind(hivm::StoreOp op) {
    if (!op.isAtomic())
      return std::nullopt;

    Type elemType = getElementTypeOrSelf(op.getDstOperandType());
    AtomicKind atomicKind = *op.getAtomicKind();
    if (auto kind =
            mapAlwaysSoftwareAtomicKindToBinary(atomicKind))
      return kind;
    if (!isHardwareSupported(elemType))
      return mapNativeUnsupportedAtomicKindToBinary(
          atomicKind);
    return std::nullopt;
  }

  static FailureOr<Value> createStoreBinary(PatternRewriter &rewriter,
                                            Location loc, hivm::StoreOp op,
                                            Value lhsTensor, Value rhsTensor,
                                            Value resultTensor) {
    auto maybeKind = getStoreDecompositionBinaryKind(op);
    if (!maybeKind.has_value())
      return failure();
    return NormalizeTraitsBase::createBinaryOp(
        rewriter, loc, lhsTensor, rhsTensor, resultTensor, *maybeKind);
  }
};

using Elemwise =
    NormalizeAtomicStoreElemwise<hivm::StoreOp, HIVMAtomicTraits>;
using CAS = NormalizeAtomicCASTemplate<hivm::AtomicCasOp, HIVMAtomicTraits>;
using XCHG = NormalizeAtomicXCHGTemplate<hivm::AtomicXchgOp, HIVMAtomicTraits>;

} // namespace NormalizeAtomicOps

void populateNormalizeAtomicPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (archisAscend950) {
    patterns.add<NormalizeAtomicOps::Elemwise>(ctx);
    patterns.add<NormalizeAtomicOps::CAS>(ctx);
    patterns.add<NormalizeAtomicOps::XCHG>(ctx);
  }
}

} // namespace mlir::hivm
