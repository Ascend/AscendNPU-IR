//===- HIVMTransformOps.cpp - Impl. of HIVM transform ops -------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

DiagnosedSilenceableFailure transform::MapForallToHIVMBlocks::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  auto forAll = dyn_cast<scf::ForallOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());
  if (!forAll)
    return emitDefiniteFailure() << "expect an scf.forall";

  ForallRewriteResult rewriteResult;
  auto diag =
      mapForallToBlocksImpl(rewriter, forAll, rewriteResult, transformOp);
  if (!diag.succeeded())
    return diag;

  results.push_back(rewriteResult.mappingId.getDefiningOp());
  return diag;
}

void MapForallToHIVMBlocks::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class HIVMTransformDialectExtension
    : public transform::TransformDialectExtension<
          HIVMTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<hivm::HIVMDialect>();
    declareDependentDialect<scf::SCFDialect>();

    declareGeneratedDialect<hivm::HIVMDialect>();
    declareGeneratedDialect<scope::ScopeDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<math::MathDialect>();
    declareGeneratedDialect<mathExt::MathExtDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    declareGeneratedDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.cpp.inc"

void mlir::hivm::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<HIVMTransformDialectExtension>();
}
