//===- HIVMTransformOps.cpp - Impl. of HIVM transform ops -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

#define DEBUG_TYPE "hivm-transform-ops"

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
