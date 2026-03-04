//====- Transforms.h - Transform Extend Fuse Into ContainingOp ---*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

#ifndef BISHENGIR_TRANSFORMS_TRANSFORMS_H
#define BISHENGIR_TRANSFORMS_TRANSFORMS_H

namespace bishengir {
void unionProducerUsers(mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
                        mlir::Operation *producerOp,
                        mlir::Operation *containingOp);
std::tuple<llvm::SmallVector<mlir::Operation *>, mlir::Operation *>

tileAndFuseFirstExtractUse(mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
                           mlir::Operation *producerOp,
                           mlir::Operation *containingOp,
                           bool duplicateProducer);
llvm::SmallVector<mlir::Operation *>

tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
    mlir::Operation *producerOp, mlir::Operation *containingOp);

mlir::Operation *cloneAndFuseFirstUse(mlir::RewriterBase &rewriter,
                                      mlir::Diagnostic &diag,
                                      mlir::Operation *producerOp,
                                      mlir::Operation *containingOp);

void normalizeLoop(mlir::RewriterBase &rewriter, mlir::scf::ForOp op,
                   mlir::Value oldStep);
} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_TRANSFORMS_H